import math
import numpy as np
from typing import Tuple, Optional, List

import torch
from torch import Tensor
import torch.nn.functional as F

try:
    from tqdm import trange
except Exception:
    def trange(n, **kwargs):
        return range(n)


def _to_device(x: np.ndarray, device: torch.device, dtype=torch.float32) -> Tensor:
    return torch.as_tensor(x, device=device, dtype=dtype)


def _as_numpy(x: Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


class MiniSom:
    """
    PyTorch GPU版 SOM（batchSOM）
      - activation_distance:
          'euclidean'（ユークリッド）
          'ssim5'    （論文仕様に近い 5x5 窓・C=0 のSSIM：移動窓平均）
          's1'       （Teweles–Wobus S1）
          's1ssim'   （S1とSSIM(5x5)の融合距離：サンプル毎min-max正規化後の等重み和）
          's1ssim5_hf'（HF-S1SSIM5: SSIM(5x5)でゲートするソフト階層化 D = u + (1-u)v）
          's1ssim5_and'（AND合成: 行方向min–max正規化後の D = max(U,V)）
          'pf_s1ssim'（比例融合: 正規化なしで D = dS1 * dSSIM）
      - 学習は「ミニバッチ版バッチSOM」：BMU→近傍重み→分子/分母累積→一括更新
      - 全ての重い計算はGPU実行
      - σ（近傍幅）は学習全体で一方向に減衰させる（セグメント学習でも継続）
      - 任意頻度で“距離一貫性”のためのメドイド置換（ノード重みを最近傍サンプルへ置換）を実行可
    """
    def __init__(self,
                 x: int,
                 y: int,
                 input_len: int,
                 sigma: float = 1.0,
                 learning_rate: float = 0.5,
                 neighborhood_function: str = 'gaussian',
                 topology: str = 'rectangular',
                 activation_distance: str = 's1',
                 random_seed: Optional[int] = None,
                 sigma_decay: str = 'asymptotic_decay',
                 s1_field_shape: Optional[Tuple[int, int]] = None,
                 device: Optional[str] = None,
                 dtype: torch.dtype = torch.float32,
                 nodes_chunk: int = 16,
                 area_weight: Optional[np.ndarray] = None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.dtype = dtype

        if random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)

        self.x = x
        self.y = y
        self.m = x * y
        self.input_len = input_len
        self.sigma0 = float(sigma)
        self.learning_rate = float(learning_rate)
        self.neighborhood_function = 'gaussian'
        self.topology = topology
        self.sigma_decay = sigma_decay
        self.nodes_chunk = int(nodes_chunk)

        # 学習全体の反復管理（σ継続減衰用）
        self.global_iter: int = 0
        self.total_iters: Optional[int] = None

        # メドイド置換頻度（None: 不使用, k: k反復ごと）
        self.medoid_replace_every: Optional[int] = None

        # 評価用の固定サンプルインデックス（QEを安定化）
        self.eval_indices: Optional[Tensor] = None

        # 距離タイプ
        activation_distance = activation_distance.lower()
        if activation_distance not in ('s1', 'euclidean', 'ssim5', 's1ssim', 's1ssim5_hf', 's1ssim5_and', 'pf_s1ssim', 's1gssim', 'gssim', 's1gl', 'gsmd', 's3d', 'cfsd', 'hff', 's1gk', 's1gcurv', 'ms_s1', 'msssim_s1g', 'spot', 'gvd', 'itcs'):
            raise ValueError('activation_distance must be one of "s1","euclidean","ssim5","s1ssim","s1ssim5_hf","s1ssim5_and","pf_s1ssim","s1gssim","gssim","s1gl","gsmd","s3d","cfsd","hff","s1gk","s1gcurv","ms_s1","msssim_s1g","spot","gvd","itcs"')
        self.activation_distance = activation_distance

        # 画像形状
        if s1_field_shape is None:
            raise ValueError('s1_field_shape=(H,W) is required for all distances in this implementation.')
        if s1_field_shape[0] * s1_field_shape[1] != input_len:
            raise ValueError(f's1_field_shape={s1_field_shape} does not match input_len={input_len}.')
        self.field_shape = s1_field_shape
        H, W = s1_field_shape
        # Optional area weight (e.g., cos(lat)); used for curvature S1 weighting on Laplacian domain
        self.area_w: Optional[Tensor] = None
        if area_weight is not None:
            aw = torch.as_tensor(area_weight, device=self.device, dtype=self.dtype)
            if aw.shape != (H, W):
                raise ValueError(f'area_weight shape {aw.shape} does not match field_shape {(H, W)}')
            self.area_w = aw

        # グリッド座標
        gx, gy = torch.meshgrid(torch.arange(x), torch.arange(y), indexing='ij')
        self.grid_coords = torch.stack([gx.flatten(), gy.flatten()], dim=1).to(self.device, torch.float32)

        # 重み（(m,H,W)）
        self.weights = (torch.rand((self.m, H, W), device=self.device, dtype=self.dtype) * 2 - 1)

        if self.neighborhood_function != 'gaussian':
            self.neighborhood_function = 'gaussian'


        # 5x5移動窓SSIM用カーネル（平均フィルタ）
        self._kernel5: Optional[Tensor] = None
        self._win5_size: int = 5
        self._win5_pad: int = 2
        # 3x3ラプラシアン（曲率）用カーネル
        self._lap_kernel: Optional[Tensor] = None
        # 正規化座標グリッド（0..1）
        self._xnorm = torch.linspace(0.0, 1.0, W, device=self.device, dtype=self.dtype).view(1, 1, W).expand(1, H, W)
        self._ynorm = torch.linspace(0.0, 1.0, H, device=self.device, dtype=self.dtype).view(1, H, 1).expand(1, H, W)
        # Normalized coordinates for projection-based distances (e.g., SPOT)
        self._xgrid = torch.linspace(0.0, 1.0, W, device=self.device, dtype=self.dtype).view(1, 1, W).expand(1, H, W)
        self._ygrid = torch.linspace(0.0, 1.0, H, device=self.device, dtype=self.dtype).view(1, H, 1).expand(1, H, W)

    # ---------- 外部制御 ----------
    def set_total_iterations(self, total_iters: int):
        """学習全体の反復回数を設定（σ減衰の基準）。複数回train_batchを呼ぶ前に設定してください。"""
        self.total_iters = int(total_iters)

    def set_medoid_replace_every(self, k: Optional[int]):
        """k反復ごとにメドイド置換（各ノード重みを距離的に最も近いサンプルへ置換）を行う。Noneまたは0で無効。"""
        if k is None or k <= 0:
            self.medoid_replace_every = None
        else:
            self.medoid_replace_every = int(k)

    def set_eval_indices(self, idx: Optional[np.ndarray]):
        """
        評価（quantization_error/predict等の固定評価で使用）用のインデックスを設定。
        Noneで解除。idxはデータ配列に対する行インデックス。
        """
        if idx is None:
            self.eval_indices = None
        else:
            self.eval_indices = torch.as_tensor(idx, device=self.device, dtype=torch.long)

    # ---------- ユーティリティ ----------
    def get_weights(self) -> np.ndarray:
        H, W = self.field_shape
        w_flat = self.weights.reshape(self.m, H * W)
        w_grid = w_flat.reshape(self.x, self.y, H * W)
        return _as_numpy(w_grid)

    def random_weights_init(self, data: np.ndarray):
        H, W = self.field_shape
        n = data.shape[0]
        if n < self.m:
            idx = np.random.choice(n, self.m, replace=True)
        else:
            idx = np.random.choice(n, self.m, replace=False)
        w0 = data[idx].reshape(self.m, H, W)
        self.weights = _to_device(w0, self.device, self.dtype).clone()

    # ---------- スケジューラ ----------
    def _sigma_at_val(self, t: int, max_iter: int) -> float:
        if self.sigma_decay == 'asymptotic_decay':
            return self.sigma0 / (1 + t / (max_iter / 2.0))
        elif self.sigma_decay == 'linear_decay':
            return max(1e-3, self.sigma0 * (1 - t / max_iter))
        else:
            return self.sigma0 / (1 + t / (max_iter / 2.0))

    # ---------- 近傍関数 ----------
    @torch.no_grad()
    def _neighborhood(self, bmu_flat: Tensor, sigma: float) -> Tensor:
        bmu_xy = self.grid_coords[bmu_flat]  # (B,2)
        d2 = ((bmu_xy.unsqueeze(1) - self.grid_coords.unsqueeze(0)) ** 2).sum(dim=-1)  # (B,m)
        h = torch.exp(-d2 / (2 * (sigma ** 2) + 1e-9))
        return h

    # ---------- 距離計算（バッチ→全ノード） ----------
    @torch.no_grad()
    def _euclidean_distance_batch(self, Xb: Tensor) -> Tensor:
        """
        Xb: (B,H,W) -> 距離 (B,m)
        d^2 = sum((X-W)^2), 戻りは sqrt(d^2)（単調変換）
        """
        B, H, W = Xb.shape
        Xf = Xb.reshape(B, -1)                # (B,D)
        Wf = self.weights.reshape(self.m, -1) # (m,D)
        x2 = (Xf * Xf).sum(dim=1, keepdim=True)         # (B,1)
        w2 = (Wf * Wf).sum(dim=1, keepdim=True).T       # (1,m)
        cross = Xf @ Wf.T                                # (B,m)
        d2 = x2 + w2 - 2 * cross
        d2 = torch.clamp(d2, min=0.0)
        return torch.sqrt(d2 + 1e-12)

    @torch.no_grad()
    def _ensure_kernel5(self):
        if self._kernel5 is None:
            k = torch.ones((1, 1, self._win5_size, self._win5_size), device=self.device, dtype=self.dtype) / float(self._win5_size * self._win5_size)
            self._kernel5 = k

    @torch.no_grad()
    def _ensure_lap_kernel(self):
        if self._lap_kernel is None:
            k = torch.tensor([[0.0, 1.0, 0.0],
                              [1.0,-4.0, 1.0],
                              [0.0, 1.0, 0.0]], device=self.device, dtype=self.dtype)
            self._lap_kernel = k.view(1, 1, 3, 3)

    @torch.no_grad()
    def _ssim5_distance_batch(self, Xb: Tensor, nodes_chunk: Optional[int] = None) -> Tensor:
        """
        論文仕様に近いSSIM: 5x5移動窓・C=0（分母のみ数値安定化）
        Xb: (B,H,W)
        戻り: (B,m) の "距離" = 1 - mean(SSIM_map)
        """
        if nodes_chunk is None:
            nodes_chunk = self.nodes_chunk
        self._ensure_kernel5()
        eps = 1e-12
        B, H, W = Xb.shape
        out = torch.empty((B, self.m), device=Xb.device, dtype=self.dtype)

        # X 側のローカル統計
        X = Xb.unsqueeze(1)  # (B,1,H,W)
        X_pad = F.pad(X, (self._win5_pad, self._win5_pad, self._win5_pad, self._win5_pad), mode='reflect')
        mu_x = F.conv2d(X_pad, self._kernel5, padding=0)                      # (B,1,H,W)
        mu_x2 = F.conv2d(X_pad * X_pad, self._kernel5, padding=0)             # (B,1,H,W)
        var_x = torch.clamp(mu_x2 - mu_x * mu_x, min=0.0)                     # (B,1,H,W)

        for start in range(0, self.m, nodes_chunk):
            end = min(start + nodes_chunk, self.m)
            Wc = self.weights[start:end].unsqueeze(1)                          # (Mc,1,H,W)
            W_pad = F.pad(Wc, (self._win5_pad, self._win5_pad, self._win5_pad, self._win5_pad), mode='reflect')
            mu_w = F.conv2d(W_pad, self._kernel5, padding=0)                  # (Mc,1,H,W)
            mu_w2 = F.conv2d(W_pad * W_pad, self._kernel5, padding=0)         # (Mc,1,H,W)
            var_w = torch.clamp(mu_w2 - mu_w * mu_w, min=0.0)                 # (Mc,1,H,W)

            # 共分散: mean(x*w) - mu_x*mu_w
            prod = (X.unsqueeze(1) * Wc.unsqueeze(0)).reshape(B * (end - start), 1, H, W)  # (B*Mc,1,H,W)
            prod_pad = F.pad(prod, (self._win5_pad, self._win5_pad, self._win5_pad, self._win5_pad), mode='reflect')
            mu_xw = F.conv2d(prod_pad, self._kernel5, padding=0).reshape(B, end - start, 1, H, W)  # (B,Mc,1,H,W)

            mu_x_b = mu_x.unsqueeze(1)                         # (B,1,1,H,W)
            mu_w_mc = mu_w.unsqueeze(0)                        # (1,Mc,1,H,W)
            var_x_b = var_x.unsqueeze(1)                       # (B,1,1,H,W)
            var_w_mc = var_w.unsqueeze(0)                      # (1,Mc,1,H,W)
            cov = mu_xw - (mu_x_b * mu_w_mc)                   # (B,Mc,1,H,W)

            # SSIMマップ（C1=C2=0だが分母にのみepsガード）
            l_num = 2 * (mu_x_b * mu_w_mc)
            l_den = (mu_x_b * mu_x_b + mu_w_mc * mu_w_mc)
            c_num = 2 * cov
            c_den = (var_x_b + var_w_mc)
            ssim_map = (l_num * c_num) / (l_den * c_den + eps)               # (B,Mc,1,H,W)

            # 空間平均
            ssim_avg = ssim_map.mean(dim=(2, 3, 4))                          # (B,Mc)
            out[:, start:end] = 1.0 - ssim_avg

        return out

    @torch.no_grad()
    def _adaptive_pool(self, T: Tensor, s: int) -> Tensor:
        # Adaptive average pool to roughly downscale by s (handles non-divisible sizes)
        H = T.shape[-2]
        W = T.shape[-1]
        h2 = max(2, max(1, H // s))
        w2 = max(2, max(1, W // s))
        return F.adaptive_avg_pool2d(T, (h2, w2))

    @torch.no_grad()
    def _s1_distance_batch_on(self, Xb: Tensor, Wc: Tensor, nodes_chunk: Optional[int] = None) -> Tensor:
        # S1 distance computed for given X and W on same resolution
        if nodes_chunk is None:
            nodes_chunk = self.nodes_chunk
        eps = 1e-12
        B, H, W = Xb.shape
        out = torch.empty((B, Wc.shape[0]), device=Xb.device, dtype=self.dtype)
        dXdx = Xb[:, :, 1:] - Xb[:, :, :-1]
        dXdy = Xb[:, 1:, :] - Xb[:, :-1, :]
        dWdx_full = Wc[:, :, 1:] - Wc[:, :, :-1]
        dWdy_full = Wc[:, 1:, :] - Wc[:, :-1, :]
        for start in range(0, Wc.shape[0], nodes_chunk):
            end = min(start + nodes_chunk, Wc.shape[0])
            dWdx = dWdx_full[start:end]
            dWdy = dWdy_full[start:end]
            num = (torch.abs(dWdx.unsqueeze(0) - dXdx.unsqueeze(1)).sum(dim=(2, 3)) +
                   torch.abs(dWdy.unsqueeze(0) - dXdy.unsqueeze(1)).sum(dim=(2, 3)))
            den = (torch.maximum(torch.abs(dWdx).unsqueeze(0), torch.abs(dXdx).unsqueeze(1)).sum(dim=(2, 3)) +
                   torch.maximum(torch.abs(dWdy).unsqueeze(0), torch.abs(dXdy).unsqueeze(1)).sum(dim=(2, 3)))
            out[:, start:end] = 100.0 * num / (den + eps)
        return out

    @torch.no_grad()
    def _ms_s1_distance_batch(self, Xb: Tensor, nodes_chunk: Optional[int] = None) -> Tensor:
        # Multi-scale S1 with row-wise min-max normalization per scale and RMS fusion
        if nodes_chunk is None:
            nodes_chunk = self.nodes_chunk
        scales = [1, 2, 4]
        weights = torch.tensor([0.5, 0.3, 0.2], device=Xb.device, dtype=self.dtype)
        weights = weights / weights.sum()
        B, H, W = Xb.shape
        # prepare pooled weights per scale
        W_all = self.weights  # (m,H,W)
        dists = []
        for s in scales:
            if s == 1:
                Xs = Xb
                Ws = W_all
            else:
                Xs = self._adaptive_pool(Xb.unsqueeze(1), s).squeeze(1)
                Ws = self._adaptive_pool(W_all.unsqueeze(1), s).squeeze(1)
            D = self._s1_distance_batch_on(Xs, Ws, nodes_chunk=nodes_chunk)  # (B,m)
            # row-wise min-max normalize per batch row
            eps = 1e-12
            dmin = D.min(dim=1, keepdim=True).values
            dmax = D.max(dim=1, keepdim=True).values
            Dn = (D - dmin) / (dmax - dmin + eps)
            dists.append(Dn)
        # weighted RMS
        out = torch.zeros_like(dists[0])
        for i, Dn in enumerate(dists):
            out = out + weights[i] * (Dn * Dn)
        return torch.sqrt(out + 1e-12)

    @torch.no_grad()
    def _ssim5_mod_distance_batch_multiscale(self, Xb: Tensor, nodes_chunk: Optional[int] = None, scales: Optional[List[int]] = None, c1: float = 1e-8, c2: float = 1e-8) -> Tensor:
        # Multi-scale "modified" SSIM (small constants) distance: 1 - MSSSIM*
        if nodes_chunk is None:
            nodes_chunk = self.nodes_chunk
        if scales is None:
            scales = [1, 2, 4]
        self._ensure_kernel5()
        B, H, W = Xb.shape
        sims = []
        weights = torch.tensor([0.5, 0.3, 0.2], device=Xb.device, dtype=self.dtype)
        weights = weights / weights.sum()
        for s in scales:
            if s == 1:
                Xs = Xb.unsqueeze(1)  # (B,1,H,W)
                Wc = self.weights.unsqueeze(1)  # (m,1,H,W)
            else:
                Xs = self._adaptive_pool(Xb.unsqueeze(1), s)
                Wc = self._adaptive_pool(self.weights.unsqueeze(1), s)
            # local stats
            pad = 2
            X_pad = F.pad(Xs, (pad, pad, pad, pad), mode='reflect')
            mu_x = F.conv2d(X_pad, self._kernel5, padding=0)
            mu_x2 = F.conv2d(X_pad * X_pad, self._kernel5, padding=0)
            var_x = torch.clamp(mu_x2 - mu_x * mu_x, min=0.0)
            # chunk nodes
            B1, _, Hs, Ws = Xs.shape
            sim_chunk = torch.empty((B1, self.m), device=Xb.device, dtype=self.dtype)
            for start in range(0, self.m, nodes_chunk):
                end = min(start + nodes_chunk, self.m)
                Wc_sub = Wc[start:end]  # (Mc,1,Hs,Ws)
                W_pad = F.pad(Wc_sub, (pad, pad, pad, pad), mode='reflect')
                mu_w = F.conv2d(W_pad, self._kernel5, padding=0)
                mu_w2 = F.conv2d(W_pad * W_pad, self._kernel5, padding=0)
                var_w = torch.clamp(mu_w2 - mu_w * mu_w, min=0.0)
                # covariance
                prod = (Xs.unsqueeze(1) * Wc_sub.unsqueeze(0)).reshape(B1 * (end - start), 1, Hs, Ws)
                prod_pad = F.pad(prod, (pad, pad, pad, pad), mode='reflect')
                mu_xw = F.conv2d(prod_pad, self._kernel5, padding=0).reshape(B1, end - start, 1, Hs, Ws)
                mu_x_b = mu_x.unsqueeze(1)
                mu_w_mc = mu_w.unsqueeze(0)
                var_x_b = var_x.unsqueeze(1)
                var_w_mc = var_w.unsqueeze(0)
                cov = mu_xw - (mu_x_b * mu_w_mc)
                # modified SSIM with small constants
                l_num = 2 * (mu_x_b * mu_w_mc)
                l_den = (mu_x_b * mu_x_b + mu_w_mc * mu_w_mc) + c1
                c_num = 2 * cov
                c_den = (var_x_b + var_w_mc) + c2
                ssim_map = (l_num * c_num) / (l_den * c_den + 1e-12)
                ssim_avg = ssim_map.mean(dim=(2, 3, 4))  # (B,Mc)
                sim_chunk[:, start:end] = ssim_avg
            sims.append(sim_chunk)
        # weighted mean of similarities, distance = 1 - sim
        sim = torch.zeros_like(sims[0])
        for i, Si in enumerate(sims):
            sim = sim + weights[i] * Si
        return 1.0 - sim.clamp(-1.0, 1.0)

    @torch.no_grad()
    def _msssim_s1g_distance_batch(self, Xb: Tensor, nodes_chunk: Optional[int] = None) -> Tensor:
        # D = dL*^2 + (1 - dL*) * sqrt( (dG^2 + dS1n^2)/2 )
        if nodes_chunk is None:
            nodes_chunk = self.nodes_chunk
        dL = self._ssim5_mod_distance_batch_multiscale(Xb, nodes_chunk=nodes_chunk, scales=[1, 2, 4])  # (B,m)
        dG = self._gssim_distance_batch(Xb, nodes_chunk=nodes_chunk)  # (B,m)
        # S1 normalized row-wise
        dS1 = self._s1_distance_batch(Xb, nodes_chunk=nodes_chunk)
        eps = 1e-12
        dmin = dS1.min(dim=1, keepdim=True).values
        dmax = dS1.max(dim=1, keepdim=True).values
        dS1n = (dS1 - dmin) / (dmax - dmin + eps)
        core = torch.sqrt((dG * dG + dS1n * dS1n) / 2.0)
        return dL * dL + (1.0 - dL) * core


    @torch.no_grad()
    def _s1gssim_distance_batch(self, Xb: Tensor, nodes_chunk: Optional[int] = None) -> Tensor:
        """
        Gradient-SSIM(+direction) と S1(行方向min-max正規化) のRMS合成距離。
        1) 勾配強度 |∇| のSSIM(5x5, C=0) => d_g ∈ [0,1]
        2) 勾配方向cosθの重み付き平均（重み=max(|∇X|,|∇W|)）=> d_dir ∈ [0,1]
        3) d_edge = 0.5*(d_g + d_dir)
        4) d_s1 = 行方向min-max正規化したS1 ∈ [0,1]
        出力: sqrt((d_edge^2 + d_s1^2)/2)
        """
        if nodes_chunk is None:
            nodes_chunk = self.nodes_chunk
        eps = 1e-12
        B, H, W = Xb.shape

        # サンプル側の勾配と強度（共通領域 (H-1, W-1)）
        dXdx = Xb[:, :, 1:] - Xb[:, :, :-1]   # (B, H, W-1)
        dXdy = Xb[:, 1:, :] - Xb[:, :-1, :]   # (B, H-1, W)
        dXdx_c = dXdx[:, :-1, :]              # (B, H-1, W-1)
        dXdy_c = dXdy[:, :, :-1]              # (B, H-1, W-1)
        magX = torch.sqrt(dXdx_c * dXdx_c + dXdy_c * dXdy_c + eps)  # (B, H-1, W-1)

        # 勾配強度のローカル統計（5x5平均畳み込み）
        self._ensure_kernel5()
        Xg = magX.unsqueeze(1)  # (B,1,H-1,W-1)
        Xg_pad = F.pad(Xg, (2, 2, 2, 2), mode='reflect')
        mu_xg = F.conv2d(Xg_pad, self._kernel5, padding=0)                  # (B,1,H-1,W-1)
        mu_xg2 = F.conv2d(Xg_pad * Xg_pad, self._kernel5, padding=0)
        var_xg = torch.clamp(mu_xg2 - mu_xg * mu_xg, min=0.0)

        # ノード側の勾配（事前計算）
        dWdx_full = self.weights[:, :, 1:] - self.weights[:, :, :-1]  # (m, H, W-1)
        dWdy_full = self.weights[:, 1:, :] - self.weights[:, :-1, :]  # (m, H-1, W)

        # S1全体を先に計算して行方向min-max正規化用のmin/maxを得る
        dS1_all = self._s1_distance_batch(Xb, nodes_chunk=self.nodes_chunk)  # (B, m)
        min_s1 = dS1_all.min(dim=1, keepdim=True).values
        max_s1 = dS1_all.max(dim=1, keepdim=True).values

        out = torch.empty((B, self.m), device=Xb.device, dtype=self.dtype)
        for start in range(0, self.m, nodes_chunk):
            end = min(start + nodes_chunk, self.m)
            dWdx = dWdx_full[start:end]    # (Mc, H, W-1)
            dWdy = dWdy_full[start:end]    # (Mc, H-1, W)
            dWdx_c = dWdx[:, :-1, :]       # (Mc, H-1, W-1)
            dWdy_c = dWdy[:, :, :-1]       # (Mc, H-1, W-1)
            magW = torch.sqrt(dWdx_c * dWdx_c + dWdy_c * dWdy_c + eps)  # (Mc, H-1, W-1)

            # 1) 勾配強度のSSIM (C=0)
            Wg = magW.unsqueeze(1)                                         # (Mc,1,H-1,W-1)
            Wg_pad = F.pad(Wg, (2, 2, 2, 2), mode='reflect')
            mu_wg = F.conv2d(Wg_pad, self._kernel5, padding=0)             # (Mc,1,H-1,W-1)
            mu_wg2 = F.conv2d(Wg_pad * Wg_pad, self._kernel5, padding=0)
            var_wg = torch.clamp(mu_wg2 - mu_wg * mu_wg, min=0.0)

            # 修正: ブロードキャスト次元を合わせるため Xg にもノード次元を追加（(B,1,1,H-1,W-1) × (1,Mc,1,H-1,W-1)）
            prod = (Xg.unsqueeze(1) * Wg.unsqueeze(0)).reshape(B * (end - start), 1, magX.shape[1], magX.shape[2])
            prod_pad = F.pad(prod, (2, 2, 2, 2), mode='reflect')
            mu_xwg = F.conv2d(prod_pad, self._kernel5, padding=0).reshape(B, end - start, 1, magX.shape[1], magX.shape[2])

            mu_xg_b = mu_xg.unsqueeze(1)       # (B,1,1,H-1,W-1)
            mu_wg_mc = mu_wg.unsqueeze(0)      # (1,Mc,1,H-1,W-1)
            var_xg_b = var_xg.unsqueeze(1)
            var_wg_mc = var_wg.unsqueeze(0)
            l_num = 2.0 * (mu_xg_b * mu_wg_mc)
            l_den = (mu_xg_b * mu_xg_b + mu_wg_mc * mu_wg_mc)
            c_num = 2.0 * (mu_xwg - mu_xg_b * mu_wg_mc)
            c_den = (var_xg_b + var_wg_mc)
            ssim_map = (l_num * c_num) / (l_den * c_den + eps)             # (B,Mc,1,H-1,W-1)
            ssim_avg = ssim_map.mean(dim=(2, 3, 4))                        # (B,Mc)
            d_g = 0.5 * (1.0 - ssim_avg)                                   # (B,Mc) in [0,1]

            # 2) 勾配方向の一致（cosθの重み付き平均）
            magW2 = torch.sqrt(dWdx_c * dWdx_c + dWdy_c * dWdy_c + eps)    # (Mc, H-1, W-1)
            dot = dXdx_c.unsqueeze(1) * dWdx_c.unsqueeze(0) + dXdy_c.unsqueeze(1) * dWdy_c.unsqueeze(0)  # (B,Mc,H-1,W-1)
            denom = magX.unsqueeze(1) * magW2.unsqueeze(0) + eps
            cos = (dot / denom).clamp(-1.0, 1.0)
            wgt = torch.maximum(magX.unsqueeze(1), magW2.unsqueeze(0))
            s_dir = (cos * wgt).sum(dim=(2, 3)) / (wgt.sum(dim=(2, 3)) + eps)   # (B,Mc)
            d_dir = 0.5 * (1.0 - s_dir)                                         # (B,Mc) in [0,1]

            d_edge = 0.5 * (d_g + d_dir)                                        # (B,Mc)

            # 3) S1 を行方向min-max正規化し、RMS合成
            dS1n = (dS1_all[:, start:end] - min_s1) / (max_s1 - min_s1 + eps)   # (B,Mc)
            out[:, start:end] = torch.sqrt((d_edge * d_edge + dS1n * dS1n) / 2.0)

        return out

    @torch.no_grad()
    def _s1gl_distance_batch(self, Xb: Tensor, nodes_chunk: Optional[int] = None) -> Tensor:
        """
        DSGC: S1 + Gradient(+direction) + Curvature(Laplacian) Structural distance
        Dedge = 1 - (SSIM(|∇|) + Sdir)/2,  Dcurv = 1 - weighted SSIM(ΔX,ΔW),  dS1n = row-wise min-max normalized S1
        Output: sqrt((Dedge^2 + Dcurv^2 + dS1n^2)/3)
        """
        if nodes_chunk is None:
            nodes_chunk = self.nodes_chunk
        eps = 1e-12
        B, H, W = Xb.shape

        # Gradients (common inner H-1 x W-1 region)
        dXdx = Xb[:, :, 1:] - Xb[:, :, :-1]      # (B,H,W-1)
        dXdy = Xb[:, 1:, :] - Xb[:, :-1, :]      # (B,H-1,W)
        gx = dXdx[:, :-1, :]                     # (B,H-1,W-1)
        gy = dXdy[:, :, :-1]                     # (B,H-1,W-1)
        gmagX = torch.sqrt(gx * gx + gy * gy + eps)  # (B,H-1,W-1)

        # 5x5 stats for gradient magnitude (X side)
        self._ensure_kernel5()
        Xg = gmagX.unsqueeze(1)                      # (B,1,H-1,W-1)
        Xg_pad = F.pad(Xg, (2, 2, 2, 2), mode='reflect')
        mu_xg = F.conv2d(Xg_pad, self._kernel5, padding=0)
        mu_xg2 = F.conv2d(Xg_pad * Xg_pad, self._kernel5, padding=0)
        var_xg = torch.clamp(mu_xg2 - mu_xg * mu_xg, min=0.0)

        # S1 distances for all nodes for min-max normalization (per row)
        dS1_all = self._s1_distance_batch(Xb, nodes_chunk=self.nodes_chunk)  # (B,m)
        min_s1 = dS1_all.min(dim=1, keepdim=True).values
        max_s1 = dS1_all.max(dim=1, keepdim=True).values

        # Precompute node gradients
        dWdx_full = self.weights[:, :, 1:] - self.weights[:, :, :-1]  # (m,H,W-1)
        dWdy_full = self.weights[:, 1:, :] - self.weights[:, :-1, :]  # (m,H-1,W)
        grx_full = dWdx_full[:, :-1, :]                                # (m,H-1,W-1)
        gry_full = dWdy_full[:, :, :-1]                                # (m,H-1,W-1)
        gmagW_full = torch.sqrt(grx_full * grx_full + gry_full * gry_full + eps)  # (m,H-1,W-1)

        # Laplacians (curvature) for X and W
        self._ensure_lap_kernel()
        Lx = F.conv2d(Xb.unsqueeze(1), self._lap_kernel, padding=0).squeeze(1)  # (B,H-2,W-2)
        Lw_full = F.conv2d(self.weights.unsqueeze(1), self._lap_kernel, padding=0).squeeze(1)  # (m,H-2,W-2)

        # 5x5 stats for Lx (X side)
        Lx1 = Lx.unsqueeze(1)                         # (B,1,H-2,W-2)
        Lx_pad = F.pad(Lx1, (2, 2, 2, 2), mode='reflect')
        mu_lx = F.conv2d(Lx_pad, self._kernel5, padding=0)
        mu_lx2 = F.conv2d(Lx_pad * Lx_pad, self._kernel5, padding=0)
        var_lx = torch.clamp(mu_lx2 - mu_lx * mu_lx, min=0.0)

        out = torch.empty((B, self.m), device=Xb.device, dtype=self.dtype)

        for start in range(0, self.m, nodes_chunk):
            end = min(start + nodes_chunk, self.m)

            # Gradient direction cosine weighted by max(|∇X|,|∇W|)
            grx = grx_full[start:end]                  # (Mc,H-1,W-1)
            gry = gry_full[start:end]                  # (Mc,H-1,W-1)
            gmagW = gmagW_full[start:end]              # (Mc,H-1,W-1)
            cos = (gx.unsqueeze(1) * grx.unsqueeze(0) + gy.unsqueeze(1) * gry.unsqueeze(0)) / (gmagX.unsqueeze(1) * gmagW.unsqueeze(0) + eps)
            cos = torch.clamp(cos, -1.0, 1.0)
            w1 = torch.maximum(gmagX.unsqueeze(1), gmagW.unsqueeze(0))
            Sdir = ((0.5 * (1.0 + cos)) * w1).sum(dim=(2, 3)) / (w1.sum(dim=(2, 3)) + eps)  # (B,Mc)

            # Gradient magnitude SSIM (5x5, C=0)
            Wg = gmagW.unsqueeze(1)                    # (Mc,1,H-1,W-1)
            Wg_pad = F.pad(Wg, (2, 2, 2, 2), mode='reflect')
            mu_wg = F.conv2d(Wg_pad, self._kernel5, padding=0)
            mu_wg2 = F.conv2d(Wg_pad * Wg_pad, self._kernel5, padding=0)
            var_wg = torch.clamp(mu_wg2 - mu_wg * mu_wg, min=0.0)

            prod = (Xg.unsqueeze(1) * Wg.unsqueeze(0)).reshape(B * (end - start), 1, gmagX.shape[1], gmagX.shape[2])
            prod_pad = F.pad(prod, (2, 2, 2, 2), mode='reflect')
            mu_xwg = F.conv2d(prod_pad, self._kernel5, padding=0).reshape(B, end - start, 1, gmagX.shape[1], gmagX.shape[2])

            mu_xg_b = mu_xg.unsqueeze(1)
            mu_wg_mc = mu_wg.unsqueeze(0)
            var_xg_b = var_xg.unsqueeze(1)
            var_wg_mc = var_wg.unsqueeze(0)

            ssim_g_map = (2 * mu_xg_b * mu_wg_mc * 2 * (mu_xwg - mu_xg_b * mu_wg_mc)) / ((mu_xg_b * mu_xg_b + mu_wg_mc * mu_wg_mc) * (var_xg_b + var_wg_mc) + eps)
            SSIMg = ssim_g_map.mean(dim=(2, 3, 4))     # (B,Mc)

            Dedge = 1.0 - 0.5 * (SSIMg + Sdir)         # (B,Mc)

            # Curvature SSIM with weight w2 = max(|ΔX|,|ΔW|)
            Lw = Lw_full[start:end]                     # (Mc,H-2,W-2)
            Lw1 = Lw.unsqueeze(1)                       # (Mc,1,H-2,W-2)
            Lw_pad = F.pad(Lw1, (2, 2, 2, 2), mode='reflect')
            mu_lw = F.conv2d(Lw_pad, self._kernel5, padding=0)
            mu_lw2 = F.conv2d(Lw_pad * Lw_pad, self._kernel5, padding=0)
            var_lw = torch.clamp(mu_lw2 - mu_lw * mu_lw, min=0.0)

            prodL = (Lx1.unsqueeze(1) * Lw1.unsqueeze(0)).reshape(B * (end - start), 1, Lx.shape[1], Lx.shape[2])
            prodL_pad = F.pad(prodL, (2, 2, 2, 2), mode='reflect')
            mu_xlw = F.conv2d(prodL_pad, self._kernel5, padding=0).reshape(B, end - start, 1, Lx.shape[1], Lx.shape[2])

            mu_lx_b = mu_lx.unsqueeze(1)
            mu_lw_mc = mu_lw.unsqueeze(0)
            var_lx_b = var_lx.unsqueeze(1)
            var_lw_mc = var_lw.unsqueeze(0)

            ssim_l_map = (2 * mu_lx_b * mu_lw_mc * 2 * (mu_xlw - mu_lx_b * mu_lw_mc)) / ((mu_lx_b * mu_lx_b + mu_lw_mc * mu_lw_mc) * (var_lx_b + var_lw_mc) + eps)
            w2 = torch.maximum(torch.abs(Lx).unsqueeze(1), torch.abs(Lw).unsqueeze(0))  # (B,Mc,H-2,W-2)
            Scurv = (ssim_l_map.squeeze(2) * w2).sum(dim=(2, 3)) / (w2.sum(dim=(2, 3)) + eps)  # (B,Mc)
            Dcurv = 1.0 - Scurv

            # Normalized S1
            dS1n = (dS1_all[:, start:end] - min_s1) / (max_s1 - min_s1 + eps)

            out[:, start:end] = torch.sqrt((Dedge * Dedge + Dcurv * Dcurv + dS1n * dS1n) / 3.0)

        return out

    @torch.no_grad()
    def _s1norm_distance_batch(self, Xb: Tensor, nodes_chunk: Optional[int] = None) -> Tensor:
        """
        Dimensionless S1-like distance in [0,1]:
          r = (sum|∂X-∂W|) / (sum max(|∂X|,|∂W|) + eps) in both x and y, D = 0.5 * r.
        """
        if nodes_chunk is None:
            nodes_chunk = self.nodes_chunk
        eps = 1e-12
        B, H, W = Xb.shape
        dXdx = Xb[:, :, 1:] - Xb[:, :, :-1]  # (B,H,W-1)
        dXdy = Xb[:, 1:, :] - Xb[:, :-1, :]  # (B,H-1,W)
        out = torch.empty((B, self.m), device=Xb.device, dtype=self.dtype)
        dWdx_full = self.weights[:, :, 1:] - self.weights[:, :, :-1]  # (m,H,W-1)
        dWdy_full = self.weights[:, 1:, :] - self.weights[:, :-1, :]  # (m,H-1,W)
        for start in range(0, self.m, nodes_chunk):
            end = min(start + nodes_chunk, self.m)
            dWdx = dWdx_full[start:end]   # (Mc,H,W-1)
            dWdy = dWdy_full[start:end]   # (Mc,H-1,W)
            num = (torch.abs(dWdx.unsqueeze(0) - dXdx.unsqueeze(1)).sum(dim=(2, 3)) +
                   torch.abs(dWdy.unsqueeze(0) - dXdy.unsqueeze(1)).sum(dim=(2, 3)))
            den = (torch.maximum(torch.abs(dWdx).unsqueeze(0), torch.abs(dXdx).unsqueeze(1)).sum(dim=(2, 3)) +
                   torch.maximum(torch.abs(dWdy).unsqueeze(0), torch.abs(dXdy).unsqueeze(1)).sum(dim=(2, 3)))
            r = num / (den + eps)  # 0..2
            out[:, start:end] = 0.5 * r
        return out

    @torch.no_grad()
    def _moment_distance_batch(self, Xb: Tensor, nodes_chunk: Optional[int] = None) -> Tensor:
        """
        Moment-based distance in [0,1]: D_mom = 0.5 * (d_pos + d_neg), centroids on normalized coords,
        with fallback to (0.5,0.5) when mass is zero.
        """
        if nodes_chunk is None:
            nodes_chunk = self.nodes_chunk
        eps = 1e-12
        B, H, W = Xb.shape
        Xp = torch.clamp(Xb, min=0.0)    # (B,H,W)
        Xn = torch.clamp(-Xb, min=0.0)   # (B,H,W)

        mp_b = Xp.sum(dim=(1, 2), keepdim=True)  # (B,1)
        mn_b = Xn.sum(dim=(1, 2), keepdim=True)  # (B,1)

        cxp_b = (Xp * self._xnorm).sum(dim=(1, 2), keepdim=True) / (mp_b + eps)
        cyp_b = (Xp * self._ynorm).sum(dim=(1, 2), keepdim=True) / (mp_b + eps)
        cxn_b = (Xn * self._xnorm).sum(dim=(1, 2), keepdim=True) / (mn_b + eps)
        cyn_b = (Xn * self._ynorm).sum(dim=(1, 2), keepdim=True) / (mn_b + eps)

        # fallback to center for zero mass
        mask_mpz = (mp_b <= eps)
        mask_mnz = (mn_b <= eps)
        cxp_b[mask_mpz] = 0.5; cyp_b[mask_mpz] = 0.5
        cxn_b[mask_mnz] = 0.5; cyn_b[mask_mnz] = 0.5

        cxp_b = cxp_b.squeeze(-1).squeeze(-1); cyp_b = cyp_b.squeeze(-1).squeeze(-1)  # (B,)
        cxn_b = cxn_b.squeeze(-1).squeeze(-1); cyn_b = cyn_b.squeeze(-1).squeeze(-1)

        out = torch.empty((B, self.m), device=Xb.device, dtype=self.dtype)
        d2norm = math.sqrt(2.0)

        for start in range(0, self.m, nodes_chunk):
            end = min(start + nodes_chunk, self.m)
            Wc = self.weights[start:end]          # (Mc,H,W)
            Wp = torch.clamp(Wc, min=0.0)
            Wn = torch.clamp(-Wc, min=0.0)

            mp = Wp.sum(dim=(1, 2), keepdim=True)  # (Mc,1)
            mn = Wn.sum(dim=(1, 2), keepdim=True)  # (Mc,1)

            cxp = (Wp * self._xnorm).sum(dim=(1, 2), keepdim=True) / (mp + eps)
            cyp = (Wp * self._ynorm).sum(dim=(1, 2), keepdim=True) / (mp + eps)
            cxn = (Wn * self._xnorm).sum(dim=(1, 2), keepdim=True) / (mn + eps)
            cyn = (Wn * self._ynorm).sum(dim=(1, 2), keepdim=True) / (mn + eps)

            mask_mp0 = (mp <= eps)
            mask_mn0 = (mn <= eps)
            cxp[mask_mp0] = 0.5; cyp[mask_mp0] = 0.5
            cxn[mask_mn0] = 0.5; cyn[mask_mn0] = 0.5

            cxp = cxp.squeeze(-1).squeeze(-1); cyp = cyp.squeeze(-1).squeeze(-1)  # (Mc,)
            cxn = cxn.squeeze(-1).squeeze(-1); cyn = cyn.squeeze(-1).squeeze(-1)

            dpos = torch.sqrt((cxp_b.unsqueeze(1) - cxp.unsqueeze(0))**2 + (cyp_b.unsqueeze(1) - cyp.unsqueeze(0))**2) / d2norm  # (B,Mc)
            dneg = torch.sqrt((cxn_b.unsqueeze(1) - cxn.unsqueeze(0))**2 + (cyn_b.unsqueeze(1) - cyn.unsqueeze(0))**2) / d2norm  # (B,Mc)

            out[:, start:end] = 0.5 * (dpos + dneg)

        return out

    @torch.no_grad()
    def _gsmd_distance_batch(self, Xb: Tensor, nodes_chunk: Optional[int] = None) -> Tensor:
        """
        GSMD: Gradient-Structural-Moment Distance
        D_grad = gssim distance in [0,1]; D_s1n in [0,1]; D_mom in [0,1]
        Output: sqrt((D_grad^2 + D_s1n^2 + D_mom^2)/3)
        """
        if nodes_chunk is None:
            nodes_chunk = self.nodes_chunk
        Dg = self._gssim_distance_batch(Xb, nodes_chunk=nodes_chunk)
        Ds = self._s1norm_distance_batch(Xb, nodes_chunk=nodes_chunk)
        Dm = self._moment_distance_batch(Xb, nodes_chunk=nodes_chunk)
        return torch.sqrt((Dg * Dg + Ds * Ds + Dm * Dm) / 3.0)

    @torch.no_grad()
    def _curv_struct_distance_batch(self, Xb: Tensor, nodes_chunk: Optional[int] = None) -> Tensor:
        """
        Curvature structural distance (0..1) using Laplacian magnitude ratio and sign agreement with weight w=max(|ΔX|,|ΔW|).
        """
        if nodes_chunk is None:
            nodes_chunk = self.nodes_chunk
        eps = 1e-12
        B, H, W = Xb.shape
        # Laplacians
        self._ensure_lap_kernel()
        Lx = F.conv2d(Xb.unsqueeze(1), self._lap_kernel, padding=0).squeeze(1)    # (B,H-2,W-2)
        LA = Lx
        KA = torch.abs(LA)                                                         # (B,h2,w2)
        sA = torch.sign(LA)
        Lw_full = F.conv2d(self.weights.unsqueeze(1), self._lap_kernel, padding=0).squeeze(1)  # (m,H-2,W-2)
        KW_full = torch.abs(Lw_full)                                               # (m,h2,w2)
        sW_full = torch.sign(Lw_full)

        out = torch.empty((B, self.m), device=Xb.device, dtype=self.dtype)
        for start in range(0, self.m, nodes_chunk):
            end = min(start + nodes_chunk, self.m)
            KW = KW_full[start:end]                                               # (Mc,h2,w2)
            sW = sW_full[start:end]                                               # (Mc,h2,w2)

            KA_b = KA.unsqueeze(1)                                                # (B,1,h2,w2)
            KW_m = KW.unsqueeze(0)                                                # (1,Mc,h2,w2)
            Smag = (2.0 * KA_b * KW_m) / (KA_b * KA_b + KW_m * KW_m + eps)       # (B,Mc,h2,w2)

            sA_b = sA.unsqueeze(1)
            sW_m = sW.unsqueeze(0)
            Ssign = 0.5 * (1.0 + sA_b * sW_m)                                     # (B,Mc,h2,w2)

            w = torch.maximum(KA_b, KW_m)                                         # (B,Mc,h2,w2)
            S = Smag * Ssign
            num = (S * w).sum(dim=(2, 3))                                         # (B,Mc)
            den = w.sum(dim=(2, 3)) + eps
            S_C = num / den
            out[:, start:end] = 1.0 - S_C

        return out

    @torch.no_grad()
    def _s3d_distance_batch(self, Xb: Tensor, nodes_chunk: Optional[int] = None) -> Tensor:
        """
        S3D: RMS of three [0,1] distances: d_L (SSIM5), d_G (gssim), d_C (curvature structural).
        """
        if nodes_chunk is None:
            nodes_chunk = self.nodes_chunk
        dL = self._ssim5_distance_batch(Xb, nodes_chunk=nodes_chunk)
        dG = self._gssim_distance_batch(Xb, nodes_chunk=nodes_chunk)
        dC = self._curv_struct_distance_batch(Xb, nodes_chunk=nodes_chunk)
        return torch.sqrt((dL * dL + dG * dG + dC * dC) / 3.0)

    @torch.no_grad()
    def _hff_distance_batch(self, Xb: Tensor, nodes_chunk: Optional[int] = None) -> Tensor:
        """
        HFF (Hierarchical Feature Fusion) distance:
          D = (1 - SSIM5)^2 + SSIM5 * (1 - GSSIM) = dL^2 + (1 - dL) * dG
        where dL = 1 - SSIM5(P,W), dG = 1 - GSSIM(∇P,∇W). Returns (B,m).
        """
        if nodes_chunk is None:
            nodes_chunk = self.nodes_chunk
        dL = self._ssim5_distance_batch(Xb, nodes_chunk=nodes_chunk)  # (B,m) in [0,1]
        dG = self._gssim_distance_batch(Xb, nodes_chunk=nodes_chunk)  # (B,m) in [0,1]
        return dL * dL + (1.0 - dL) * dG

    @torch.no_grad()
    def _kappa_field(self, X: Tensor) -> Tensor:
        """
        Curvature field κ = div(∇X/|∇X|) computed with centered differences on the inner common grid.
        Input: X (B,H,W), Output: (B,H-3,W-3)
        """
        eps = 1e-12
        B, H, W = X.shape
        dXdx = X[:, :, 1:] - X[:, :, :-1]        # (B,H,W-1)
        dXdy = X[:, 1:, :] - X[:, :-1, :]        # (B,H-1,W)
        gx = dXdx[:, :-1, :]                     # (B,H-1,W-1)
        gy = dXdy[:, :, :-1]                     # (B,H-1,W-1)
        mag = torch.sqrt(gx * gx + gy * gy + eps)
        nx = gx / (mag + eps)
        ny = gy / (mag + eps)
        dnx_dx = 0.5 * (nx[:, :, 2:] - nx[:, :, :-2])   # (B,H-1,W-3)
        dny_dy = 0.5 * (ny[:, 2:, :] - ny[:, :-2, :])   # (B,H-3,W-1)
        dnx_dx_c = dnx_dx[:, 1:-1, :]                   # (B,H-3,W-3)
        dny_dy_c = dny_dy[:, :, 1:-1]                   # (B,H-3,W-3)
        kappa = dnx_dx_c + dny_dy_c
        return kappa

    @torch.no_grad()
    def _kappa_distance_batch(self, Xb: Tensor, nodes_chunk: Optional[int] = None) -> Tensor:
        """
        Kappa curvature distance in [0,1]: D_k = 0.5 * Σ|κ(X)-κ(W)| / Σ max(|κ(X)|,|κ(W)|)
        Uses inner grid (H-3, W-3).
        """
        if nodes_chunk is None:
            nodes_chunk = self.nodes_chunk
        eps = 1e-12
        B, H, W = Xb.shape
        kx = self._kappa_field(Xb)                               # (B,hk,wk)
        out = torch.empty((B, self.m), device=Xb.device, dtype=self.dtype)
        for start in range(0, self.m, nodes_chunk):
            end = min(start + nodes_chunk, self.m)
            kw = self._kappa_field(self.weights[start:end])      # (Mc,hk,wk)
            diff = torch.abs(kx.unsqueeze(1) - kw.unsqueeze(0))  # (B,Mc,hk,wk)
            num = diff.flatten(2).sum(dim=2)                     # (B,Mc)
            den = torch.maximum(kx.abs().unsqueeze(1), kw.abs().unsqueeze(0)).flatten(2).sum(dim=2)
            out[:, start:end] = 0.5 * (num / (den + eps))
        return out

    @torch.no_grad()
    def _s1gk_distance_batch(self, Xb: Tensor, nodes_chunk: Optional[int] = None) -> Tensor:
        """
        S1GK distance: RMS of row-normalized S1, G-SSIM distance, and row-normalized Kappa curvature distance.
        Returns (B,m).
        """
        if nodes_chunk is None:
            nodes_chunk = self.nodes_chunk
        eps = 1e-12
        d1 = self._s1_distance_batch(Xb, nodes_chunk=nodes_chunk)     # (B,m)
        dg = self._gssim_distance_batch(Xb, nodes_chunk=nodes_chunk)  # (B,m) in [0,1]
        dk = self._kappa_distance_batch(Xb, nodes_chunk=nodes_chunk)  # (B,m)
        d1_min = d1.min(dim=1, keepdim=True).values
        d1_max = d1.max(dim=1, keepdim=True).values
        dk_min = dk.min(dim=1, keepdim=True).values
        dk_max = dk.max(dim=1, keepdim=True).values
        d1n = (d1 - d1_min) / (d1_max - d1_min + eps)
        dkn = (dk - dk_min) / (dk_max - dk_min + eps)
        return torch.sqrt((d1n * d1n + dg * dg + dkn * dkn) / 3.0)

    @torch.no_grad()
    def _gssim_distance_batch(self, Xb: Tensor, nodes_chunk: Optional[int] = None) -> Tensor:
        """
        Gradient-Structure Similarity (G-SSIM-S1の勾配構造部分) に基づく距離。
        S_GS = Σ[w · S_mag · S_dir] / (Σ w + ε),  D = 1 - S_GS
          - S_mag = 2|∇X||∇W| / (|∇X|^2 + |∇W|^2 + ε)
          - S_dir = (1 + cosθ)/2,  cosθ = (∇X·∇W)/(||∇X||||∇W|| + ε)
          - w = max(|∇X|, |∇W|)
        """
        if nodes_chunk is None:
            nodes_chunk = self.nodes_chunk
        eps = 1e-12
        B, H, W = Xb.shape

        # サンプル側勾配（共通領域）
        dXdx = Xb[:, :, 1:] - Xb[:, :, :-1]
        dXdy = Xb[:, 1:, :] - Xb[:, :-1, :]
        gx = dXdx[:, :-1, :]
        gy = dXdy[:, :, :-1]
        gmagX = torch.sqrt(gx * gx + gy * gy + eps)  # (B, H-1, W-1)

        out = torch.empty((B, self.m), device=Xb.device, dtype=self.dtype)

        dWdx_full = self.weights[:, :, 1:] - self.weights[:, :, :-1]  # (m, H, W-1)
        dWdy_full = self.weights[:, 1:, :] - self.weights[:, :-1, :]  # (m, H-1, W)

        for start in range(0, self.m, nodes_chunk):
            end = min(start + nodes_chunk, self.m)
            grx = dWdx_full[start:end, :-1, :]   # (Mc, H-1, W-1)
            gry = dWdy_full[start:end, :, :-1]   # (Mc, H-1, W-1)
            gmagW = torch.sqrt(grx * grx + gry * gry + eps)  # (Mc, H-1, W-1)

            gx_b = gx.unsqueeze(1)     # (B,1,H-1,W-1)
            gy_b = gy.unsqueeze(1)
            gX_b = gmagX.unsqueeze(1)  # (B,1,H-1,W-1)
            grx_m = grx.unsqueeze(0)   # (1,Mc,H-1,W-1)
            gry_m = gry.unsqueeze(0)
            gW_m = gmagW.unsqueeze(0)  # (1,Mc,H-1,W-1)

            dot = gx_b * grx_m + gy_b * gry_m
            cos = (dot / (gX_b * gW_m + eps)).clamp(-1.0, 1.0)
            Sdir = 0.5 * (1.0 + cos)
            Smag = (2.0 * gX_b * gW_m) / (gX_b * gX_b + gW_m * gW_m + eps)
            S = Smag * Sdir
            w = torch.maximum(gX_b, gW_m)
            sim = (S * w).sum(dim=(2, 3)) / (w.sum(dim=(2, 3)) + eps)  # (B,Mc)
            out[:, start:end] = 1.0 - sim

        return out

    @torch.no_grad()
    def _s1_distance_batch(self, Xb: Tensor, nodes_chunk: Optional[int] = None) -> Tensor:
        """
        Xb: (B,H,W) 戻り (B,m) S1距離
        """
        if nodes_chunk is None:
            nodes_chunk = self.nodes_chunk
        B, H, W = Xb.shape
        dXdx = Xb[:, :, 1:] - Xb[:, :, :-1]  # (B,H,W-1)
        dXdy = Xb[:, 1:, :] - Xb[:, :-1, :]  # (B,H-1,W)

        out = torch.empty((B, self.m), device=Xb.device, dtype=self.dtype)
        dWdx_full = self.weights[:, :, 1:] - self.weights[:, :, :-1]
        dWdy_full = self.weights[:, 1:, :] - self.weights[:, :-1, :]

        for start in range(0, self.m, nodes_chunk):
            end = min(start + nodes_chunk, self.m)
            dWdx = dWdx_full[start:end]   # (Mc,H,W-1)
            dWdy = dWdy_full[start:end]   # (Mc,H-1,W)
            num_dx = (torch.abs(dWdx.unsqueeze(0) - dXdx.unsqueeze(1))).sum(dim=(2, 3))
            num_dy = (torch.abs(dWdy.unsqueeze(0) - dXdy.unsqueeze(1))).sum(dim=(2, 3))
            num = num_dx + num_dy

            den_dx = (torch.maximum(torch.abs(dWdx).unsqueeze(0), torch.abs(dXdx).unsqueeze(1))).sum(dim=(2, 3))
            den_dy = (torch.maximum(torch.abs(dWdy).unsqueeze(0), torch.abs(dXdy).unsqueeze(1))).sum(dim=(2, 3))
            denom = den_dx + den_dy
            s1 = 100.0 * num / (denom + 1e-12)
            out[:, start:end] = s1

        return out

    @torch.no_grad()
    def _curv_s1_distance_batch(self, Xb: Tensor, nodes_chunk: Optional[int] = None) -> Tensor:
        """
        Curvature S1-like ratio on Laplacian fields (not normalized), optionally area-weighted.
        Returns (B,m) with values roughly in [0,2]; downstream callers may row-wise normalize to [0,1].
        """
        if nodes_chunk is None:
            nodes_chunk = self.nodes_chunk
        eps = 1e-12
        B, H, W = Xb.shape
        self._ensure_lap_kernel()
        Lx = F.conv2d(Xb.unsqueeze(1), self._lap_kernel, padding=0).squeeze(1)   # (B,H-2,W-2)
        Lw_full = F.conv2d(self.weights.unsqueeze(1), self._lap_kernel, padding=0).squeeze(1)  # (m,H-2,W-2)
        if self.area_w is not None:
            w_inner = self.area_w[1:-1, 1:-1]                                     # (H-2,W-2)
        else:
            w_inner = None
        out = torch.empty((B, self.m), device=Xb.device, dtype=self.dtype)
        for start in range(0, self.m, nodes_chunk):
            end = min(start + nodes_chunk, self.m)
            Lw = Lw_full[start:end]                                              # (Mc,H-2,W-2)
            diff = torch.abs(Lx.unsqueeze(1) - Lw.unsqueeze(0))                  # (B,Mc,H-2,W-2)
            denom = torch.maximum(torch.abs(Lx).unsqueeze(1), torch.abs(Lw).unsqueeze(0))  # (B,Mc,H-2,W-2)
            if w_inner is not None:
                win = w_inner.view(1, 1, H-2, W-2)
                num_s = (diff * win).sum(dim=(2, 3))
                den_s = (denom * win).sum(dim=(2, 3)) + eps
            else:
                num_s = diff.sum(dim=(2, 3))
                den_s = denom.sum(dim=(2, 3)) + eps
            out[:, start:end] = num_s / den_s
        return out

    @torch.no_grad()
    def _cfsd_distance_batch(self, Xb: Tensor, nodes_chunk: Optional[int] = None) -> Tensor:
        """
        CFSD: RMS of three components:
          - D_G: G-SSIM distance in [0,1]
          - D_S1^n: row-wise min–max normalized S1
          - D_CURV^n: row-wise min–max normalized curvature S1 on Laplacians
        """
        eps = 1e-12
        # Compute all components (each function internally chunks)
        Dg = self._gssim_distance_batch(Xb, nodes_chunk=nodes_chunk)               # (B,m) in [0,1]
        Ds1 = self._s1_distance_batch(Xb, nodes_chunk=nodes_chunk)                 # (B,m) arbitrary scale
        Dc_raw = self._curv_s1_distance_batch(Xb, nodes_chunk=nodes_chunk)         # (B,m) ratio
        min_s1 = Ds1.min(dim=1, keepdim=True).values
        max_s1 = Ds1.max(dim=1, keepdim=True).values
        Ds1n = (Ds1 - min_s1) / (max_s1 - min_s1 + eps)
        min_c = Dc_raw.min(dim=1, keepdim=True).values
        max_c = Dc_raw.max(dim=1, keepdim=True).values
        Dcn = (Dc_raw - min_c) / (max_c - min_c + eps)
        return torch.sqrt((Dg * Dg + Ds1n * Ds1n + Dcn * Dcn) / 3.0)

    @torch.no_grad()
    def _s1gcurv_distance_batch(self, Xb: Tensor, nodes_chunk: Optional[int] = None) -> Tensor:
        """
        S1GCURV: RMS of three [0,1] distances:
          - D_s1 = S1/200 clamped to [0,1]
          - D_edge = G-SSIM distance in [0,1]
          - D_curv = 0.5 * curvature S1-like ratio on Laplacians in [0,1]
        Returns (B,m).
        """
        if nodes_chunk is None:
            nodes_chunk = self.nodes_chunk
        D1 = self._s1_distance_batch(Xb, nodes_chunk=nodes_chunk)                 # (B,m) ~0..200
        D1n = torch.clamp(D1 / 200.0, min=0.0, max=1.0)                           # (B,m) in [0,1]
        Dg = self._gssim_distance_batch(Xb, nodes_chunk=nodes_chunk)              # (B,m) in [0,1]
        Dc = self._curv_s1_distance_batch(Xb, nodes_chunk=nodes_chunk)            # (B,m) ~0..2
        Dcurv = 0.5 * Dc                                                          # (B,m) in [0,1]
        return torch.sqrt((D1n * D1n + Dg * Dg + Dcurv * Dcurv) / 3.0)

    @torch.no_grad()
    def _spot_distance_batch(self, Xb: Tensor, nodes_chunk: Optional[int] = None) -> Tensor:
        """
        SPOT: Spherical (grid-normalized) sliced Wasserstein-1 distance on positive/negative anomaly masses.
        - Fixed 16 projections, 64 bins, no exposed hyperparameters.
        - Area weight (cosφ) if provided.
        Returns (B,m) in [0,1].
        """
        if nodes_chunk is None:
            nodes_chunk = self.nodes_chunk
        device = Xb.device; dtype = Xb.dtype
        B, H, W = Xb.shape
        m = self.m
        # Mass split
        Ap = torch.clamp(Xb, min=0.0)
        An = torch.clamp(-Xb, min=0.0)
        Wp = torch.clamp(self.weights, min=0.0)
        Wn = torch.clamp(-self.weights, min=0.0)
        if self.area_w is not None:
            aw = self.area_w.view(1, H, W)
            Ap = Ap * aw; An = An * aw
            Wp = Wp * self.area_w; Wn = Wn * self.area_w
        # Normalize mass (avoid div0)
        def _normalize_mass(T):
            s = T.flatten(1).sum(dim=1, keepdim=True).clamp(min=1e-12)
            return (T.flatten(1) / s).view(T.shape)
        Ap = _normalize_mass(Ap); An = _normalize_mass(An)
        Wp = _normalize_mass(Wp); Wn = _normalize_mass(Wn)
        # Projections
        K = 16
        thetas = torch.linspace(0.0, 3.141592653589793, K, device=device, dtype=dtype)  # [0,π]
        xg = self._xgrid; yg = self._ygrid
        # Binning support
        N_BINS = 64
        # projection length (max of x*cos+ y*sin in [0,1]^2) is <= sqrt(2)
        L = (2.0 ** 0.5)
        bin_edges = torch.linspace(0.0, L, N_BINS + 1, device=device, dtype=dtype)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_width = (L / N_BINS)
        # Precompute s per theta
        S_list = []
        for t in thetas:
            s = xg * torch.cos(t) + yg * torch.sin(t)  # (1,H,W)
            S_list.append(s.reshape(-1))  # (H*W,)
        # Hist builder
        def _mass_histograms(MassB: Tensor, MassM: Tensor):
            # MassB: (B,H,W), MassM: (m,H,W)
            HB = torch.zeros((B, K, N_BINS), device=device, dtype=dtype)
            HM = torch.zeros((m, K, N_BINS), device=device, dtype=dtype)
            massB_flat = MassB.view(B, -1)
            massM_flat = MassM.view(m, -1)
            for ki, s_flat in enumerate(S_list):
                # bucketize
                idx = torch.bucketize(s_flat, bin_edges) - 1  # in [0,N_BINS-1]
                idx = idx.clamp(min=0, max=N_BINS - 1)
                # scatter-add for batch B
                # Create (B, H*W) → (B, N_BINS)
                HB[:, ki, :] = torch.zeros((B, N_BINS), device=device, dtype=dtype).scatter_add(
                    1, idx.unsqueeze(0).expand(B, -1), massB_flat
                )
                HM[:, ki, :] = torch.zeros((m, N_BINS), device=device, dtype=dtype).scatter_add(
                    1, idx.unsqueeze(0).expand(m, -1), massM_flat
                )
            # Normalize histograms to sum to 1 (already normalized mass, but bins may introduce small drift)
            HB = HB / (HB.sum(dim=2, keepdim=True).clamp(min=1e-12))
            HM = HM / (HM.sum(dim=2, keepdim=True).clamp(min=1e-12))
            # CDFs
            CDFB = torch.cumsum(HB, dim=2)
            CDFM = torch.cumsum(HM, dim=2)
            # EMD per projection between B and m: sum |cdfB - cdfM| * bin_width over bins
            # Expand to (B,K,N_BINS) and (m,K,N_BINS)
            # Compute pairwise via broadcasting: (B,K,N) vs (m,K,N) -> (B,m,K,N)
            diff = (CDFB.unsqueeze(1) - CDFM.unsqueeze(0)).abs()  # (B,m,K,N_BINS)
            emd = diff.sum(dim=3) * bin_width  # (B,m,K)
            # normalize by L to map to [0,1]
            emd = emd / L
            # average over projections
            return emd.mean(dim=2)  # (B,m)
        Dp = _mass_histograms(Ap, Wp)
        Dn = _mass_histograms(An, Wn)
        return 0.5 * (Dp + Dn)

    @torch.no_grad()
    def _gvd_distance_batch(self, Xb: Tensor, nodes_chunk: Optional[int] = None) -> Tensor:
        """
        GVD: Geostrophic Vorticity–Deformation Invariants distance.
        Uses second derivatives (inner grid), compares:
          - normalized Laplacian L,
          - normalized deformation magnitude S,
          - principal axis orientation θ (π-periodic via cos(2Δθ)).
        Returns (B,m) in [0,1].
        """
        if nodes_chunk is None:
            nodes_chunk = self.nodes_chunk
        device = Xb.device; dtype = Xb.dtype
        B, H, W = Xb.shape
        m = self.m

        def _second_derivs(Z: Tensor):
            # Z: (...,H,W) → inner grid derivs of shape (...,H-2,W-2)
            Zc = Z
            Z_xx = Zc[..., 2:, 1:-1] - 2.0 * Zc[..., 1:-1, 1:-1] + Zc[..., :-2, 1:-1]
            Z_yy = Zc[..., 1:-1, 2:] - 2.0 * Zc[..., 1:-1, 1:-1] + Zc[..., 1:-1, :-2]
            Z_xy = (Zc[..., 2:, 2:] - Zc[..., 2:, :-2] - Zc[..., :-2, 2:] + Zc[..., :-2, :-2]) * 0.25
            return Z_xx, Z_yy, Z_xy

        X_xx, X_yy, X_xy = _second_derivs(Xb)
        W_xx, W_yy, W_xy = _second_derivs(self.weights)

        def _invariants(Z_xx, Z_yy, Z_xy):
            L = Z_xx + Z_yy
            S = torch.sqrt((Z_xx - Z_yy) * (Z_xx - Z_yy) + (2.0 * Z_xy) * (2.0 * Z_xy) + 1e-12)
            theta = 0.5 * torch.atan2(2.0 * Z_xy, (Z_xx - Z_yy + 1e-12))
            return L, S, theta

        Lx, Sx, thx = _invariants(X_xx, X_yy, X_xy)    # (B,h,w)
        Lw, Sw, thw = _invariants(W_xx, W_yy, W_xy)    # (m,h,w)

        # Area weights (inner)
        if self.area_w is not None:
            w_inner = self.area_w[1:-1, 1:-1]
        else:
            w_inner = torch.ones((H - 2, W - 2), device=device, dtype=dtype)

        def _normalize_LS(L, S):
            # Normalize by total magnitude to be scale free.
            Lsum = (L.abs() * w_inner).flatten(1).sum(dim=1, keepdim=True).clamp(min=1e-12) if L.dim() == 3 else (L.abs() * w_inner).flatten().sum().clamp(min=1e-12)
            Ssum = (S * w_inner).flatten(1).sum(dim=1, keepdim=True).clamp(min=1e-12) if S.dim() == 3 else (S * w_inner).flatten().sum().clamp(min=1e-12)
            return L / Lsum.view((-1, 1, 1)), S / Ssum.view((-1, 1, 1))

        Lx, Sx = _normalize_LS(Lx, Sx)  # (B,h,w)
        Lw, Sw = _normalize_LS(Lw, Sw)  # (m,h,w)

        Wsum = w_inner.sum()

        # d_vort: mean abs difference of normalized L
        d_vort = ((Lx.unsqueeze(1) - Lw.unsqueeze(0)).abs() * w_inner).flatten(2).sum(dim=2) / Wsum  # (B,m)
        # d_def: mean abs difference of normalized S
        d_def = ((Sx.unsqueeze(1) - Sw.unsqueeze(0)).abs() * w_inner).flatten(2).sum(dim=2) / Wsum   # (B,m)
        # d_axis: 1 - cos(2Δθ) averaged
        d_axis = (1.0 - torch.cos(2.0 * (thx.unsqueeze(1) - thw.unsqueeze(0))))  # (B,m,h,w)
        d_axis = (d_axis * w_inner).flatten(2).sum(dim=2) / Wsum                # (B,m)

        return torch.sqrt((d_vort * d_vort + d_def * d_def + d_axis * d_axis) / 3.0)

    @torch.no_grad()
    def _itcs_distance_batch(self, Xb: Tensor, nodes_chunk: Optional[int] = None) -> Tensor:
        """
        ITCS: Isobaric Topology & Centroid Signature (simplified: area fraction φ and centroid c).
        - Fixed quantiles q=0.1..0.9
        - Positive and negative sides averaged
        Returns (B,m) in [0,1].
        """
        if nodes_chunk is None:
            nodes_chunk = self.nodes_chunk
        device = Xb.device; dtype = Xb.dtype
        B, H, W = Xb.shape
        m = self.m

        if self.area_w is not None:
            w = self.area_w
        else:
            w = torch.ones((H, W), device=device, dtype=dtype)

        xg = self._xgrid.squeeze(0)  # (H,W)
        yg = self._ygrid.squeeze(0)  # (H,W)

        qs = torch.linspace(0.1, 0.9, 9, device=device, dtype=dtype)

        def _signature(Z: Tensor):
            # Z: (N,H,W)
            N = Z.shape[0]
            Zf = Z.view(N, -1)
            # Quantile thresholds per sample
            t = torch.quantile(Zf, qs, dim=1, interpolation='linear').transpose(0, 1)  # (N,9)
            # Positive superlevel sets
            sig_phi_p = []
            cx_p = []; cy_p = []
            # Negative sublevel sets
            sig_phi_n = []
            cx_n = []; cy_n = []
            w_flat = w.reshape(-1)
            x_flat = xg.reshape(-1)
            y_flat = yg.reshape(-1)
            for qi in range(qs.numel()):
                tp = t[:, qi].view(N, 1, 1)
                # positive mask
                Mp = (Z >= tp)
                wp = (Mp * w).view(N, -1)
                ap = wp.sum(dim=1).clamp(min=1e-12)
                sig_phi_p.append((ap / w.sum()).view(N, 1))
                cx_p.append(((wp * x_flat) .sum(dim=1) / ap).view(N, 1))
                cy_p.append(((wp * y_flat) .sum(dim=1) / ap).view(N, 1))
                # negative mask (use threshold of -tp on -Z to keep monotonic)
                tn = (-Zf).quantile(q=qs[qi].item(), dim=1, interpolation='linear').view(N, 1, 1)
                Mn = (Z <= -tn)
                wn = (Mn * w).view(N, -1)
                an = wn.sum(dim=1).clamp(min=1e-12)
                sig_phi_n.append((an / w.sum()).view(N, 1))
                cx_n.append(((wn * x_flat).sum(dim=1) / an).view(N, 1))
                cy_n.append(((wn * y_flat).sum(dim=1) / an).view(N, 1))
            # Stack over q: (N,9)
            phi_p = torch.cat(sig_phi_p, dim=1); phi_n = torch.cat(sig_phi_n, dim=1)
            cx_p = torch.cat(cx_p, dim=1);       cy_p = torch.cat(cy_p, dim=1)
            cx_n = torch.cat(cx_n, dim=1);       cy_n = torch.cat(cy_n, dim=1)
            return (phi_p, cx_p, cy_p, phi_n, cx_n, cy_n)

        # Batch and nodes signatures
        phi_p_B, cx_p_B, cy_p_B, phi_n_B, cx_n_B, cy_n_B = _signature(Xb)
        phi_p_M, cx_p_M, cy_p_M, phi_n_M, cx_n_M, cy_n_M = _signature(self.weights)

        # Distances over q (average absolute diffs); centroid distance in normalized coordinates; combine pos/neg and components equally
        def _dist_sig(phiA, cxA, cyA, phiB, cxB, cyB):
            dphi = (phiA.unsqueeze(1) - phiB.unsqueeze(0)).abs().mean(dim=2)  # (B,m)
            dc  = torch.sqrt((cxA.unsqueeze(1) - cxB.unsqueeze(0))**2 + (cyA.unsqueeze(1) - cyB.unsqueeze(0))**2).mean(dim=2)  # (B,m)
            # Normalize centroid distance by sqrt(2) to [0,1]
            dc = dc / (2.0 ** 0.5)
            return 0.5 * dphi + 0.5 * dc

        Dp = _dist_sig(phi_p_B, cx_p_B, cy_p_B, phi_p_M, cx_p_M, cy_p_M)
        Dn = _dist_sig(phi_n_B, cx_n_B, cy_n_B, phi_n_M, cx_n_M, cy_n_M)
        return 0.5 * (Dp + Dn)

    @torch.no_grad()
    def _distance_batch(self, Xb: Tensor, nodes_chunk: Optional[int] = None) -> Tensor:
        if self.activation_distance == 's1':
            return self._s1_distance_batch(Xb, nodes_chunk=nodes_chunk)
        elif self.activation_distance == 'euclidean':
            return self._euclidean_distance_batch(Xb)
        elif self.activation_distance == 'ssim5':
            return self._ssim5_distance_batch(Xb, nodes_chunk=nodes_chunk)
        elif self.activation_distance == 's1ssim':
            # 融合：各サンプル毎にノード方向min-max正規化して等重み和
            d1 = self._s1_distance_batch(Xb, nodes_chunk=nodes_chunk)    # (B,m)
            d2 = self._ssim5_distance_batch(Xb, nodes_chunk=nodes_chunk) # (B,m)
            min1 = d1.min(dim=1, keepdim=True).values
            max1 = d1.max(dim=1, keepdim=True).values
            min2 = d2.min(dim=1, keepdim=True).values
            max2 = d2.max(dim=1, keepdim=True).values
            dn1 = (d1 - min1) / (max1 - min1 + 1e-12)
            dn2 = (d2 - min2) / (max2 - min2 + 1e-12)
            return 0.5 * (dn1 + dn2)
        elif self.activation_distance == 's1ssim5_hf':
            # HF-S1SSIM5: SSIM(5x5)でゲートするソフト階層化 D = u + (1-u)*v
            dS1 = self._s1_distance_batch(Xb, nodes_chunk=nodes_chunk)      # (B,m)
            dSSIM = self._ssim5_distance_batch(Xb, nodes_chunk=nodes_chunk) # (B,m) = 1-SSIM
            eps = 1e-12
            min_s1 = dS1.min(dim=1, keepdim=True).values
            max_s1 = dS1.max(dim=1, keepdim=True).values
            min_ss = dSSIM.min(dim=1, keepdim=True).values
            max_ss = dSSIM.max(dim=1, keepdim=True).values
            v = (dS1 - min_s1) / (max_s1 - min_s1 + eps)   # normalized S1
            u = (dSSIM - min_ss) / (max_ss - min_ss + eps) # normalized dSSIM
            return u + (1.0 - u) * v
        elif self.activation_distance == 's1ssim5_and':
            # AND合成: 行方向min–max正規化 U,V を用いて D = max(U,V)
            dS1 = self._s1_distance_batch(Xb, nodes_chunk=nodes_chunk)
            dSSIM = self._ssim5_distance_batch(Xb, nodes_chunk=nodes_chunk)
            eps = 1e-12
            min_s1 = dS1.min(dim=1, keepdim=True).values
            max_s1 = dS1.max(dim=1, keepdim=True).values
            min_ss = dSSIM.min(dim=1, keepdim=True).values
            max_ss = dSSIM.max(dim=1, keepdim=True).values
            V = (dS1 - min_s1) / (max_s1 - min_s1 + eps)
            U = (dSSIM - min_ss) / (max_ss - min_ss + eps)
            return torch.maximum(U, V)
        elif self.activation_distance == 'pf_s1ssim':
            # 比例融合: 正規化なしで積 D = dS1 * dSSIM
            dS1 = self._s1_distance_batch(Xb, nodes_chunk=nodes_chunk)
            dSSIM = self._ssim5_distance_batch(Xb, nodes_chunk=nodes_chunk)
            return dS1 * dSSIM
        elif self.activation_distance == 's3d':
            return self._s3d_distance_batch(Xb, nodes_chunk=nodes_chunk)
        elif self.activation_distance == 'cfsd':
            return self._cfsd_distance_batch(Xb, nodes_chunk=nodes_chunk)
        elif self.activation_distance == 'hff':
            return self._hff_distance_batch(Xb, nodes_chunk=nodes_chunk)
        elif self.activation_distance == 's1gk':
            return self._s1gk_distance_batch(Xb, nodes_chunk=nodes_chunk)
        elif self.activation_distance == 's1gcurv':
            return self._s1gcurv_distance_batch(Xb, nodes_chunk=nodes_chunk)
        elif self.activation_distance == 'gsmd':
            return self._gsmd_distance_batch(Xb, nodes_chunk=nodes_chunk)
        elif self.activation_distance == 's1gl':
            return self._s1gl_distance_batch(Xb, nodes_chunk=nodes_chunk)
        elif self.activation_distance == 's1gssim':
            return self._s1gssim_distance_batch(Xb, nodes_chunk=nodes_chunk)
        elif self.activation_distance == 'gssim':
            return self._gssim_distance_batch(Xb, nodes_chunk=nodes_chunk)
        elif self.activation_distance == 'ms_s1':
            return self._ms_s1_distance_batch(Xb, nodes_chunk=nodes_chunk)
        elif self.activation_distance == 'msssim_s1g':
            return self._msssim_s1g_distance_batch(Xb, nodes_chunk=nodes_chunk)
        elif self.activation_distance == 'spot':
            return self._spot_distance_batch(Xb, nodes_chunk=nodes_chunk)
        elif self.activation_distance == 'gvd':
            return self._gvd_distance_batch(Xb, nodes_chunk=nodes_chunk)
        elif self.activation_distance == 'itcs':
            return self._itcs_distance_batch(Xb, nodes_chunk=nodes_chunk)
        else:
            raise RuntimeError('Unknown activation_distance')

    @torch.no_grad()
    def bmu_indices(self, Xb: Tensor, nodes_chunk: Optional[int] = None) -> Tensor:
        dists = self._distance_batch(Xb, nodes_chunk=nodes_chunk)
        bmu = torch.argmin(dists, dim=1)
        return bmu

    # ---------- 距離計算（バッチ→単一参照：メドイド置換等で使用） ----------
    @torch.no_grad()
    def _euclidean_to_ref(self, Xb: Tensor, ref: Tensor) -> Tensor:
        # Xb: (B,H,W), ref: (H,W) -> (B,)
        diff = Xb - ref.view(1, *ref.shape)
        d2 = (diff * diff).sum(dim=(1, 2))
        return torch.sqrt(d2 + 1e-12)


    @torch.no_grad()
    def _ssim5_to_ref(self, Xb: Tensor, ref: Tensor) -> Tensor:
        # 1 - mean(SSIM_map(5x5, C=0)) 対参照
        self._ensure_kernel5()
        eps = 1e-12
        B, H, W = Xb.shape
        X = Xb.unsqueeze(1)
        R = ref.view(1, 1, H, W)

        X_pad = F.pad(X, (self._win5_pad, self._win5_pad, self._win5_pad, self._win5_pad), mode='reflect')
        R_pad = F.pad(R, (self._win5_pad, self._win5_pad, self._win5_pad, self._win5_pad), mode='reflect')

        mu_x = F.conv2d(X_pad, self._kernel5, padding=0)                 # (B,1,H,W)
        mu_r = F.conv2d(R_pad, self._kernel5, padding=0)                 # (1,1,H,W)

        mu_x2 = F.conv2d(X_pad * X_pad, self._kernel5, padding=0)
        mu_r2 = F.conv2d(R_pad * R_pad, self._kernel5, padding=0)
        var_x = torch.clamp(mu_x2 - mu_x * mu_x, min=0.0)
        var_r = torch.clamp(mu_r2 - mu_r * mu_r, min=0.0)

        mu_xr = F.conv2d(F.pad(X * R, (self._win5_pad, self._win5_pad, self._win5_pad, self._win5_pad), mode='reflect'),
                         self._kernel5, padding=0)
        cov = mu_xr - mu_x * mu_r

        l_num = 2 * (mu_x * mu_r)
        l_den = (mu_x * mu_x + mu_r * mu_r)
        c_num = 2 * cov
        c_den = (var_x + var_r)
        ssim_map = (l_num * c_num) / (l_den * c_den + eps)               # (B,1,H,W)
        ssim_avg = ssim_map.mean(dim=(1, 2, 3))                          # (B,)
        return 1.0 - ssim_avg

    @torch.no_grad()
    def _ms_s1_to_ref(self, Xb: Tensor, ref: Tensor) -> Tensor:
        # Multi-scale S1 (to reference), row-wise (batch) normalization per scale and RMS fusion
        scales = [1, 2, 4]
        weights = torch.tensor([0.5, 0.3, 0.2], device=Xb.device, dtype=self.dtype)
        weights = weights / weights.sum()
        dists = []
        for s in scales:
            if s == 1:
                Xs = Xb
                R = ref
            else:
                Xs = self._adaptive_pool(Xb.unsqueeze(1), s).squeeze(1)
                R = self._adaptive_pool(ref.unsqueeze(0).unsqueeze(0), s).squeeze(0).squeeze(0)
            d = self._s1_to_ref(Xs, R)  # (B,)
            # batch-wise min-max normalize
            eps = 1e-12
            dmin = d.min(dim=0, keepdim=True).values
            dmax = d.max(dim=0, keepdim=True).values
            dn = (d - dmin) / (dmax - dmin + eps)
            dists.append(dn)
        out = torch.zeros_like(dists[0])
        for i, dn in enumerate(dists):
            out = out + weights[i] * (dn * dn)
        return torch.sqrt(out + 1e-12)

    @torch.no_grad()
    def _msssim_s1g_to_ref(self, Xb: Tensor, ref: Tensor) -> Tensor:
        # D = dL*^2 + (1 - dL*) * sqrt( (dG^2 + dS1n^2)/2 )
        dL = []
        scales = [1, 2, 4]
        weights = torch.tensor([0.5, 0.3, 0.2], device=Xb.device, dtype=self.dtype)
        weights = weights / weights.sum()
        # modified multi-scale SSIM (to ref)
        for s in scales:
            if s == 1:
                Xs = Xb
                R = ref
            else:
                Xs = self._adaptive_pool(Xb.unsqueeze(1), s).squeeze(1)
                R = self._adaptive_pool(ref.unsqueeze(0).unsqueeze(0), s).squeeze(0).squeeze(0)
            # reuse existing _ssim5_to_ref but with small constants effect approximated by adding eps to denominator inside orig fn
            # As a proxy, we use the existing implementation (C=0) and rely on multi-scale robustness
            dl = self._ssim5_to_ref(Xs, R)  # (B,)
            dL.append(dl.clamp(0.0, 2.0))
        dL = torch.stack(dL, dim=1)  # (B,S)
        dL = (weights.view(1, -1) * dL).sum(dim=1)  # weighted mean distance per row (B,)
        # dG (to ref)
        dG = self._gssim_to_ref(Xb, ref)  # (B,)
        # dS1 normalized within batch
        dS1 = self._s1_to_ref(Xb, ref)  # (B,)
        eps = 1e-12
        dmin = dS1.min(dim=0, keepdim=True).values
        dmax = dS1.max(dim=0, keepdim=True).values
        dS1n = (dS1 - dmin) / (dmax - dmin + eps)
        core = torch.sqrt((dG * dG + dS1n * dS1n) / 2.0)
        return dL * dL + (1.0 - dL) * core

    @torch.no_grad()
    def _s1_to_ref(self, Xb: Tensor, ref: Tensor) -> Tensor:
        dXdx = Xb[:, :, 1:] - Xb[:, :, :-1]
        dXdy = Xb[:, 1:, :] - Xb[:, :-1, :]
        dRdx = ref[:, 1:] - ref[:, :-1]
        dRdy = ref[1:, :] - ref[:-1, :]
        num_dx = (torch.abs(dXdx - dRdx.view(1, *dRdx.shape))).sum(dim=(1, 2))
        num_dy = (torch.abs(dXdy - dRdy.view(1, *dRdy.shape))).sum(dim=(1, 2))
        den_dx = torch.maximum(torch.abs(dXdx), torch.abs(dRdx).view(1, *dRdx.shape)).sum(dim=(1, 2))
        den_dy = torch.maximum(torch.abs(dXdy), torch.abs(dRdy).view(1, *dRdy.shape)).sum(dim=(1, 2))
        s1 = 100.0 * (num_dx + num_dy) / (den_dx + den_dy + 1e-12)
        return s1

    @torch.no_grad()
    def _s1gssim_to_ref(self, Xb: Tensor, ref: Tensor) -> Tensor:
        """
        s1gssim の対参照距離：
          d_edge = 0.5*(d_g + d_dir),  d_s1 = 行方向min-max正規化したS1
          D = sqrt((d_edge^2 + d_s1^2)/2)
        """
        eps = 1e-12
        B, H, W = Xb.shape

        # 勾配（共通領域）
        dXdx = Xb[:, :, 1:] - Xb[:, :, :-1]
        dXdy = Xb[:, 1:, :] - Xb[:, :-1, :]
        dRdx = ref[:, 1:] - ref[:, :-1]
        dRdy = ref[1:, :] - ref[:-1, :]

        dXdx_c, dXdy_c = dXdx[:, :-1, :], dXdy[:, :, :-1]
        dRdx_c, dRdy_c = dRdx[:-1, :], dRdy[:, :-1]
        magX = torch.sqrt(dXdx_c * dXdx_c + dXdy_c * dXdy_c + eps)  # (B, H-1, W-1)
        magR = torch.sqrt(dRdx_c * dRdx_c + dRdy_c * dRdy_c + eps)  # (H-1, W-1)

        # 勾配強度のSSIM(5x5, C=0)
        self._ensure_kernel5()
        Xg = magX.unsqueeze(1)                          # (B,1,.,.)
        Rg = magR.unsqueeze(0).unsqueeze(1)             # (1,1,.,.)
        Xg_pad = F.pad(Xg, (2, 2, 2, 2), mode='reflect')
        Rg_pad = F.pad(Rg, (2, 2, 2, 2), mode='reflect')
        mu_x = F.conv2d(Xg_pad, self._kernel5, padding=0)
        mu_r = F.conv2d(Rg_pad, self._kernel5, padding=0)
        mu_x2 = F.conv2d(Xg_pad * Xg_pad, self._kernel5, padding=0)
        mu_r2 = F.conv2d(Rg_pad * Rg_pad, self._kernel5, padding=0)
        var_x = torch.clamp(mu_x2 - mu_x * mu_x, min=0.0)
        var_r = torch.clamp(mu_r2 - mu_r * mu_r, min=0.0)
        mu_xr = F.conv2d(F.pad(Xg * Rg, (2, 2, 2, 2), mode='reflect'), self._kernel5, padding=0)
        cov = mu_xr - mu_x * mu_r
        ssim_map = (2 * mu_x * mu_r * 2 * cov) / ((mu_x * mu_x + mu_r * mu_r) * (var_x + var_r) + eps)
        ssim_avg = ssim_map.mean(dim=(1, 2, 3))
        d_g = 0.5 * (1.0 - ssim_avg)                    # (B,)

        # 勾配方向（加重平均）
        dot = dXdx_c * dRdx_c.unsqueeze(0) + dXdy_c * dRdy_c.unsqueeze(0)     # (B,.,.)
        denom = magX * magR.unsqueeze(0) + eps
        cos = (dot / denom).clamp(-1.0, 1.0)
        wgt = torch.maximum(magX, magR.unsqueeze(0))
        s_dir = (cos * wgt).flatten(1).sum(dim=1) / (wgt.flatten(1).sum(dim=1) + eps)     # (B,)
        d_dir = 0.5 * (1.0 - s_dir)

        d_edge = 0.5 * (d_g + d_dir)                    # (B,)

        # S1 の行方向min-max正規化（バッチ内）
        dS1 = self._s1_to_ref(Xb, ref)                  # (B,)
        min_s1, _ = dS1.min(dim=0, keepdim=True)
        max_s1, _ = dS1.max(dim=0, keepdim=True)
        dS1n = (dS1 - min_s1) / (max_s1 - min_s1 + eps)

        return torch.sqrt((d_edge * d_edge + dS1n * dS1n) / 2.0)

    @torch.no_grad()
    def _s1gl_to_ref(self, Xb: Tensor, ref: Tensor) -> Tensor:
        """
        DSGC (to ref): Dedge (grad structure) + Dcurv (curvature SSIM) + normalized S1, combined by RMS.
        Returns (B,)
        """
        eps = 1e-12
        B, H, W = Xb.shape

        # Gradients
        dXdx = Xb[:, :, 1:] - Xb[:, :, :-1]
        dXdy = Xb[:, 1:, :] - Xb[:, :-1, :]
        dRdx = ref[:, 1:] - ref[:, :-1]
        dRdy = ref[1:, :] - ref[:-1, :]

        dXdx_c, dXdy_c = dXdx[:, :-1, :], dXdy[:, :, :-1]            # (B,H-1,W-1)
        dRdx_c, dRdy_c = dRdx[:-1, :], dRdy[:, :-1]                  # (H-1,W-1)
        magX = torch.sqrt(dXdx_c * dXdx_c + dXdy_c * dXdy_c + eps)   # (B,H-1,W-1)
        magR = torch.sqrt(dRdx_c * dRdx_c + dRdy_c * dRdy_c + eps)   # (H-1,W-1)

        # Gradient magnitude SSIM (5x5, C=0)
        self._ensure_kernel5()
        Xg = magX.unsqueeze(1)                          # (B,1,.,.)
        Rg = magR.unsqueeze(0).unsqueeze(1)             # (1,1,.,.)
        Xg_pad = F.pad(Xg, (2, 2, 2, 2), mode='reflect')
        Rg_pad = F.pad(Rg, (2, 2, 2, 2), mode='reflect')
        mu_x = F.conv2d(Xg_pad, self._kernel5, padding=0)
        mu_r = F.conv2d(Rg_pad, self._kernel5, padding=0)
        mu_x2 = F.conv2d(Xg_pad * Xg_pad, self._kernel5, padding=0)
        mu_r2 = F.conv2d(Rg_pad * Rg_pad, self._kernel5, padding=0)
        var_x = torch.clamp(mu_x2 - mu_x * mu_x, min=0.0)
        var_r = torch.clamp(mu_r2 - mu_r * mu_r, min=0.0)
        mu_xr = F.conv2d(F.pad(Xg * Rg, (2, 2, 2, 2), mode='reflect'), self._kernel5, padding=0)
        cov = mu_xr - mu_x * mu_r
        ssim_map = (2 * mu_x * mu_r * 2 * cov) / ((mu_x * mu_x + mu_r * mu_r) * (var_x + var_r) + eps)
        ssim_avg = ssim_map.mean(dim=(1, 2, 3))
        d_g = 0.5 * (1.0 - ssim_avg)                    # (B,)

        # Gradient direction (weighted cosine)
        dot = dXdx_c * dRdx_c.unsqueeze(0) + dXdy_c * dRdy_c.unsqueeze(0)
        denom = magX * magR.unsqueeze(0) + eps
        cos = (dot / denom).clamp(-1.0, 1.0)
        wgt = torch.maximum(magX, magR.unsqueeze(0))
        s_dir = (cos * wgt).flatten(1).sum(dim=1) / (wgt.flatten(1).sum(dim=1) + eps)
        d_dir = 0.5 * (1.0 - s_dir)
        d_edge = 0.5 * (d_g + d_dir)                    # (B,)

        # Curvature (Laplacian) SSIM with weight w2
        self._ensure_lap_kernel()
        Lx = F.conv2d(Xb.unsqueeze(1), self._lap_kernel, padding=0).squeeze(1)  # (B,H-2,W-2)
        Lr = F.conv2d(ref.view(1, 1, H, W), self._lap_kernel, padding=0).squeeze(0).squeeze(0)  # (H-2,W-2)

        Lx1 = Lx.unsqueeze(1)                       # (B,1,H-2,W-2)
        Lr1 = Lr.unsqueeze(0).unsqueeze(1)          # (1,1,H-2,W-2)
        Lx_pad = F.pad(Lx1, (2, 2, 2, 2), mode='reflect')
        Lr_pad = F.pad(Lr1, (2, 2, 2, 2), mode='reflect')
        mu_lx = F.conv2d(Lx_pad, self._kernel5, padding=0)
        mu_lr = F.conv2d(Lr_pad, self._kernel5, padding=0)
        mu_lx2 = F.conv2d(Lx_pad * Lx_pad, self._kernel5, padding=0)
        mu_lr2 = F.conv2d(Lr_pad * Lr_pad, self._kernel5, padding=0)
        var_lx = torch.clamp(mu_lx2 - mu_lx * mu_lx, min=0.0)
        var_lr = torch.clamp(mu_lr2 - mu_lr * mu_lr, min=0.0)
        mu_xl = F.conv2d(F.pad(Lx1 * Lr1, (2, 2, 2, 2), mode='reflect'), self._kernel5, padding=0)
        cov_l = mu_xl - mu_lx * mu_lr

        ssim_l = (2 * mu_lx * mu_lr * 2 * cov_l) / ((mu_lx * mu_lx + mu_lr * mu_lr) * (var_lx + var_lr) + eps)  # (B,1,H-2,W-2)
        w2 = torch.maximum(torch.abs(Lx), torch.abs(Lr).unsqueeze(0))  # (B,H-2,W-2)
        Scurv = (ssim_l.squeeze(1) * w2).flatten(1).sum(dim=1) / (w2.flatten(1).sum(dim=1) + eps)
        Dcurv = 1.0 - Scurv                                            # (B,)

        # S1 row-wise min-max normalization within batch
        dS1 = self._s1_to_ref(Xb, ref)                  # (B,)
        min_s1, _ = dS1.min(dim=0, keepdim=True)
        max_s1, _ = dS1.max(dim=0, keepdim=True)
        dS1n = (dS1 - min_s1) / (max_s1 - min_s1 + eps)

        return torch.sqrt((d_edge * d_edge + Dcurv * Dcurv + dS1n * dS1n) / 3.0)

    @torch.no_grad()
    def _s1norm_to_ref(self, Xb: Tensor, ref: Tensor) -> Tensor:
        """
        Dimensionless S1-like distance to ref in [0,1]: D = 0.5 * r
        """
        eps = 1e-12
        dXdx = Xb[:, :, 1:] - Xb[:, :, :-1]
        dXdy = Xb[:, 1:, :] - Xb[:, :-1, :]
        dRdx = ref[:, 1:] - ref[:, :-1]
        dRdy = ref[1:, :] - ref[:-1, :]
        num = (torch.abs(dXdx - dRdx.view(1, *dRdx.shape)).sum(dim=(1, 2)) +
               torch.abs(dXdy - dRdy.view(1, *dRdy.shape)).sum(dim=(1, 2)))
        den = (torch.maximum(torch.abs(dXdx), torch.abs(dRdx).view(1, *dRdx.shape)).sum(dim=(1, 2)) +
               torch.maximum(torch.abs(dXdy), torch.abs(dRdy).view(1, *dRdy.shape)).sum(dim=(1, 2)))
        r = num / (den + eps)
        return 0.5 * r

    @torch.no_grad()
    def _moment_to_ref(self, Xb: Tensor, ref: Tensor) -> Tensor:
        """
        Moment-based distance to ref in [0,1].
        """
        eps = 1e-12
        B, H, W = Xb.shape
        d2norm = math.sqrt(2.0)

        # batch side centroids
        Xp = torch.clamp(Xb, min=0.0); Xn = torch.clamp(-Xb, min=0.0)
        mp_b = Xp.sum(dim=(1, 2), keepdim=True)
        mn_b = Xn.sum(dim=(1, 2), keepdim=True)
        cxp_b = (Xp * self._xnorm).sum(dim=(1, 2), keepdim=True) / (mp_b + eps)
        cyp_b = (Xp * self._ynorm).sum(dim=(1, 2), keepdim=True) / (mp_b + eps)
        cxn_b = (Xn * self._xnorm).sum(dim=(1, 2), keepdim=True) / (mn_b + eps)
        cyn_b = (Xn * self._ynorm).sum(dim=(1, 2), keepdim=True) / (mn_b + eps)
        cxp_b[mp_b <= eps] = 0.5; cyp_b[mp_b <= eps] = 0.5
        cxn_b[mn_b <= eps] = 0.5; cyn_b[mn_b <= eps] = 0.5
        cxp_b = cxp_b.squeeze(-1).squeeze(-1); cyp_b = cyp_b.squeeze(-1).squeeze(-1)
        cxn_b = cxn_b.squeeze(-1).squeeze(-1); cyn_b = cyn_b.squeeze(-1).squeeze(-1)

        # ref side centroids
        Rp = torch.clamp(ref, min=0.0); Rn = torch.clamp(-ref, min=0.0)
        mrp = Rp.sum(); mrn = Rn.sum()
        cxp_r = (Rp * self._xnorm.squeeze(0)).sum() / (mrp + eps)
        cyp_r = (Rp * self._ynorm.squeeze(0)).sum() / (mrp + eps)
        cxn_r = (Rn * self._xnorm.squeeze(0)).sum() / (mrn + eps)
        cyn_r = (Rn * self._ynorm.squeeze(0)).sum() / (mrn + eps)
        if mrp <= eps:
            cxp_r = torch.tensor(0.5, device=Xb.device, dtype=Xb.dtype)
            cyp_r = torch.tensor(0.5, device=Xb.device, dtype=Xb.dtype)
        if mrn <= eps:
            cxn_r = torch.tensor(0.5, device=Xb.device, dtype=Xb.dtype)
            cyn_r = torch.tensor(0.5, device=Xb.device, dtype=Xb.dtype)

        dpos = torch.sqrt((cxp_b - cxp_r)**2 + (cyp_b - cyp_r)**2) / d2norm
        dneg = torch.sqrt((cxn_b - cxn_r)**2 + (cyn_b - cyn_r)**2) / d2norm
        return 0.5 * (dpos + dneg)

    @torch.no_grad()
    def _gsmd_to_ref(self, Xb: Tensor, ref: Tensor) -> Tensor:
        Dg = self._gssim_to_ref(Xb, ref)
        Ds = self._s1norm_to_ref(Xb, ref)
        Dm = self._moment_to_ref(Xb, ref)
        return torch.sqrt((Dg * Dg + Ds * Ds + Dm * Dm) / 3.0)

    @torch.no_grad()
    def _kappa_to_ref(self, Xb: Tensor, ref: Tensor) -> Tensor:
        """
        Kappa curvature distance to ref in [0,1]: D_k = 0.5 * Σ|κ(X)-κ(ref)| / Σ max(|κ(X)|,|κ(ref)|)
        """
        eps = 1e-12
        kx = self._kappa_field(Xb)                       # (B,hk,wk)
        kr = self._kappa_field(ref.unsqueeze(0)).squeeze(0)  # (hk,wk)
        num = torch.abs(kx - kr.unsqueeze(0)).flatten(1).sum(dim=1)
        den = torch.maximum(kx.abs(), kr.abs().unsqueeze(0)).flatten(1).sum(dim=1)
        return 0.5 * (num / (den + eps))

    @torch.no_grad()
    def _s1gk_to_ref(self, Xb: Tensor, ref: Tensor) -> Tensor:
        """
        S1GK (to ref): RMS of row-normalized S1, G-SSIM distance, and row-normalized Kappa curvature distance.
        """
        eps = 1e-12
        d1 = self._s1_to_ref(Xb, ref)        # (B,)
        dg = self._gssim_to_ref(Xb, ref)     # (B,)
        dk = self._kappa_to_ref(Xb, ref)     # (B,)
        d1_min, _ = d1.min(dim=0, keepdim=True)
        d1_max, _ = d1.max(dim=0, keepdim=True)
        dk_min, _ = dk.min(dim=0, keepdim=True)
        dk_max, _ = dk.max(dim=0, keepdim=True)
        d1n = (d1 - d1_min) / (d1_max - d1_min + eps)
        dkn = (dk - dk_min) / (dk_max - dk_min + eps)
        return torch.sqrt((d1n * d1n + dg * dg + dkn * dkn) / 3.0)

    @torch.no_grad()
    def _hff_to_ref(self, Xb: Tensor, ref: Tensor) -> Tensor:
        """
        HFF (to ref): D = (1 - SSIM5)^2 + SSIM5 * (1 - GSSIM) = dL^2 + (1 - dL) * dG
        Returns (B,)
        """
        dL = self._ssim5_to_ref(Xb, ref)  # (B,)
        dG = self._gssim_to_ref(Xb, ref)  # (B,)
        return dL * dL + (1.0 - dL) * dG

    @torch.no_grad()
    def _s3d_to_ref(self, Xb: Tensor, ref: Tensor) -> Tensor:
        """
        S3D to ref: sqrt((dL^2 + dG^2 + dC^2)/3), where
          dL = 1 - SSIM5(A,B), dG = gssim distance, dC = curvature structural distance.
        """
        dL = self._ssim5_to_ref(Xb, ref)
        dG = self._gssim_to_ref(Xb, ref)
        # curvature structural to ref
        eps = 1e-12
        B, H, W = Xb.shape
        self._ensure_lap_kernel()
        LA = F.conv2d(Xb.unsqueeze(1), self._lap_kernel, padding=0).squeeze(1)  # (B,h2,w2)
        LR = F.conv2d(ref.view(1, 1, H, W), self._lap_kernel, padding=0).squeeze(0).squeeze(0)  # (h2,w2)
        KA = torch.abs(LA); sA = torch.sign(LA)
        KR = torch.abs(LR); sR = torch.sign(LR)
        KR_b = KR.unsqueeze(0)
        Smag = (2.0 * KA * KR_b) / (KA * KA + KR_b * KR_b + eps)
        Ssign = 0.5 * (1.0 + sA * sR.unsqueeze(0))
        w = torch.maximum(KA, KR_b)
        S = (Smag * Ssign * w).flatten(1).sum(dim=1) / (w.flatten(1).sum(dim=1) + eps)
        dC = 1.0 - S
        return torch.sqrt((dL * dL + dG * dG + dC * dC) / 3.0)

    @torch.no_grad()
    def _gssim_to_ref(self, Xb: Tensor, ref: Tensor) -> Tensor:
        """
        gssim の対参照距離： D = 1 - S_GS （勾配強度・方向の重み付き類似度）
        """
        eps = 1e-12
        B, H, W = Xb.shape
        dXdx = Xb[:, :, 1:] - Xb[:, :, :-1]
        dXdy = Xb[:, 1:, :] - Xb[:, :-1, :]
        gx = dXdx[:, :-1, :]
        gy = dXdy[:, :, :-1]
        gmagX = torch.sqrt(gx * gx + gy * gy + eps)     # (B, H-1, W-1)

        dRdx = ref[:, 1:] - ref[:, :-1]
        dRdy = ref[1:, :] - ref[:-1, :]
        grx = dRdx[:-1, :]
        gry = dRdy[:, :-1]
        gmagR = torch.sqrt(grx * grx + gry * gry + eps) # (H-1, W-1)

        dot = gx * grx.unsqueeze(0) + gy * gry.unsqueeze(0)                     # (B,.,.)
        cos = (dot / (gmagX * gmagR.unsqueeze(0) + eps)).clamp(-1.0, 1.0)
        Sdir = 0.5 * (1.0 + cos)
        Smag = (2.0 * gmagX * gmagR.unsqueeze(0)) / (gmagX * gmagX + gmagR.unsqueeze(0) * gmagR.unsqueeze(0) + eps)
        S = Smag * Sdir
        w = torch.maximum(gmagX, gmagR.unsqueeze(0))
        sim = (S * w).flatten(1).sum(dim=1) / (w.flatten(1).sum(dim=1) + eps)   # (B,)
        return 1.0 - sim

    @torch.no_grad()
    def _curv_s1_to_ref(self, Xb: Tensor, ref: Tensor) -> Tensor:
        """
        Curvature S1-like ratio to reference on Laplacian fields, optionally area-weighted.
        Returns (B,)
        """
        eps = 1e-12
        B, H, W = Xb.shape
        self._ensure_lap_kernel()
        Lx = F.conv2d(Xb.unsqueeze(1), self._lap_kernel, padding=0).squeeze(1)  # (B,H-2,W-2)
        Lr = F.conv2d(ref.view(1,1,H,W), self._lap_kernel, padding=0).squeeze(0).squeeze(0)  # (H-2,W-2)
        diff = torch.abs(Lx - Lr.unsqueeze(0))                                   # (B,H-2,W-2)
        denom = torch.maximum(torch.abs(Lx), torch.abs(Lr).unsqueeze(0))         # (B,H-2,W-2)
        if self.area_w is not None:
            w_inner = self.area_w[1:-1,1:-1].view(1, H-2, W-2)
            num_s = (diff * w_inner).flatten(1).sum(dim=1)
            den_s = (denom * w_inner).flatten(1).sum(dim=1) + eps
        else:
            num_s = diff.flatten(1).sum(dim=1)
            den_s = denom.flatten(1).sum(dim=1) + eps
        return num_s / den_s

    @torch.no_grad()
    def _cfsd_to_ref(self, Xb: Tensor, ref: Tensor) -> Tensor:
        """
        CFSD to reference: RMS of Dg (G-SSIM), normalized S1, normalized curvature S1 across batch rows.
        """
        eps = 1e-12
        Dg = self._gssim_to_ref(Xb, ref)            # (B,)
        Ds1 = self._s1_to_ref(Xb, ref)              # (B,)
        Dc = self._curv_s1_to_ref(Xb, ref)          # (B,)
        min_s1, _ = Ds1.min(dim=0, keepdim=True)
        max_s1, _ = Ds1.max(dim=0, keepdim=True)
        min_c, _ = Dc.min(dim=0, keepdim=True)
        max_c, _ = Dc.max(dim=0, keepdim=True)
        Ds1n = (Ds1 - min_s1) / (max_s1 - min_s1 + eps)
        Dcn = (Dc - min_c) / (max_c - min_c + eps)
        return torch.sqrt((Dg * Dg + Ds1n * Ds1n + Dcn * Dcn) / 3.0)

    @torch.no_grad()
    def _s1gcurv_to_ref(self, Xb: Tensor, ref: Tensor) -> Tensor:
        """
        S1GCURV (to ref): RMS of D_s1 (=S1/200), D_edge (=G-SSIM distance), and D_curv (=0.5 * curvature S1-like ratio).
        Returns (B,)
        """
        d1 = self._s1_to_ref(Xb, ref)       # (B,) percent
        d1n = torch.clamp(d1 / 200.0, min=0.0, max=1.0)
        dEdge = self._gssim_to_ref(Xb, ref) # (B,)
        dCurv = 0.5 * self._curv_s1_to_ref(Xb, ref)  # (B,)
        return torch.sqrt((d1n * d1n + dEdge * dEdge + dCurv * dCurv) / 3.0)

    @torch.no_grad()
    def _distance_to_ref(self, Xb: Tensor, ref: Tensor) -> Tensor:
        """
        現在のactivation_distanceに対応した「Xb vs 単一参照ref」の距離ベクトル(B,)
        """
        if self.activation_distance == 'euclidean':
            return self._euclidean_to_ref(Xb, ref)
        elif self.activation_distance == 'ssim5':
            return self._ssim5_to_ref(Xb, ref)
        elif self.activation_distance == 's1':
            return self._s1_to_ref(Xb, ref)
        elif self.activation_distance == 's1ssim':
            d1 = self._s1_to_ref(Xb, ref)
            d2 = self._ssim5_to_ref(Xb, ref)
            min1, _ = d1.min(dim=0, keepdim=True)
            max1, _ = d1.max(dim=0, keepdim=True)
            min2, _ = d2.min(dim=0, keepdim=True)
            max2, _ = d2.max(dim=0, keepdim=True)
            dn1 = (d1 - min1) / (max1 - min1 + 1e-12)
            dn2 = (d2 - min2) / (max2 - min2 + 1e-12)
            return 0.5 * (dn1 + dn2)
        elif self.activation_distance == 's1ssim5_hf':
            # HF-S1SSIM5 (to ref): 行方向（バッチ内）でmin-max正規化後、D = u + (1-u)*v
            dS1 = self._s1_to_ref(Xb, ref)       # (B,)
            dSSIM = self._ssim5_to_ref(Xb, ref)  # (B,)
            eps = 1e-12
            min_s1, _ = dS1.min(dim=0, keepdim=True)
            max_s1, _ = dS1.max(dim=0, keepdim=True)
            min_ss, _ = dSSIM.min(dim=0, keepdim=True)
            max_ss, _ = dSSIM.max(dim=0, keepdim=True)
            v = (dS1 - min_s1) / (max_s1 - min_s1 + eps)    # normalized S1
            u = (dSSIM - min_ss) / (max_ss - min_ss + eps)  # normalized dSSIM
            return u + (1.0 - u) * v
        elif self.activation_distance == 's1ssim5_and':
            # AND合成 (to ref): 行方向min–max正規化後、D = max(U,V)
            dS1 = self._s1_to_ref(Xb, ref)
            dSSIM = self._ssim5_to_ref(Xb, ref)
            eps = 1e-12
            min_s1, _ = dS1.min(dim=0, keepdim=True)
            max_s1, _ = dS1.max(dim=0, keepdim=True)
            min_ss, _ = dSSIM.min(dim=0, keepdim=True)
            max_ss, _ = dSSIM.max(dim=0, keepdim=True)
            V = (dS1 - min_s1) / (max_s1 - min_s1 + eps)
            U = (dSSIM - min_ss) / (max_ss - min_ss + eps)
            return torch.maximum(U, V)
        elif self.activation_distance == 'pf_s1ssim':
            # 比例融合 (to ref): 正規化なしで積
            dS1 = self._s1_to_ref(Xb, ref)
            dSSIM = self._ssim5_to_ref(Xb, ref)
            return dS1 * dSSIM
        elif self.activation_distance == 's3d':
            return self._s3d_to_ref(Xb, ref)
        elif self.activation_distance == 'cfsd':
            return self._cfsd_to_ref(Xb, ref)
        elif self.activation_distance == 'hff':
            return self._hff_to_ref(Xb, ref)
        elif self.activation_distance == 's1gk':
            return self._s1gk_to_ref(Xb, ref)
        elif self.activation_distance == 's1gcurv':
            return self._s1gcurv_to_ref(Xb, ref)
        elif self.activation_distance == 'gsmd':
            return self._gsmd_to_ref(Xb, ref)
        elif self.activation_distance == 's1gl':
            return self._s1gl_to_ref(Xb, ref)
        elif self.activation_distance == 's1gssim':
            return self._s1gssim_to_ref(Xb, ref)
        elif self.activation_distance == 'gssim':
            return self._gssim_to_ref(Xb, ref)
        elif self.activation_distance == 'ms_s1':
            return self._ms_s1_to_ref(Xb, ref)
        elif self.activation_distance == 'msssim_s1g':
            return self._msssim_s1g_to_ref(Xb, ref)
        else:
            raise RuntimeError('Unknown activation_distance')

    # ---------- 学習 ----------
    @torch.no_grad()
    def train_batch(self,
                    data: np.ndarray,
                    num_iteration: int,
                    batch_size: int = 32,
                    verbose: bool = True,
                    log_interval: int = 50,
                    update_per_iteration: bool = False,
                    shuffle: bool = True):
        """
        σは self.total_iters を基準に self.global_iter + it で一方向に減衰。
        複数回に分けて呼んでも、set_total_iterations(total) 済みなら継続減衰します。
        """
        N, D = data.shape
        H, W = self.field_shape
        if D != H * W:
            raise ValueError(f'data dimension {D} != H*W {H*W}')
        Xall = _to_device(data, self.device, self.dtype).reshape(N, H, W)

        qhist: List[float] = []
        rng_idx = torch.arange(N, device=self.device)

        # total iters 未設定なら今回のnum_iterationを総回数とみなす
        if self.total_iters is None:
            self.total_iters = int(num_iteration)

        iterator = trange(num_iteration) if verbose else range(num_iteration)
        for it in iterator:
            # 学習全体での反復数に基づくσ
            sigma = self._sigma_at_val(self.global_iter + it, self.total_iters)

            numerator = torch.zeros_like(self.weights)  # (m,H,W)
            denominator = torch.zeros((self.m,), device=self.device, dtype=self.dtype)

            if shuffle:
                perm = torch.randperm(N, device=self.device)
                idx_all = rng_idx[perm]
            else:
                idx_all = rng_idx

            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                batch_idx = idx_all[start:end]
                Xb = Xall[batch_idx]

                bmu = self.bmu_indices(Xb, nodes_chunk=self.nodes_chunk)    # (B,)
                h = self._neighborhood(bmu, sigma)                          # (B,m)
                numerator += (h.unsqueeze(-1).unsqueeze(-1) * Xb.unsqueeze(1)).sum(dim=0)
                denominator += h.sum(dim=0)

                if update_per_iteration:
                    mask = (denominator > 0)
                    denom_safe = denominator.clone()
                    denom_safe[~mask] = 1.0
                    new_w = numerator / denom_safe.view(-1, 1, 1)
                    self.weights[mask] = new_w[mask]
                    numerator.zero_(); denominator.zero_()

            # 1イテレーションの最後に一括更新
            mask = (denominator > 0)
            if mask.any():
                denom_safe = denominator.clone()
                denom_safe[~mask] = 1.0
                new_w = numerator / denom_safe.view(-1, 1, 1)
                self.weights[mask] = new_w[mask]

            # 任意頻度のメドイド置換（距離一貫性の改善）
            if (self.medoid_replace_every is not None) and (((self.global_iter + it + 1) % self.medoid_replace_every) == 0):
                # 全データでBMUを計算して各ノードの最近傍サンプルで置換
                bmu_all = self.bmu_indices(Xall, nodes_chunk=self.nodes_chunk)  # (N,)
                for node in range(self.m):
                    idxs = (bmu_all == node).nonzero(as_tuple=False).flatten()
                    if idxs.numel() == 0:
                        continue
                    Xn = Xall[idxs]                                   # (Bn,H,W)
                    ref = self.weights[node]                          # (H,W)
                    d = self._distance_to_ref(Xn, ref)                # (Bn,)
                    pos = int(torch.argmin(d).item())
                    self.weights[node] = Xn[pos]

            # ログ用QE（固定サブセットを外部から渡すのが推奨だが、API互換のためここはそのまま）
            if (it % log_interval == 0) or (it == num_iteration - 1):
                qe = self.quantization_error(data, sample_limit=2048)
                qhist.append(qe)
                if verbose and hasattr(iterator, 'set_postfix'):
                    iterator.set_postfix(q_error=f"{qe:.6f}", sigma=f"{sigma:.3f}")

        # グローバル反復を進める
        self.global_iter += num_iteration

        return qhist

    # ---------- 評価 ----------
    @torch.no_grad()
    def quantization_error(self, data: np.ndarray, sample_limit: Optional[int] = None, batch_size: int = 64) -> float:
        N, D = data.shape
        H, W = self.field_shape
        if self.eval_indices is not None:
            # 固定評価
            X = _to_device(data, self.device, self.dtype)[self.eval_indices].reshape(-1, H, W)
        else:
            if sample_limit is not None and sample_limit < N:
                idx = np.random.choice(N, sample_limit, replace=False)
                X = data[idx]
            else:
                X = data
            X = _to_device(X, self.device, self.dtype).reshape(-1, H, W)
        total = 0.0; cnt = 0
        for start in range(0, X.shape[0], batch_size):
            end = min(start + batch_size, X.shape[0])
            Xb = X[start:end]
            d = self._distance_batch(Xb, nodes_chunk=self.nodes_chunk)
            mins = torch.min(d, dim=1).values
            total += float(mins.sum().item())
            cnt += Xb.shape[0]
        return total / max(1, cnt)

    @torch.no_grad()
    def predict(self, data: np.ndarray, batch_size: int = 64) -> np.ndarray:
        N, D = data.shape
        H, W = self.field_shape
        X = _to_device(data, self.device, self.dtype).reshape(N, H, W)
        bmu_all = []
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            Xb = X[start:end]
            bmu = self.bmu_indices(Xb, nodes_chunk=self.nodes_chunk)
            bmu_all.append(bmu)
        bmu_flat = torch.cat(bmu_all, dim=0)
        y = (bmu_flat % self.y).to(torch.long)
        x = (bmu_flat // self.y).to(torch.long)
        out = torch.stack([x, y], dim=1)
        return _as_numpy(out)
