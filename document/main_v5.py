# main_v5.py
# -*- coding: utf-8 -*-
import os
import sys
import json
import logging
import time
import argparse
from typing import Optional, List, Dict, Tuple
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import xarray as xr
import torch
import torch.nn.functional as F
import math

import matplotlib
matplotlib.use('Agg')  # サーバ上でも保存できるように
import matplotlib.pyplot as plt

# cartopy: 可視化に使用
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import Normalize

# 3type版 SOM（Euclidean/SSIM/S1対応, batchSOM）
from minisom import MiniSom as MultiDistMiniSom

# =====================================================
# ユーザ調整パラメータ
# =====================================================
SEED = 1

# SOM学習・推論（全期間版：3方式）
SOM_X, SOM_Y = 10, 10
NUM_ITER = 1000
BATCH_SIZE = 256
NODES_CHUNK = 32 # VRAM16GB:2, VRAM24GB:4
LOG_INTERVAL = 10
EVAL_SAMPLE_LIMIT = 4000
SOM_EVAL_SEGMENTS = 100  # NUM_ITER をこの個数の区間に分割して評価（区切り数）

# データ
DATA_FILE = './prmsl_era5_all_data_seasonal_large.nc'

# 期間（学習/検証）
LEARN_START = '1991-01-01'
LEARN_END   = '1999-12-31'
VALID_START = '2000-01-01'
VALID_END   = '2000-12-31'

# 出力先（v5）
RESULT_DIR   = './results_v5_iter1000_batch256_seed1'
LEARNING_ROOT = os.path.join(RESULT_DIR, 'learning_result')
VERIF_ROOT    = os.path.join(RESULT_DIR, 'verification_results')

# 基本ラベル（15）
BASE_LABELS = [
    '1', '2A', '2B', '2C', '2D', '3A', '3B', '3C', '3D', '4A', '4B', '5', '6A', '6B', '6C'
]


# =====================================================
# 再現性・ログ
# =====================================================
def set_reproducibility(seed: int = 42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    try:
        torch.set_num_threads(1)
    except Exception:
        pass


def setup_logging_v5():
    for d in [RESULT_DIR, LEARNING_ROOT, VERIF_ROOT]:
        os.makedirs(d, exist_ok=True)
    log_path = os.path.join(RESULT_DIR, 'run_v5.log')
    if os.path.exists(log_path):
        os.remove(log_path)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()]
    )
    logging.info("ログ初期化完了（run_v5.log）。")


# =====================================================
# 共通ユーティリティ
# =====================================================
def format_date_yyyymmdd(ts_val) -> str:
    """
    ndarrayのdatetime64や文字列混在に対しても YYYY/MM/DD に整形。
    失敗時は str(ts_val) を返す。
    """
    if ts_val is None:
        return ''
    try:
        ts = pd.to_datetime(ts_val)
        if pd.isna(ts):
            return ''
        return ts.strftime('%Y/%m/%d')
    except Exception:
        try:
            s = str(ts_val)
            if 'T' in s:
                s = s.split('T')[0]
            s = s.replace('-', '/')
            if len(s) >= 10:
                return s[:10]
            return s
        except Exception:
            return str(ts_val)


# =====================================================
# データ読み込み ＆ 前処理（hPa偏差、空間平均差し引き）
# =====================================================
def load_and_prepare_data_unified(filepath: str,
                                  start_date: Optional[str],
                                  end_date: Optional[str],
                                  device: str = 'cpu'):
    logging.info(f"データ読み込み: {filepath}")
    ds = xr.open_dataset(filepath, decode_times=True)

    # time座標名の検出
    if 'valid_time' in ds:
        time_coord = 'valid_time'
    elif 'time' in ds:
        time_coord = 'time'
    else:
        raise ValueError('No time coordinate named "valid_time" or "time".')

    # 期間指定
    if (start_date is not None) or (end_date is not None):
        sub = ds.sel({time_coord: slice(start_date, end_date)})
    else:
        sub = ds

    if 'msl' not in sub:
        raise ValueError('Variable "msl" not found in dataset.')

    msl = sub['msl'].astype('float32')

    # 次元名の標準化
    lat_name = 'latitude'
    lon_name = 'longitude'
    for dn in msl.dims:
        if 'lat' in dn.lower(): lat_name = dn
        if 'lon' in dn.lower(): lon_name = dn

    msl = msl.transpose(time_coord, lat_name, lon_name)  # (N,H,W)
    ntime = msl.sizes[time_coord]
    nlat = msl.sizes[lat_name]
    nlon = msl.sizes[lon_name]

    arr = msl.values  # (N,H,W) in Pa
    arr2 = arr.reshape(ntime, nlat*nlon)  # (N,D)

    # NaN行除外
    valid_mask = ~np.isnan(arr2).any(axis=1)
    arr2 = arr2[valid_mask]
    times = msl[time_coord].values[valid_mask]
    lat = sub[lat_name].values
    lon = sub[lon_name].values

    # ラベル（あれば）
    labels = None
    if 'label' in sub.variables:
        raw = sub['label'].values
        raw = raw[valid_mask]
        labels = [v.decode('utf-8') if isinstance(v, (bytes, bytearray)) else str(v) for v in raw]
        logging.info("ラベルを読み込みました。")

    # hPaへ換算 → 空間平均差し引き
    msl_hpa_flat = (arr2 / 100.0).astype(np.float32)  # (N,D)
    mean_per_sample = np.nanmean(msl_hpa_flat, axis=1, keepdims=True)
    anomaly_flat = msl_hpa_flat - mean_per_sample  # (N,D)
    X_for_s1 = torch.from_numpy(anomaly_flat).to(device=device, dtype=torch.float32)

    # 3D形状に戻す
    n = anomaly_flat.shape[0]
    msl_hpa = msl_hpa_flat.reshape(n, nlat, nlon)
    anomaly_hpa = anomaly_flat.reshape(n, nlat, nlon)

    logging.info(f"期間: {str(times.min()) if len(times)>0 else '?'} 〜 {str(times.max()) if len(times)>0 else '?'}")
    logging.info(f"サンプル数={n}, 解像度={nlat}x{nlon}")
    return X_for_s1, msl_hpa, anomaly_hpa, lat, lon, nlat, nlon, times, labels


# =====================================================
# 評価ユーティリティ（s1_clustering.pyから必要部分を移植）
# =====================================================
def _normalize_to_base_candidate(label_str: Optional[str]) -> Optional[str]:
    import unicodedata, re
    if label_str is None:
        return None
    s = str(label_str)
    s = unicodedata.normalize('NFKC', s)
    s = s.upper().strip()
    s = s.replace('＋', '+').replace('－', '-').replace('−', '-')
    s = re.sub(r'[^0-9A-Z]', '', s)
    return s if s != '' else None


def basic_label_or_none(label_str: Optional[str], base_labels: List[str]) -> Optional[str]:
    import re
    cand = _normalize_to_base_candidate(label_str)
    if cand is None:
        return None
    # 完全一致を優先
    if cand in base_labels:
        return cand
    # '2A+' → '2A' のようなパターンを許容（残りに英数字がなければOK）
    for bl in base_labels:
        if cand == bl:
            return bl
        if cand.startswith(bl):
            rest = cand[len(bl):]
            if re.search(r'[0-9A-Z]', rest) is None:
                return bl
    return None


def extract_base_components(raw_label: Optional[str], base_labels: List[str]) -> List[str]:
    import unicodedata, re
    if raw_label is None:
        return []
    s = unicodedata.normalize('NFKC', str(raw_label)).upper().strip()
    s = s.replace('＋', '+').replace('－', '-').replace('−', '-')
    tokens = re.split(r'[^0-9A-Z]+', s)
    comps: List[str] = []
    for t in tokens:
        if t in base_labels and t not in comps:
            comps.append(t)
    return comps


def primary_base_label(raw_label: Optional[str], base_labels: List[str]) -> Optional[str]:
    parts = extract_base_components(raw_label, base_labels)
    return parts[0] if parts else None


def build_confusion_matrix_only_base(clusters: List[List[int]],
                                     all_labels: List[Optional[str]],
                                     base_labels: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    num_clusters = len(clusters)
    cluster_names = [f'Cluster_{i+1}' for i in range(num_clusters)]
    cm = pd.DataFrame(0, index=base_labels, columns=cluster_names, dtype=int)
    for i, idxs in enumerate(clusters):
        col = cluster_names[i]
        cnt = Counter()
        for j in idxs:
            lbl = basic_label_or_none(all_labels[j], base_labels)
            if lbl is not None:
                cnt[lbl] += 1
        for lbl, k in cnt.items():
            cm.loc[lbl, col] = k
    return cm, cluster_names


def evaluate_clusters_only_base(clusters: List[List[int]],
                                all_labels: List[Optional[str]],
                                base_labels: List[str],
                                title: str = "評価（基本ラベルのみ）") -> Optional[Dict[str, float]]:
    logging.info(f"\n--- {title} ---")
    if not all_labels:
        logging.warning("ラベル無しのため評価をスキップします。")
        return None

    cm, cluster_names = build_confusion_matrix_only_base(clusters, all_labels, base_labels)
    present_labels = [l for l in base_labels if cm.loc[l].sum() > 0]
    if len(present_labels) == 0:
        logging.warning("基本ラベルに該当するサンプルがありません。評価をスキップします。")
        return None

    logging.info("【混同行列（基本ラベルのみ）】\n" + "\n" + cm.loc[present_labels, :].to_string())

    # 各クラスタの多数決（代表ラベル）
    cluster_majority: Dict[int, Optional[str]] = {}
    logging.info("\n【各クラスタの多数決（代表ラベル）】")
    for k in range(len(cluster_names)):
        col = cluster_names[k]
        col_counts = cm[col]
        col_sum = int(col_counts.sum())
        if col_sum == 0:
            cluster_majority[k] = None
            logging.info(f" - {col:<12}: 代表ラベル=None（基本ラベル出現なし）")
            continue
        top_label = col_counts.idxmax()
        top_count = int(col_counts.max())
        share = top_count / col_sum if col_sum > 0 else 0.0
        top3 = col_counts.sort_values(ascending=False)[:3]
        top3_str = ", ".join([f"{lbl}:{int(cnt)}" for lbl, cnt in top3.items()])
        logging.info(f" - {col:<12}: 代表={top_label:<3} 件数={top_count:4d} シェア={share:5.2f} | 上位: {top3_str}")
        cluster_majority[k] = top_label

    # Macro Recall (基本ラベル)
    logging.info("\n【各ラベルの再現率（代表クラスタ群ベース）】")
    per_label = {}
    for lbl in present_labels:
        row_sum = int(cm.loc[lbl, :].sum())
        cols_for_lbl = [cluster_names[k] for k in range(len(cluster_names)) if cluster_majority.get(k, None) == lbl]
        correct = int(cm.loc[lbl, cols_for_lbl].sum()) if cols_for_lbl else 0
        recall = correct / row_sum if row_sum > 0 else 0.0
        per_label[lbl] = {'N': row_sum, 'Correct': correct, 'Recall': recall}
        logging.info(f" - {lbl:<3}: N={row_sum:4d} Correct={correct:4d} Recall={recall:.4f} 代表={cols_for_lbl if cols_for_lbl else 'なし'}")
    macro_recall = float(np.mean([per_label[l]['Recall'] for l in present_labels]))

    # 複合ラベル考慮（基本+応用）
    logging.info("\n【複合ラベル考慮の再現率（基本+応用）】")
    composite_totals = Counter()
    for j, raw_label in enumerate(all_labels):
        components = extract_base_components(raw_label, base_labels)
        for comp in components:
            composite_totals[comp] += 1
    present_labels_composite = sorted([l for l in base_labels if composite_totals[l] > 0])

    macro_recall_composite = np.nan
    if present_labels_composite:
        composite_correct_recall = Counter()
        # 各サンプルの予測＝割当クラスタの代表（基本ラベル）
        # 代表が複合ラベルの構成に含まれていれば正解
        n_samples = sum(len(idxs) for idxs in clusters)
        sample_to_cluster = {}
        for ci, idxs in enumerate(clusters):
            for j in idxs:
                sample_to_cluster[j] = ci

        for j, raw_label in enumerate(all_labels):
            comps = extract_base_components(raw_label, base_labels)
            if not comps:
                continue
            ci = sample_to_cluster.get(j, -1)
            if ci < 0:
                continue
            pred = cluster_majority.get(ci)
            if pred is None:
                continue
            if pred in comps:
                composite_correct_recall[pred] += 1

        recalls_composite = []
        for lbl in present_labels_composite:
            total = int(composite_totals[lbl])
            correct = int(composite_correct_recall[lbl])
            recall = correct / total if total > 0 else 0.0
            recalls_composite.append(recall)
            logging.info(f" - {lbl:<3}: N={total:4d} Correct={correct:4d} Recall={recall:.4f}")
        if recalls_composite:
            macro_recall_composite = float(np.mean(recalls_composite))
    else:
        logging.warning("複合ラベル考慮での評価対象ラベルがありません.")

    metrics: Dict[str, float] = {
        'MacroRecall_majority': macro_recall,
        'MacroRecall_composite': macro_recall_composite
    }

    logging.info("\n【集計】")
    logging.info(f"Macro Recall (基本ラベル) = {macro_recall:.4f}")
    logging.info(f"Macro Recall (基本+応用) = {macro_recall_composite:.4f}")
    logging.info(f"--- {title} 終了 ---\n")
    return metrics


def plot_iteration_metrics(history: Dict[str, List[float]], save_path: str) -> None:
    iters = history.get('iteration', [])
    metrics_names = [k for k in history.keys() if k != 'iteration']
    n = len(metrics_names)
    n_cols = 2
    n_rows = (n + n_cols - 1) // n_cols if n > 0 else 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axes = np.atleast_1d(axes).flatten()
    for idx, mname in enumerate(metrics_names):
        ax = axes[idx]
        ax.plot(iters, history.get(mname, []), marker='o')
        ax.set_title(mname)
        ax.set_xlabel('Iteration')
        ax.set_ylabel(mname)
        ax.grid(True)
    for i in range(n, len(axes)):
        axes[i].axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_iteration_metrics_single(history: Dict[str, List[float]], out_dir: str, filename_prefix: str) -> None:
    """
    各指標ごとに1枚の画像を保存する（従来の4指標まとめ画像に加えて出力）
    """
    os.makedirs(out_dir, exist_ok=True)
    iters = history.get('iteration', [])
    for mname, values in history.items():
        if mname == 'iteration':
            continue
        fig = plt.figure(figsize=(6, 4))
        ax = plt.gca()
        ax.plot(iters, values, marker='o')
        ax.set_title(mname)
        ax.set_xlabel('Iteration')
        ax.set_ylabel(mname)
        ax.grid(True)
        fpath = os.path.join(out_dir, f'{filename_prefix}_iteration_vs_{mname}.png')
        plt.tight_layout()
        plt.savefig(fpath, dpi=200)
        plt.close(fig)


def save_metrics_history_to_csv(history: Dict[str, List[float]], out_csv: str) -> None:
    df = pd.DataFrame(history)
    df.to_csv(out_csv, index=False)


# =====================================================
# 3type_som側のユーティリティ（ログ・評価・可視化）
# =====================================================
class Logger:
    def __init__(self, path):
        self.path = path
        self.f = open(path, 'w', encoding='utf-8')
    def write(self, s):
        sys.stdout.write(s)
        self.f.write(s)
        self.f.flush()
    def close(self):
        self.f.close()


def winners_to_clusters(winners_xy, som_shape):
    clusters = [[] for _ in range(som_shape[0]*som_shape[1])]
    for i,(ix,iy) in enumerate(winners_xy):
        k = ix*som_shape[1] + iy
        clusters[k].append(i)
    return clusters


def plot_som_node_average_patterns(data_flat, winners_xy, lat, lon, som_shape, save_path, title):
    """
    ノード平均（セントロイド）マップ（偏差[hPa]）
    """
    H, W = len(lat), len(lon)
    X2 = data_flat.reshape(-1, H, W)  # 偏差[hPa]

    map_x, map_y = som_shape
    mean_patterns = np.full((map_x, map_y, H, W), np.nan, dtype=np.float32)
    counts = np.zeros((map_x, map_y), dtype=int)
    for ix in range(map_x):
        for iy in range(map_y):
            mask = (winners_xy[:,0]==ix) & (winners_xy[:,1]==iy)
            idxs = np.where(mask)[0]
            counts[ix,iy] = len(idxs)
            if len(idxs)>0:
                mean_patterns[ix,iy] = np.nanmean(X2[idxs], axis=0)

    vmin, vmax = -40, 40
    levels = np.linspace(vmin, vmax, 21)
    cmap = 'RdBu_r'

    nrows, ncols = som_shape[1], som_shape[0]
    figsize=(ncols*2.6, nrows*2.6)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize,
                               subplot_kw={'projection': ccrs.PlateCarree()})
    axes = np.atleast_2d(axes)
    axes = axes.T[::-1,:]

    last_cf=None
    for ix in range(map_x):
        for iy in range(map_y):
            ax = axes[ix,iy]
            mp = mean_patterns[ix,iy]
            if np.isnan(mp).all():
                ax.set_axis_off(); continue
            cf = ax.contourf(lon, lat, mp, levels=levels, cmap=cmap, extend='both', transform=ccrs.PlateCarree())
            ax.contour(lon, lat, mp, levels=levels, colors='k', linewidths=0.3, transform=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE.with_scale('50m'), edgecolor='black', linewidth=0.8)
            ax.set_extent([115, 155, 15, 55], ccrs.PlateCarree())
            ax.text(0.02,0.96,f'({ix},{iy}) N={counts[ix,iy]}', transform=ax.transAxes,
                    fontsize=9, va='top', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
            ax.set_xticks([]); ax.set_yticks([])
            last_cf=cf
    if last_cf is not None:
        # よりコンパクトな余白と大きめの色バー文字
        fig.subplots_adjust(left=0.04, right=0.88, top=0.95, bottom=0.04, wspace=0.05, hspace=0.05)
        cax = fig.add_axes([0.90, 0.12, 0.02, 0.76])
        cb = fig.colorbar(last_cf, cax=cax, label='Sea Level Pressure Anomaly (hPa)')
        cb.ax.tick_params(labelsize=12)
        cb.set_label('Sea Level Pressure Anomaly (hPa)', fontsize=42)
    plt.suptitle(title, fontsize=42, fontweight='bold')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def save_each_node_mean_image(data_flat, winners_xy, lat, lon, som_shape, out_dir, prefix):
    """
    ノード平均（セントロイド）の個別図を保存
    """
    os.makedirs(out_dir, exist_ok=True)
    H, W = len(lat), len(lon)
    X2 = data_flat.reshape(-1, H, W)  # 偏差[hPa]
    map_x, map_y = som_shape

    vmin, vmax = -40, 40
    levels = np.linspace(vmin, vmax, 21)
    cmap = 'RdBu_r'

    for ix in range(map_x):
        for iy in range(map_y):
            mask = (winners_xy[:,0]==ix) & (winners_xy[:,1]==iy)
            idxs = np.where(mask)[0]
            if len(idxs)>0:
                mean_img = np.nanmean(X2[idxs], axis=0)
            else:
                mean_img = np.full((H,W), np.nan, dtype=np.float32)

            fig = plt.figure(figsize=(4,3))
            ax = plt.axes(projection=ccrs.PlateCarree())
            cf = ax.contourf(lon, lat, mean_img, levels=levels, cmap=cmap, transform=ccrs.PlateCarree(), extend='both')
            ax.contour(lon, lat, mean_img, levels=levels, colors='k', linewidths=0.3, transform=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.8, edgecolor='black')
            ax.set_extent([115, 155, 15, 55], ccrs.PlateCarree())
            plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04, label='Sea Level Pressure Anomaly (hPa)')
            ax.set_title(f'({ix},{iy}) N={len(idxs)}')
            ax.set_xticks([]); ax.set_yticks([])
            fpath = os.path.join(out_dir, f'{prefix}_node_{ix}_{iy}.png')
            plt.tight_layout()
            plt.savefig(fpath, dpi=180)
            plt.close(fig)


def plot_label_distributions_base(winners_xy, labels_raw: List[Optional[str]],
                                  base_labels: List[str], som_shape: Tuple[int,int],
                                  save_dir: str, title_prefix: str):
    """
    基本ラベルのみの分布ヒートマップ（15種類を1枚にまとめる）
    """
    os.makedirs(save_dir, exist_ok=True)
    node_counts = {lbl: np.zeros((som_shape[0], som_shape[1]), dtype=int) for lbl in base_labels}
    for i,(ix,iy) in enumerate(winners_xy):
        lab = basic_label_or_none(labels_raw[i], base_labels)
        if lab in node_counts:
            node_counts[lab][ix,iy] += 1
    cols = 5
    rows = int(np.ceil(len(base_labels)/cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2.6, rows*2.6))
    axes = np.atleast_2d(axes)
    for idx,lbl in enumerate(base_labels):
        r = idx//cols; c=idx%cols
        ax = axes[r,c]
        # SOMノードグリッドを90度右回転（現在の「左」を「上」に）
        arr = node_counts[lbl]
        arr_rot = np.rot90(arr, -1)  # 90° clockwise
        local_max = int(arr_rot.max()) if arr_rot.size > 0 else 0
        vmax_local = local_max if local_max > 0 else 1
        im = ax.imshow(arr_rot, cmap='viridis', origin='upper', interpolation='nearest', aspect='equal', vmin=0, vmax=vmax_local)
        ax.set_title(lbl); ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for k in range(len(base_labels), rows*cols):
        r = k//cols; c=k%cols
        axes[r,c].axis('off')
    plt.suptitle(f'{title_prefix} Label Distributions on SOM nodes (Base only)', fontsize=14)
    plt.tight_layout(rect=[0,0,1,0.95])
    fpath = os.path.join(save_dir, f'{title_prefix}_label_distributions_base.png')
    plt.savefig(fpath, dpi=250)
    plt.close(fig)


def save_label_distributions_base_individual(winners_xy, labels_raw: List[Optional[str]],
                                             base_labels: List[str], som_shape: Tuple[int,int],
                                             save_dir: str, title_prefix: str):
    """
    基本ラベルのみの分布ヒートマップ（各ラベルごとの個別画像を追加保存）
    """
    os.makedirs(save_dir, exist_ok=True)
    node_counts = {lbl: np.zeros((som_shape[0], som_shape[1]), dtype=int) for lbl in base_labels}
    for i,(ix,iy) in enumerate(winners_xy):
        lab = basic_label_or_none(labels_raw[i], base_labels)
        if lab in node_counts:
            node_counts[lab][ix,iy] += 1
    for lbl in base_labels:
        fig = plt.figure(figsize=(4, 3))
        ax = plt.gca()
        # SOMノードグリッドを90度右回転（現在の「左」を「上」に）
        arr = node_counts[lbl]
        arr_rot = np.rot90(arr, -1)  # 90° clockwise
        im = ax.imshow(arr_rot, cmap='viridis', origin='upper', interpolation='nearest', aspect='equal', vmin=0)
        ax.set_title(lbl)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fpath = os.path.join(save_dir, f'{title_prefix}_label_dist_base_{lbl}.png')
        plt.tight_layout()
        plt.savefig(fpath, dpi=200)
        plt.close(fig)


def analyze_nodes_detail_to_log(clusters: List[List[int]],
                                labels: List[Optional[str]],
                                timestamps: np.ndarray,
                                base_labels: List[str],
                                som_shape: Tuple[int, int],
                                log: Logger,
                                title: str):
    """
    ノードごとの詳細（基本ラベル構成・月別分布・純度・代表ラベル[raw]）を results.log に追記。
    """
    log.write(f'\n--- {title} ---\n')
    for k, idxs in enumerate(clusters):
        ix, iy = k // som_shape[1], k % som_shape[1]
        n = len(idxs)
        if n == 0:
            continue
        log.write(f'\n[Node ({ix},{iy})] N={n}\n')
        # 代表ラベル（元ラベル：複合含む）
        cnt_raw = Counter([labels[j] for j in idxs if labels[j] is not None])
        if len(cnt_raw) > 0:
            top_raw, top_count_raw = cnt_raw.most_common(1)[0]
            log.write(f'  - 代表ラベル（元ラベル）: {top_raw}  ({top_count_raw}/{n}, {top_count_raw/n*100:5.1f}%)\n')

        # 基本ラベル構成
        cnt = Counter()
        for j in idxs:
            bl = basic_label_or_none(labels[j], base_labels)
            if bl is not None:
                cnt[bl] += 1
        if cnt:
            log.write('  - ラベル構成（基本ラベルのみ）:\n')
            for lbl, c in sorted(cnt.items(), key=lambda x: x[1], reverse=True):
                log.write(f'    {lbl:<3}: {c:4d} ({c/n*100:5.1f}%)\n')
            purity = max(cnt.values()) / n
            log.write(f'  - ノード純度（基本ラベル多数決）: {purity:.3f}\n')

        # 月別分布
        if timestamps is not None and n > 0:
            months = pd.to_datetime(timestamps[idxs]).month
            mon_c = Counter(months)
            log.write('  - 月別分布:\n')
            for m in range(1, 13):
                c = mon_c.get(m, 0)
                log.write(f'    {m:2d}月: {c:4d} ({c/n*100:5.1f}%)\n')
    log.write(f'--- {title} 終了 ---\n')


def log_som_recall_by_label_with_nodes(
    log: Logger,
    winners_xy: np.ndarray,
    labels_all: List[Optional[str]],
    base_labels: List[str],
    som_shape: Tuple[int, int],
    section_title: str
):
    if labels_all is None or len(labels_all) == 0:
        log.write("\nラベルが無いため、代表ノード群ベースの再現率出力をスキップします。\n")
        return

    H_nodes, W_nodes = som_shape
    n_nodes = H_nodes * W_nodes
    node_index_arr = winners_xy[:, 0] * W_nodes + winners_xy[:, 1]  # (N,)

    # ノード毎の基本ラベル分布
    node_counters = [Counter() for _ in range(n_nodes)]
    for i, k in enumerate(node_index_arr):
        bl = basic_label_or_none(labels_all[i], base_labels)
        if bl is not None:
            node_counters[int(k)][bl] += 1

    # ノードの代表（多数決）ラベル
    node_majority: List[Optional[str]] = [None] * n_nodes
    for k in range(n_nodes):
        if len(node_counters[k]) > 0:
            node_majority[k] = node_counters[k].most_common(1)[0][0]

    # ラベル→代表ノード一覧
    label_to_nodes: Dict[str, List[Tuple[int, int]]] = {lbl: [] for lbl in base_labels}
    for k, rep in enumerate(node_majority):
        if rep is None:
            continue
        ix, iy = k // W_nodes, k % W_nodes
        label_to_nodes[rep].append((ix, iy))

    # 基本ラベルベースの再現率
    total_base = Counter()
    correct_base = Counter()
    for i, k in enumerate(node_index_arr):
        bl = basic_label_or_none(labels_all[i], base_labels)
        if bl is None:
            continue
        total_base[bl] += 1
        pred = node_majority[int(k)]
        if pred is None:
            continue
        if pred == bl:
            correct_base[bl] += 1

    # 複合ラベル考慮（基本+応用）
    total_comp = Counter()
    correct_comp = Counter()
    for i, k in enumerate(node_index_arr):
        comps = extract_base_components(labels_all[i], base_labels)
        if not comps:
            continue
        for c in comps:
            total_comp[c] += 1
        pred = node_majority[int(k)]
        if pred is None:
            continue
        if pred in comps:
            correct_comp[pred] += 1

    log.write(f"\n【{section_title}】\n")
    log.write("【各ラベルの再現率（代表ノード群ベース）】\n")
    recalls_base = []
    for lbl in base_labels:
        N = int(total_base[lbl])
        C = int(correct_base[lbl])
        rec = (C / N) if N > 0 else 0.0
        recalls_base.append(rec if N > 0 else np.nan)
        nodes_disp = label_to_nodes.get(lbl, [])
        rep_str = "なし" if len(nodes_disp) == 0 else "[" + ", ".join([f"({ix},{iy})" for ix, iy in nodes_disp]) + "]"
        log.write(f" - {lbl:<3}: N={N:4d} Correct={C:4d} Recall={rec:.4f} 代表={rep_str}\n")

    log.write("\n【複合ラベル考慮の再現率（基本+応用）】\n")
    recalls_comp = []
    for lbl in base_labels:
        Nt = int(total_comp[lbl])
        Ct = int(correct_comp[lbl])
        rec_t = (Ct / Nt) if Nt > 0 else 0.0
        if Nt > 0:
            recalls_comp.append(rec_t)
        log.write(f" - {lbl:<3}: N={Nt:4d} Correct={Ct:4d} Recall={rec_t:.4f}\n")

    base_valid = [r for r, lbl in zip(recalls_base, base_labels) if not np.isnan(r) and total_base[lbl] > 0]
    macro_base = float(np.mean(base_valid)) if len(base_valid) > 0 else float('nan')
    macro_comp = float(np.mean(recalls_comp)) if len(recalls_comp) > 0 else float('nan')
    log.write(f"\n[Summary] Macro Recall (基本ラベル)   = {macro_base:.4f}\n")
    log.write(f"[Summary] Macro Recall (基本+応用) = {macro_comp:.4f}\n")


# =====================================================
# True Medoid / 距離計算ユーティリティ（SSIM/S1等）
# =====================================================
def _euclidean_dist_to_ref(Xb: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    # Xb: (B,H,W), ref: (H,W) -> (B,)
    diff = Xb - ref.view(1, *ref.shape)
    d2 = (diff*diff).sum(dim=(1,2))
    return torch.sqrt(d2 + 1e-12)




def _ssim5_dist_to_ref(Xb: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    1 - mean(SSIM_map) with 5x5 moving window, C1=C2=0 (denominator epsilon guarded)
    """
    B, H, W = Xb.shape
    device = Xb.device
    dtype = Xb.dtype
    pad = 2
    kernel = torch.ones((1, 1, 5, 5), device=device, dtype=dtype) / 25.0

    X = Xb.unsqueeze(1)              # (B,1,H,W)
    R = ref.view(1, 1, H, W)         # (1,1,H,W)

    X_pad = F.pad(X, (pad, pad, pad, pad), mode='reflect')
    R_pad = F.pad(R, (pad, pad, pad, pad), mode='reflect')

    mu_x = F.conv2d(X_pad, kernel, padding=0)            # (B,1,H,W)
    mu_r = F.conv2d(R_pad, kernel, padding=0)            # (1,1,H,W)

    mu_x2 = F.conv2d(X_pad * X_pad, kernel, padding=0)
    mu_r2 = F.conv2d(R_pad * R_pad, kernel, padding=0)
    var_x = torch.clamp(mu_x2 - mu_x * mu_x, min=0.0)
    var_r = torch.clamp(mu_r2 - mu_r * mu_r, min=0.0)

    prod = X * R
    prod_pad = F.pad(prod, (pad, pad, pad, pad), mode='reflect')
    mu_xr = F.conv2d(prod_pad, kernel, padding=0)
    cov = mu_xr - mu_x * mu_r

    eps = 1e-12
    l_num = 2 * (mu_x * mu_r)
    l_den = (mu_x * mu_x + mu_r * mu_r)
    c_num = 2 * cov
    c_den = (var_x + var_r)
    ssim_map = (l_num * c_num) / (l_den * c_den + eps)    # (B,1,H,W)
    ssim_avg = ssim_map.mean(dim=(1, 2, 3))               # (B,)
    return 1.0 - ssim_avg

def _s1_dist_to_ref(Xb: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    # Teweles–Wobus S1: 100 * sum|∇X-∇ref| / sum max(|∇X|,|∇ref|)
    dXdx = Xb[:,:,1:] - Xb[:,:,:-1]
    dXdy = Xb[:,1:,:] - Xb[:,:-1,:]
    dRdx = ref[:,1:] - ref[:,:-1]
    dRdy = ref[1:,:] - ref[:-1,:]
    num_dx = (torch.abs(dXdx - dRdx.view(1, *dRdx.shape))).sum(dim=(1,2))
    num_dy = (torch.abs(dXdy - dRdy.view(1, *dRdy.shape))).sum(dim=(1,2))
    den_dx = torch.maximum(torch.abs(dXdx), torch.abs(dRdx).view(1, *dRdx.shape)).sum(dim=(1,2))
    den_dy = torch.maximum(torch.abs(dXdy), torch.abs(dRdy).view(1, *dRdy.shape)).sum(dim=(1,2))
    s1 = 100.0 * (num_dx + num_dy) / (den_dx + den_dy + 1e-12)
    return s1


def _grad_ssim_dir_to_ref(Xb: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    勾配構造（強度SSIM + 方向cosθ）の合成不一致 d_edge を返す [0,1]。
    - d_g = 0.5 * (1 - SSIM(|∇X|, |∇R|)) with 5x5 window, C=0
    - d_dir = 0.5 * (1 - s_dir), s_dir = 加重平均_{w=max(|∇X|,|∇R|)}(cosθ)
    - d_edge = 0.5 * (d_g + d_dir)
    """
    eps = 1e-12
    B, H, W = Xb.shape
    # 勾配（中心を揃える）
    dXdx = Xb[:, :, 1:] - Xb[:, :, :-1]
    dXdy = Xb[:, 1:, :] - Xb[:, :-1, :]
    dRdx = ref[:, 1:] - ref[:, :-1]
    dRdy = ref[1:, :] - ref[:-1, :]
    dXdx_c, dXdy_c = dXdx[:, :-1, :], dXdy[:, :, :-1]
    dRdx_c, dRdy_c = dRdx[:-1, :], dRdy[:, :-1]
    magX = torch.sqrt(dXdx_c * dXdx_c + dXdy_c * dXdy_c + eps)   # (B,H-1,W-1)
    magR = torch.sqrt(dRdx_c * dRdx_c + dRdy_c * dRdy_c + eps)   # (H-1,W-1)

    # SSIM(|∇|) with 5x5, C=0
    pad = 2
    kernel = torch.ones((1, 1, 5, 5), device=Xb.device, dtype=Xb.dtype) / 25.0
    Xg = magX.unsqueeze(1)                          # (B,1,.,.)
    Rg = magR.unsqueeze(0).unsqueeze(1)             # (1,1,.,.)
    Xg_pad = F.pad(Xg, (pad, pad, pad, pad), mode='reflect')
    Rg_pad = F.pad(Rg, (pad, pad, pad, pad), mode='reflect')
    mu_x = F.conv2d(Xg_pad, kernel, padding=0)
    mu_r = F.conv2d(Rg_pad, kernel, padding=0)
    mu_x2 = F.conv2d(Xg_pad * Xg_pad, kernel, padding=0)
    mu_r2 = F.conv2d(Rg_pad * Rg_pad, kernel, padding=0)
    var_x = torch.clamp(mu_x2 - mu_x * mu_x, min=0.0)
    var_r = torch.clamp(mu_r2 - mu_r * mu_r, min=0.0)
    mu_xr = F.conv2d(F.pad(Xg * Rg, (pad, pad, pad, pad), mode='reflect'), kernel, padding=0)
    cov = mu_xr - mu_x * mu_r
    ssim_map = (2 * mu_x * mu_r * 2 * cov) / ((mu_x * mu_x + mu_r * mu_r) * (var_x + var_r) + eps)
    ssim_avg = ssim_map.mean(dim=(1, 2, 3))
    d_g = 0.5 * (1.0 - ssim_avg)                    # (B,)

    # 方向cosθの重み付き平均
    dot = dXdx_c * dRdx_c.unsqueeze(0) + dXdy_c * dRdy_c.unsqueeze(0)
    denom = magX * magR.unsqueeze(0) + eps
    cos = (dot / denom).clamp(-1.0, 1.0)
    wgt = torch.maximum(magX, magR.unsqueeze(0))
    s_dir = (cos * wgt).flatten(1).sum(dim=1) / (wgt.flatten(1).sum(dim=1) + eps)
    d_dir = 0.5 * (1.0 - s_dir)
    return 0.5 * (d_g + d_dir)                      # (B,)


def _gssim_dist_to_ref(Xb: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    G-SSIM（勾配強度・方向の重み付き類似 S を max(|∇|)で加重平均）に対する距離 D = 1 - S。
    """
    eps = 1e-12
    B, H, W = Xb.shape
    dXdx = Xb[:, :, 1:] - Xb[:, :, :-1]
    dXdy = Xb[:, 1:, :] - Xb[:, :-1, :]
    gx = dXdx[:, :-1, :]
    gy = dXdy[:, :, :-1]
    gmagX = torch.sqrt(gx * gx + gy * gy + eps)     # (B,H-1,W-1)

    dRdx = ref[:, 1:] - ref[:, :-1]
    dRdy = ref[1:, :] - ref[:-1, :]
    grx = dRdx[:-1, :]
    gry = dRdy[:, :-1]
    gmagR = torch.sqrt(grx * grx + gry * gry + eps) # (H-1,W-1)

    dot = gx * grx.unsqueeze(0) + gy * gry.unsqueeze(0)
    cos = (dot / (gmagX * gmagR.unsqueeze(0) + eps)).clamp(-1.0, 1.0)
    Sdir = 0.5 * (1.0 + cos)
    Smag = (2.0 * gmagX * gmagR.unsqueeze(0)) / (gmagX * gmagX + gmagR.unsqueeze(0) * gmagR.unsqueeze(0) + eps)
    S = Smag * Sdir
    w = torch.maximum(gmagX, gmagR.unsqueeze(0))
    sim = (S * w).flatten(1).sum(dim=1) / (w.flatten(1).sum(dim=1) + eps)   # (B,)
    return 1.0 - sim

def _curv_ssim_weighted_to_ref(Xb: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    Curvature structural dissimilarity Dcurv = 1 - weighted SSIM(ΔX, Δref)
    - Laplacian kernel 3x3 (5-point)
    - SSIM with 5x5 window, C=0 (denominator epsilon guarded)
    - Weighted by w2 = max(|ΔX|, |Δref|) per-pixel
    Returns: (B,)
    """
    eps = 1e-12
    B, H, W = Xb.shape
    device = Xb.device
    dtype = Xb.dtype
    pad = 2
    lap = torch.tensor([[0.0, 1.0, 0.0],
                        [1.0,-4.0, 1.0],
                        [0.0, 1.0, 0.0]], device=device, dtype=dtype).view(1,1,3,3)
    kernel = torch.ones((1,1,5,5), device=device, dtype=dtype) / 25.0

    # Laplacians
    Lx1 = F.conv2d(Xb.unsqueeze(1), lap, padding=0)                    # (B,1,H-2,W-2)
    Lr1 = F.conv2d(ref.view(1,1,H,W), lap, padding=0)                  # (1,1,H-2,W-2)

    # 5x5 local stats
    Lx_pad = F.pad(Lx1, (pad,pad,pad,pad), mode='reflect')
    Lr_pad = F.pad(Lr1, (pad,pad,pad,pad), mode='reflect')
    mu_lx  = F.conv2d(Lx_pad, kernel, padding=0)                       # (B,1,H-2,W-2)
    mu_lr  = F.conv2d(Lr_pad, kernel, padding=0)                       # (1,1,H-2,W-2)
    mu_lx2 = F.conv2d(Lx_pad*Lx_pad, kernel, padding=0)
    mu_lr2 = F.conv2d(Lr_pad*Lr_pad, kernel, padding=0)
    var_lx = torch.clamp(mu_lx2 - mu_lx*mu_lx, min=0.0)
    var_lr = torch.clamp(mu_lr2 - mu_lr*mu_lr, min=0.0)
    mu_xl  = F.conv2d(F.pad(Lx1*Lr1, (pad,pad,pad,pad), mode='reflect'), kernel, padding=0)
    cov_l  = mu_xl - mu_lx*mu_lr

    ssim_l = (2*mu_lx*mu_lr * 2*cov_l) / ((mu_lx*mu_lx + mu_lr*mu_lr)*(var_lx + var_lr) + eps)  # (B,1,H-2,W-2)

    # Weighted average with w2 = max(|ΔX|, |Δref|)
    w2 = torch.maximum(torch.abs(Lx1), torch.abs(Lr1))                  # (B,1,H-2,W-2) via broadcast
    num = (ssim_l * w2).flatten(2).sum(dim=2)                           # (B,1)
    den = (w2).flatten(2).sum(dim=2) + eps                              # (B,1)
    Scurv = (num / den).squeeze(1)                                      # (B,)
    return 1.0 - Scurv

def _s1norm_dist_to_ref(Xb: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    Dimensionless S1-like distance in [0,1]: D = 0.5 * r
      r = (sum|∂X-∂ref|) / (sum max(|∂X|,|∂ref|) + eps) for x and y directions combined.
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

def _moment_dist_to_ref(Xb: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    Moment-based distance in [0,1]: D_mom = 0.5 * (d_pos + d_neg)
    using normalized coordinate grid [0,1] for both axes, diagonal length sqrt(2).
    """
    eps = 1e-12
    B, H, W = Xb.shape
    device = Xb.device
    dtype = Xb.dtype
    d2 = 2.0 ** 0.5

    # Normalized coordinate grids (broadcastable)
    xnorm = torch.linspace(0.0, 1.0, W, device=device, dtype=dtype).view(1, 1, W).expand(B, H, W)
    ynorm = torch.linspace(0.0, 1.0, H, device=device, dtype=dtype).view(1, H, 1).expand(B, H, W)

    # Batch side positive/negative masses and centroids
    Xp = torch.clamp(Xb, min=0.0); Xn = torch.clamp(-Xb, min=0.0)
    mp = Xp.sum(dim=(1, 2), keepdim=True)  # (B,1)
    mn = Xn.sum(dim=(1, 2), keepdim=True)  # (B,1)

    cxp_b = (Xp * xnorm).sum(dim=(1, 2), keepdim=True) / (mp + eps)
    cyp_b = (Xp * ynorm).sum(dim=(1, 2), keepdim=True) / (mp + eps)
    cxn_b = (Xn * xnorm).sum(dim=(1, 2), keepdim=True) / (mn + eps)
    cyn_b = (Xn * ynorm).sum(dim=(1, 2), keepdim=True) / (mn + eps)
    cxp_b[mp <= eps] = 0.5; cyp_b[mp <= eps] = 0.5
    cxn_b[mn <= eps] = 0.5; cyn_b[mn <= eps] = 0.5
    cxp_b = cxp_b.squeeze(-1).squeeze(-1); cyp_b = cyp_b.squeeze(-1).squeeze(-1)  # (B,)
    cxn_b = cxn_b.squeeze(-1).squeeze(-1); cyn_b = cyn_b.squeeze(-1).squeeze(-1)

    # Reference side (single map)
    Rp = torch.clamp(ref, min=0.0); Rn = torch.clamp(-ref, min=0.0)
    mrp = Rp.sum(); mrn = Rn.sum()
    # For ref, reuse 1-sample coordinate grids
    xnorm1 = torch.linspace(0.0, 1.0, W, device=device, dtype=dtype).view(1, 1, W).expand(1, H, W).squeeze(0)
    ynorm1 = torch.linspace(0.0, 1.0, H, device=device, dtype=dtype).view(1, H, 1).expand(1, H, W).squeeze(0)
    cxp_r = (Rp * xnorm1).sum() / (mrp + eps)
    cyp_r = (Rp * ynorm1).sum() / (mrp + eps)
    cxn_r = (Rn * xnorm1).sum() / (mrn + eps)
    cyn_r = (Rn * ynorm1).sum() / (mrn + eps)
    if mrp <= eps:
        cxp_r = torch.tensor(0.5, device=device, dtype=dtype)
        cyp_r = torch.tensor(0.5, device=device, dtype=dtype)
    if mrn <= eps:
        cxn_r = torch.tensor(0.5, device=device, dtype=dtype)
        cyn_r = torch.tensor(0.5, device=device, dtype=dtype)

    dpos = torch.sqrt((cxp_b - cxp_r)**2 + (cyp_b - cyp_r)**2) / d2
    dneg = torch.sqrt((cxn_b - cxn_r)**2 + (cyn_b - cyn_r)**2) / d2
    return 0.5 * (dpos + dneg)

def _gsmd_dist_to_ref(Xb: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    GSMD distance to reference: sqrt((Dg^2 + Ds^2 + Dm^2)/3)
      - Dg = G-SSIM distance (0..1)
      - Ds = dimensionless S1-like distance (0..1)
      - Dm = moment centroid distance (0..1)
    """
    Dg = _gssim_dist_to_ref(Xb, ref)
    Ds = _s1norm_dist_to_ref(Xb, ref)
    Dm = _moment_dist_to_ref(Xb, ref)
    return torch.sqrt((Dg*Dg + Ds*Ds + Dm*Dm) / 3.0)

def _curv_struct_dist_to_ref(Xb: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    Curvature structural dissimilarity d_C in [0,1] (S3D component):
      - Laplacian magnitude ratio Smag = 2|ΔX||ΔR|/(|ΔX|^2+|ΔR|^2+eps)
      - Sign agreement Ssign = (1 + sign(ΔX)*sign(ΔR))/2
      - Weighted by w = max(|ΔX|, |ΔR|)
      - d_C = 1 - weighted_mean(Smag * Ssign)
    """
    eps = 1e-12
    B, H, W = Xb.shape
    device = Xb.device; dtype = Xb.dtype
    lap = torch.tensor([[0.0, 1.0, 0.0],
                        [1.0,-4.0, 1.0],
                        [0.0, 1.0, 0.0]], device=device, dtype=dtype).view(1,1,3,3)

    LA = F.conv2d(Xb.unsqueeze(1), lap, padding=0).squeeze(1)  # (B,H-2,W-2)
    LR = F.conv2d(ref.view(1,1,H,W), lap, padding=0).squeeze(0).squeeze(0)  # (H-2,W-2)

    KA = torch.abs(LA); sA = torch.sign(LA)
    KR = torch.abs(LR); sR = torch.sign(LR)

    KR_b = KR.unsqueeze(0)    # (1,h2,w2)
    Smag = (2.0 * KA * KR_b) / (KA*KA + KR_b*KR_b + eps)  # (B,h2,w2)
    Ssign = 0.5 * (1.0 + sA * sR.unsqueeze(0))
    w = torch.maximum(KA, KR_b)

    S = (Smag * Ssign * w).flatten(1).sum(dim=1) / (w.flatten(1).sum(dim=1) + eps)  # (B,)
    return 1.0 - S

def _curv_s1_dist_to_ref(Xb: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    Curvature S1-like ratio on Laplacian fields: sum|ΔX-ΔR| / sum max(|ΔX|,|ΔR|).
    Returns (B,) roughly in [0,2]; for CFSD we row-normalize to [0,1].
    """
    eps = 1e-12
    B, H, W = Xb.shape
    device = Xb.device; dtype = Xb.dtype
    lap = torch.tensor([[0.0, 1.0, 0.0],
                        [1.0,-4.0, 1.0],
                        [0.0, 1.0, 0.0]], device=device, dtype=dtype).view(1,1,3,3)
    Lx = F.conv2d(Xb.unsqueeze(1), lap, padding=0).squeeze(1)                 # (B,H-2,W-2)
    Lr = F.conv2d(ref.view(1,1,H,W), lap, padding=0).squeeze(0).squeeze(0)    # (H-2,W-2)
    diff = torch.abs(Lx - Lr.unsqueeze(0))                                     # (B,H-2,W-2)
    denom = torch.maximum(torch.abs(Lx), torch.abs(Lr).unsqueeze(0))           # (B,H-2,W-2)
    return diff.flatten(1).sum(dim=1) / (denom.flatten(1).sum(dim=1) + eps)

def _kappa_to_ref(Xb: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    Kappa curvature distance to reference in [0,1]:
      κ(Z) = div(∇Z/|∇Z|) computed on inner grid via centered differences.
      D_k = 0.5 * sum|κ(X)-κ(R)| / sum max(|κ(X)|, |κ(R)|)
    """
    eps = 1e-12
    B, H, W = Xb.shape
    device = Xb.device
    dtype = Xb.dtype

    def kappa_field(Z: torch.Tensor) -> torch.Tensor:
        # Z: (B,H,W)
        dZdx = Z[:, :, 1:] - Z[:, :, :-1]    # (B,H,W-1)
        dZdy = Z[:, 1:, :] - Z[:, :-1, :]    # (B,H-1,W)
        gx = dZdx[:, :-1, :]                 # (B,H-1,W-1)
        gy = dZdy[:, :, :-1]                 # (B,H-1,W-1)
        mag = torch.sqrt(gx*gx + gy*gy + eps)
        nx = gx / (mag + eps)
        ny = gy / (mag + eps)
        dnx_dx = 0.5 * (nx[:, :, 2:] - nx[:, :, :-2])   # (B,H-1,W-3)
        dny_dy = 0.5 * (ny[:, 2:, :] - ny[:, :-2, :])   # (B,H-3,W-1)
        dnx_dx_c = dnx_dx[:, 1:-1, :]                   # (B,H-3,W-3)
        dny_dy_c = dny_dy[:, :, 1:-1]                   # (B,H-3,W-3)
        return dnx_dx_c + dny_dy_c

    Xk = kappa_field(Xb)                                  # (B,hk,wk)
    Rk = kappa_field(ref.unsqueeze(0)).squeeze(0)         # (hk,wk)
    num = torch.abs(Xk - Rk.unsqueeze(0)).flatten(1).sum(dim=1)                        # (B,)
    den = torch.maximum(Xk.abs(), Rk.abs().unsqueeze(0)).flatten(1).sum(dim=1) + eps   # (B,)
    return 0.5 * (num / den)

def _s3d_dist_to_ref(Xb: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    S3D distance to reference: sqrt((dL^2 + dG^2 + dC^2)/3)
      - dL = 1 - SSIM5 (luminance/contrast/structure, C=0 denom-guard)
      - dG = G-SSIM distance (gradient structural)
      - dC = curvature structural distance (magnitude ratio + sign, weighted)
    """
    dL = _ssim5_dist_to_ref(Xb, ref)
    dG = _gssim_dist_to_ref(Xb, ref)
    dC = _curv_struct_dist_to_ref(Xb, ref)
    return torch.sqrt((dL*dL + dG*dG + dC*dC) / 3.0)

def _spot_to_ref(Xb: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    SPOT: Sliced Wasserstein-1 distance (fixed 16 projections, 64 bins), positive/negative mass average.
    Grid-normalized coordinates [0,1]^2, area weight未使用（近似）。
    Returns (B,) in [0,1].
    """
    device = Xb.device; dtype = Xb.dtype
    B, H, W = Xb.shape
    # mass split and normalization
    Ap = torch.clamp(Xb, min=0.0).flatten(1)
    An = torch.clamp(-Xb, min=0.0).flatten(1)
    Wp = torch.clamp(ref, min=0.0).flatten()
    Wn = torch.clamp(-ref, min=0.0).flatten()
    Ap = Ap / (Ap.sum(dim=1, keepdim=True).clamp(min=1e-12))
    An = An / (An.sum(dim=1, keepdim=True).clamp(min=1e-12))
    Wp = Wp / (Wp.sum().clamp(min=1e-12))
    Wn = Wn / (Wn.sum().clamp(min=1e-12))
    # coords
    x = torch.linspace(0.0, 1.0, W, device=device, dtype=dtype).view(1, 1, W).expand(1, H, W)
    y = torch.linspace(0.0, 1.0, H, device=device, dtype=dtype).view(1, H, 1).expand(1, H, W)
    x = x.reshape(-1); y = y.reshape(-1)  # (H*W,)
    # projections
    K = 16; N_BINS = 64; L = (2.0 ** 0.5)
    thetas = torch.linspace(0.0, math.pi, K, device=device, dtype=dtype)
    bin_edges = torch.linspace(0.0, L, N_BINS + 1, device=device, dtype=dtype)
    bin_width = (L / N_BINS)
    def _emd_side(Mb_flat: torch.Tensor, Mm_flat: torch.Tensor):
        # Mb_flat: (B,HW), Mm_flat: (HW,)
        emds = torch.zeros((B, K), device=device, dtype=dtype)
        for ki, t in enumerate(thetas):
            s = (x * torch.cos(t) + y * torch.sin(t))  # (HW,)
            idx = torch.bucketize(s, bin_edges) - 1
            idx = idx.clamp(min=0, max=N_BINS - 1)
            # batch hist
            HB = torch.zeros((B, N_BINS), device=device, dtype=dtype)
            HB.scatter_add_(1, idx.unsqueeze(0).expand(B, -1), Mb_flat)
            HM = torch.zeros((N_BINS,), device=device, dtype=dtype)
            HM.scatter_add_(0, idx, Mm_flat)
            # normalize to 1
            HB = HB / (HB.sum(dim=1, keepdim=True).clamp(min=1e-12))
            HM = HM / (HM.sum().clamp(min=1e-12))
            # CDFs
            CDFB = torch.cumsum(HB, dim=1)
            CDFM = torch.cumsum(HM, dim=0)
            diff = (CDFB - CDFM.view(1, -1)).abs().sum(dim=1) * bin_width
            # normalize by L
            emds[:, ki] = diff / L
        return emds.mean(dim=1)  # (B,)
    Dp = _emd_side(Ap, Wp)
    Dn = _emd_side(An, Wn)
    return 0.5 * (Dp + Dn)

def _gvd_to_ref(Xb: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    GVD to ref: second-derivative invariants distance averaged over inner grid.
    Returns (B,) in [0,1].
    """
    device = Xb.device; dtype = Xb.dtype
    B, H, W = Xb.shape
    def _second(Z):
        Z_xx = Z[..., 2:, 1:-1] - 2.0 * Z[..., 1:-1, 1:-1] + Z[..., :-2, 1:-1]
        Z_yy = Z[..., 1:-1, 2:] - 2.0 * Z[..., 1:-1, 1:-1] + Z[..., 1:-1, :-2]
        Z_xy = (Z[..., 2:, 2:] - Z[..., 2:, :-2] - Z[..., :-2, 2:] + Z[..., :-2, :-2]) * 0.25
        return Z_xx, Z_yy, Z_xy
    X_xx, X_yy, X_xy = _second(Xb)
    R_xx, R_yy, R_xy = _second(ref.view(1, H, W))
    def _inv(Z_xx, Z_yy, Z_xy):
        L = Z_xx + Z_yy
        S = torch.sqrt((Z_xx - Z_yy) * (Z_xx - Z_yy) + (2.0 * Z_xy) * (2.0 * Z_xy) + 1e-12)
        theta = 0.5 * torch.atan2(2.0 * Z_xy, (Z_xx - Z_yy + 1e-12))
        return L, S, theta
    Lx, Sx, thx = _inv(X_xx, X_yy, X_xy)
    Lr, Sr, thr = _inv(R_xx, R_yy, R_xy)
    # normalize L,S per sample/ref
    w_inner = torch.ones((H - 2, W - 2), device=device, dtype=dtype)
    def _norm_LS(L, S):
        Lsum = (L.abs() * w_inner).flatten(1).sum(dim=1, keepdim=True).clamp(min=1e-12)
        Ssum = (S * w_inner).flatten(1).sum(dim=1, keepdim=True).clamp(min=1e-12)
        return L / Lsum.view(-1, 1, 1), S / Ssum.view(-1, 1, 1)
    Lx, Sx = _norm_LS(Lx, Sx)
    Lr0, Sr0 = _norm_LS(Lr, Sr)
    Lr = Lr0[0]; Sr = Sr0[0]; thr = thr[0]
    Wsum = w_inner.sum()
    d_vort = ((Lx - Lr.unsqueeze(0)).abs() * w_inner).flatten(1).sum(dim=1) / Wsum
    d_def  = ((Sx - Sr.unsqueeze(0)).abs() * w_inner).flatten(1).sum(dim=1) / Wsum
    d_axis = (1.0 - torch.cos(2.0 * (thx - thr.unsqueeze(0)))) * w_inner
    d_axis = d_axis.flatten(1).sum(dim=1) / Wsum
    return torch.sqrt((d_vort * d_vort + d_def * d_def + d_axis * d_axis) / 3.0)

def _itcs_to_ref(Xb: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    ITCS to ref: quantile-based area fraction & centroid signature distance, pos/neg average.
    Returns (B,) in [0,1].
    """
    device = Xb.device; dtype = Xb.dtype
    B, H, W = Xb.shape
    qs = torch.linspace(0.1, 0.9, 9, device=device, dtype=dtype)
    x = torch.linspace(0.0, 1.0, W, device=device, dtype=dtype).view(1, 1, W).expand(1, H, W).reshape(-1)
    y = torch.linspace(0.0, 1.0, H, device=device, dtype=dtype).view(1, H, 1).expand(1, H, W).reshape(-1)
    w = torch.ones((H * W,), device=device, dtype=dtype)
    def _sig(Z: torch.Tensor):
        N = Z.shape[0]
        Zf = Z.reshape(N, -1)
        t = torch.quantile(Zf, qs, dim=1, interpolation='linear')  # (N,9)
        phi = []; cx = []; cy = []
        for qi in range(qs.numel()):
            thr = t[:, qi].view(N, 1)
            M = (Zf >= thr).to(dtype)
            wM = M * w.view(1, -1)
            a = wM.sum(dim=1).clamp(min=1e-12)
            phi.append((a / w.sum()).view(N, 1))
            cx.append(((wM * x.view(1, -1)).sum(dim=1) / a).view(N, 1))
            cy.append(((wM * y.view(1, -1)).sum(dim=1) / a).view(N, 1))
        return torch.cat(phi, dim=1), torch.cat(cx, dim=1), torch.cat(cy, dim=1)
    # positive side
    phiB_p, cxB_p, cyB_p = _sig(Xb)
    phiR_p, cxR_p, cyR_p = _sig(ref.view(1, H, W))
    dphi_p = (phiB_p - phiR_p.view(1, -1)).abs().mean(dim=1)
    dc_p = torch.sqrt((cxB_p - cxR_p.view(1, -1))**2 + (cyB_p - cyR_p.view(1, -1))**2).mean(dim=1) / (2.0 ** 0.5)
    Dp = 0.5 * dphi_p + 0.5 * dc_p
    # negative side (apply on -Z with same routine)
    phiB_n, cxB_n, cyB_n = _sig(-Xb)
    phiR_n, cxR_n, cyR_n = _sig((-ref).view(1, H, W))
    dphi_n = (phiB_n - phiR_n.view(1, -1)).abs().mean(dim=1)
    dc_n = torch.sqrt((cxB_n - cxR_n.view(1, -1))**2 + (cyB_n - cyR_n.view(1, -1))**2).mean(dim=1) / (2.0 ** 0.5)
    Dn = 0.5 * dphi_n + 0.5 * dc_n
    return 0.5 * (Dp + Dn)


# 追加: マルチスケール系/複合距離の「対参照」関数群（medoid算出で使用）
def _adaptive_pool_ms(T: torch.Tensor, s: int) -> torch.Tensor:
    """
    Adaptive average pool for 3D tensors (B,H,W) to roughly downscale by factor s.
    Ensures output size is at least 2x2.
    """
    B, H, W = T.shape
    h2 = max(2, max(1, H // s))
    w2 = max(2, max(1, W // s))
    return F.adaptive_avg_pool2d(T.unsqueeze(1), (h2, w2)).squeeze(1)

def _ms_s1_dist_to_ref(Xb: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    Multi-scale S1 distance to reference with batch-wise min-max normalization per scale and weighted RMS fusion.
    Returns (B,)
    """
    device = Xb.device; dtype = Xb.dtype
    scales = [1, 2, 4]
    weights = torch.tensor([0.5, 0.3, 0.2], device=device, dtype=dtype)
    weights = weights / weights.sum()
    dists = []
    eps = 1e-12
    for s in scales:
        if s == 1:
            Xs = Xb
            R = ref
        else:
            Xs = _ms_s1_down_X = _adaptive_pool_ms(Xb, s)
            R = _ms_s1_down_R = _adaptive_pool_ms(ref.unsqueeze(0), s).squeeze(0)
        d = _s1_dist_to_ref(Xs, R)  # (B,)
        dmin = torch.min(d)
        dmax = torch.max(d)
        dn = (d - dmin) / (dmax - dmin + eps)
        dists.append(dn)
    out = torch.zeros_like(dists[0])
    for i, dn in enumerate(dists):
        out = out + weights[i] * (dn * dn)
    return torch.sqrt(out + 1e-12)

def _msssim_s1g_to_ref(Xb: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    MSSSIM*-S1 Gate distance to reference:
      D = dL*^2 + (1 - dL*) * sqrt( (dG^2 + dS1n^2)/2 )
    where dL* is multi-scale 1-SSIM (5x5, C=0 proxy), dG is G-SSIM distance, dS1n is batch-wise min-max normalized S1.
    Returns (B,)
    """
    device = Xb.device; dtype = Xb.dtype
    scales = [1, 2, 4]
    w = torch.tensor([0.5, 0.3, 0.2], device=device, dtype=dtype)
    w = w / w.sum()
    dL_list = []
    for s in scales:
        if s == 1:
            Xs = Xb
            R = ref
        else:
            Xs = _adaptive_pool_ms(Xb, s)
            R = _adaptive_pool_ms(ref.unsqueeze(0), s).squeeze(0)
        dl = _ssim5_dist_to_ref(Xs, R)  # (B,)
        dL_list.append(dl.clamp(0.0, 2.0))
    # weighted mean for dL*
    dL = torch.zeros_like(dL_list[0])
    for i, dl in enumerate(dL_list):
        dL = dL + w[i] * dl
    dG = _gssim_dist_to_ref(Xb, ref)  # (B,)
    dS1 = _s1_dist_to_ref(Xb, ref)    # (B,)
    eps = 1e-12
    dS1n = (dS1 - torch.min(dS1)) / (torch.max(dS1) - torch.min(dS1) + eps)
    core = torch.sqrt((dG * dG + dS1n * dS1n) / 2.0)
    return dL * dL + (1.0 - dL) * core

def _s1gcurv_to_ref(Xb: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    S1GCurv: RMS of three [0,1] distances:
      - D_s1 = clamp(S1/200, 0,1)
      - D_edge = G-SSIM distance in [0,1]
      - D_curv = 0.5 * curvature S1-like ratio on Laplacians in [0,1]
    Returns (B,)
    """
    D1 = _s1_dist_to_ref(Xb, ref)             # (B,) ~0..200
    D1n = torch.clamp(D1 / 200.0, min=0.0, max=1.0)
    Dg = _gssim_dist_to_ref(Xb, ref)          # (B,) in [0,1]
    Dc = _curv_s1_dist_to_ref(Xb, ref)        # (B,) ~0..2
    Dcurv = 0.5 * Dc
    return torch.sqrt((D1n * D1n + Dg * Dg + Dcurv * Dcurv) / 3.0)

def compute_node_true_medoids(
    method_name: str,
    data_flat: np.ndarray,
    winners_xy: np.ndarray,
    som_shape: Tuple[int, int],
    field_shape: Tuple[int, int],
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    fusion_alpha: float = 0.5
) -> Tuple[Dict[Tuple[int,int], int], Dict[Tuple[int,int], float]]:
    """
    各ノードについて「総距離最小（true medoid）」のサンプルを選ぶ。
    - 各ノードの割当集合 Ic の中で、候補 i∈Ic について cost(i) = Σ_{j∈Ic} d(X_j, X_i) を計算し最小の i を選ぶ。
    - method_name: 'euclidean' | 'ssim' | 'ssim5' | 's1' | 's1ssim' | 's1ssim5_hf' | 's1ssim5_and' | 'pf_s1ssim' | 's1gssim' | 'gssim'
    - 戻りの距離は、選ばれたメドイドに対する平均距離（sum/|Ic|）。
    """
    H, W = field_shape
    X2 = data_flat.reshape(-1, H, W)
    X_t = torch.as_tensor(X2, device=device, dtype=torch.float32)

    node_to_medoid_idx: Dict[Tuple[int,int], int] = {}
    node_to_medoid_avgdist: Dict[Tuple[int,int], float] = {}

    def pairwise_euclidean(Xc: torch.Tensor) -> torch.Tensor:
        C = Xc.shape[0]
        Xf = Xc.reshape(C, -1)
        x2 = (Xf * Xf).sum(dim=1, keepdim=True)           # (C,1)
        d2 = x2 + x2.T - 2.0 * (Xf @ Xf.T)                # (C,C)
        d2 = torch.clamp(d2, min=0.0)
        return torch.sqrt(d2 + 1e-12)

    def pairwise_by_ref(dist_fn, Xc: torch.Tensor) -> torch.Tensor:
        # 行ごとに「ref=Xc[i]」として距離ベクトルを計算して積み上げる
        C = Xc.shape[0]
        D = torch.empty((C, C), device=Xc.device, dtype=Xc.dtype)
        for i in range(C):
            ref = Xc[i]
            D[i] = dist_fn(Xc, ref)  # (C,)
        return D

    for ix in range(som_shape[0]):
        for iy in range(som_shape[1]):
            mask = (winners_xy[:, 0] == ix) & (winners_xy[:, 1] == iy)
            idxs = np.where(mask)[0]
            if len(idxs) == 0:
                continue
            # ITCS は計算量が非常に大きいため、各ノードの候補数を上限でサンプリングして近似メドイドを計算
            if method_name == 'itcs' and len(idxs) > 120:
                try:
                    rng = np.random.RandomState(SEED)
                except Exception:
                    rng = np.random.RandomState(1)
                sel = np.sort(rng.choice(len(idxs), size=120, replace=False))
                idxs = idxs[sel]
            Xcand = X_t[idxs]  # (C,H,W)
            Ccand = Xcand.shape[0]

            if Ccand == 1:
                node_to_medoid_idx[(ix, iy)] = int(idxs[0])
                node_to_medoid_avgdist[(ix, iy)] = 0.0
                continue

            # 距離行列 D (C,C) を作る
            if method_name == 'euclidean':
                D = pairwise_euclidean(Xcand)
            elif method_name == 'ssim5':
                D = pairwise_by_ref(lambda Xb, ref: _ssim5_dist_to_ref(Xb, ref), Xcand)
            elif method_name == 's1':
                D = pairwise_by_ref(lambda Xb, ref: _s1_dist_to_ref(Xb, ref), Xcand)
            elif method_name == 's1ssim':
                # まず個別の距離行列
                D1 = pairwise_by_ref(lambda Xb, ref: _s1_dist_to_ref(Xb, ref), Xcand)
                D2 = pairwise_by_ref(lambda Xb, ref: _ssim5_dist_to_ref(Xb, ref), Xcand)
                # 行ごとにmin-max正規化（i行に対して）
                eps = 1e-12
                d1_min = D1.min(dim=1, keepdim=True).values
                d1_max = D1.max(dim=1, keepdim=True).values
                d2_min = D2.min(dim=1, keepdim=True).values
                d2_max = D2.max(dim=1, keepdim=True).values
                D1n = (D1 - d1_min) / (d1_max - d1_min + eps)
                D2n = (D2 - d2_min) / (d2_max - d2_min + eps)
                D = fusion_alpha * D1n + (1.0 - fusion_alpha) * D2n
            elif method_name == 's1ssim5_hf':
                # HF-S1SSIM5: 行ごとに min-max 正規化し、D = U + (1-U)⊙V
                # U: 正規化 dSSIM = 1-SSIM5, V: 正規化 S1
                D1 = pairwise_by_ref(lambda Xb, ref: _s1_dist_to_ref(Xb, ref), Xcand)      # (C,C)
                D2 = pairwise_by_ref(lambda Xb, ref: _ssim5_dist_to_ref(Xb, ref), Xcand)   # (C,C)
                eps = 1e-12
                d1_min = D1.min(dim=1, keepdim=True).values
                d1_max = D1.max(dim=1, keepdim=True).values
                d2_min = D2.min(dim=1, keepdim=True).values
                d2_max = D2.max(dim=1, keepdim=True).values
                V = (D1 - d1_min) / (d1_max - d1_min + eps)   # normalized S1
                U = (D2 - d2_min) / (d2_max - d2_min + eps)   # normalized dSSIM
                D = U + (1.0 - U) * V
            elif method_name == 's1ssim5_and':
                # AND合成: 行ごとにmin-max正規化して D = max(U,V)
                D1 = pairwise_by_ref(lambda Xb, ref: _s1_dist_to_ref(Xb, ref), Xcand)      # (C,C)
                D2 = pairwise_by_ref(lambda Xb, ref: _ssim5_dist_to_ref(Xb, ref), Xcand)   # (C,C)
                eps = 1e-12
                d1_min = D1.min(dim=1, keepdim=True).values
                d1_max = D1.max(dim=1, keepdim=True).values
                d2_min = D2.min(dim=1, keepdim=True).values
                d2_max = D2.max(dim=1, keepdim=True).values
                V = (D1 - d1_min) / (d1_max - d1_min + eps)
                U = (D2 - d2_min) / (d2_max - d2_min + eps)
                D = torch.maximum(U, V)
            elif method_name == 'pf_s1ssim':
                # 比例融合: 正規化なしの積 D = dS1 * dSSIM
                D1 = pairwise_by_ref(lambda Xb, ref: _s1_dist_to_ref(Xb, ref), Xcand)      # (C,C)
                D2 = pairwise_by_ref(lambda Xb, ref: _ssim5_dist_to_ref(Xb, ref), Xcand)   # (C,C)
                D = D1 * D2
            elif method_name == 's1gl':
                # DSGC: combine D_edge, D_curv, and normalized S1 by RMS
                D_edge = pairwise_by_ref(lambda Xb, ref: _grad_ssim_dir_to_ref(Xb, ref), Xcand)     # (C,C)
                D_s1   = pairwise_by_ref(lambda Xb, ref: _s1_dist_to_ref(Xb, ref), Xcand)          # (C,C)
                D_curv = pairwise_by_ref(lambda Xb, ref: _curv_ssim_weighted_to_ref(Xb, ref), Xcand)  # (C,C)
                eps = 1e-12
                d1_min = D_s1.min(dim=1, keepdim=True).values
                d1_max = D_s1.max(dim=1, keepdim=True).values
                D1n = (D_s1 - d1_min) / (d1_max - d1_min + eps)
                D = torch.sqrt((D_edge * D_edge + D_curv * D_curv + D1n * D1n) / 3.0)
            elif method_name == 'gsmd':
                # GSMD: sqrt((Dg^2 + Ds^2 + Dm^2)/3)
                Dg = pairwise_by_ref(lambda Xb, ref: _gssim_dist_to_ref(Xb, ref), Xcand)          # (C,C)
                Ds = pairwise_by_ref(lambda Xb, ref: _s1norm_dist_to_ref(Xb, ref), Xcand)         # (C,C)
                Dm = pairwise_by_ref(lambda Xb, ref: _moment_dist_to_ref(Xb, ref), Xcand)         # (C,C)
                D = torch.sqrt((Dg*Dg + Ds*Ds + Dm*Dm) / 3.0)
            elif method_name == 's3d':
                # S3D: sqrt((dL^2 + dG^2 + dC^2)/3)
                dL = pairwise_by_ref(lambda Xb, ref: _ssim5_dist_to_ref(Xb, ref), Xcand)          # (C,C)
                dG = pairwise_by_ref(lambda Xb, ref: _gssim_dist_to_ref(Xb, ref), Xcand)          # (C,C)
                dC = pairwise_by_ref(lambda Xb, ref: _curv_struct_dist_to_ref(Xb, ref), Xcand)    # (C,C)
                D = torch.sqrt((dL*dL + dG*dG + dC*dC) / 3.0)
            elif method_name == 'cfsd':
                # CFSD: sqrt((Dg^2 + Ds1n^2 + Dcurvn^2)/3)
                Dg = pairwise_by_ref(lambda Xb, ref: _gssim_dist_to_ref(Xb, ref), Xcand)          # (C,C)
                Ds = pairwise_by_ref(lambda Xb, ref: _s1_dist_to_ref(Xb, ref), Xcand)             # (C,C)
                Dc = pairwise_by_ref(lambda Xb, ref: _curv_s1_dist_to_ref(Xb, ref), Xcand)        # (C,C)
                eps = 1e-12
                d1_min = Ds.min(dim=1, keepdim=True).values
                d1_max = Ds.max(dim=1, keepdim=True).values
                Dc_min = Dc.min(dim=1, keepdim=True).values
                Dc_max = Dc.max(dim=1, keepdim=True).values
                Dsn = (Ds - d1_min) / (d1_max - d1_min + eps)
                Dcn = (Dc - Dc_min) / (Dc_max - Dc_min + eps)
                D = torch.sqrt((Dg*Dg + Dsn*Dsn + Dcn*Dcn) / 3.0)
            elif method_name == 'hff':
                # HFF: D = dL^2 + (1 - dL) * dG, where dL = 1-SSIM5, dG = 1-GSSIM
                dL = pairwise_by_ref(lambda Xb, ref: _ssim5_dist_to_ref(Xb, ref), Xcand)          # (C,C)
                dG = pairwise_by_ref(lambda Xb, ref: _gssim_dist_to_ref(Xb, ref), Xcand)          # (C,C)
                D = dL*dL + (1.0 - dL) * dG
            elif method_name == 's1gk':
                # S1GK: RMS of row-normalized S1, G-SSIM distance, and row-normalized kappa curvature distance
                D1 = pairwise_by_ref(lambda Xb, ref: _s1_dist_to_ref(Xb, ref), Xcand)             # (C,C)
                Dg = pairwise_by_ref(lambda Xb, ref: _gssim_dist_to_ref(Xb, ref), Xcand)          # (C,C)
                Dk = pairwise_by_ref(lambda Xb, ref: _kappa_to_ref(Xb, ref), Xcand)               # (C,C)
                eps = 1e-12
                D1n = (D1 - D1.min(dim=1, keepdim=True).values) / (D1.max(dim=1, keepdim=True).values - D1.min(dim=1, keepdim=True).values + eps)
                Dkn = (Dk - Dk.min(dim=1, keepdim=True).values) / (Dk.max(dim=1, keepdim=True).values - Dk.min(dim=1, keepdim=True).values + eps)
                D = torch.sqrt((D1n*D1n + Dg*Dg + Dkn*Dkn) / 3.0)
            elif method_name == 's1gssim':
                # d_edge（[0,1]）と 行方向min–max正規化したS1 のRMS合成
                D_edge = pairwise_by_ref(lambda Xb, ref: _grad_ssim_dir_to_ref(Xb, ref), Xcand)  # (C,C)
                D_s1   = pairwise_by_ref(lambda Xb, ref: _s1_dist_to_ref(Xb, ref), Xcand)       # (C,C)
                eps = 1e-12
                d1_min = D_s1.min(dim=1, keepdim=True).values
                d1_max = D_s1.max(dim=1, keepdim=True).values
                D1n = (D_s1 - d1_min) / (d1_max - d1_min + eps)
                D = torch.sqrt((D_edge * D_edge + D1n * D1n) / 2.0)
            elif method_name == 'gssim':
                # 勾配構造類似ベースの距離（[0,1]）
                D = pairwise_by_ref(lambda Xb, ref: _gssim_dist_to_ref(Xb, ref), Xcand)
            elif method_name == 'spot':
                # SPOT: sliced Wasserstein-1 (正負平均)
                D = pairwise_by_ref(lambda Xb, ref: _spot_to_ref(Xb, ref), Xcand)
            elif method_name == 'gvd':
                # GVD: 二階微分不変量
                D = pairwise_by_ref(lambda Xb, ref: _gvd_to_ref(Xb, ref), Xcand)
            elif method_name == 'itcs':
                # ITCS: トポロジー＆重心シグネチャ
                D = pairwise_by_ref(lambda Xb, ref: _itcs_to_ref(Xb, ref), Xcand)
            elif method_name == 'ms_s1':
                # Multi-Scale S1
                D = pairwise_by_ref(lambda Xb, ref: _ms_s1_dist_to_ref(Xb, ref), Xcand)
            elif method_name == 'msssim_s1g':
                # MSSSIM*-S1 Gate
                D = pairwise_by_ref(lambda Xb, ref: _msssim_s1g_to_ref(Xb, ref), Xcand)
            elif method_name == 's1gcurv':
                # S1 + G-SSIM + curvature S1 (scaled)
                D = pairwise_by_ref(lambda Xb, ref: _s1gcurv_to_ref(Xb, ref), Xcand)
            else:
                raise ValueError(f'Unknown method_name: {method_name}')

            # cost(i) = Σ_j D[i,j]
            costs = D.sum(dim=1)  # (C,)
            imin = int(torch.argmin(costs).item())
            node_to_medoid_idx[(ix, iy)] = int(idxs[imin])
            node_to_medoid_avgdist[(ix, iy)] = float((costs[imin] / Ccand).item())

    return node_to_medoid_idx, node_to_medoid_avgdist


def plot_som_node_medoid_patterns(
    data_flat: np.ndarray,
    node_to_medoid_idx: Dict[Tuple[int,int], int],
    lat: np.ndarray, lon: np.ndarray,
    som_shape: Tuple[int,int],
    times_all: Optional[np.ndarray],
    save_path: str,
    title: str
):
    """
    ノードのメドイド（代表サンプル）のマップを保存（標準は True Medoid）
    各サブプロットのタイトルを (x,y)_YYYY/MM/DD にする。
    """
    H, W = len(lat), len(lon)

    vmin, vmax = -40, 40
    levels = np.linspace(vmin, vmax, 21)
    cmap = 'RdBu_r'

    map_x, map_y = som_shape
    nrows, ncols = map_y, map_x
    figsize=(ncols*2.6, nrows*2.6)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize,
                             subplot_kw={'projection': ccrs.PlateCarree()})
    axes = np.atleast_2d(axes)
    axes = axes.T[::-1,:]  # 表示並びの整合

    last_cf = None
    for ix in range(map_x):
        for iy in range(map_y):
            ax = axes[ix, iy]
            key = (ix, iy)
            if key not in node_to_medoid_idx:
                ax.set_axis_off()
                continue
            mi = node_to_medoid_idx[key]
            pat = data_flat[mi].reshape(H, W)
            cf = ax.contourf(lon, lat, pat, levels=levels, cmap=cmap, extend='both', transform=ccrs.PlateCarree())
            ax.contour(lon, lat, pat, levels=levels, colors='k', linewidths=0.3, transform=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE.with_scale('50m'), edgecolor='black', linewidth=0.8)
            ax.set_extent([115, 155, 15, 55], ccrs.PlateCarree())
            # タイトルに日付を追加（YYYY/MM/DD）
            dstr = format_date_yyyymmdd(times_all[mi]) if times_all is not None and len(times_all)>mi else ''
            ax.set_title(f'({ix},{iy})_{dstr}')
            ax.set_xticks([]); ax.set_yticks([])
            last_cf = cf

    if last_cf is not None:
        # よりコンパクトな余白と大きめの色バー文字
        fig.subplots_adjust(left=0.04, right=0.88, top=0.95, bottom=0.04, wspace=0.05, hspace=0.05)
        cax = fig.add_axes([0.90, 0.12, 0.02, 0.76])
        cb = fig.colorbar(last_cf, cax=cax, label='Sea Level Pressure Anomaly (hPa)')
        cb.ax.tick_params(labelsize=12)
        cb.set_label('Sea Level Pressure Anomaly (hPa)', fontsize=42)
    plt.suptitle(title, fontsize=42, fontweight='bold')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def save_each_node_medoid_image(
    data_flat: np.ndarray,
    node_to_medoid_idx: Dict[Tuple[int,int], int],
    lat: np.ndarray, lon: np.ndarray,
    som_shape: Tuple[int,int],
    out_dir: str,
    prefix: str
):
    os.makedirs(out_dir, exist_ok=True)
    H, W = len(lat), len(lon)

    vmin, vmax = -40, 40
    levels = np.linspace(vmin, vmax, 21)
    cmap = 'RdBu_r'

    for ix in range(som_shape[0]):
        for iy in range(som_shape[1]):
            key = (ix, iy)
            if key not in node_to_medoid_idx:
                continue
            mi = node_to_medoid_idx[key]
            pat = data_flat[mi].reshape(H, W)
            fig = plt.figure(figsize=(4,3))
            ax = plt.axes(projection=ccrs.PlateCarree())
            cf = ax.contourf(lon, lat, pat, levels=levels, cmap=cmap, transform=ccrs.PlateCarree(), extend='both')
            ax.contour(lon, lat, pat, levels=levels, colors='k', linewidths=0.3, transform=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.8, edgecolor='black')
            ax.set_extent([115, 155, 15, 55], ccrs.PlateCarree())
            cb = plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04, label='Sea Level Pressure Anomaly (hPa)')
            cb.ax.tick_params(labelsize=11)
            cb.set_label('Sea Level Pressure Anomaly (hPa)', fontsize=13)
            ax.set_title(f'Medoid ({ix},{iy}) sample={mi}', fontsize=12)
            ax.set_xticks([]); ax.set_yticks([])
            fpath = os.path.join(out_dir, f'{prefix}_node_{ix}_{iy}_medoid.png')
            plt.tight_layout(pad=0.5)
            plt.savefig(fpath, dpi=200, bbox_inches='tight')
            plt.close(fig)


def plot_nodewise_match_map(
    winners_xy: np.ndarray,
    labels_all: List[str],
    node_to_medoid_idx: Dict[Tuple[int,int], int],
    times_all: np.ndarray,
    som_shape: Tuple[int,int],
    save_path: str,
    title: str
):
    """
    各ノードについて、
      - ノード内の最頻出ラベル（元ラベル、複合含む）= majority
      - medoid（True Medoid）の元ラベル
    が一致すれば「薄い緑」でノード四角を塗り、不一致なら「薄い赤」にする。
    N, majority, medoid, date(YYYY/MM/DD) を少し大きめに表示。
    """
    # ノードごとのサンプルリスト
    clusters = winners_to_clusters(winners_xy, som_shape)
    node_to_majority_raw: Dict[Tuple[int,int], Optional[str]] = {}
    node_to_count: Dict[Tuple[int,int], int] = {}

    for k, idxs in enumerate(clusters):
        ix, iy = k // som_shape[1], k % som_shape[1]
        node_to_count[(ix,iy)] = len(idxs)
        if len(idxs) == 0:
            node_to_majority_raw[(ix,iy)] = None
            continue
        cnt = Counter([labels_all[j] for j in idxs if labels_all[j] is not None])
        if len(cnt) == 0:
            node_to_majority_raw[(ix,iy)] = None
        else:
            node_to_majority_raw[(ix,iy)] = cnt.most_common(1)[0][0]

    # グリッド図
    map_x, map_y = som_shape
    nrows, ncols = map_y, map_x
    figsize = (ncols*2.6, nrows*2.6)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = np.atleast_2d(axes)
    axes = axes.T[::-1, :]

    for ix in range(map_x):
        for iy in range(map_y):
            ax = axes[ix, iy]
            ax.set_xticks([]); ax.set_yticks([])
            maj = node_to_majority_raw.get((ix,iy), None)
            if (ix,iy) not in node_to_medoid_idx:
                # データ無し
                ax.set_facecolor((0.95, 0.95, 0.95, 1.0))  # 薄い灰色
                ax.text(0.5, 0.5, '-', ha='center', va='center', fontsize=16, color='gray')
                ax.text(0.02, 0.98, f'({ix},{iy}) N={node_to_count.get((ix,iy),0)}',
                        transform=ax.transAxes, ha='left', va='top', fontsize=10)
                continue
            mi = node_to_medoid_idx[(ix,iy)]
            med_raw = labels_all[mi] if labels_all is not None else None
            dstr = format_date_yyyymmdd(times_all[mi]) if times_all is not None and len(times_all)>mi else ''
            match = (med_raw == maj) if (med_raw is not None and maj is not None) else False

            # 背景色を一致/不一致で塗る（薄い緑/薄い赤）
            if match:
                ax.set_facecolor((0.85, 1.00, 0.85, 1.0))  # light green
            else:
                ax.set_facecolor((1.00, 0.85, 0.85, 1.0))  # light red

            # ヘッダ
            ax.text(0.02, 0.98, f'({ix},{iy}) N={node_to_count.get((ix,iy),0)}',
                    transform=ax.transAxes, ha='left', va='top', fontsize=10, fontweight='bold')
            # 付記（少し大きめ）
            ax.text(0.02, 0.78, f'Majority: {maj}', transform=ax.transAxes, ha='left', va='top', fontsize=10)
            ax.text(0.02, 0.62, f'Medoid  : {med_raw}', transform=ax.transAxes, ha='left', va='top', fontsize=10)
            ax.text(0.02, 0.46, f'Date    : {dstr}', transform=ax.transAxes, ha='left', va='top', fontsize=10)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


# =====================================================
# Nodewise match rate（rawラベルでの一致率）計算ユーティリティ
# =====================================================
def compute_nodewise_match_rate(
    winners_xy: np.ndarray,
    labels_all: List[Optional[str]],
    node_to_medoid_idx: Dict[Tuple[int,int], int],
    som_shape: Tuple[int, int]
) -> Tuple[float, int, int]:
    """
    ノード代表ラベル（raw: 複合含む）と medoid の raw ラベルの一致率。
    戻り値: (match_rate, matched_nodes, counted_nodes)
    """
    clusters = winners_to_clusters(winners_xy, som_shape)
    node_to_majority_raw: Dict[Tuple[int,int], Optional[str]] = {}
    for k, idxs in enumerate(clusters):
        ix, iy = k // som_shape[1], k % som_shape[1]
        if len(idxs) == 0:
            node_to_majority_raw[(ix,iy)] = None
        else:
            cnt_raw = Counter([labels_all[j] for j in idxs if labels_all[j] is not None])
            node_to_majority_raw[(ix,iy)] = cnt_raw.most_common(1)[0][0] if len(cnt_raw)>0 else None

    matched = 0
    counted = 0
    for key, mi in node_to_medoid_idx.items():
        maj = node_to_majority_raw.get(key, None)
        med_raw = labels_all[mi] if labels_all is not None else None
        if maj is None or med_raw is None:
            continue
        counted += 1
        if maj == med_raw:
            matched += 1
    rate = (matched / counted) if counted > 0 else np.nan
    return float(rate), matched, counted


# =====================================================
# 学習時のノード代表（raw/基本）ラベルを構築
# =====================================================
def compute_training_node_majorities(
    winners_xy: np.ndarray,
    labels_all: List[Optional[str]],
    base_labels: List[str],
    som_shape: Tuple[int, int]
) -> Tuple[Dict[Tuple[int,int], Optional[str]], Dict[Tuple[int,int], Optional[str]]]:
    """
    学習データ上で、各ノードの代表ラベルを raw と基本の2種類で計算
    戻り値: (node_to_majority_raw, node_to_majority_base)
    """
    clusters = winners_to_clusters(winners_xy, som_shape)
    node_to_majority_raw: Dict[Tuple[int,int], Optional[str]] = {}
    node_to_majority_base: Dict[Tuple[int,int], Optional[str]] = {}

    for k, idxs in enumerate(clusters):
        ix, iy = k // som_shape[1], k % som_shape[1]
        # raw
        cnt_raw = Counter([labels_all[j] for j in idxs if labels_all[j] is not None])
        node_to_majority_raw[(ix, iy)] = cnt_raw.most_common(1)[0][0] if len(cnt_raw) > 0 else None
        # base
        cnt_base = Counter()
        for j in idxs:
            bl = basic_label_or_none(labels_all[j], base_labels)
            if bl is not None:
                cnt_base[bl] += 1
        node_to_majority_base[(ix, iy)] = cnt_base.most_common(1)[0][0] if len(cnt_base) > 0 else None

    return node_to_majority_raw, node_to_majority_base

def compute_training_node_base_counts(
    winners_xy: np.ndarray,
    labels_all: List[Optional[str]],
    base_labels: List[str],
    som_shape: Tuple[int, int]
) -> Dict[Tuple[int, int], Dict[str, int]]:
    """
    学習データ上で、各ノードに出現した基本ラベルの出現数を辞書で返す。
    戻り値: {(ix,iy): {label: count, ...}, ...}
    """
    clusters = winners_to_clusters(winners_xy, som_shape)
    node_to_counts: Dict[Tuple[int, int], Dict[str, int]] = {}
    for k, idxs in enumerate(clusters):
        ix, iy = k // som_shape[1], k % som_shape[1]
        cnt = Counter()
        for j in idxs:
            bl = basic_label_or_none(labels_all[j], base_labels)
            if bl is not None:
                cnt[bl] += 1
        node_to_counts[(ix, iy)] = dict(cnt)
    return node_to_counts


# =====================================================
# 検証（学習時代表ラベルに基づく推論）ユーティリティ
# =====================================================
def evaluate_verification_with_training_majority(
    winners_xy_valid: np.ndarray,
    labels_valid: List[Optional[str]],
    times_valid: np.ndarray,
    base_labels: List[str],
    som_shape: Tuple[int, int],
    node_to_majority_base_train: Dict[Tuple[int,int], Optional[str]],
    out_dir: str,
    method_name: str,
    logger: Logger,
    node_to_base_counts_train: Optional[Dict[Tuple[int,int], Dict[str,int]]] = None,
    sigma_pred: float = 1.2
):
    """
    学習時の「ノード代表（基本ラベル）」または「近傍重み付きラベル分布推定（学習時カウント×SOMグリッド距離ガウス重み）」を
    予測ラベルとして使用し、検証データの正解率を評価。
    - 混同行列（基本ラベル vs クラスタ列）CSV
    - per-label 再現率（基本/複合）CSV
    - 割当CSV（予測/正誤フラグ含む）
    - 棒グラフ（基本/複合）PNG
    """
    os.makedirs(out_dir, exist_ok=True)
    Hn, Wn = som_shape

    # 混同行列（基本ラベル vs クラスタ列）: 検証データベース
    clusters_val = winners_to_clusters(winners_xy_valid, som_shape)
    cm_val, cluster_names = build_confusion_matrix_only_base(clusters_val, labels_valid, base_labels)
    conf_csv = os.path.join(out_dir, f'{method_name}_verification_confusion_matrix.csv')
    cm_val.to_csv(conf_csv, encoding='utf-8-sig')

    # 学習代表（基本）を予測ラベルとして、検証の per-label 再現率（基本/複合）を算出
    total_base = Counter()
    correct_base = Counter()
    total_comp = Counter()
    correct_comp = Counter()

    # 割当CSV用
    rows_assign = []

    for i, (ix, iy) in enumerate(winners_xy_valid):
        # 実ラベル（基本/複合）
        raw = labels_valid[i]
        bl = basic_label_or_none(raw, base_labels)
        comps = extract_base_components(raw, base_labels)

        # 予測
        pred: Optional[str] = None
        if node_to_base_counts_train is not None and len(node_to_base_counts_train) > 0:
            # 近傍重み付き推定（学習時のノード別基本ラベル出現数 × ガウス重み）
            ix_i, iy_i = int(ix), int(iy)
            scores = Counter()
            Hn, Wn = som_shape
            for nx in range(Hn):
                for ny in range(Wn):
                    cnts = node_to_base_counts_train.get((nx, ny), None)
                    if not cnts:
                        continue
                    d2 = (ix_i - nx) * (ix_i - nx) + (iy_i - ny) * (iy_i - ny)
                    w = math.exp(-d2 / (2.0 * (sigma_pred ** 2)))
                    if w <= 0.0:
                        continue
                    for lbl_k, c in cnts.items():
                        if lbl_k in base_labels and c > 0:
                            scores[lbl_k] += w * float(c)
            if len(scores) > 0:
                pred = max(scores.items(), key=lambda x: x[1])[0]

        # フォールバック：BMUノードの多数決（従来仕様）
        if pred is None:
            pred = node_to_majority_base_train.get((int(ix), int(iy)), None)

        # 基本ラベル再現率用
        if bl is not None:
            total_base[bl] += 1
            if pred is not None and pred == bl:
                correct_base[bl] += 1

        # 複合ラベル再現率用
        if comps:
            for c in comps:
                total_comp[c] += 1
            if pred is not None and pred in comps:
                correct_comp[pred] += 1

        # 割当CSV 1行
        rows_assign.append({
            'time': format_date_yyyymmdd(times_valid[i]),
            'bmu_x': int(ix), 'bmu_y': int(iy),
            'label_raw': raw if raw is not None else '',
            'actual_base': bl if bl is not None else '',
            'pred_base_from_train': pred if pred is not None else '',
            'correct_base': int(1 if (bl is not None and pred == bl) else 0),
            'correct_composite': int(1 if (pred is not None and pred in comps) else 0)
        })

    # per-label 再現率テーブルとマクロ平均
    per_label_rows = []
    recalls_base = []
    recalls_comp = []
    for lbl in base_labels:
        N_base = int(total_base[lbl])
        C_base = int(correct_base[lbl])
        rec_base = (C_base / N_base) if N_base > 0 else np.nan

        N_comp = int(total_comp[lbl])
        C_comp = int(correct_comp[lbl])
        rec_comp = (C_comp / N_comp) if N_comp > 0 else np.nan

        per_label_rows.append({
            'label': lbl,
            'N_base': N_base, 'Correct_base': C_base, 'Recall_base': rec_base,
            'N_composite': N_comp, 'Correct_composite': C_comp, 'Recall_composite': rec_comp
        })
        if not np.isnan(rec_base):
            recalls_base.append(rec_base)
        if not np.isnan(rec_comp):
            recalls_comp.append(rec_comp)

    macro_base = float(np.mean(recalls_base)) if len(recalls_base) > 0 else float('nan')
    macro_comp = float(np.mean(recalls_comp)) if len(recalls_comp) > 0 else float('nan')

    # 保存（CSV）
    assign_csv = os.path.join(out_dir, f'{method_name}_verification_assign.csv')
    pd.DataFrame(rows_assign).to_csv(assign_csv, index=False, encoding='utf-8-sig')

    per_label_csv = os.path.join(out_dir, f'{method_name}_verification_per_label_recall.csv')
    df_pl = pd.DataFrame(per_label_rows)
    df_pl.to_csv(per_label_csv, index=False, encoding='utf-8-sig')

    # 棒グラフ出力
    def plot_per_label_bars(labels, values, title, out_png):
        fig = plt.figure(figsize=(max(8, 0.5*len(labels)+2), 4))
        ax = plt.gca()
        ax.bar(labels, values, color='steelblue')
        ax.set_ylim(0, 1.0)
        ax.set_ylabel('Recall')
        ax.set_title(title)
        for i, v in enumerate(values):
            if not np.isnan(v):
                ax.text(i, v+0.02, f"{v:.2f}", ha='center', va='bottom', fontsize=8)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close(fig)

    base_vals = [float(df_pl.loc[df_pl['label']==lbl, 'Recall_base'].values[0]) if lbl in df_pl['label'].values else np.nan for lbl in base_labels]
    comp_vals = [float(df_pl.loc[df_pl['label']==lbl, 'Recall_composite'].values[0]) if lbl in df_pl['label'].values else np.nan for lbl in base_labels]
    plot_per_label_bars(base_labels, base_vals, f'{method_name.upper()} Verification Recall (Base)', os.path.join(out_dir, f'{method_name}_verification_per_label_recall_base.png'))
    plot_per_label_bars(base_labels, comp_vals, f'{method_name.upper()} Verification Recall (Composite)', os.path.join(out_dir, f'{method_name}_verification_per_label_recall_composite.png'))

    # ログ出力（詳細）
    logger.write("\n=== [Verification Evaluation] ===\n")
    logger.write(f"Confusion matrix (base vs clusters) -> {conf_csv}\n")
    logger.write(f"Assignments with prediction/flags -> {assign_csv}\n")
    logger.write(f"Per-label recall CSV -> {per_label_csv}\n")
    logger.write("\n【各ラベルの再現率（学習時の代表基本ラベルを予測として）】\n")
    for r in per_label_rows:
        logger.write(f" - {r['label']:<3}: N_base={r['N_base']:4d} Correct_base={r['Correct_base']:4d} Recall_base={0.0 if np.isnan(r['Recall_base']) else r['Recall_base']:.4f} | "
                     f"N_comp={r['N_composite']:4d} Correct_comp={r['Correct_composite']:4d} Recall_comp={0.0 if np.isnan(r['Recall_composite']) else r['Recall_composite']:.4f}\n")
    logger.write(f"\n[Summary] Macro Recall (基本ラベル)   = {macro_base:.4f}\n")
    logger.write(f"[Summary] Macro Recall (基本+応用) = {macro_comp:.4f}\n")
    logger.write("=== [Verification Evaluation End] ===\n")

    return {
        'MacroRecall_majority': macro_base,
        'MacroRecall_composite': macro_comp
    }


# =====================================================
# 3種類のbatchSOM（学習：一方式分）
# =====================================================
def run_one_method_learning(method_name, activation_distance, data_all, labels_all, times_all,
                            field_shape, lat, lon, out_dir, device: str):
    """
    学習（learning）：method_name: 'euclidean' | 'ssim5' | 's1' | 's1ssim'
    activation_distance: 同上
    data_all は「空間平均を引いた偏差[hPa]」（N,D）
    """
    os.makedirs(out_dir, exist_ok=True)
    log = Logger(os.path.join(out_dir, f'{method_name}_results.log'))
    log.write(f'=== {method_name} SOM (Learning period) ===\n')
    if device.startswith('cuda') and torch.cuda.is_available():
        try:
            dev_index = torch.device(device).index if torch.device(device).index is not None else 0
            gpu_name = torch.cuda.get_device_name(dev_index)
        except Exception:
            dev_index = 0
            gpu_name = torch.cuda.get_device_name(0)
        log.write(f'Device: {device} ({gpu_name})\n')
    else:
        log.write(f'Device: {device.upper()}\n')
    log.write(f'SOM size: {SOM_X} x {SOM_Y}, iter={NUM_ITER}, batch={BATCH_SIZE}, nodes_chunk={NODES_CHUNK}\n')
    log.write(f'All samples: {data_all.shape[0]}\n')
    log.write('Input representation: SLP anomaly [hPa], spatial-mean removed per sample\n')
    if len(times_all) > 0:
        tmin = pd.to_datetime(times_all.min()).strftime('%Y-%m-%d')
        tmax = pd.to_datetime(times_all.max()).strftime('%Y-%m-%d')
        log.write(f'Period: {tmin} to {tmax}\n')

    # SOM構築（3距離対応版）
    # Optional area weights (cos φ) for curvature S1-based distances
    try:
        H_, W_ = field_shape
        lat_arr = np.asarray(lat, dtype=np.float32)
        aw_line = np.cos(np.deg2rad(lat_arr)).astype(np.float32)  # (H_,)
        area_w_map = np.repeat(aw_line.reshape(H_, 1), W_, axis=1)
    except Exception:
        area_w_map = None

    som = MultiDistMiniSom(
        x=SOM_X, y=SOM_Y, input_len=data_all.shape[1],
        sigma=2.5, learning_rate=0.5,
        neighborhood_function='gaussian',
        topology='rectangular',
        activation_distance=activation_distance,                # 'euclidean'/'ssim'/'s1'
        random_seed=SEED,
        sigma_decay='asymptotic_decay',
        s1_field_shape=field_shape,
        device=device,
        dtype=torch.float32,
        nodes_chunk=NODES_CHUNK,
        area_weight=area_w_map
    )
    som.random_weights_init(data_all)
    # σを学習全体で一方向に減衰（セグメント跨ぎで継続）
    som.set_total_iterations(NUM_ITER)

    # 固定QE評価サンプルを設定（再現性あり）
    n_eval = min(EVAL_SAMPLE_LIMIT if EVAL_SAMPLE_LIMIT else data_all.shape[0], data_all.shape[0])
    if n_eval > 0:
        rng = np.random.RandomState(SEED)
        eval_idx = rng.choice(data_all.shape[0], size=n_eval, replace=False)
        som.set_eval_indices(eval_idx)

    # ====== 学習を区切って実施し、各区切りで評価（履歴プロット用） ======
    step = max(1, NUM_ITER // SOM_EVAL_SEGMENTS)
    iter_history: Dict[str, List[float]] = {
        'iteration': [],
        'MacroRecall_majority': [],
        'MacroRecall_composite': [],
        'QuantizationError': [],
        'NodewiseMatchRate': []  # 追加：ノード代表（raw）vs medoid（raw）の一致率
    }

    current_iter = 0
    for seg in range(SOM_EVAL_SEGMENTS):
        n_it = min(step, NUM_ITER - current_iter)
        if n_it <= 0:
            break
        som.train_batch(
            data_all, num_iteration=n_it,
            batch_size=BATCH_SIZE, verbose=True, log_interval=LOG_INTERVAL,
            update_per_iteration=False, shuffle=True
        )
        current_iter += n_it

        # 量子化誤差
        qe_now = som.quantization_error(data_all, sample_limit=EVAL_SAMPLE_LIMIT, batch_size=max(32, BATCH_SIZE))

        # 評価
        winners_now = som.predict(data_all, batch_size=max(64, BATCH_SIZE))
        clusters_now = winners_to_clusters(winners_now, (SOM_X, SOM_Y))
        metrics = evaluate_clusters_only_base(
            clusters=clusters_now,
            all_labels=labels_all,
            base_labels=BASE_LABELS,
            title=f"[{method_name.upper()}] Iteration={current_iter} Evaluation (Base labels)"
        )

        # 追加: 現時点の True Medoid を再計算して、match rate を算出
        if method_name == 'itcs':
            # ITCS は距離計算が高コストなため、学習中の各セグメントではメドイド再計算をスキップ
            # （最終モデルでのみ算出）
            node_to_true_medoid_idx_now = {}
            matched_nodes_now = 0
            counted_nodes_now = 0
            match_rate_now = np.nan
            log.write('  [skip] ITCS: per-iteration True Medoid computation is skipped to avoid heavy O(C^2) cost.\n')
        else:
            node_to_true_medoid_idx_now, _ = compute_node_true_medoids(
                method_name=method_name,
                data_flat=data_all, winners_xy=winners_now,
                som_shape=(SOM_X, SOM_Y), field_shape=field_shape,
                device=device,
                fusion_alpha=0.5
            )
            match_rate_now, matched_nodes_now, counted_nodes_now = compute_nodewise_match_rate(
                winners_xy=winners_now,
                labels_all=labels_all,
                node_to_medoid_idx=node_to_true_medoid_idx_now,
                som_shape=(SOM_X, SOM_Y)
            )

        # ログ（results.log）に集約指標を追記
        log.write(f'\n[Iteration {current_iter}] QuantizationError={qe_now:.6f}\n')
        if metrics is not None:
            for k in ['MacroRecall_majority', 'MacroRecall_composite']:
                if k in metrics:
                    log.write(f'  {k} = {metrics[k]:.6f}\n')
        # 追加: match rate
        if not np.isnan(match_rate_now):
            log.write(f'  NodewiseMatchRate = {match_rate_now:.6f} (matched {matched_nodes_now}/{counted_nodes_now} nodes)\n')
        else:
            log.write(f'  NodewiseMatchRate = NaN (no countable nodes)\n')

        # 履歴に保存
        iter_history['iteration'].append(current_iter)
        iter_history['QuantizationError'].append(qe_now)
        if metrics is not None:
            for k in ['MacroRecall_majority', 'MacroRecall_composite']:
                iter_history[k].append(metrics.get(k, np.nan))
        else:
            for k in ['MacroRecall_majority', 'MacroRecall_composite']:
                iter_history[k].append(np.nan)
        iter_history['NodewiseMatchRate'].append(match_rate_now if not np.isnan(match_rate_now) else np.nan)

    # イテレーション履歴の保存（CSV/PNG）
    iter_csv = os.path.join(out_dir, f'{method_name}_iteration_metrics.csv')
    save_metrics_history_to_csv(iter_history, iter_csv)
    iter_png = os.path.join(out_dir, f'{method_name}_iteration_vs_metrics.png')
    plot_iteration_metrics(iter_history, iter_png)
    # 追加：各メトリクスの単独画像
    plot_iteration_metrics_single(iter_history, out_dir, filename_prefix=method_name)

    log.write(f'\nIteration metrics saved: CSV={iter_csv}, PNG={iter_png} and per-metric PNGs\n')

    # ====== 最終モデルでの割当 ======
    winners_all = som.predict(data_all, batch_size=max(64, BATCH_SIZE))

    # 割当CSV
    assign_csv_all = os.path.join(out_dir, f'{method_name}_assign_all.csv')
    pd.DataFrame({
        'time': times_all,
        'bmu_x': winners_all[:,0], 'bmu_y': winners_all[:,1],
        'label_raw': labels_all if labels_all is not None else ['']*len(winners_all)
    }).to_csv(assign_csv_all, index=False, encoding='utf-8-sig')
    log.write(f'\nAssigned BMU (all) -> {assign_csv_all}\n')

    # ノード平均パターン（偏差[hPa]）
    bigmap_all = os.path.join(out_dir, f'{method_name}_som_node_avg_all.png')
    plot_som_node_average_patterns(
        data_all, winners_all, lat, lon, (SOM_X,SOM_Y),
        save_path=bigmap_all,
        title=f'{method_name.upper()} SOM Node Avg SLP Anomaly (All)'
    )
    log.write(f'Node average patterns (all) -> {bigmap_all}\n')

    # 各ノード平均画像（個別）
    pernode_dir_all = os.path.join(out_dir, f'{method_name}_pernode_all')
    save_each_node_mean_image(data_all, winners_all, lat, lon, (SOM_X,SOM_Y),
                              out_dir=pernode_dir_all, prefix='all')
    log.write(f'Per-node mean images (all) -> {pernode_dir_all}\n')

    # ===== ノードの多数決（元ラベル/基本ラベル） =====
    node_to_majority_raw, node_to_majority_base = compute_training_node_majorities(
        winners_all, labels_all, BASE_LABELS, (SOM_X, SOM_Y)
    )
    # 学習時のノード別 基本ラベル出現数
    node_to_base_counts = compute_training_node_base_counts(
        winners_all, labels_all, BASE_LABELS, (SOM_X, SOM_Y)
    )

    # ===== True Medoid（総距離最小）も計算・保存 =====
    node_to_true_medoid_idx, node_to_true_medoid_avgdist = compute_node_true_medoids(
        method_name=method_name,
        data_flat=data_all, winners_xy=winners_all,
        som_shape=(SOM_X, SOM_Y), field_shape=field_shape,
        device=device,
        fusion_alpha=0.5
    )

    # True Medoid マップ
    true_medoid_bigmap = os.path.join(out_dir, f'{method_name}_som_node_true_medoid_all.png')
    plot_som_node_medoid_patterns(
        data_flat=data_all,
        node_to_medoid_idx=node_to_true_medoid_idx,
        lat=lat, lon=lon, som_shape=(SOM_X, SOM_Y),
        times_all=times_all,
        save_path=true_medoid_bigmap,
        title=f'{method_name.upper()} SOM Node True Medoid (min-sum-of-distances)'
    )
    log.write(f'True medoid map (all) -> {true_medoid_bigmap}\n')

    # True Medoid 個別図
    pernode_true_medoid_dir = os.path.join(out_dir, f'{method_name}_pernode_true_medoid_all')
    save_each_node_medoid_image(
        data_flat=data_all,
        node_to_medoid_idx=node_to_true_medoid_idx,
        lat=lat, lon=lon, som_shape=(SOM_X, SOM_Y),
        out_dir=pernode_true_medoid_dir, prefix='all_true'
    )
    log.write(f'Per-node true medoid images (all) -> {pernode_true_medoid_dir}\n')

    # True Medoid CSV
    rows_true = []
    for (ix, iy), mi in sorted(node_to_true_medoid_idx.items()):
        t_str = format_date_yyyymmdd(times_all[mi]) if len(times_all)>0 else ''
        raw = labels_all[mi] if labels_all is not None else ''
        label_base_or_none = basic_label_or_none(raw, BASE_LABELS)
        avgdist = node_to_true_medoid_avgdist.get((ix,iy), np.nan)
        rows_true.append({
            'node_x': ix, 'node_y': iy, 'node_flat': ix*SOM_Y+iy,
            'true_medoid_index': mi, 'time': t_str,
            'label_raw': raw,
            'label': label_base_or_none,
            'avg_distance_in_node': avgdist
        })
    true_medoid_csv = os.path.join(out_dir, f'{method_name}_node_true_medoids.csv')
    pd.DataFrame(rows_true).to_csv(true_medoid_csv, index=False, encoding='utf-8-sig')
    log.write(f'Node true medoid CSV -> {true_medoid_csv}\n')


    # ===== 「SOM Node-wise Analysis」の一致可視化（背景色） =====
    nodewise_vis_path = os.path.join(out_dir, f'{method_name}_nodewise_analysis_match.png')
    plot_nodewise_match_map(
        winners_xy=winners_all,
        labels_all=labels_all,
        node_to_medoid_idx=node_to_true_medoid_idx,
        times_all=times_all,
        som_shape=(SOM_X, SOM_Y),
        save_path=nodewise_vis_path,
        title=f'{method_name.upper()} SOM Node-wise Analysis (True Medoid: green=match / red=not)'
    )
    log.write(f'Node-wise match visualization -> {nodewise_vis_path}\n')

    # ===== 評価（ラベルがあれば） =====
    final_match_rate, final_matched_nodes, final_counted_nodes = compute_nodewise_match_rate(
        winners_xy=winners_all,
        labels_all=labels_all,
        node_to_medoid_idx=node_to_true_medoid_idx,
        som_shape=(SOM_X, SOM_Y)
    )

    if labels_all is not None:
        # 混同行列（基本ラベルのみ）を構築・保存
        clusters_all = winners_to_clusters(winners_all, (SOM_X, SOM_Y))
        cm, cluster_names = build_confusion_matrix_only_base(clusters_all, labels_all, BASE_LABELS)
        conf_csv = os.path.join(out_dir, f'{method_name}_confusion_matrix_all.csv')
        cm.to_csv(conf_csv, encoding='utf-8-sig')
        log.write(f'\nConfusion matrix (base vs clusters) -> {conf_csv}\n')

        # 集約指標（基本/複合）
        metrics = evaluate_clusters_only_base(
            clusters=clusters_all,
            all_labels=labels_all,
            base_labels=BASE_LABELS,
            title=f"[{method_name.upper()}] SOM Final Evaluation (Base labels)"
        )
        if metrics is not None:
            log.write('\n[Final Metrics]\n')
            for k in ['MacroRecall_majority', 'MacroRecall_composite']:
                if k in metrics:
                    log.write(f'  {k} = {metrics[k]:.6f}\n')
            # 追加：match rate
            if not np.isnan(final_match_rate):
                log.write(f'  NodewiseMatchRate = {final_match_rate:.6f} (matched {final_matched_nodes}/{final_counted_nodes} nodes)\n')
            else:
                log.write(f'  NodewiseMatchRate = NaN (no countable nodes)\n')

        # ラベル分布ヒートマップ（基本ラベルのみ） 1枚
        dist_dir_all = os.path.join(out_dir, f'{method_name}_label_dist_all')
        plot_label_distributions_base(winners_all, labels_all, BASE_LABELS, (SOM_X,SOM_Y), dist_dir_all, title_prefix='All')
        # 追加：基本ラベルごとの個別画像
        save_label_distributions_base_individual(winners_all, labels_all, BASE_LABELS, (SOM_X,SOM_Y), dist_dir_all, title_prefix='All')

        log.write(f'Label-distribution heatmaps (base only) -> {dist_dir_all}\n')

        # ノード詳細（構成・月分布など）を results.log に（代表ラベルrawも出力）
        analyze_nodes_detail_to_log(
            clusters_all, labels_all, times_all, BASE_LABELS, (SOM_X, SOM_Y),
            log, title=f'[{method_name.upper()}] SOM Node-wise Analysis'
        )

        # 代表ノード群ベースの再現率（基本/複合）と、各ラベルの代表ノード一覧を results.log に出力
        log_som_recall_by_label_with_nodes(
            log=log,
            winners_xy=winners_all,
            labels_all=labels_all,
            base_labels=BASE_LABELS,
            som_shape=(SOM_X, SOM_Y),
            section_title='SOM代表ノード群ベースの再現率（基本/複合）'
        )
    else:
        # ラベルが無いケースでも match rate は NaN になるだけ
        log.write('Labels not found; skip evaluation.\n')
        if not np.isnan(final_match_rate):
            log.write('\n[Final Metrics]\n')
            log.write(f'  NodewiseMatchRate = {final_match_rate:.6f} (matched {final_matched_nodes}/{final_counted_nodes} nodes)\n')

    # 学習時ノード代表（raw/基本）ラベルをJSONに保存（検証時に利用）
    majority_rows = []
    for ix in range(SOM_X):
        for iy in range(SOM_Y):
            majority_rows.append({
                'node_x': ix, 'node_y': iy,
                'majority_raw': node_to_majority_raw.get((ix,iy)),
                'majority_base': node_to_majority_base.get((ix,iy))
            })
    maj_json = os.path.join(out_dir, 'node_majorities.json')
    with open(maj_json, 'w', encoding='utf-8') as f:
        json.dump(majority_rows, f, ensure_ascii=False, indent=2)
    log.write(f'Node majorities (raw/base) saved -> {maj_json}\n')

    # ノード別 基本ラベル出現数も保存
    counts_rows = []
    for ix in range(SOM_X):
        for iy in range(SOM_Y):
            counts_rows.append({
                'node_x': ix,
                'node_y': iy,
                'base_counts': node_to_base_counts.get((ix, iy), {})
            })
    counts_json = os.path.join(out_dir, 'node_base_counts.json')
    with open(counts_json, 'w', encoding='utf-8') as f:
        json.dump(counts_rows, f, ensure_ascii=False, indent=2)
    log.write(f'Node base-label counts (training) saved -> {counts_json}\n')

    log.write('\n=== Done (learning) ===\n')
    log.close()

    # 検証で使用するためにSOMインスタンスと学習代表（raw/基本）辞書を返却
    return som, node_to_majority_raw, node_to_majority_base, node_to_base_counts


# =====================================================
# 検証（学習済モデルを使って、検証期間データで評価）
# =====================================================
def run_one_method_verification(method_name: str,
                                som: MultiDistMiniSom,
                                train_majority_raw: Dict[Tuple[int,int], Optional[str]],
                                train_majority_base: Dict[Tuple[int,int], Optional[str]],
                                data_valid: np.ndarray,
                                labels_valid: Optional[List[Optional[str]]],
                                times_valid: np.ndarray,
                                field_shape: Tuple[int,int],
                                lat: np.ndarray, lon: np.ndarray,
                                out_dir: str,
                                train_base_counts: Optional[Dict[Tuple[int,int], Dict[str,int]]] = None):
    """
    学習済み SOM を用いて検証データをBMU割当し、学習時のノード代表（基本）ラベルで検証を評価
    """
    os.makedirs(out_dir, exist_ok=True)
    vlog = Logger(os.path.join(out_dir, f'{method_name}_verification.log'))
    vlog.write(f'=== {method_name} SOM (Verification period) ===\n')
    vlog.write(f'Verification samples: {data_valid.shape[0]}\n')
    if len(times_valid) > 0:
        tmin = pd.to_datetime(times_valid.min()).strftime('%Y-%m-%d')
        tmax = pd.to_datetime(times_valid.max()).strftime('%Y-%m-%d')
        vlog.write(f'Verification Period: {tmin} to {tmax}\n')

    # 検証データのBMU予測
    winners_val = som.predict(data_valid, batch_size=max(64, BATCH_SIZE))

    # ラベルがあれば、検証評価（学習代表基本ラベルを予測とする）
    if labels_valid is not None and len(labels_valid) == len(winners_val):
        # 混同行列や per-label 再現率など
        metrics = evaluate_verification_with_training_majority(
            winners_xy_valid=winners_val,
            labels_valid=labels_valid,
            times_valid=times_valid,
            base_labels=BASE_LABELS,
            som_shape=(SOM_X, SOM_Y),
            node_to_majority_base_train=train_majority_base,
            out_dir=out_dir,
            method_name=method_name,
            logger=vlog,
            node_to_base_counts_train=train_base_counts,
            sigma_pred=1.2
        )
        # ラベル分布ヒートマップ（基本/個別）
        dist_dir_val = os.path.join(out_dir, f'{method_name}_verification_label_dist')
        plot_label_distributions_base(winners_val, labels_valid, BASE_LABELS, (SOM_X, SOM_Y), dist_dir_val, title_prefix='Verification')
        save_label_distributions_base_individual(winners_val, labels_valid, BASE_LABELS, (SOM_X, SOM_Y), dist_dir_val, title_prefix='Verification')
        vlog.write(f'Label-distribution heatmaps for verification -> {dist_dir_val}\n')
    else:
        vlog.write('Labels not found for verification; skip evaluation.\n')

    # 参考：検証データのノード平均（必要であれば）
    avg_map_path = os.path.join(out_dir, f'{method_name}_verification_node_avg.png')
    plot_som_node_average_patterns(
        data_valid, winners_val, lat, lon, (SOM_X,SOM_Y),
        save_path=avg_map_path,
        title=f'{method_name.upper()} SOM Node Avg SLP Anomaly (Verification)'
    )
    vlog.write(f'Verification node average patterns -> {avg_map_path}\n')

    vlog.write('=== Done (verification) ===\n')
    vlog.close()


# =====================================================
# 評価集約ユーティリティ: 各手法の *_results.log から代表ノード群ベースの再現率ブロックを集約
# =====================================================
def write_evaluation_summary(learning_root: str, result_root: str, methods: List[Tuple[str, str]]) -> str:
    """
    learning_result/{method}_som/{method}_results.log の中から
    「【SOM代表ノード群ベースの再現率（基本/複合）】」ブロックを抜き出して集約ログを作成。
    出力先: {result_root}/evaluation_v5.log
    戻り値: 出力ファイルパス
    """
    out_path = os.path.join(result_root, 'evaluation_v5.log')
    start_tokens = [
        '【SOM代表ノード群ベースの再現率（基本/複合）】',
        'SOM代表ノード群ベースの再現率'  # フォールバック（括弧の有無などに強くする）
    ]
    with open(out_path, 'w', encoding='utf-8') as fout:
        fout.write('=== Evaluation Summary (SOM代表ノード群ベース) ===\n\n')
        for mname, _ in methods:
            log_path = os.path.join(learning_root, f'{mname}_som', f'{mname}_results.log')
            if not os.path.exists(log_path):
                continue
            try:
                with open(log_path, 'r', encoding='utf-8') as fin:
                    text = fin.read()
            except Exception:
                continue

            # 開始位置（所望の見出し）を特定
            start_idx = -1
            for tok in start_tokens:
                start_idx = text.find(tok)
                if start_idx != -1:
                    break
            if start_idx == -1:
                # 見つからなければスキップ
                continue

            # 開始以降の本文を取り出し、次のセクション見出し（=== や --- など）以降はカット
            snippet = text[start_idx:].strip()
            terminators = ['\n=== ', '\n--- [', '\n--- ']
            cut_idx = len(snippet)
            for t in terminators:
                j = snippet.find(t, 1)  # 先頭は見出しなので 1 から
                if j != -1:
                    cut_idx = min(cut_idx, j)
            snippet = snippet[:cut_idx].rstrip()

            fout.write(f'--- [{mname.upper()}] ---\n')
            fout.write(snippet + '\n\n')

    return out_path

# =====================================================
# メイン
# =====================================================
def main():
    global SEED, RESULT_DIR, LEARNING_ROOT, VERIF_ROOT
    parser = argparse.ArgumentParser(description="PressurePattern SOM v5")
    parser.add_argument('--seed', type=int, default=SEED, help='random seed')
    parser.add_argument('--gpu', type=int, default=None, help='GPU index to use (e.g., 0 or 1). Ignored if --device is set.')
    parser.add_argument('--device', type=str, default=None, help="Device to use: 'cpu', 'cuda', or 'cuda:N'")
    parser.add_argument('--result-dir', type=str, default=None, help='Output directory root. If not set, auto-generated by seed and device.')
    args = parser.parse_args()

    # デバイス解決
    dev = args.device
    if dev is None:
        if args.gpu is not None and torch.cuda.is_available():
            dev = f'cuda:{args.gpu}'
        else:
            dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    if isinstance(dev, str) and dev.startswith('cuda') and not torch.cuda.is_available():
        print("Warning: CUDA is not available; falling back to CPU.")
        dev = 'cpu'

    if isinstance(dev, str) and dev.startswith('cuda') and torch.cuda.is_available():
        try:
            parts = dev.split(':')
            idx = int(parts[1]) if len(parts) > 1 else torch.cuda.current_device()
            if idx >= torch.cuda.device_count():
                print(f"Warning: requested GPU index {idx} >= device_count {torch.cuda.device_count()}, using GPU 0 instead.")
                dev = 'cuda:0'
                idx = 0
            selected_gpu_index = idx
            gpu_name = torch.cuda.get_device_name(idx)
        except Exception:
            selected_gpu_index = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(selected_gpu_index)
    else:
        selected_gpu_index = None
        gpu_name = 'CPU'

    # グローバルを更新（SEEDと出力先）
    SEED = int(args.seed)

    if args.result_dir:
        RESULT_DIR = args.result_dir
    else:
        dev_tag = 'cpu' if dev == 'cpu' else f'cuda{selected_gpu_index if selected_gpu_index is not None else 0}'
        RESULT_DIR = f'./results_v5_iter{NUM_ITER}_batch{BATCH_SIZE}_seed{SEED}_{dev_tag}'
    LEARNING_ROOT = os.path.join(RESULT_DIR, 'learning_result')
    VERIF_ROOT = os.path.join(RESULT_DIR, 'verification_results')

    set_reproducibility(SEED)
    setup_logging_v5()
    device = dev
    logging.info(f"使用デバイス: {device} ({gpu_name})")
    logging.info(f"SOM(3type): size={SOM_X}x{SOM_Y}, iters={NUM_ITER}, batch={BATCH_SIZE}, nodes_chunk={NODES_CHUNK}")

    # 学習データ（1991-01-01〜1999-12-31）
    X_for_s1_L, X_original_hpa_L, X_anomaly_hpa_L, lat, lon, d_lat, d_lon, ts_L, labels_L = load_and_prepare_data_unified(
        DATA_FILE, LEARN_START, LEARN_END, device
    )
    data_learn = X_anomaly_hpa_L.reshape(X_anomaly_hpa_L.shape[0], -1).astype(np.float32)

    # 検証データ（2000-01-01〜2000-12-31）
    X_for_s1_V, X_original_hpa_V, X_anomaly_hpa_V, lat2, lon2, d_lat2, d_lon2, ts_V, labels_V = load_and_prepare_data_unified(
        DATA_FILE, VALID_START, VALID_END, device
    )
    data_valid = X_anomaly_hpa_V.reshape(X_anomaly_hpa_V.shape[0], -1).astype(np.float32)

    # 次元・座標は同じはずだが、保険で確認
    assert d_lat == d_lat2 and d_lon == d_lon2, "学習/検証でグリッドサイズが異なります。"
    assert np.allclose(lat, lat2) and np.allclose(lon, lon2), "学習/検証でlat/lonが異なります。"

    field_shape = (d_lat, d_lon)

    # 3種類のbatchSOM（学習→検証）
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if isinstance(device, str) and device.startswith('cuda') and torch.cuda.is_available():
        try:
            parts = device.split(':')
            idx = int(parts[1]) if len(parts) > 1 else torch.cuda.current_device()
        except Exception:
            idx = torch.cuda.current_device()
        print(f"GPU: {torch.cuda.get_device_name(idx)} (index={idx})")

    methods = [
        ('euclidean',   'euclidean'),
        ('ssim5',       'ssim5'),        # 論文仕様（5x5窓・C=0）版SSIM
        ('s1',          's1'),
        ('ms_s1',       'ms_s1'),        # Multi-Scale S1
        ('msssim_s1g',  'msssim_s1g'),   # MSSSIM*-S1 Gate
        ('s1ssim',      's1ssim'),       # S1とSSIM(5x5)の等重み融合
        ('s1ssim5_hf',  's1ssim5_hf'),   # 提案: HF-S1SSIM5（SSIMゲートのソフト階層化）
        ('s1ssim5_and', 's1ssim5_and'),  # 新提案: AND合成（max融合）
        ('pf_s1ssim',   'pf_s1ssim'),    # 新提案: 比例融合（積）
        ('s1gssim',     's1gssim'),      # 新提案: 勾配SSIM(+方向)+S1のRMS合成
        ('s1gl',        's1gl'),         # 新提案: DSGC (S1 + Gradient + Curvature)
        ('gsmd',        'gsmd'),         # 新提案: GSMD (Gradient–Structural–Moment)
        ('s3d',         's3d'),          # 新提案: S3D (SSIM + Gradient structure + Curvature structure)
        ('cfsd',        'cfsd'),         # 新提案: CFSD (G-SSIM + normalized S1 + normalized curvature S1)
        ('hff',         'hff'),          # 新提案: HFF (Hierarchical Feature Fusion)
        ('s1gk',        's1gk'),         # 新提案: S1GK (S1 + G-SSIM + Kappa curvature)
        ('gssim',       'gssim'),        # 新提案: 勾配構造類似距離
        ('spot',        'spot'),         # 新提案: SPOT (Optimal Transport)
        ('gvd',         'gvd'),          # 新提案: GVD (Vorticity–Deformation)
        ('s1gcurv',    's1gcurv')       # 新提案: S1GCurv (S1 + Gradient Curvature)
    ]
    for mname, adist in methods:
        # 学習
        out_dir_learn = os.path.join(LEARNING_ROOT, f'{mname}_som')
        som, majority_raw_train, majority_base_train, base_counts_train = run_one_method_learning(
            method_name=mname, activation_distance=adist,
            data_all=data_learn, labels_all=labels_L, times_all=ts_L,
            field_shape=field_shape, lat=lat, lon=lon, out_dir=out_dir_learn, device=device
        )
        # 検証
        out_dir_verif = os.path.join(VERIF_ROOT, f'{mname}_som')
        run_one_method_verification(
            method_name=mname,
            som=som,
            train_majority_raw=majority_raw_train,
            train_majority_base=majority_base_train,
            data_valid=data_valid,
            labels_valid=labels_V,
            times_valid=ts_V,
            field_shape=field_shape,
            lat=lat, lon=lon,
            out_dir=out_dir_verif,
            train_base_counts=base_counts_train
        )

    # 各手法の結果ログから集約評価ログを生成
    eval_log_path = write_evaluation_summary(LEARNING_ROOT, RESULT_DIR, methods)
    logging.info(f"Evaluation summary saved -> {eval_log_path}")

    logging.info("全処理完了（学習＋検証）。")


if __name__ == '__main__':
    main()
