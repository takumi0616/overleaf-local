# 気圧配置分類（SOM, Multi-distance）

本プログラムは ERA5 の海面更正気圧（msl）から偏差を作り、SOM（自己組織化マップ）でクラスタリングします。  
距離は以下に対応しています（コード内 methods 参照）:

- euclidean（ユークリッド）
- ssim5（5x5 移動窓, C=0）
- s1（Teweles–Wobus S1）
- s1ssim / s1ssim5_hf / s1ssim5_and / pf_s1ssim（S1 と SSIM の合成）
- s1gssim / gssim（勾配構造類似）

学習後は代表ラベルや True Medoid、混同行列、各種メトリクスを CSV/PNG/ログに出力します。

データファイル:

- `./prmsl_era5_all_data_seasonal_large.nc`（同ディレクトリに配置）

# 実行方法（GPU/CPU 切替と複数同時実行）

main_v5.py に以下の引数を追加しました:

- `--seed INT` 乱数シード（デフォルト: コード側の SEED）
- `--gpu INT` 使う GPU 番号（0 や 1 など）。`--device` 指定がある場合は無視されます
- `--device STR` `'cpu'`, `'cuda'`, `'cuda:N'` を直接指定（例: `cuda:0`）
- `--result-dir PATH` 出力先ルートを明示指定（未指定なら `results_v5_iter{NUM}_batch{BATCH}_seed{SEED}_{devtag}` が自動生成）

優先順位:

1. `--device` を指定したらそれを使用
2. `--device` 未指定かつ `--gpu` 指定があれば `cuda:{gpu}` を使用
3. どちらも未指定なら、CUDA 利用可能時は `cuda:0`、なければ `cpu`

出力先:

- ルート: `RESULT_DIR`（引数で明示指定可、未指定時は seed と device から自動命名）
- 配下: `learning_result/` と `verification_results/`
- 実行全体ログ: `RESULT_DIR/run_v4.log`（既存命名を踏襲）
- 距離別ログ/CSV/PNG が個別フォルダに出ます

## 1GPU 環境（自動で GPU0 を使用）

```bash
# 前提: カレントは src/PressurePattern
python main_v5.py --seed 1
# あるいは明示
python main_v5.py --gpu 0 --seed 1
```

CPU 実行に切り替えたい場合:

```bash
python main_v5.py --device cpu --seed 1
```

出力例（自動命名）:

```
./results_v5_iter1000_batch256_seed1_cuda0/
  ├─ run_v4.log
  ├─ learning_result/
  └─ verification_results/
```

## 2GPU 環境（GPU の 0 番 と 1 番 を同時実行）

以下のように別のシード・別の GPU 番号を与えると、同時に 2 本走らせられます。  
1GPU 環境でも `--gpu 0` のみを使えば従来通り動作します。

- フォアグラウンドで開始（例）

```bash
python main_v5.py --gpu 0 --seed 1
python main_v5.py --gpu 1 --seed 2
```

- バックグラウンドで開始（nohup + ログ標準出力 redirection）

```bash
nohup python main_v5.py --gpu 0 --seed 1 > seed1_gpu0.out 2>&1 &
nohup python main_v5.py --gpu 1 --seed 2 > seed2_gpu1.out 2>&1 &
```

- notify-run を使う（通知が欲しい場合）

```bash
notify-run gpu02 -- nohup python main_v5.py --gpu 0 --seed 1 > seed1.out 2>&1 &
notify-run gpu02 -- nohup python main_v5.py --gpu 1 --seed 20 > seed20.out 2>&1 &
```

出力先を明示したい場合（同名衝突を避けたいとき等）:

```bash
nohup python main_v5.py --gpu 0 --seed 1 --result-dir ./results_gpu0_seed1 > s1_g0.out 2>&1 &
nohup python main_v5.py --gpu 1 --seed 2 --result-dir ./results_gpu1_seed2 > s2_g1.out 2>&1 &
```

補足:

- `--device cuda:0` / `--device cuda:1` と書いても同様に動作します（`--device` が優先されます）
- `CUDA が利用不可` の環境で `--device cuda:*` を渡した場合は自動で `cpu` にフォールバックします（警告出力あり）

## 代表的なプロセス管理

- 実行中のプロセス確認:

```bash
ps aux | grep main_v5.py | grep -v grep
```

- 全て停止（強制）:

```bash
pkill -f "main_v5.py"
```

- 指定 GPU だけを止めたい場合はコマンドラインを絞り込む:

```bash
pkill -f "main_v5.py.*--gpu 0"
# または明示 device 指定で動かしているなら
pkill -f "main_v5.py.*--device cuda:0"
```

- 個別に PID を kill（安全策）:

```bash
# 確認
ps aux | grep "main_v5.py" | grep -v grep
# 停止
kill -9 <PID>
```

# 出力物の場所と内容

- ルート: `RESULT_DIR`（例: `results_v5_iter1000_batch256_seed1_cuda0`）
  - `run_v4.log`: 実行全体ログ（開始/デバイス/メトリクス要約など）
  - `learning_result/*`: 各距離法ごとのログ・CSV・図
    - `*_results.log`（学習ログ、QE・MacroRecall・NodewiseMatchRate など）
    - `*_iteration_metrics.csv / .png`（学習イテレーションごとの履歴）
    - `*_assign_all.csv`（全サンプルの BMU 割当）
    - `*_som_node_avg_all.png`（ノード平均マップ）
    - `*_som_node_true_medoid_all.png` / `*_node_true_medoids.csv` など（True Medoid 関連）
    - ラベル分布図（base のみ）など
  - `verification_results/*`:
    - 検証混同行列, per-label 再現率 CSV, バー図
    - 検証データのノード平均マップ, ラベル分布図（base のみ）など

# よく使う実行例

- 1GPU 環境で seed を変えて順番に実行:

```bash
python main_v5.py --gpu 0 --seed 1
python main_v5.py --gpu 0 --seed 2
```

- 2GPU 環境で seed を変えて同時に 2 本:

```bash
nohup python main_v5.py --gpu 0 --seed 1 > s1_g0.out 2>&1 &
nohup python main_v5.py --gpu 1 --seed 2 > s2_g1.out 2>&1 &
```

- CPU で試す:

```bash
python main_v5.py --device cpu --seed 1
```

- 出力を明示し（分かりやすいフォルダ名で）保存:

```bash
python main_v5.py --gpu 0 --seed 7 --result-dir ./results_gpu0_seed7
```

# 結果を mac に転送（rsync 例）

プロジェクトルート例:

- gpu01/gpu02: `/home/devel/work_takasuka_git/docker_miniconda/src/PressurePattern`
- via-tml2: `/home/s233319/docker_miniconda/src/PressurePattern`
- wsl-ubuntu: `/home/takumi/docker_miniconda/src/PressurePattern`

フォルダ名は seed / device により変化します（例: `results_v5_iter1000_batch256_seed1_cuda0`）。  
パターンマッチや個別フォルダを指定して rsync してください。

- wsl-ubuntu → mac

```bash
rsync -avz --progress \
  'wsl-ubuntu:/home/takumi/docker_miniconda/src/PressurePattern/results_v5_iter1000_batch256_seed{1,2}_*' \
  /Users/takumi0616/Develop/docker_miniconda/src/PressurePattern/result_wsl-ubuntu
```

- via-tml2 → mac

```bash
rsync -avz --progress \
  'via-tml2:/home/s233319/docker_miniconda/src/PressurePattern/results_v5_iter1000_batch256_seed{1,2}_*' \
  /Users/takumi0616/Develop/docker_miniconda/src/PressurePattern/result_via-tml2
```

- gpu01 → mac

```bash
rsync -avz --progress \
  'gpu01:/home/devel/work_takasuka_git/docker_miniconda/src/PressurePattern/results_v5_iter1000_batch256_seed{19,20}_*' \
  /Users/takumi0616/Develop/docker_miniconda/src/PressurePattern/result_gpu01
```

- gpu02 → mac

```bash
rsync -avz --progress \
  'gpu02:/home/devel/work_takasuka_git/docker_miniconda/src/PressurePattern/results_v5_iter1000_batch256_seed{11,12}_*' \
  /Users/takumi0616/Develop/docker_miniconda/src/PressurePattern/result_gpu02
```

注意:

- `results_v5_*` のパターンで一括転送できます。個別に指定する場合は実行時に表示された `RESULT_DIR` をそのまま使うと確実です。
- リモート側のシェルのグロブ展開に依存するため、必要に応じて引用符/エスケープを調整してください。

# 変更点（今回の改善）

- GPU/CPU デバイス指定の CLI 対応
  - `--device`（例: `cuda:0`, `cuda:1`, `cpu`）
  - `--gpu`（整数で GPU 番号。`--device` 指定時は無視）
- 1GPU/2GPU 双方での同一コード動作
  - 2GPU 環境では `--gpu 0` と `--gpu 1` で同時に別条件実行が可能
  - 1GPU 環境では自動で `cuda:0`（ない場合は `cpu` にフォールバック）
- `--seed` と `--result-dir` を追加
  - シードや出力先を変えて複数ジョブを並走/整理しやすく
- True Medoid 計算等、内部で用いる torch デバイスをユーザ指定デバイスに統一
  - GPU 0/1 の使い分けが混在しないよう安全に反映

## 自動化: 2GPU で seed を順次実行（notify-run 通知付き）

手動で `seed` を 2 つずつ増やしながら実行する作業を自動化するスクリプトを追加しました。

- スクリプト: `src/PressurePattern/run_seeds.sh`
- 既定動作:
  - `seed=19,20` のペアから開始し、`seed=49,50` まで実行（デフォルトは 19..50）
  - GPU0 に奇数（19, 21, 23, ...）、GPU1 に偶数（20, 22, 24, ...）を割当
  - 各ペア（2 本）の終了を待ってから次ペアへ進む
  - ログは `seed{SEED}.out` に保存（例: `seed19.out`, `seed20.out`）
  - `notify-run <channel> -- <command>` で実行開始/完了の通知（チャンネル既定: `gpu02`）
- 実行場所はどこでも OK（スクリプトが自動で `src/PressurePattern` に `cd` します）

使い方（基本）:

```bash
chmod +x run_seeds.sh
```

```bash
# 範囲やGPU、チャネル、対象スクリプトを明示
nohup bash run_seeds.sh --start 13 --end 14 --gpu0 0 --gpu1 1 --channel gpu02 --script main_v5.py > orchestrator.log 2>&1 &
```

そのコマンドは正常に動きます。前提として src/PressurePattern に移動してから実行する場合、以下が成立します。

- nohup bash run_seeds.sh --start 21 --end 50 --gpu0 0 --gpu1 1 --channel gpu02 --script main_v5.py > orchestrator.log 2>&1 &

  - オーケストレータ(run_seeds.sh)がバックグラウンドで継続
  - orchestrator.log はカレントの src/PressurePattern に出力
  - 各ジョブの標準出力・標準エラーは seed{SEED}.out（例: seed21.out, seed22.out）として src/PressurePattern に作成
  - スクリプト内で自身のあるディレクトリに cd するため、ジョブの実行とログの場所は安定（相対パスの main_v5.py も解決可能）

補足:

- 同じ内容をリポジトリルートから実行したい場合は以下でも可:

  - nohup bash src/PressurePattern/run_seeds.sh --start 21 --end 50 --gpu0 0 --gpu1 1 --channel gpu02 --script main_v5.py > src/PressurePattern/orchestrator.log 2>&1 &

- 内側のコマンドに nohup は不要です（スクリプト自体を nohup で起動しているため、子プロセスも SIGHUP を受けません）。

- 動作確認:

  - tail -f src/PressurePattern/orchestrator.log
  - ps aux | grep main_v5.py | grep -v grep

- 事前条件:

  - notify-run が PATH 上にあり、チャンネル gpu02 が有効
  - main_v5.py が src/PressurePattern に存在し --gpu/--seed を受け付ける

このコマンドを打った時の中断の仕方は？ (swinunet_env) root@bace913e623d:/app/src/PressurePattern# nohup bash run_seeds.sh --start 21 --end 50 --gpu0 0 --gpu1 1 --channel gpu02 --script main_v5.py > orchestrator.log 2>&1 &

\*\*Checkpoint

Compare

Restore

API Request$0.0855

Cline wants to edit this file:

src/PressurePattern/run_seeds.sh‎

3

\*\*Checkpoint

Compare

Restore

API Request$0.0662

Cline wants to edit this file:

src/PressurePattern/run_seeds.sh‎

3

\*\*Checkpoint

Compare

Restore

API Request$0.0589

Task Completed

中断方法は以下です（nohup でバックグラウンド起動した run_seeds.sh を安全に止める手順）。

前提:

- スクリプトは PID を src/PressurePattern/run_seeds.pid に保存します
- TERM/INT シグナルを受けると、実行中の GPU ジョブもまとめて安全に停止するよう実装済みです

最も簡単な停止（推奨）:

- 実行中のコンテナ/ホストで src/PressurePattern にいる場合:
  - kill "$(cat run_seeds.pid)"
- どこからでもフルパスで:
  - kill "$(cat /app/src/PressurePattern/run_seeds.pid)"

確認:

- tail -n 50 -f orchestrator.log
  - "[INFO] Received signal, terminating active jobs..." が出て、ペア終了の待機を切り上げて停止します
- ps -p "$(cat run_seeds.pid)"
  - プロセスがいなければ停止済みです

補足（PID ファイルが見つからない/壊れている場合）:

- オーケストレータを直接探して止める:

  - pgrep -af 'run_seeds.sh'
  - kill <見つかった PID>

- まだ子プロセス（学習ジョブ）が残ってしまった場合は個別停止:

  - pkill -f 'main_v5.py.\*--gpu 0'
  - pkill -f 'main_v5.py.\*--gpu 1'

強制停止（最後の手段）:

- 正常停止できない場合のみ使用してください

  - pkill -9 -f 'main_v5.py'
  - kill -9 "$(cat run_seeds.pid)"; rm -f run_seeds.pid

再開方法:

- 途中まで実行済みなら、次の seed から再開

  - 例: seed 33 まで完了していれば、--start 35 を指定
  - nohup bash run_seeds.sh --start 35 --end 50 --gpu0 0 --gpu1 1 --channel gpu02 --script main_v5.py > orchestrator.log 2>&1 &

# results_v5 集計スクリプトの使い方（search_results_v5.py）

目的:

- `results_v5` 配下の各 seed 実験ディレクトリにある `evaluation_v5.log` を再帰的に探索し、手法ごとに次を集計します。
  - [Summary] Macro Recall（基本ラベル）の平均
  - [Summary] Macro Recall（基本+応用）の平均
  - ラベル 6A/6B/6C の統計（Correct の合計、Recall の平均）※「各ラベルの再現率（代表ノード群ベース）」の値を使用

スクリプト:

- `PressurePattern/search_results_v5.py`

基本的な使い方（プロジェクトルートから）:

```bash
nohup python search_results_v5.py  --root ./results_v5 > search_results_v5.log 2>&1 &
```

主なオプション:

- `--root PATH`:
  - 探索対象の `results_v5` ルートを指定（省略時は `search_results_v5.py` と同じディレクトリ直下の `results_v5`）。
- `--sort {rank,name,basic_combo}`:
  - 並び順を指定（デフォルト: `rank`）。
  - `rank`: 各表で平均値の降順（ランキング）で表示。
  - `name`: 手法名の昇順。
  - `basic_combo`: 「基本」表は基本の平均降順、「基本+応用」表は基本+応用の平均降順（`rank` と同じ挙動）。
- `--precision INT`:
  - 小数点以下の表示桁数（デフォルト: 4）。

出力（例）:

- 2 つの表を分けて表示します。
  1. 手法別 Macro Recall 平均（基本ラベル）
  2. 手法別 Macro Recall 平均（基本+応用）
- さらに、ラベル 6A / 6B / 6C について、手法別に「Correct 合計 / Recall 平均」を表示します。

注意:

- ログ内の該当値が欠損している場合、そのログは平均算出から自動的に除外されます。
- 6A/6B/6C は「代表ノード群ベース」の表からのみ集計します（複合ラベル表は使用しません）。
