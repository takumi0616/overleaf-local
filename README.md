# ローカル Overleaf CE (Community Edition) 構築手順 (Apple Silicon, macOS)

この README は、Apple M2 搭載 macOS 上で Overleaf CE をローカル構築し、日本語 TeX（pLaTeX/upLaTeX + dvipdfmx）でのコンパイル、VSCode + Git 連携（Docker で Overleaf と同一環境ビルド）までを一通りカバーします。  
作業ルートは以下とします。

- 作業ルート: `/Users/takumi0616/Develop/overleaf-local`
- リポジトリ: `overleaf`（本体）と `overleaf-toolkit`（ツールキット）

目次

- 1. 前提条件
- 2. リポジトリのクローン
- 3. Docker イメージのビルド（server-ce）
- 4. ツールキット初期化と起動
- 5. 初回アクセス（管理者作成）
- 6. 日本語 TeX 環境の導入（scheme-full）と PATH 反映
- 7. イメージ更新の永続化（任意）
- 8. データ保存場所の構造と注意点（CE のアーキテクチャ）
- 9. VSCode + Git での編集・ビルド（Docker で Overleaf と同一環境）
- 10. Overleaf 拡張での自己ホスト接続（任意）
- 11. よくあるエラーと対処
- 12. 起動・停止・ログ・クリーンアップ
- 13. ローカル →Overleaf 同期運用（scripts/overleaf-sync.sh）
- 14. トラブルシューティング（同期・アップロード周り）
- 15. 参考：サーバ側ログの確認（デバッグ）
- 16. Project_id の取得（一覧とスクリプト）

---

## 1. 前提条件

- Docker Desktop（Apple Silicon 向け）をインストール・起動し、Docker ログイン済みにしておく。
- Git が使えること。
- 作業用ディレクトリ（ここでは `/Users/takumi0616/Develop/overleaf-local`）を用意。
- ARM64 固定（Apple Silicon 前提）でのビルド推奨:
  ```bash
  # zsh 等に入れておくと安全
  export DOCKER_DEFAULT_PLATFORM=linux/arm64/v8
  ```
- macOS の `sed` は GNU と互換がない箇所があるため、必要に応じて `brew install coreutils`（realpath など）や `gsed` の利用を検討。

カレントディレクトリは常に:

```bash
cd /Users/takumi0616/Develop/overleaf-local
```

---

## 2. リポジトリのクローン

```bash
cd /Users/takumi0616/Develop/overleaf-local
git clone https://github.com/overleaf/overleaf
git clone https://github.com/overleaf/toolkit.git ./overleaf-toolkit
```

---

## 3. Docker イメージのビルド（server-ce）

`overleaf/server-ce/Makefile` により、以下のイメージが作成されます（ブランチ名 + コミット SHA に応じたタグ）。

- `sharelatex/sharelatex-base:<branch>` および `<branch>-<sha>`
- `sharelatex/sharelatex:<branch>` および `<branch>-<sha>`

手順:

```bash
cd /Users/takumi0616/Develop/overleaf-local/overleaf/server-ce

# ベースイメージ
make build-base

# CE 本体（コミュニティ）
make build-community
```

ビルド後のイメージ確認:

```bash
docker images | grep -E 'sharelatex/sharelatex|sharelatex/sharelatex-base'
```

（例）

```
sharelatex/sharelatex        main
sharelatex/sharelatex        main-cfcb9f32abfb017d82eb65ab555fd4e3fcaf2d24
sharelatex/sharelatex-base   main
sharelatex/sharelatex-base   main-cfcb9f32abfb017d82eb65ab555fd4e3fcaf2d24
```

ツールキットの `config/version` と合わせるため、必要に応じてタグを付け替えます（例: 5.5.4 を使う）。

```bash
cd /Users/takumi0616/Develop/overleaf-local
docker tag sharelatex/sharelatex:main sharelatex/sharelatex:5.5.4
```

---

## 4. ツールキット初期化と起動

初回のみ設定ファイルを生成します。

```bash
cd /Users/takumi0616/Develop/overleaf-local/overleaf-toolkit
bin/init
```

生成物:

- `overleaf-toolkit/config/overleaf.rc`
- `overleaf-toolkit/config/variables.env`
- `overleaf-toolkit/config/version`（例: `5.5.4`）

起動前に、`config/version` とイメージタグを一致させます（例: `5.5.4` へタグ付け済み）。そのまま起動:

```bash
cd /Users/takumi0616/Develop/overleaf-local/overleaf-toolkit
bin/up -d
```

Mongo/Redis/ShareLaTeX コンテナが立ち上がります。

### macOS sed 非互換への対策（参考）

- ツールキット内部の設定値抽出で `sed -r` が原因となりうるため、`-E` へ置換する等の対策が必要な場合があります。
- 本手順では、`overleaf-toolkit/lib/shared-functions.sh` の抽出を堅牢化して対処しています（realpath/grep/パラメータ展開で KEY=value から値のみ取り出す）。環境により異なるため、エラーが出る場合は `bin/doctor` を参照し、差分を見ながら `sed -E` への置換を検討してください。

---

## 5. 初回アクセス（管理者作成）

起動後、ブラウザで以下にアクセス:

- http://localhost/launchpad

管理者アカウントを作成後、ダッシュボードは http://localhost/ にアクセス。

---

## 6. 日本語 TeX 環境の導入（scheme-full）と PATH 反映

CE のイメージは最小構成の TeX Live のため、日本語用にフル導入します。

コンテナへ入って実行:

```bash
# コンテナへ入る
docker exec -it sharelatex bash

# TeX Live の自己更新 → フル導入（時間がかかります）
tlmgr update --self
tlmgr install scheme-full

# TeX Live 2025 以降はインストール後に PATH へリンクを張る必要あり
tlmgr path add

# 必要に応じてフォーマット再生成
fmtutil-sys --all

# platex/uplatex/pbibtex/upbibtex/mendex が使えることを確認
which platex uplatex pbibtex upbibtex mendex

exit
```

コンパイル設定（pLaTeX/upLaTeX + dvipdfmx）は Overleaf 側プロジェクトに `latexmkrc` を用意するか、ローカルでビルドする場合にプロジェクト直下へ配置します（後述）。

---

## 7. イメージ更新の永続化（任意）

TeX Live 等をコンテナで更新した状態を新しいイメージに保存しておくと再作成時に便利です。

```bash
# 実行中コンテナIDを確認
docker ps

# 例: 現在の version ファイルに合わせて保存
cd /Users/takumi0616/Develop/overleaf-local/overleaf-toolkit
cat config/version   # 例: 5.5.4
docker commit sharelatex sharelatex/sharelatex:5.5.4-with-texlive-full

# 今後そのタグを使いたければ version を合わせる
echo 5.5.4-with-texlive-full > config/version

# 再起動
bin/stop
bin/up -d
```

---

## 8. データ保存場所の構造と注意点（CE のアーキテクチャ）

- Overleaf CE の「プロジェクト正本」は MongoDB に保存されます。  
  ホスト上に「そのまま Git 管理できるプロジェクト本体フォルダ」は**用意されません**。
- `overleaf-toolkit/config/overleaf.rc` の `OVERLEAF_DATA_PATH` で、ホスト側にコンパイルや成果物、ファイルストア一部の永続化が行われます（デフォルトは `overleaf-toolkit/data/overleaf` がコンテナ内 `/var/lib/overleaf` にマウント）。
- 例（ホスト側のディレクトリと意味）:
  - `overleaf-toolkit/data/overleaf/compiles/<projectId>-<userId>/`  
    → コンパイルのために展開された**一時作業コピー**（`v1.tex` や `fig/` などが見えても正本ではない）
  - `overleaf-toolkit/data/overleaf/output/<projectId>-<userId>/generated-files/<build-id>/`  
    → `output.pdf` などの成果物
  - `overleaf-toolkit/data/overleaf/user_files/`  
    → ファイルストア実体の一部
- したがって、**ホスト上の特定ディレクトリ（例 `/project/学会1`）にプロジェクト正本を直接置く構造へ変更することは CE では非対応**です（Server Pro の Git Bridge とは別物）。  
  完全同期の代替案は §9/§10 のワークフロー参照。

---

## 9. VSCode + Git での編集・ビルド（Docker で Overleaf と同一環境）

CE ではサーバ側 Git 連携が無効のため、**ローカル Git 主導**が現実的です。Overleaf への反映は ZIP アップロード（置換）か拡張の同期機能を利用します。

### 9.1 ローカル Git リポジトリ作成

例: `~/Documents/papers/SITA2025` に ZIP を展開して初期化:

```bash
cd ~/Documents/papers
mkdir -p SITA2025
cd SITA2025
# ZIP 展開後
git init
git add .
git commit -m "Initial import"
```

`.gitignore` 例（生成物を除外）:

```
*.aux
*.bbl
*.blg
*.fls
*.fdb_latexmk
*.log
*.toc
*.out
*.dvi
*.synctex*
*.pdf
__MACOSX/
.DS_Store
```

### 9.2 latexmkrc（pLaTeX/upLaTeX + dvipdfmx）

プロジェクト直下に **拡張子なし**で作成。  
upLaTeX 推奨例:

```perl
$latex = 'uplatex -synctex=1 -halt-on-error -interaction=nonstopmode %O %S';
$bibtex = 'upbibtex %O %S';   # biblatex+biber なら: $biber = 'biber %O %S';
$dvipdf = 'dvipdfmx %O -o %D %S';
$makeindex = 'mendex %O -o %D %S';
$pdf_mode = 3;
```

ZIP に `output.pdf` が含まれていると Overleaf 側で No PDF 抑止になるため、含めない。

### 9.3 Docker で Overleaf と同一環境ビルド

```bash
# プロジェクト直下で
docker run --rm -t \
  -v "$(pwd)":/work -w /work \
  sharelatex/sharelatex:5.5.4 \
  bash -lc "tlmgr path add && latexmk"
```

VSCode のタスク例（`.vscode/tasks.json`）:

```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "Overleaf image: latexmk",
      "type": "shell",
      "command": "docker run --rm -t -v \"${workspaceFolder}\":/work -w /work sharelatex/sharelatex:5.5.4 bash -lc \"tlmgr path add && latexmk\"",
      "group": "build",
      "problemMatcher": []
    }
  ]
}
```

---

## 10. Overleaf 拡張での自己ホスト接続（任意）

- VSCode の「Overleaf Workshop」等の拡張を使い、**自己ホスト URL を http://localhost** に設定すると、拡張が VSCode 側フォルダと Overleaf を同期してくれる場合があります。
- 拡張の仕様によってはローカル保存先が指定できるため、**Git 管理中のフォルダ**に保存できるなら、それを使うのが便利です（保存先を変えられない拡張もあるため、各拡張のドキュメントを参照）。
- 404 や 401 などが出る場合は、ベース URL のミス（`/launchpad` を付けない）、CE 非対応 API を呼んでいる等が原因です。前述のローカル Git 主導 + ZIP 置換が最も堅実です。

---

## 11. よくあるエラーと対処

- `docker tag sharelatex/sharelatex:$(cat config/version)` で `No such image`  
  → ビルド直後は `sharelatex/sharelatex:main`（や `main-<sha>`）のため、`config/version` と一致するタグへ明示的に `docker tag` する。
- `Invalid MONGO_VERSION: MONGO_VERSION=6.0` 等  
  → macOS の `sed -r` 非互換が関与することあり。`sed -E` に置換、または抽出ロジックをパラメータ展開へ修正。`bin/doctor` で検出も可。
- `No PDF` / `output.pdf` がある  
  → プロジェクト内 `output.pdf` は削除/リネーム。`Main document` が正しい `.tex` 指定か確認。
- `LaTeX Error: File 'jarticle.cls' not found.`  
  → `tlmgr install scheme-full` 実行。pLaTeX/upLaTeX + dvipdfmx でコンパイルするため `latexmkrc` を用意。
- `platex/uplatex not found`  
  → `tlmgr path add` を実行（TeX Live 2025 以降はインストール後に必要）。`which platex uplatex` で確認。
- UI の `TeX Live 2024` 表示  
  → ラベル表示に過ぎず、実体はコンテナ内で確認した TeX Live（この例では 2025）。`latex --version` 等で実体を確認可。

---

## 12. 起動・停止・ログ・クリーンアップ

```bash
# 起動（既存設定で再起動）
cd /Users/takumi0616/Develop/overleaf-local/overleaf-toolkit
bin/start

# 停止
bin/stop

# ログ（別タブで追う等）
bin/logs

# 開発/デバッグ：コンテナに入る
bin/shell   # or: docker exec -it sharelatex bash
```

---

## 13. ローカル →Overleaf 同期運用（scripts/overleaf-sync.sh）

ローカルの任意フォルダを、ブラウザログイン済みの Overleaf CE に対して「ワンコマンド」で反映するためのスクリプト群（`/Users/takumi0616/Develop/overleaf-local/scripts/overleaf-sync.sh`）です。  
CSRF/Cookie/Referer/Origin など Overleaf 側の要件を満たしつつ、既存プロジェクトへのアップロードを安定化しています。

### 13.1 導入

1. ブラウザで http://localhost へログイン（通常の Overleaf UI）。
2. 開発者ツールなどで Cookie を確認し、`scripts/overleaf-sync.env` を用意（例）:

```bash
# /Users/takumi0616/Develop/overleaf-local/scripts/overleaf-sync.env
BASE_URL=http://localhost
COOKIE='overleaf.sid=...; _csrf=...'   # 属性(Path/HttpOnly等)は不要。name=value; を ; 区切りで。
SLEEP_SEC=0                            # 連投時のスリープ(秒)。レートリミット回避に有効。
# ROOT_FOLDER_ID=root-folder-id        # 基本未設定でOK（スクリプトが自動解決）
```

3. 実行権限を付与:

```bash
chmod +x /Users/takumi0616/Develop/overleaf-local/scripts/overleaf-sync.sh
```

### 13.2 仕組み（要点）

- 認証/CSRF:
  - Cookie は属性を除去した `"name=value; name2=value2"` を送信。
  - Cookie Jar（`-b/-c`）を併用して Set-Cookie ローテーションに追随。
  - CSRF は `/dev/csrf` → `/login` → `/project` の順で取得。
- 既存プロジェクトへのアップロード:

  - まず `/project/:id/folder` で「直下に一時フォルダ」を作成（`parent_folder_id` 未指定）。
  - レスポンス JSON の `_id`（作成フォルダ）を抽出し、アップロード先 `folder_id` として使用。
  - `/Project/:id/upload` に multipart を送信（`name`, `qqfile`, `relativePath=null`, `_csrf` など）。
  - これにより、`folder_not_found`（422）の根本原因であるフォルダ解決の不整合を回避。

- 保存先:
  - .tex/.bib/.sty などテキストは Docstore（Mongo）→ エディタで「ドキュメント」扱い。
  - 画像・JSON などは Filestore（/var/lib/overleaf/data/user_files/...）→ 「ファイル」扱い。

### 13.3 使い方（どのコマンドをいつ叩くか）

Project_id は「16. Project_id の取得（一覧とスクリプト）」を参照。

- 接続確認（ログイン/CSRF）

```bash
bash scripts/overleaf-sync.sh status
# 期待出力:
# Login OK (/project HTTP 200)
# CSRF OK: xxxxxxxx...
```

- プロジェクト一覧から対象 ID を調べた後、その構成を確認（任意）

```bash
bash scripts/overleaf-sync.sh entities <PROJECT_ID>
# 例:
# {"project_id":"...","entities":[{"path":"/v1.tex","type":"doc"}, ... ]}
```

- 既存プロジェクトへローカルディレクトリを反映（頻用）

```bash
bash scripts/overleaf-sync.sh update <PROJECT_ID> /path/to/local/dir
# 例: bash scripts/overleaf-sync.sh update 68b0022621c3446b6bee0f6c SITA2025_高須賀
# 各ファイル毎にHTTPステータス行と末尾JSONが出力される
```

- 新規プロジェクト作成（ZIP アップロード）

```bash
bash scripts/overleaf-sync.sh new /path/to/local/dir
# /project/new/upload にZIPを送信。成功時は {success:true, project_id: "..."} を返す
```

- CSRF トークン値だけを取得（デバッグ）

```bash
bash scripts/overleaf-sync.sh csrf
```

### 13.4 コマンドの意味と注意点

- `status`:
  - Cookie の形式チェック + `/project` 200 でログイン確認。
  - `/dev/csrf` 等から CSRF を取得できるかテスト。
- `entities <PROJECT_ID>`:
  - `/project/:id/entities` を GET。root からの `path` と `type(doc/file)` を列挙。
  - 認可の確認や、アップロード後の反映確認に便利。
- `new <DIR>`:
  - `<DIR>` を ZIP 化して `/project/new/upload` へ送信。新規プロジェクトを作る。
  - ZIP 内に `output.pdf` 等の生成物は含めないこと（No PDF の原因）。
- `update <PROJECT_ID> <DIR>`:
  - `<DIR>` 配下の全ファイルを走査し、`/project/:id/folder` で作成したフォルダ直下にアップロード。
  - `relativePath=null`（=サブフォルダを作らない）。必要に応じて相対パス対応へ拡張可能。
  - レートリミット回避として `SLEEP_SEC` を設定可能（デフォルト 0）。

### 13.5 よく使う例

```bash
# 1) まずは接続確認
bash scripts/overleaf-sync.sh status

# 2) 既存プロジェクトの構成確認
bash scripts/overleaf-sync.sh entities 68b0022621c3446b6bee0f6c

# 3) ローカルの SITA2025_高須賀 を既存プロジェクトへ反映
bash scripts/overleaf-sync.sh update 68b0022621c3446b6bee0f6c SITA2025_高須賀
```

### 13.6 スクリプトの内部ログ（デバッグ）

- フォルダ作成 API の raw レスポンスと、抽出した `NEW_ID` / `PARENT_ID` / `TARGET_ID` を stderr に出力（URL 汚染防止）。
- `update` の各ファイル POST で、HTTP ステータス行 + 末尾の JSON を出力（`success`/`error` を直接確認可能）。

---

## 14. トラブルシューティング（同期・アップロード周り）

- 403/401:
  - `status` で `/project` が 200 か、Cookie が `"name=value; ..."` 形式になっているか確認。
  - CSRF が取得できるかを `csrf` で確認。
- 404:
  - エンドポイントの大文字小文字に注意。`/Project/:id/upload`（先頭 P が大文字）。
- 422 `folder_not_found`:
  - スクリプトは、アップロード直前に `/project/:id/folder` で一時フォルダを作って `_id` を取得し、そこへアップロードするよう対策済み。
  - それでも発生する場合、ログ（raw BODY/抽出 ID）を確認。環境差や整合性の問題を切り分ける。
- 422 `duplicate_file_name`:
  - 同名の doc/file が既に存在。`entities` で一覧確認し、名称の衝突を解消。
- レートリミット:
  - `SLEEP_SEC` を 0.2〜1.0s などに設定して再実行。

---

## 15. 参考：サーバ側ログの確認（デバッグ）

`ProjectEntityUpdateHandler.upsertFile` 前後にログを追加済み（`projectId`/`folderId`/例外内容）。  
反映には Web コンテナの再ビルド/再起動が必要です（手元の CE 構成に合わせて実施）。

- ログ確認（Toolkit）:
  ```bash
  cd /Users/takumi0616/Develop/overleaf-local/overleaf-toolkit
  bin/logs
  ```
- 必要に応じて `bin/stop` → `bin/start` で再起動。

---

## 16. Project_id の取得（一覧とスクリプト）

プロジェクト ID（\_id）を /user/projects から取得するスクリプトを追加済みです。

- スクリプト: `scripts/overleaf-list-projects.sh`
- 概要: CookieJar が有効ならそのまま一覧。未ログインでも EMAIL/PASSWORD を渡せば CSRF → /login → /user/projects の順で取得します。
- 出力: 既定で「PROJECT_ID[TAB]NAME」。jq があれば整形、なければ node があれば整形、どちらも無ければ JSON のまま。

### 16.1 クイックスタート（典型シナリオ）

```bash
bash -lc 'set -a; source scripts/overleaf-sync.env; set +a; BASE_URL="${BASE_URL:-http://localhost}"; ech
o "[INFO] BASE_URL=$BASE_URL"; curl -sS -H "Cookie: ${COOKIE}" "$BASE_URL/user/projects" | (if command -v jq >/dev/null 2>&1; then jq -r ".proje
cts[] | [._id, .name] | @tsv"; else cat; fi)'
```

- 既に有効な CookieJar を使って一覧

```bash
bash scripts/overleaf-list-projects.sh -b http://localhost -c /tmp/overleaf_ce.cookie
```

- メール/パスワードでログインして一覧

```bash
bash scripts/overleaf-list-projects.sh -b http://localhost -u "s233319@stn.nagaokaut.ac.jp" -p "eIwq4aHDT6"
```

- 整形出力（jq 推奨）

```bash
bash scripts/overleaf-list-projects.sh | column -t
```

- 生 JSON を強制出力

```bash
bash scripts/overleaf-list-projects.sh --raw
```

取得した PROJECT_ID は、`scripts/overleaf-sync.sh` の `entities` や `update` にそのまま渡せます。

### 16.2 オプション

- `-b, --base-url`: ベース URL（既定: `http://localhost`）
- `-c, --cookie-jar`: CookieJar のパス（既定: `/tmp/overleaf_ce.cookie`）
- `-u, --email`: ログイン用メール（Cookie が無効で、ログインしたい場合に指定）
- `-p, --password`: ログイン用パスワード
- `--raw`: 整形せず JSON のまま出力

環境変数でも指定可能: `BASE_URL`, `COOKIE_JAR`, `EMAIL`, `PASSWORD`。

### 16.3 出力例

```
PROJECT_ID	NAME
68b0022621c3446b6bee0f6c	SITA2025
68aef7cc21c3446b6bedfa43	MyTestProject
```

### 16.4 jq/node が無い場合

jq も node も無い環境では raw JSON を出力します。できれば jq の導入を推奨します。

### 16.5 curl の最小手順（スクリプトを使わない場合）

1. Cookie 初期化 + CSRF 取得

```bash
curl -sS -c /tmp/ol.cookie -b /tmp/ol.cookie http://localhost/login -o /dev/null
CSRF=$(curl -sS -c /tmp/ol.cookie -b /tmp/ol.cookie http://localhost/dev/csrf)
```

2. ログイン

```bash
curl -sS -i -c /tmp/ol.cookie -b /tmp/ol.cookie \
  -H "Content-Type: application/x-www-form-urlencoded" \
  --data-urlencode "_csrf=$CSRF" \
  --data-urlencode "email=you@example.com" \
  --data-urlencode "password=your_password" \
  http://localhost/login >/dev/null
```

3. 一覧取得

```bash
curl -sS -c /tmp/ol.cookie -b /tmp/ol.cookie http://localhost/user/projects \
  | jq -r '.projects[] | [.["_id"], .name] | @tsv'
```

### 16.6 トラブルシュート

- 非 JSON の応答（HTML 等）が返る:
  - 未ログインの可能性。`-u/-p` を渡すか、CookieJar を正しく指定してください。
- `Login failed: HTTP ...`:
  - メール/パスワードの確認。`/dev/csrf` が取得できているかも確認してください。
- 200 だが `projects` が空:
  - ログインユーザーで可視なプロジェクトが無い可能性があります。対象ユーザーで作成済みか確認してください。
