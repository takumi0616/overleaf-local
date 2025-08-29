#!/usr/bin/env bash
# overleaf-zip-upload.sh
#
# 一方向同期（ローカル → Overleaf）を zip/post または個別ファイルPOSTで行うユーティリティ
# - 新規プロジェクト作成: /project/new/upload に zip を送信
# - 既存プロジェクト更新: /Project/:Project_id/upload にファイルごと送信（relativePath でディレクトリ再現）
#
# 要件:
#   - macOS / bash / curl / zip / find / xargs / file（mimeタイプ取得; なければデフォルトを使う）
#   - Overleaf CE にブラウザでログイン済みのセッション Cookie（connect.sid 等）
#
# 使い方:
#   1) ブラウザで Overleaf CE にログイン（http://localhost:3000 など）
#   2) セッション Cookie を取得（例: 開発環境ならブラウザの DevTools または cookie コマンド等）
#   3) 下記のように実行
#
#   新規（zip で丸ごとアップロード）:
#     BASE_URL=http://localhost:3000 \
#     COOKIE="connect.sid=xxxx; _csrf=yyyy" \
#     ./scripts/overleaf-zip-upload.sh new /path/to/project-dir
#
#   既存プロジェクトにツリ―を上書き（個別ファイル POST）:
#    BASE_URL=http://localhost
#    COOKIE="overleaf.sid=s%3AEUs0ZvdR6Tu8RCkdPXrGekM9AMzSlL1y.GSqQuP8FzcwBcDgiCvnpj8aLqbMAwwttiqmHn7%2B%2BsfQ; Path=/; Expires=Tue, 02 Sep 2025 07:04:19 GMT; HttpOnly; SameSite=Lax" 
#     ./scripts/overleaf-zip-upload.sh update <PROJECT_ID> /path/to/project-dir
#
#   注意:
#     - COOKIE は "name=value; name2=value2" のような 1 行で渡してください（curl の -b にそのまま渡します）
#     - update は既存ファイルと混在します（差分削除はしません）。完全上書きをしたい場合は、Overleaf 側で
#       事前にファイル削除するか「new」で別プロジェクトとして作成してください。
#     - レート制限あり（UploadsRouter: projectUpload / fileUpload）。大量アップロード時は待ち時間を挟むなど調整してください。
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  BASE_URL=<base url> COOKIE="<cookie string>" overleaf-zip-upload.sh new <DIR>
  BASE_URL=<base url> COOKIE="<cookie string>" overleaf-zip-upload.sh update <PROJECT_ID> <DIR>

Args:
  MODE:
    new                 Create a new project from the zipped directory
    update              Upload files recursively into an existing project
  PROJECT_ID           Overleaf project id (for update mode)
  DIR                  Local directory to upload

Env:
  BASE_URL             e.g., http://localhost
  COOKIE               e.g., "connect.sid=...; _csrf=..."
  SLEEP_SEC            optional sleep between file uploads (default: 0)
  ROOT_FOLDER_ID       upload target root id (default: "root-folder-id")
USAGE
  exit 1
}

BASE_URL="${BASE_URL:-}"
COOKIE="${COOKIE:-}"
SLEEP_SEC="${SLEEP_SEC:-0}"
ROOT_FOLDER_ID="${ROOT_FOLDER_ID:-root-folder-id}"

if [[ -z "${BASE_URL}" || -z "${COOKIE}" ]]; then
  echo "ERROR: BASE_URL and COOKIE must be set (env)" >&2
  usage
fi

MODE="${1:-}"
shift || true

if [[ "${MODE}" != "new" && "${MODE}" != "update" ]]; then
  echo "ERROR: MODE must be 'new' or 'update'" >&2
  usage
fi

# mime ヘルパー
mime_of() {
  local f="$1"
  if command -v file >/dev/null 2>&1; then
    file --mime-type -b "$f"
  else
    # なければデフォルト
    echo application/octet-stream
  fi
}

# COOKIE から余計な属性(Path, Expires, HttpOnly, Secure, SameSite など)を取り除き、
# name=value; name2=value2 のみに整形する
sanitize_cookie() {
  local raw="${1:-}"
  local out=()
  IFS=';' read -ra parts <<< "$raw"
  for part in "${parts[@]}"; do
    part="$(echo "$part" | xargs)"
    [[ -z "$part" ]] && continue
    case "$part" in
      [Pp]ath=*|[Ee]xpires=*|[Mm]ax-[Aa]ge=*|[Ss]ame[Ss]ite=*|[Dd]omain=* )
        continue
        ;;
      [Ss]ecure|[Hh]ttp[Oo]nly )
        continue
        ;;
    esac
    if [[ "$part" == *"="* ]]; then
      out+=("$part")
    fi
  done
  local joined=""
  local IFS2='; '
  printf -v joined "%s" "${out[*]}"
  echo "$joined"
}

# --- new: ディレクトリを zip 化して /project/new/upload に POST ---
# サニタイズ済み COOKIE を作成
SANITIZED_COOKIE="$(sanitize_cookie "${COOKIE}")"
if [[ -z "${SANITIZED_COOKIE}" ]]; then
  echo "ERROR: COOKIE sanitized empty. Provide cookie like 'connect.sid=...; _csrf=...'" >&2
  exit 1
fi

# --- CSRF トークンを取得 ---
echo "[csrf] GET ${BASE_URL%/}/dev/csrf"
CSRF_TOKEN="$(curl -sS -b "${SANITIZED_COOKIE}" "${BASE_URL%/}/dev/csrf")" || {
  echo "ERROR: failed to fetch CSRF token (GET /dev/csrf)" >&2
  exit 1
}
if [[ -z "${CSRF_TOKEN}" ]]; then
  echo "ERROR: empty CSRF token. Are you logged in on ${BASE_URL%/}?" >&2
  exit 1
fi
echo "[csrf] token fetched"

if [[ "${MODE}" == "new" ]]; then
  DIR="${1:-}"
  if [[ -z "${DIR}" || ! -d "${DIR}" ]]; then
    echo "ERROR: DIR not found: ${DIR}" >&2
    usage
  fi

  TMP_ZIP="$(mktemp -t olzip.XXXXXX).zip"
  NAME="$(basename "$DIR")"

  echo "[new] Zipping '${DIR}' -> ${TMP_ZIP}"
  # -q 静か -r 再帰 -y シンボリックはシンボリックのまま
  (cd "$DIR" && zip -qry "$TMP_ZIP" .)

  echo "[new] POST ${BASE_URL}/project/new/upload"
  # multer はフィールド名 qqfile を要求
  # uploadProject は body.name の拡張子 .zip を落としてプロジェクト名を決めるため名前は任意
  RESP="$(curl -sS -i \
    -b "${SANITIZED_COOKIE}" \
    -H "Accept: application/json" \
    -H "Origin: ${BASE_URL%/}" \
    -H "Referer: ${BASE_URL%/}/project" \
    -H "X-Requested-With: XMLHttpRequest" \
    -H "X-CSRF-Token: ${CSRF_TOKEN}" \
    -F "name=${NAME}" \
    -F "qqfile=@${TMP_ZIP};type=application/zip" \
    "${BASE_URL%/}/project/new/upload")" || {
      echo "ERROR: curl failed" >&2
      rm -f "$TMP_ZIP"
      exit 1
    }

  echo "[new] Response: ${RESP}"
  rm -f "$TMP_ZIP"
  exit 0
fi

# --- update: ディレクトリ以下の全ファイルを /Project/:Project_id/upload に順次 POST ---
if [[ "${MODE}" == "update" ]]; then
  PROJECT_ID="${1:-}"
  DIR="${2:-}"
  if [[ -z "${PROJECT_ID}" || -z "${DIR}" || ! -d "${DIR}" ]]; then
    echo "ERROR: update requires PROJECT_ID and DIR (dir must exist)" >&2
    usage
  fi

  echo "[update] Uploading tree from '${DIR}' to project ${PROJECT_ID}"
  # find でファイル一覧を取得し、1件ずつ POST
  # Note: relativePath はプロジェクト内でのネスト再現に使われる
  IFS=$'\n'
  FILES=( $(cd "$DIR" && find . -type f -print | sed 's#^\./##') )
  unset IFS
  COUNT="${#FILES[@]}"
  i=0

  for rel in "${FILES[@]}"; do
    i=$((i+1))
    abs="${DIR%/}/${rel}"
    base="$(basename "$rel")"
    mime="$(mime_of "$abs")"

    echo "[${i}/${COUNT}] POST ${BASE_URL}/Project/${PROJECT_ID}/upload?folder_id=${ROOT_FOLDER_ID} rel=${rel}"
    RESP="$(curl -sS -i \
      -b "${SANITIZED_COOKIE}" \
      -H "Accept: application/json" \
      -H "Origin: ${BASE_URL%/}" \
      -H "Referer: ${BASE_URL%/}/project/${PROJECT_ID}" \
      -H "X-Requested-With: XMLHttpRequest" \
      -H "X-CSRF-Token: ${CSRF_TOKEN}" \
      -F "name=${base}" \
      -F "relativePath=${rel}" \
      -F "qqfile=@${abs};type=${mime}" \
      "${BASE_URL%/}/Project/${PROJECT_ID}/upload?folder_id=${ROOT_FOLDER_ID}")" || {
        echo "ERROR: curl failed (file: ${rel})" >&2
        exit 1
      }

    echo "  > ${RESP}"
    # 適宜スリープ（レートリミット・負荷対策）
    if [[ "${SLEEP_SEC}" != "0" ]]; then
      sleep "${SLEEP_SEC}"
    fi
  done

  echo "[update] Done."
  exit 0
fi
