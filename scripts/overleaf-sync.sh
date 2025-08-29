#!/usr/bin/env bash
# overleaf-sync.sh
#
# ローカルのディレクトリを Overleaf CE に「ワンコマンド」で反映するための統合スクリプト
# - new:     ローカルDIR → zip → /project/new/upload で新規プロジェクト作成
# - update:  既存プロジェクトへファイル単位でPOST（相対パスを維持）
# - csrf:    CSRFトークン取得確認
# - status:  接続確認（Cookie/CSRF）
#
# 事前準備:
#   1) ブラウザで Overleaf CE (例: http://localhost) にログイン
#   2) DevTools等で Cookie を取得し、scripts/overleaf-sync.env に記述
#        BASE_URL=http://localhost
#        COOKIE='overleaf.sid=...; _csrf=...'
#   3) 実行権限を付与
#        chmod +x scripts/overleaf-sync.sh
#
# 使い方:
#   新規作成:
#     scripts/overleaf-sync.sh new /path/to/dir
#
#   既存プロジェクトを更新:
#     scripts/overleaf-sync.sh update <PROJECT_ID> /path/to/dir
#
#   接続確認:
#     scripts/overleaf-sync.sh status
#
# 注意:
#   - update は Overleaf 側の不要ファイルを削除しません（差分削除は仕様上非対応）。
#   - 完全上書きしたい場合は new で別プロジェクトとして作成するか、Overleaf 側で不要物を削除してください。
#   - 大量ファイルアップロードは Rate Limit にかかる可能性があるため、SLEEP_SEC を設定して間隔を空けられます。
#
set -euo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ROOT_DIR="$(cd "$THIS_DIR/.." >/dev/null 2>&1 && pwd)"

# 既定の設定ファイル（カスタマイズ用）
ENV_FILE_DEFAULT="$THIS_DIR/overleaf-sync.env"
if [[ -n "${ENV_FILE:-}" && -f "${ENV_FILE}" ]]; then
  # 明示指定があればそれを使う
  # shellcheck source=/dev/null
  source "${ENV_FILE}"
elif [[ -f "${ENV_FILE_DEFAULT}" ]]; then
  # 既定のenv
  # shellcheck source=/dev/null
  source "${ENV_FILE_DEFAULT}"
fi


# 既定値
BASE_URL="${BASE_URL:-http://localhost}"
SLEEP_SEC="${SLEEP_SEC:-0}"
ROOT_FOLDER_ID="${ROOT_FOLDER_ID:-root-folder-id}"

# --- 共通ユーティリティ ---
log()  { printf '[%s] %s\n' "$(date +'%H:%M:%S')" "$*" ; }
die()  { printf '[%s] ERROR: %s\n' "$(date +'%H:%M:%S')" "$*" >&2; exit 1; }
# stderr へ出すログ（コマンド置換の結果に混入させないため）
log_err() { printf '[%s] %s\n' "$(date +'%H:%M:%S')" "$*" >&2; }

usage() {
  cat <<'USAGE'
Usage:
  overleaf-sync.sh new <DIR>
  overleaf-sync.sh update <PROJECT_ID> <DIR>
  overleaf-sync.sh entities <PROJECT_ID>
  overleaf-sync.sh status
  overleaf-sync.sh csrf
  overleaf-sync.sh help

Env (scripts/overleaf-sync.env 推奨):
  BASE_URL=http://localhost
  COOKIE='overleaf.sid=...; _csrf=...'
  SLEEP_SEC=0
  ROOT_FOLDER_ID=root-folder-id
USAGE
}

# COOKIE から属性（Path/Expires/SameSite/Domain など）を除去し "name=value; name2=value2" 形式に整形
sanitize_cookie() {
  local raw="${1:-}"
  local out=()
  IFS=';' read -ra parts <<< "$raw"
  for part in "${parts[@]}"; do
    part="$(echo "$part" | xargs)"
    [[ -z "$part" ]] && continue
    case "$part" in
      [Pp]ath=*|[Ee]xpires=*|[Mm]ax-[Aa]ge=*|[Ss]ame[Ss]ite=*|[Dd]omain=* ) continue ;;
      [Ss]ecure|[Hh]ttp[Oo]nly ) continue ;;
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

mime_of() {
  local f="$1"
  if command -v file >/dev/null 2>&1; then
    file --mime-type -b "$f"
  else
    echo application/octet-stream
  fi
}

require_cookie() {
  [[ -n "${COOKIE:-}" ]] || die "COOKIE が未設定です。scripts/overleaf-sync.env に COOKIE を設定してください"
  SANITIZED_COOKIE="$(sanitize_cookie "${COOKIE}")"
  [[ -n "${SANITIZED_COOKIE}" ]] || die "COOKIE の整形結果が空です。'name=value; name2=value2' 形式で指定してください"
  # サーバが Set-Cookie でセッションをローテーションする場合に備え、クッキージャーを使って追随する
  COOKIE_JAR="${COOKIE_JAR:-$(mktemp -t olcookie.XXXXXX)}"
}

# helper to parse csrf token from HTML meta tag on pages like /login or /project
get_csrf_from_html() {
  local endpoint="${1:-/login}"
  local html
  html="$(curl -sS -L -b "${COOKIE_JAR}" -c "${COOKIE_JAR}" "${BASE_URL%/}${endpoint}")" || return 1
  # extract <meta name="ol-csrfToken" content="...">
  printf "%s" "${html}" | sed -n 's/.*<meta name="ol-csrfToken" content="\([^"]*\)".*/\1/p' | head -n 1
}

get_csrf() {
  # 1) try /dev/csrf and ensure 200 OK
  local resp status body
  resp="$(curl -sS -b "${SANITIZED_COOKIE}" -c "${COOKIE_JAR}" -w '\nHTTP_STATUS:%{http_code}' "${BASE_URL%/}/dev/csrf")" || return 1
  status="${resp##*HTTP_STATUS:}"
  body="${resp%HTTP_STATUS:*}"
  if [[ "${status}" == "200" && -n "${body}" && "${body}" != *"Redirecting to /login"* ]]; then
    printf "%s" "${body}"
    return 0
  fi
  # 2) fallback: parse from HTML meta on /login, then /project
  local token
  token="$(get_csrf_from_html "/login")" || true
  if [[ -n "${token}" ]]; then
    printf "%s" "${token}"
    return 0
  fi
  token="$(get_csrf_from_html "/project")" || true
  if [[ -n "${token}" ]]; then
    printf "%s" "${token}"
    return 0
  fi
  return 1
}

require_csrf() {
  require_cookie
  CSRF_TOKEN="$(get_csrf)" || die "CSRF取得に失敗 (/dev/csrf or /login)。ブラウザで ${BASE_URL%/} にログインし、scripts/overleaf-sync.env の COOKIE を更新してください"
  [[ -n "${CSRF_TOKEN}" ]] || die "CSRFトークンが空です。ログイン状態や COOKIE を確認してください"
}

status() {
  log "BASE_URL=${BASE_URL}"
  require_cookie
  log "Cookie OK（sanitized済み）"

  # ログイン状態を簡易チェック（/project はログイン必須）
  local http_dashboard
  http_dashboard="$(curl -s -o /dev/null -w '%{http_code}' -b "${SANITIZED_COOKIE}" "${BASE_URL%/}/project" || true)"
  if [[ "${http_dashboard}" == "200" ]]; then
    log "Login OK (/project HTTP 200)"
  else
    log "Login NG (/project HTTP ${http_dashboard}) → ブラウザで ${BASE_URL%/} にログインし、scripts/overleaf-sync.env の COOKIE を更新してください"
  fi

  CSRF="$(get_csrf || true)"
  if [[ -n "${CSRF}" ]]; then
    log "CSRF OK: $(printf '%s' "${CSRF}" | cut -c1-8)..."
    exit 0
  else
    die "CSRF 取得に失敗しました（/dev/csrf or /login）。ブラウザで ${BASE_URL%/} にログインし、scripts/overleaf-sync.env の COOKIE を更新してください"
  fi
}

cmd_csrf() {
  require_cookie
  CSRF="$(get_csrf)" || die "CSRF取得に失敗"
  echo "${CSRF}"
}

entities() {
  local project_id="${1:-}"
  [[ -n "${project_id}" ]] || die "PROJECT_ID を指定してください"
  require_cookie
  log "GET ${BASE_URL%/}/project/${project_id}/entities"
  local RESP
  RESP="$(curl -sS -i \
    -b "${SANITIZED_COOKIE}" -c "${COOKIE_JAR}" \
    -H "Accept: application/json" \
    "${BASE_URL%/}/project/${project_id}/entities")" || {
      die "curl 失敗（entities）"
    }
  echo "${RESP}"
}

# rootFolder._id を解決する（公開APIのみで取得）
# 手順:
#  - ルート直下に一時フォルダを作成（parent_folder_id 未指定）
#  - レスポンスの parentFolder_id が rootFolder._id
#  - 直後に作成フォルダは削除（ベストエフォート）
resolve_root_folder_id() {
  local project_id="${1:-}"
  [[ -n "${project_id}" ]] || die "resolve_root_folder_id: PROJECT_ID が未指定です"
  require_csrf

  local FOLDER_NAME="__ol_probe_$(date +%s)"
  local RESP BODY NEW_ID PARENT_ID

  RESP="$(curl -sS -L \
    -b "${COOKIE_JAR}" -c "${COOKIE_JAR}" \
    -H "Accept: application/json" \
    -H "Origin: ${BASE_URL%/}" \
    -H "Referer: ${BASE_URL%/}/Project/${project_id}" \
    -H "X-Requested-With: XMLHttpRequest" \
    -H "x-csrf-token: ${CSRF_TOKEN}" \
    -H "Content-Type: application/x-www-form-urlencoded; charset=utf-8" \
    --data-urlencode "name=${FOLDER_NAME}" \
    --data-urlencode "_csrf=${CSRF_TOKEN}" \
    "${BASE_URL%/}/project/${project_id}/folder")" || return 1

  BODY="$(printf "%s" "${RESP}" | tr -d '\r' )"
  log_err "resolve_root_folder_id: raw BODY (first 400 bytes): $(printf "%s" "${BODY}" | head -c 400)"

  # BSD sed 対応のため -E を使い、\+ を避ける
  NEW_ID="$(printf "%s" "${BODY}" | sed -E 's/.*"_id"[[:space:]]*:[[:space:]]*"([^"]+)".*/\1/')" || true
  PARENT_ID="$(printf "%s" "${BODY}" | sed -E 's/.*"parentFolder_id"[[:space:]]*:[[:space:]]*"([^"]+)".*/\1/')" || true

  # フォールバック（grep -o → sed）
  if [[ -z "${PARENT_ID}" ]]; then
    PARENT_ID="$(printf "%s" "${BODY}" | grep -o '"parentFolder_id"[[:space:]]*:[[:space:]]*"[^"]*"' | head -n 1 | sed -E 's/.*"parentFolder_id"[[:space:]]*:[[:space:]]*"([^"]*)".*/\1/')" || true
  fi
  if [[ -z "${NEW_ID}" ]]; then
    NEW_ID="$(printf "%s" "${BODY}" | grep -o '"_id"[[:space:]]*:[[:space:]]*"[^"]*"' | head -n 1 | sed -E 's/.*"_id"[[:space:]]*:[[:space:]]*"([^"]*)".*/\1/')" || true
  fi

  # sanitize to Mongo ObjectId-like 24-hex (防御的に抽出)
  PARENT_ID="$(printf "%s" "${PARENT_ID}" | grep -Eo '[0-9a-f]{24}' | head -n 1)"
  NEW_ID="$(printf "%s" "${NEW_ID}" | grep -Eo '[0-9a-f]{24}' | head -n 1)"

  if [[ -z "${PARENT_ID}" ]]; then
    echo "resolve_root_folder_id: unexpected response (first 400 bytes):" >&2
    printf "%s" "${BODY}" | head -c 400 >&2
    echo >&2
    return 1
  fi

  # アップロード先として使うフォルダIDを決定（新規作成したフォルダがあればそれを使う）
  # 備考: NEW_ID は直下に作成された実フォルダの _id、PARENT_ID は rootFolder._id
  #       upsertFile() で folder_not_found を避けるため、実在するフォルダID（NEW_ID）を優先して使用する
  local TARGET_ID=""
  if [[ -n "${NEW_ID}" ]]; then
    TARGET_ID="${NEW_ID}"
  else
    TARGET_ID="${PARENT_ID}"
  fi
  log_err "resolve_root_folder_id: parsed NEW_ID=${NEW_ID:-} PARENT_ID=${PARENT_ID:-} -> TARGET_ID=${TARGET_ID}"

  printf "%s" "${TARGET_ID}"
}

new_project() {
  local dir="${1:-}"
  [[ -n "${dir}" && -d "${dir}" ]] || die "ディレクトリが見つかりません: ${dir}"
  require_csrf

  local TMP_ZIP NAME RESP
  TMP_ZIP="$(mktemp -t olzip.XXXXXX).zip"
  NAME="$(basename "$dir")"

  log "Zipping '${dir}' -> ${TMP_ZIP}"
  (cd "$dir" && zip -qry "$TMP_ZIP" .)

  log "POST ${BASE_URL%/}/project/new/upload"
  RESP="$(curl -sS -i \
    -b "${COOKIE_JAR}" -c "${COOKIE_JAR}" \
    -H "Accept: application/json" \
    -H "Origin: ${BASE_URL%/}" \
    -H "Referer: ${BASE_URL%/}/project" \
    -H "X-Requested-With: XMLHttpRequest" \
    -H "x-csrf-token: ${CSRF_TOKEN}" \
    -F "_csrf=${CSRF_TOKEN}" \
    -F "name=${NAME}" \
    -F "qqfile=@${TMP_ZIP};type=application/zip" \
    "${BASE_URL%/}/project/new/upload")" || {
      rm -f "$TMP_ZIP"
      die "curl 失敗（新規アップロード）"
    }

  rm -f "$TMP_ZIP"
  echo "${RESP}"
}

update_project() {
  local project_id="${1:-}"
  local dir="${2:-}"
  [[ -n "${project_id}" ]] || die "PROJECT_ID を指定してください"
  [[ -n "${dir}" && -d "${dir}" ]] || die "ディレクトリが見つかりません: ${dir}"
  require_csrf

  log "Uploading tree from '${dir}' to project ${project_id}"
  local rel abs base mime RESP i count upload_url
  local FILES=()
  while IFS= read -r -d '' rel; do
    rel="${rel#./}"
    FILES+=("$rel")
  done < <(cd "$dir" && find . -type f -print0)
  count="${#FILES[@]}"
  i=0

  # folder_id を解決（明示が無ければ rootFolder._id を取得して付与）
  local effective_folder_id="${ROOT_FOLDER_ID:-}"
  if [[ -z "${effective_folder_id}" || "${effective_folder_id}" == "root-folder-id" ]]; then
    log "Resolving root folder id for project ${project_id}..."
    effective_folder_id="$(resolve_root_folder_id "${project_id}")" || die "root folder id の解決に失敗しました"
    log "root folder id=${effective_folder_id}"
  fi
  upload_url="${BASE_URL%/}/Project/${project_id}/upload?folder_id=${effective_folder_id}"

  for rel in "${FILES[@]}"; do
    i=$((i+1))
    abs="${dir%/}/${rel}"
    base="$(basename "$rel")"
    mime="$(mime_of "$abs")"

    log "[${i}/${count}] POST ${upload_url} rel=${rel}"
    RESP="$(curl -sS -i \
      -b "${COOKIE_JAR}" -c "${COOKIE_JAR}" \
      -H "Accept: application/json" \
      -H "Origin: ${BASE_URL%/}" \
      -H "Referer: ${BASE_URL%/}/Project/${project_id}" \
      -H "X-Requested-With: XMLHttpRequest" \
      -H "x-csrf-token: ${CSRF_TOKEN}" \
      -F "_csrf=${CSRF_TOKEN}" \
      -F "name=${base}" \
      -F "relativePath=null" \
      -F "folder_id=${effective_folder_id}" \
      -F "qqfile=@${abs};type=${mime}" \
      "${upload_url}")" || {
        die "curl 失敗（ファイル: ${rel}）"
      }
    # 結果の1行だけログに残す（HTTPステータス等）
    echo "  > $(printf "%s" "${RESP}" | sed -n '1,1p')"
    # レスポンス本文も出力（末尾行を想定、JSONエラー詳細を確認）
    echo "  >> $(printf "%s" "${RESP}" | tail -n 1)"

    if [[ "${SLEEP_SEC}" != "0" ]]; then
      sleep "${SLEEP_SEC}"
    fi
  done

  log "Done."
}

MODE="${1:-}"
case "${MODE}" in
  new )
    shift || true
    new_project "${1:-}" ;;
  update )
    shift || true
    update_project "${1:-}" "${2:-}" ;;
  status )
    status ;;
  csrf )
    cmd_csrf ;;
  entities )
    shift || true
    entities "${1:-}" ;;
  help|--help|-h|"" )
    usage ;;
  * )
    usage; exit 1 ;;
esac
