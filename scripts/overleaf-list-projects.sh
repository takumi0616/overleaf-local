#!/usr/bin/env bash
# List Overleaf CE projects via /user/projects and print Project_id and name.
# - Logs in programmatically using CSRF if EMAIL/PASSWORD are provided.
# - If already logged in (valid cookie jar), login is skipped.
#
# Usage:
#   bash scripts/overleaf-list-projects.sh [-b BASE_URL] [-c COOKIE_JAR] [-u EMAIL] [-p PASSWORD] [--raw]
#
# Examples:
#   # 1) 既にログイン済み CookieJar を使って一覧（Cookie が有効な場合）
#   bash scripts/overleaf-list-projects.sh -b http://localhost -c /tmp/overleaf.cookie
#
#   # 2) メール/パスワードでログインして一覧（Cookie 不要）
#   bash scripts/overleaf-list-projects.sh -b http://localhost -u "you@example.com" -p "password"
#
#   # 3) 環境変数で指定（BASE_URL/COOKIE_JAR/EMAIL/PASSWORD）
#   BASE_URL=http://localhost EMAIL=you@example.com PASSWORD=pass bash scripts/overleaf-list-projects.sh
#
# 備考:
# - jq があれば整形出力、なければ node があれば整形、どちらも無ければ生 JSON を出力します。
# - Overleaf Toolkit 既定では BASE_URL は http://localhost（ポート80）です。

set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost}"
COOKIE_JAR="${COOKIE_JAR:-/tmp/overleaf_ce.cookie}"
EMAIL="${EMAIL:-}"
PASSWORD="${PASSWORD:-}"
FORCE_RAW=0

usage() {
  sed -n '2,50p' "$0" | sed -e 's/^# \{0,1\}//'
  echo
  echo "Options:"
  echo "  -b, --base-url URL       Base URL (default: ${BASE_URL})"
  echo "  -c, --cookie-jar PATH    Cookie jar file (default: ${COOKIE_JAR})"
  echo "  -u, --email EMAIL        Login email (optional if cookie already valid)"
  echo "  -p, --password PASS      Login password (optional if cookie already valid)"
  echo "      --raw                Always print raw JSON (skip pretty formatting)"
}

# ---- parse args ----
while [[ $# -gt 0 ]]; do
  case "$1" in
    -b|--base-url)
      BASE_URL="$2"; shift 2;;
    -c|--cookie-jar)
      COOKIE_JAR="$2"; shift 2;;
    -u|--email)
      EMAIL="$2"; shift 2;;
    -p|--password)
      PASSWORD="$2"; shift 2;;
    --raw)
      FORCE_RAW=1; shift 1;;
    -h|--help)
      usage; exit 0;;
    *)
      echo "Unknown argument: $1" >&2
      usage; exit 2;;
  esac
done

# ---- helpers ----
http_status() {
  # curl the URL and print only HTTP status code (follow no redirects)
  local method="$1"
  local url="$2"
  shift 2
  curl -sS -o /dev/null -w "%{http_code}" -X "$method" "$url" "$@"
}

fetch_csrf() {
  # ensure session + get CSRF token text from /dev/csrf
  curl -sS -c "$COOKIE_JAR" -b "$COOKIE_JAR" "$BASE_URL/login" -o /dev/null
  curl -sS -c "$COOKIE_JAR" -b "$COOKIE_JAR" "$BASE_URL/dev/csrf"
}

login_if_needed() {
  # If /user/projects is 200 with current cookie, skip login
  local status
  status=$(http_status GET "$BASE_URL/user/projects" -c "$COOKIE_JAR" -b "$COOKIE_JAR")
  if [[ "$status" == "200" ]]; then
    return 0
  fi

  if [[ -z "$EMAIL" || -z "$PASSWORD" ]]; then
    echo "[INFO] Cookie で未ログインまたは期限切れ。EMAIL/PASSWORD が未指定のためログイン省略。" >&2
    echo "[HINT] EMAIL/PASSWORD を渡すか、ブラウザログイン済み Cookie を COOKIE_JAR にエクスポートして再実行してください。" >&2
    return 0
  fi

  local csrf
  csrf="$(fetch_csrf)"

  # POST /login (form-urlencoded)
  # 成功時は 302 リダイレクトが返る想定（200 でも可とする）
  local login_code
  login_code=$(curl -sS -i -c "$COOKIE_JAR" -b "$COOKIE_JAR" \
    -H "Content-Type: application/x-www-form-urlencoded" \
    --data-urlencode "_csrf=${csrf}" \
    --data-urlencode "email=${EMAIL}" \
    --data-urlencode "password=${PASSWORD}" \
    "$BASE_URL/login" | awk 'BEGIN{code=0} /^HTTP\/.[.]./ {code=$2} END{print code}')

  if [[ "$login_code" != "302" && "$login_code" != "200" ]]; then
    echo "[ERROR] Login failed: HTTP $login_code" >&2
    exit 1
  fi
}

fetch_projects_json() {
  curl -sS -c "$COOKIE_JAR" -b "$COOKIE_JAR" "$BASE_URL/user/projects"
}

print_projects_table() {
  local json="$1"

  if [[ "$FORCE_RAW" -eq 1 ]]; then
    echo "$json"
    return 0
  fi

  if command -v jq >/dev/null 2>&1; then
    echo -e "PROJECT_ID\tNAME"
    echo "$json" | jq -r '.projects[] | [._id, .name] | @tsv'
    return 0
  fi

  if command -v node >/dev/null 2>&1; then
    node -e 'const fs=require("fs");
const d=JSON.parse(fs.readFileSync(0,"utf8"));
if(!d.projects){console.log(JSON.stringify(d));process.exit(0)}
console.log("PROJECT_ID\tNAME");
for(const p of d.projects){console.log(`${p._id}\t${p.name}`)}' <<< "$json"
    return 0
  fi

  echo "[WARN] jq/node が無いため raw JSON を出力します。" >&2
  echo "$json"
}

# ---- main ----
login_if_needed
JSON="$(fetch_projects_json)"

# 簡易チェック: 未ログイン時はHTML等（200以外や login ページ）を返す場合がある
if [[ "${JSON:0:1}" != "{" ]]; then
  echo "[ERROR] 非JSON応答。未ログインまたは権限不足の可能性。" >&2
  echo "[HINT] -u/--email と -p/--password でログインを実施してください。" >&2
  echo "[HINT] 例) bash scripts/overleaf-list-projects.sh -b http://localhost -u you@example.com -p 'password'" >&2
  exit 1
fi

print_projects_table "$JSON"
