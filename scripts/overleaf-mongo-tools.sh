#!/usr/bin/env bash
# Overleaf MongoDB helper: shell/eval/dump/restore from Overleaf Toolkit stack
# Usage:
#   scripts/overleaf-mongo-tools.sh shell [-d DB]
#   scripts/overleaf-mongo-tools.sh eval  [-d DB] -f file.js
#   scripts/overleaf-mongo-tools.sh eval  [-d DB] -- 'js code here'
#   scripts/overleaf-mongo-tools.sh dump  [-d DB] [-o backups/file.archive]
#   scripts/overleaf-mongo-tools.sh restore [-i backups/file.archive] [--force-drop]
#   scripts/overleaf-mongo-tools.sh stats [-d DB]
#   scripts/overleaf-mongo-tools.sh list-collections [-d DB]
#
# Notes:
# - Requires Overleaf Toolkit running (overleaf-toolkit/bin/docker-compose up/start).
# - Uses docker-compose exec into the 'mongo' service.
# - Default DB is 'sharelatex'.

set -euo pipefail

# ----- realpath fallback -----
command -v realpath >/dev/null 2>&1 || realpath() {
  [[ $1 = /* ]] && echo "$1" || echo "$PWD/${1#./}"
}

SCRIPT_PATH="$(realpath "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"
PROJECT_ROOT="$(realpath "$SCRIPT_DIR/..")"
DOCKER_COMPOSE="$PROJECT_ROOT/overleaf-toolkit/bin/docker-compose"

default_db="sharelatex"

usage() {
  sed -n '2,35p' "$SCRIPT_PATH" | sed 's/^# \{0,1\}//'
  cat <<'EOF'

Examples:
  # Interactive shell
  scripts/overleaf-mongo-tools.sh shell
  scripts/overleaf-mongo-tools.sh shell -d sharelatex

  # Evaluate inline JS
  scripts/overleaf-mongo-tools.sh eval -- 'db.runCommand({ ping: 1 })'
  scripts/overleaf-mongo-tools.sh eval -d sharelatex -- 'db.projects.countDocuments()'

  # Evaluate from file
  scripts/overleaf-mongo-tools.sh eval -f ./queries/list-projects.js

  # Dump/Restore (archive format)
  scripts/overleaf-mongo-tools.sh dump -o backups/sharelatex-$(date +%Y%m%d-%H%M%S).archive
  scripts/overleaf-mongo-tools.sh restore -i backups/sharelatex-20250828-010000.archive --force-drop

  # Quick stats / list collections
  scripts/overleaf-mongo-tools.sh stats
  scripts/overleaf-mongo-tools.sh list-collections
EOF
  exit 1
}

require_toolkit() {
  if [[ ! -x "$DOCKER_COMPOSE" ]]; then
    echo "ERROR: '$DOCKER_COMPOSE' not found or not executable."
    echo "Ensure you run this script from the Overleaf repo root and the Toolkit is present."
    exit 1
  fi
}

check_stack() {
  # Non-fatal: warn if mongo is not up
  if ! "$DOCKER_COMPOSE" ps | grep -qE '^mongo.*\bUp\b'; then
    echo "WARNING: mongo service is not reported as Up. Commands may fail. Start the stack:"
    echo "  overleaf-toolkit/bin/start   # or: overleaf-toolkit/bin/up"
  fi
}

shell_cmd() {
  local db="$default_db"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -d|--db) db="$2"; shift 2;;
      -h|--help) usage;;
      *) echo "Unknown arg: $1"; usage;;
    esac
  done

  exec "$DOCKER_COMPOSE" exec mongo mongosh "$db"
}

eval_cmd() {
  local db="$default_db"
  local from_file=""
  local inline_js=""

  while [[ $# -gt 0 ]]; do
    case "$1" in
      -d|--db) db="$2"; shift 2;;
      -f|--file) from_file="$2"; shift 2;;
      --) shift; inline_js="$*"; break;;
      -h|--help) usage;;
      *) # allow positional to be treated as inline after --
         echo "Unknown arg: $1"; usage;;
    esac
  done

  if [[ -n "$from_file" && -n "$inline_js" ]]; then
    echo "ERROR: specify either -f FILE or inline JS with --, not both"; exit 1
  fi
  if [[ -z "$from_file" && -z "$inline_js" ]]; then
    echo "ERROR: provide -f FILE or inline JS with --"; exit 1
  fi

  if [[ -n "$from_file" ]]; then
    if [[ ! -f "$from_file" ]]; then
      echo "ERROR: JS file not found: $from_file"; exit 1
    fi
    # Use mongosh --file via a shell to support -T
    # shellcheck disable=SC2016
    "$DOCKER_COMPOSE" exec -T mongo sh -lc 'mongosh '"$db"' --quiet --file /dev/stdin' < "$from_file"
  else
    "$DOCKER_COMPOSE" exec -T mongo mongosh "$db" --quiet --eval "$inline_js"
  fi
}

dump_cmd() {
  local db="$default_db"
  local out_file=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -d|--db) db="$2"; shift 2;;
      -o|--out) out_file="$2"; shift 2;;
      -h|--help) usage;;
      *) echo "Unknown arg: $1"; usage;;
    esac
  done
  if [[ -z "$out_file" ]]; then
    mkdir -p "$PROJECT_ROOT/backups"
    out_file="$PROJECT_ROOT/backups/${db}-$(date +%Y%m%d-%H%M%S).archive"
  else
    mkdir -p "$(dirname "$out_file")"
  fi

  echo "Dumping db '$db' to $out_file ..."
  # Use archive over stdout into local file
  "$DOCKER_COMPOSE" exec -T mongo mongodump --db "$db" --archive > "$out_file"
  echo "Done."
}

restore_cmd() {
  local in_file=""
  local force_drop="false"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -i|--in) in_file="$2"; shift 2;;
      --force-drop) force_drop="true"; shift 1;;
      -h|--help) usage;;
      *) echo "Unknown arg: $1"; usage;;
    esac
  done
  if [[ -z "$in_file" ]]; then
    echo "ERROR: input archive is required: -i backups/xxx.archive"; exit 1
  fi
  if [[ ! -f "$in_file" ]]; then
    echo "ERROR: archive not found: $in_file"; exit 1
  fi

  local drop_flag=""
  if [[ "$force_drop" == "true" ]]; then
    drop_flag="--drop"
  else
    echo "INFO: restore will not use --drop unless you pass --force-drop"
  fi

  echo "Restoring from $in_file ..."
  # Stream archive into mongorestore
  "$DOCKER_COMPOSE" exec -T mongo mongorestore $drop_flag --archive < "$in_file"
  echo "Done."
}

stats_cmd() {
  local db="$default_db"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -d|--db) db="$2"; shift 2;;
      -h|--help) usage;;
      *) echo "Unknown arg: $1"; usage;;
    esac
  done
  "$DOCKER_COMPOSE" exec -T mongo mongosh "$db" --quiet --eval \
    'const collInfos=db.getCollectionInfos(); const collections=collInfos.map(ci=>ci.name); function count(name){ try { const c=db.getCollection(name); return (typeof c.countDocuments==="function") ? c.countDocuments() : c.count(); } catch(e){ return "N/A"; } } printjson({ db: db.getName(), collections, counts: { projects: count("projects"), users: count("users") } });'
}

list_collections_cmd() {
  local db="$default_db"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -d|--db) db="$2"; shift 2;;
      -h|--help) usage;;
      *) echo "Unknown arg: $1"; usage;;
    esac
  done
  "$DOCKER_COMPOSE" exec -T mongo mongosh "$db" --quiet --eval 'db.getCollectionInfos().map(ci=>ci.name)'
}

main() {
  require_toolkit
  check_stack

  local cmd="${1:-}"
  shift || true

  case "$cmd" in
    shell)            shell_cmd "$@";;
    eval)             eval_cmd "$@";;
    dump)             dump_cmd "$@";;
    restore)          restore_cmd "$@";;
    stats)            stats_cmd "$@";;
    list-collections) list_collections_cmd "$@";;
    ""|-h|--help)     usage;;
    *) echo "Unknown subcommand: $cmd"; usage;;
  esac
}

main "$@"
