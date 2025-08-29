#!/usr/bin/env bash
# Mirror the latest (or specified) Overleaf CE "compiles" working copy to a local folder
# so local tools (e.g., Cline, other LLMs) can read project files from disk even when
# editing happens via Overleaf/VSCode extension (projects stored in MongoDB).
#
# USAGE:
#   scripts/overleaf-compiles-mirror.sh                      # mirror the most recently compiled project
#   scripts/overleaf-compiles-mirror.sh <projectId-userId> [dest-folder-name]
#
# EXAMPLES:
#   scripts/overleaf-compiles-mirror.sh
#   scripts/overleaf-compiles-mirror.sh 68aef7cc21c3-68aef75221c3 SITA2025_high
#
# NOTES:
# - Overleaf CE stores the canonical project in MongoDB. The "compiles" directory is a
#   transient working copy that reflects the latest compile input. This is read-only in spirit.
# - Running this script will rsync the chosen compiles directory into /project/<dest> so local
#   tools can read it. To push changes back to Overleaf, continue to edit via the Overleaf UI
#   or use the extension's sync, or re-upload a ZIP to Overleaf.
# - If you need a stable mapping, provide the <projectId-userId> explicitly and choose a
#   destination name; otherwise the script picks the most recently modified compiles folder.
#
# This version reads OVERLEAF_DATA_PATH from overleaf-toolkit/config/overleaf.rc (fallback to lib/default.rc)
# so it works whether your data path is "data/overleaf" or "data/sharelatex".

set -euo pipefail

ROOT="/Users/takumi0616/Develop/overleaf-local"
TOOLKIT_ROOT="${ROOT}/overleaf-toolkit"

# Resolve OVERLEAF_DATA_PATH from config
read_overleaf_data_path() {
  local val=""
  local line=""
  if [ -f "${TOOLKIT_ROOT}/config/overleaf.rc" ]; then
    line="$(grep -E '^OVERLEAF_DATA_PATH=' "${TOOLKIT_ROOT}/config/overleaf.rc" | head -n1 || true)"
  fi
  if [ -z "${line}" ] && [ -f "${TOOLKIT_ROOT}/lib/default.rc" ]; then
    line="$(grep -E '^OVERLEAF_DATA_PATH=' "${TOOLKIT_ROOT}/lib/default.rc" | head -n1 || true)"
  fi
  if [ -n "${line}" ]; then
    # take everything after the first '='
    val="${line#*=}"
    # strip optional surrounding quotes
    if [ "${val#\"}" != "${val}" ] && [ "${val%\"}" != "${val}" ]; then
      val="${val%\"}"; val="${val#\"}"
    elif [ "${val#\'}" != "${val}" ] && [ "${val%\'}" != "${val}" ]; then
      val="${val%\'}"; val="${val#\'}"
    fi
    # strip possible CR (Windows line ending)
    val="${val%$'\r'}"
  fi
  if [ -z "${val}" ]; then
    val="data/sharelatex"
  fi
  echo "${val}"
}

OVERLEAF_DATA_PATH="$(read_overleaf_data_path)"
BASE_DATA_DIR="${TOOLKIT_ROOT}/${OVERLEAF_DATA_PATH}"
# Resolve compiles dir (Overleaf CE v5+ stores under BASE/data/compiles)
COMPILES_DIR_CANDIDATE1="${BASE_DATA_DIR}/compiles"
COMPILES_DIR_CANDIDATE2="${BASE_DATA_DIR}/data/compiles"
if [ -d "${COMPILES_DIR_CANDIDATE1}" ]; then
  COMPILES_DIR="${COMPILES_DIR_CANDIDATE1}"
else
  COMPILES_DIR="${COMPILES_DIR_CANDIDATE2}"
fi
DEST_ROOT="${ROOT}/project"

if [ ! -d "${BASE_DATA_DIR}" ]; then
  echo "ERROR: base data dir not found: ${BASE_DATA_DIR}" >&2
  echo "Start Overleaf Toolkit (bin/up) first. Current OVERLEAF_DATA_PATH='${OVERLEAF_DATA_PATH}'" >&2
  exit 1
fi

if [ ! -d "${COMPILES_DIR}" ]; then
  echo "ERROR: compiles directory not found: ${COMPILES_DIR}" >&2
  echo "Start Overleaf Toolkit (bin/up) and compile a project at least once." >&2
  echo "Hint: You can check available dirs under: ${BASE_DATA_DIR}" >&2
  exit 1
fi

mkdir -p "${DEST_ROOT}"

ARG_COMPILES="${1:-}"
ARG_DESTNAME="${2:-}"

# Resolve source compiles directory
if [ -n "${ARG_COMPILES}" ]; then
  SRC="${COMPILES_DIR}/${ARG_COMPILES}"
  if [ ! -d "${SRC}" ]; then
    echo "ERROR: specified compiles dir not found: ${SRC}" >&2
    echo "Hint: Available compiles dirs:" >&2
    ls -1 "${COMPILES_DIR}" >&2 || true
    exit 1
  fi
else
  # macOS-friendly latest directory selection by mtime
  # (assumes compiles/ contains only project dirs)
  LATEST_BASENAME="$(ls -1t "${COMPILES_DIR}" | head -n 1 || true)"
  if [ -z "${LATEST_BASENAME}" ]; then
    echo "ERROR: no compiles directories found under ${COMPILES_DIR}" >&2
    echo "Compile a project once from Overleaf/extension, then re-run this script." >&2
    exit 1
  fi
  SRC="${COMPILES_DIR}/${LATEST_BASENAME}"
fi

# Destination folder name
if [ -n "${ARG_DESTNAME}" ]; then
  DEST="${DEST_ROOT}/${ARG_DESTNAME}"
else
  # Default to the compiles dir basename (projectId-userId)
  DEST="${DEST_ROOT}/$(basename "${SRC}")"
fi

echo "OVERLEAF_DATA_PATH   : ${OVERLEAF_DATA_PATH}"
echo "Source compiles dir  : ${SRC}"
echo "Destination mirror   : ${DEST}"
mkdir -p "${DEST}"

# Perform the mirror
# -a: archive mode, -v: verbose, --delete: make exact mirror (removes stale files)
rsync -av --delete "${SRC}/" "${DEST}/"

echo
echo "Mirror complete."
echo "Local path for Cline/LLMs: ${DEST}"
echo
echo "Tips:"
echo " - Edit within Overleaf/extension as usual, then re-run this script to refresh the local mirror."
echo " - The mirror is read-only in concept; it does not push changes back to Overleaf."
echo " - To identify a specific project, pass its compiles directory name (projectId-userId) explicitly."
echo " - To locate compiles folders quickly:"
echo "     ls -1 \"${COMPILES_DIR}\""
