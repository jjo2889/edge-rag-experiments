#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXCLUDE_FILE="${ROOT_DIR}/.git/info/exclude"

if [ ! -f "${EXCLUDE_FILE}" ]; then
  echo "[fail] ${EXCLUDE_FILE} not found. Run 'git init' first."
  exit 1
fi

add_pattern() {
  local pattern="$1"
  if ! grep -Fqx "${pattern}" "${EXCLUDE_FILE}"; then
    echo "${pattern}" >> "${EXCLUDE_FILE}"
    echo "[ok] added local exclude: ${pattern}"
  else
    echo "[skip] already exists: ${pattern}"
  fi
}

if [ "$#" -eq 0 ]; then
  echo "[usage] scripts/setup_local_exclude.sh <pattern1> [pattern2 ...]"
  echo "[hint] example: scripts/setup_local_exclude.sh '/.private/*' '/*.private.md'"
  exit 0
fi

for pattern in "$@"; do
  add_pattern "${pattern}"
done

echo "[done] local exclude configured"
