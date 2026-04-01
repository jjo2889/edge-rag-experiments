#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_PATH="${ROOT_DIR}/data/raw/scifact_corpus.jsonl"
DATASET_URL="https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
FORCE=0

usage() {
  cat <<'USAGE'
Usage: scripts/setup_scifact_data.sh [--force]

Downloads BEIR scifact and extracts corpus.jsonl into:
  data/raw/scifact_corpus.jsonl
USAGE
}

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "[fail] missing command: $1"
    exit 1
  fi
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --force)
      FORCE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[fail] unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

need_cmd curl
need_cmd python3
need_cmd mktemp

if [ -f "${OUT_PATH}" ] && [ "${FORCE}" -ne 1 ]; then
  echo "[skip] already exists: ${OUT_PATH}"
  echo "[hint] rerun with --force to refresh"
  exit 0
fi

TMP_DIR="$(mktemp -d)"
ARCHIVE_PATH="${TMP_DIR}/scifact.zip"
mkdir -p "${ROOT_DIR}/data/raw"

cleanup() {
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

echo "[download] ${DATASET_URL}"
curl -fL "${DATASET_URL}" -o "${ARCHIVE_PATH}"

python3 - "${ARCHIVE_PATH}" "${OUT_PATH}" <<'PY'
import pathlib
import shutil
import sys
import zipfile

archive_path = sys.argv[1]
out_path = pathlib.Path(sys.argv[2])

with zipfile.ZipFile(archive_path) as zf:
    members = [m for m in zf.namelist() if m.endswith("/corpus.jsonl") or m == "corpus.jsonl"]
    if not members:
        raise SystemExit("[fail] corpus.jsonl not found in scifact archive")
    member = sorted(members, key=len)[0]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zf.open(member) as src, out_path.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    print(f"[ok] extracted {member} -> {out_path}")
PY

echo "[done] data/raw/scifact_corpus.jsonl ready"
