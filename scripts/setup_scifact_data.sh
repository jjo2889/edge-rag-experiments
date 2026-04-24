#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_URL="${DATASET_URL:-https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip}"
CORPUS_OUT="${ROOT_DIR}/data/beir/scifact_corpus/corpus.jsonl"
QUERIES_OUT="${ROOT_DIR}/data/beir/scifact/queries.jsonl"
QRELS_OUT="${ROOT_DIR}/data/beir/scifact/qrels/test.tsv"
FORCE=0

usage() {
  cat <<'USAGE'
Usage: scripts/setup_scifact_data.sh [--force]

Downloads BEIR scifact and prepares files into:
  data/beir/scifact_corpus/corpus.jsonl
  data/beir/scifact/queries.jsonl
  data/beir/scifact/qrels/test.tsv
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

if [ -f "${CORPUS_OUT}" ] && [ -f "${QUERIES_OUT}" ] && [ -f "${QRELS_OUT}" ] && [ "${FORCE}" -ne 1 ]; then
  echo "[skip] already exists:"
  echo "  ${CORPUS_OUT}"
  echo "  ${QUERIES_OUT}"
  echo "  ${QRELS_OUT}"
  echo "[hint] rerun with --force to refresh"
  exit 0
fi

TMP_DIR="$(mktemp -d)"
ARCHIVE_PATH="${TMP_DIR}/scifact.zip"

cleanup() {
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

if [ -f "${DATASET_URL}" ]; then
  echo "[copy] ${DATASET_URL}"
  cp "${DATASET_URL}" "${ARCHIVE_PATH}"
else
  echo "[download] ${DATASET_URL}"
  curl -fL "${DATASET_URL}" -o "${ARCHIVE_PATH}"
fi

python3 - "${ARCHIVE_PATH}" "${CORPUS_OUT}" "${QUERIES_OUT}" "${QRELS_OUT}" <<'PY'
import pathlib
import shutil
import sys
import zipfile

archive_path = sys.argv[1]
corpus_out = pathlib.Path(sys.argv[2])
queries_out = pathlib.Path(sys.argv[3])
qrels_out = pathlib.Path(sys.argv[4])


def pick_member(names, suffixes):
    matches = []
    for name in names:
        for suffix in suffixes:
            if name.endswith(suffix):
                matches.append(name)
                break
    if not matches:
        return None
    return sorted(matches, key=len)[0]

with zipfile.ZipFile(archive_path) as zf:
    names = zf.namelist()
    corpus_member = pick_member(names, ["/corpus.jsonl", "corpus.jsonl"])
    queries_member = pick_member(names, ["/queries.jsonl", "queries.jsonl"])
    qrels_member = pick_member(names, ["/qrels/test.tsv", "qrels/test.tsv", "/qrels.tsv", "qrels.tsv"])

    if corpus_member is None:
        raise SystemExit("[fail] corpus.jsonl not found in scifact archive")
    if queries_member is None:
        raise SystemExit("[fail] queries.jsonl not found in scifact archive")
    if qrels_member is None:
        raise SystemExit("[fail] qrels/test.tsv not found in scifact archive")

    for member, out_path in (
        (corpus_member, corpus_out),
        (queries_member, queries_out),
        (qrels_member, qrels_out),
    ):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(member) as src, out_path.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        print(f"[ok] extracted {member} -> {out_path}")
PY

echo "[done] SciFact data ready"
echo "  corpus:  data/beir/scifact_corpus/corpus.jsonl"
echo "  queries: data/beir/scifact/queries.jsonl"
echo "  qrels:   data/beir/scifact/qrels/test.tsv"
