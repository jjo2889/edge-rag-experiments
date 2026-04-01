#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

EMBEDDING_REPO="${EMBEDDING_REPO:-sentence-transformers/all-MiniLM-L6-v2}"
# Pinned default for reproducible setup across fresh clones.
PINNED_EMBEDDING_REVISION="5641a7880f40ebf4035d05e60c5f9b7a9c272c84"
EMBEDDING_REVISION="${EMBEDDING_REVISION:-${PINNED_EMBEDDING_REVISION}}"
EMBEDDING_MODEL_FILE="${EMBEDDING_MODEL_FILE:-onnx/model.onnx}"
EMBEDDING_TOKENIZER_FILE="${EMBEDDING_TOKENIZER_FILE:-tokenizer.json}"

DEST_DIR="${ROOT_DIR}/models/all-MiniLM-L6-v2-onnx"
MODEL_OUT="${DEST_DIR}/model.onnx"
TOKENIZER_OUT="${DEST_DIR}/tokenizer.json"

MODEL_SHA256="${MODEL_SHA256:-}"
TOKENIZER_SHA256="${TOKENIZER_SHA256:-}"
FORCE=0

usage() {
  cat <<'USAGE'
Usage: scripts/setup_embedding_model.sh [--force]

Downloads ONNX embedding model + tokenizer.json from the same Hugging Face repo/revision.
Default source:
  repo: sentence-transformers/all-MiniLM-L6-v2
  revision: 5641a7880f40ebf4035d05e60c5f9b7a9c272c84
  model file: onnx/model.onnx
  tokenizer file: tokenizer.json

Output:
  models/all-MiniLM-L6-v2-onnx/model.onnx
  models/all-MiniLM-L6-v2-onnx/tokenizer.json

Optional env:
  EMBEDDING_REPO
  EMBEDDING_REVISION
  EMBEDDING_MODEL_FILE
  EMBEDDING_TOKENIZER_FILE
  MODEL_SHA256
  TOKENIZER_SHA256
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
need_cmd mktemp

MODEL_URL="https://huggingface.co/${EMBEDDING_REPO}/resolve/${EMBEDDING_REVISION}/${EMBEDDING_MODEL_FILE}"
TOKENIZER_URL="https://huggingface.co/${EMBEDDING_REPO}/resolve/${EMBEDDING_REVISION}/${EMBEDDING_TOKENIZER_FILE}"

echo "[setup] embedding repo=${EMBEDDING_REPO}"
echo "[setup] embedding revision=${EMBEDDING_REVISION}"
if [ "${EMBEDDING_REVISION}" = "${PINNED_EMBEDDING_REVISION}" ]; then
  echo "[setup] embedding revision mode=pinned-default"
else
  echo "[setup] embedding revision mode=override"
fi
echo "[setup] model url=${MODEL_URL}"
echo "[setup] tokenizer url=${TOKENIZER_URL}"

if [ "${FORCE}" -ne 1 ] && [ -f "${MODEL_OUT}" ] && [ -f "${TOKENIZER_OUT}" ]; then
  echo "[skip] model and tokenizer already exist under ${DEST_DIR}"
  echo "[hint] rerun with --force to refresh"
  exit 0
fi

TMP_DIR="$(mktemp -d)"
TMP_MODEL="${TMP_DIR}/model.onnx"
TMP_TOKENIZER="${TMP_DIR}/tokenizer.json"

cleanup() {
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

mkdir -p "${DEST_DIR}"

echo "[download] model"
curl -fL "${MODEL_URL}" -o "${TMP_MODEL}"

echo "[download] tokenizer"
curl -fL "${TOKENIZER_URL}" -o "${TMP_TOKENIZER}"

if [ -n "${MODEL_SHA256}" ] || [ -n "${TOKENIZER_SHA256}" ]; then
  need_cmd sha256sum
fi
if [ -n "${MODEL_SHA256}" ]; then
  echo "${MODEL_SHA256}  ${TMP_MODEL}" | sha256sum -c -
  echo "[ok] model checksum verified"
fi
if [ -n "${TOKENIZER_SHA256}" ]; then
  echo "${TOKENIZER_SHA256}  ${TMP_TOKENIZER}" | sha256sum -c -
  echo "[ok] tokenizer checksum verified"
fi

mv "${TMP_MODEL}" "${MODEL_OUT}"
mv "${TMP_TOKENIZER}" "${TOKENIZER_OUT}"

echo "[done] embedding artifacts ready"
echo "[out] ${MODEL_OUT}"
echo "[out] ${TOKENIZER_OUT}"
