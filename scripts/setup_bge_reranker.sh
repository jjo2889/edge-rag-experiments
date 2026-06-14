#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${BGE_RERANKER_DIR:-${ROOT_DIR}/models/bge-reranker-base-onnx}"
MODEL_REPO="${BGE_RERANKER_ONNX_REPO:-onnx-community/bge-reranker-base-ONNX}"
TOKENIZER_REPO="${BGE_RERANKER_TOKENIZER_REPO:-BAAI/bge-reranker-base}"
MODEL_FILE="${BGE_RERANKER_ONNX_FILE:-onnx/model_q4.onnx}"

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "[fail] missing command: $1"
    exit 1
  fi
}

download_if_missing() {
  local url="$1"
  local path="$2"
  if [ -s "${path}" ]; then
    echo "[skip] ${path}"
    return
  fi
  echo "[download] ${url}"
  curl -fL "${url}" -o "${path}"
}

need_cmd curl
mkdir -p "${OUT_DIR}"

download_if_missing \
  "https://huggingface.co/${MODEL_REPO}/resolve/main/${MODEL_FILE}" \
  "${OUT_DIR}/model.onnx"

download_if_missing \
  "https://huggingface.co/${TOKENIZER_REPO}/resolve/main/sentencepiece.bpe.model" \
  "${OUT_DIR}/sentencepiece.bpe.model"

download_if_missing \
  "https://huggingface.co/${MODEL_REPO}/resolve/main/config.json" \
  "${OUT_DIR}/config.json"

echo "[done] BGE reranker artifacts: ${OUT_DIR}"
echo "[config] model_path: ${OUT_DIR}/model.onnx"
echo "[config] sentencepiece_model_path: ${OUT_DIR}/sentencepiece.bpe.model"
