#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL_REPO="${LLM_MODEL_REPO:-unsloth/gemma-3-270m-it-GGUF}"
# Pinned default for reproducible setup across fresh clones.
PINNED_LLM_REVISION="05136ac288e220e64bb5731cc67c2a7355fc4e79"
LLM_REVISION="${LLM_REVISION:-${PINNED_LLM_REVISION}}"
LLM_MODEL_FILE="${LLM_MODEL_FILE:-gemma-3-270m-it-Q4_K_S.gguf}"

DEST_DIR="${ROOT_DIR}/models/gemma"
MODEL_OUT="${DEST_DIR}/${LLM_MODEL_FILE}"

# Verified 2026-06-22 against this exact pinned revision: matches the file
# already in use for this project (HF x-linked-etag / sha256 both confirmed).
MODEL_SHA256="${MODEL_SHA256:-ac352a5e8efd59d794fd2faa8f864dd624d4392e87448060814ef2611a368ac8}"

FORCE=0

usage() {
  cat <<'USAGE'
Usage: scripts/setup_llm_model.sh [--force]

Downloads the generation model (GGUF) from Hugging Face.
Default source:
  repo: unsloth/gemma-3-270m-it-GGUF
  revision: 05136ac288e220e64bb5731cc67c2a7355fc4e79
  file: gemma-3-270m-it-Q4_K_S.gguf

Output:
  models/gemma/gemma-3-270m-it-Q4_K_S.gguf

Optional env:
  LLM_MODEL_REPO
  LLM_REVISION
  LLM_MODEL_FILE
  MODEL_SHA256 (set to empty to skip verification, e.g. when overriding the file/revision)
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

MODEL_URL="https://huggingface.co/${MODEL_REPO}/resolve/${LLM_REVISION}/${LLM_MODEL_FILE}?download=true"

echo "[setup] llm repo=${MODEL_REPO}"
echo "[setup] llm revision=${LLM_REVISION}"
if [ "${LLM_REVISION}" = "${PINNED_LLM_REVISION}" ]; then
  echo "[setup] llm revision mode=pinned-default"
else
  echo "[setup] llm revision mode=override"
fi
echo "[setup] model url=${MODEL_URL}"

if [ "${FORCE}" -ne 1 ] && [ -f "${MODEL_OUT}" ]; then
  echo "[skip] already exists: ${MODEL_OUT}"
  echo "[hint] rerun with --force to refresh"
  exit 0
fi

TMP_DIR="$(mktemp -d)"
TMP_MODEL="${TMP_DIR}/${LLM_MODEL_FILE}"

cleanup() {
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

mkdir -p "${DEST_DIR}"

echo "[download] model (~250MB)"
curl -fL "${MODEL_URL}" -o "${TMP_MODEL}"

if [ -n "${MODEL_SHA256}" ]; then
  need_cmd sha256sum
  echo "${MODEL_SHA256}  ${TMP_MODEL}" | sha256sum -c -
  echo "[ok] model checksum verified"
fi

mv "${TMP_MODEL}" "${MODEL_OUT}"

echo "[done] llm model ready"
echo "[out] ${MODEL_OUT}"
