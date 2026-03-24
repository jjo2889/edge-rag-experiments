#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_DIR="${LLAMA_CPP_ROOT:-${ROOT_DIR}/third_party/llama.cpp}"
BUILD_DIR="${LLAMA_DIR}/build"

echo "[rebuild] ROOT_DIR=${ROOT_DIR}"
echo "[rebuild] LLAMA_DIR=${LLAMA_DIR}"

rm -rf "${BUILD_DIR}"

cmake -S "${LLAMA_DIR}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=ON

cmake --build "${BUILD_DIR}" -j"$(nproc)"

echo "[done] llama.cpp rebuild complete"
echo "[hint] run: source ${ROOT_DIR}/scripts/activate_env.sh"
echo "[hint] check: ${BUILD_DIR}/bin/llama-cli --version"
