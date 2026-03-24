#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_ONNXRUNTIME_ROOT="${ROOT_DIR}/third_party/onnxruntime-linux-x64-1.24.3"
DEFAULT_LLAMA_CPP_ROOT="${ROOT_DIR}/third_party/llama.cpp"
DEFAULT_FAISS_ROOT="/usr/local"

export ONNXRUNTIME_ROOT="${ONNXRUNTIME_ROOT:-${DEFAULT_ONNXRUNTIME_ROOT}}"
export LLAMA_CPP_ROOT="${LLAMA_CPP_ROOT:-${DEFAULT_LLAMA_CPP_ROOT}}"
export FAISS_ROOT="${FAISS_ROOT:-${DEFAULT_FAISS_ROOT}}"

ORT_LIB_PATHS="${ONNXRUNTIME_ROOT}/lib"
if [ -d "${ONNXRUNTIME_ROOT}/lib64" ]; then
  ORT_LIB_PATHS="${ORT_LIB_PATHS}:${ONNXRUNTIME_ROOT}/lib64"
fi
export LD_LIBRARY_PATH="${LLAMA_CPP_ROOT}/build/bin:${ORT_LIB_PATHS}:${LD_LIBRARY_PATH:-}"

FAISS_HINTS="${FAISS_ROOT}:${FAISS_ROOT}/lib/cmake/faiss:${FAISS_ROOT}/share/faiss"
export CMAKE_PREFIX_PATH="${FAISS_HINTS}:${ONNXRUNTIME_ROOT}:${CMAKE_PREFIX_PATH:-}"

echo "[env] ROOT_DIR=${ROOT_DIR}"
echo "[env] LLAMA_CPP_ROOT=${LLAMA_CPP_ROOT}"
echo "[env] ONNXRUNTIME_ROOT=${ONNXRUNTIME_ROOT}"
echo "[env] FAISS_ROOT=${FAISS_ROOT}"
echo "[env] LD_LIBRARY_PATH updated for llama.cpp + onnxruntime"
echo "[env] CMAKE_PREFIX_PATH updated for faiss + onnxruntime"
