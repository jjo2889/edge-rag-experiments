#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ONNXRUNTIME_ROOT="${ONNXRUNTIME_ROOT:-${ROOT_DIR}/third_party/onnxruntime-linux-x64-1.24.3}"
LLAMA_CPP_ROOT="${LLAMA_CPP_ROOT:-${ROOT_DIR}/third_party/llama.cpp}"
FAISS_ROOT="${FAISS_ROOT:-/usr/local}"
CMAKE_HINTS="${FAISS_ROOT};${FAISS_ROOT}/lib/cmake/faiss;${FAISS_ROOT}/share/faiss;${ONNXRUNTIME_ROOT}"

echo "[check] root: ${ROOT_DIR}"
echo "[check] ONNXRUNTIME_ROOT=${ONNXRUNTIME_ROOT}"
echo "[check] LLAMA_CPP_ROOT=${LLAMA_CPP_ROOT}"
echo "[check] FAISS_ROOT=${FAISS_ROOT}"

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "[fail] missing command: $1"
    exit 1
  fi
  echo "[ok] command: $1"
}

need_file() {
  if [ ! -e "$1" ]; then
    echo "[fail] missing path: $1"
    exit 1
  fi
  echo "[ok] path: $1"
}

need_cmd cmake
need_cmd g++
need_cmd python3

need_file "${ROOT_DIR}/models/gemma/gemma-3-270m-it-Q4_K_S.gguf"
need_file "${ROOT_DIR}/models/all-MiniLM-L6-v2-onnx/model.onnx"
need_file "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so"

LLAMA_LIB_PATH="$(find "${LLAMA_CPP_ROOT}/build/bin" -maxdepth 1 -type f -name 'libllama.so*' | sort | head -n 1 || true)"
if [ -z "${LLAMA_LIB_PATH}" ]; then
  echo "[fail] missing llama shared lib under ${LLAMA_CPP_ROOT}/build/bin"
  exit 1
fi
echo "[ok] path: ${LLAMA_LIB_PATH}"

FAISS_STATIC_LIB="$(find "${FAISS_ROOT}" -type f -name 'libfaiss.a' 2>/dev/null | sort | head -n 1 || true)"
if [ -n "${FAISS_STATIC_LIB}" ]; then
  echo "[info] detected static faiss: ${FAISS_STATIC_LIB}"
else
  echo "[info] static faiss not found under ${FAISS_ROOT} (shared faiss may still be valid)"
fi

echo "[check] testing CMake package: onnxruntime"
TMP_ORT="$(mktemp -d)"
cat > "${TMP_ORT}/CMakeLists.txt" <<'CMAKE'
cmake_minimum_required(VERSION 3.20)
if(POLICY CMP0144)
  cmake_policy(SET CMP0144 NEW)
endif()
project(check_ort CXX)
find_package(onnxruntime CONFIG REQUIRED)
add_executable(check main.cpp)
target_link_libraries(check PRIVATE onnxruntime::onnxruntime)
CMAKE
cat > "${TMP_ORT}/main.cpp" <<'CPP'
int main(){return 0;}
CPP
cmake -S "${TMP_ORT}" -B "${TMP_ORT}/build" -DCMAKE_PREFIX_PATH="${CMAKE_HINTS}" >/dev/null
echo "[ok] onnxruntime CMake package"

echo "[check] testing CMake package: faiss (OpenMP required)"
TMP_FAISS="$(mktemp -d)"
cat > "${TMP_FAISS}/CMakeLists.txt" <<'CMAKE'
cmake_minimum_required(VERSION 3.20)
if(POLICY CMP0144)
  cmake_policy(SET CMP0144 NEW)
endif()
project(check_faiss CXX)
find_package(OpenMP REQUIRED)
find_package(faiss CONFIG REQUIRED)
add_executable(check main.cpp)
target_link_libraries(check PRIVATE faiss)
CMAKE
cat > "${TMP_FAISS}/main.cpp" <<'CPP'
#include <faiss/IndexFlat.h>
int main(){ faiss::IndexFlatIP idx(384); return 0; }
CPP
cmake -S "${TMP_FAISS}" -B "${TMP_FAISS}/build" -DCMAKE_PREFIX_PATH="${CMAKE_HINTS}" >/dev/null
cmake --build "${TMP_FAISS}/build" -j >/dev/null
echo "[ok] faiss CMake package + link test"

echo "[done] environment check passed"
