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

need_dir() {
  if [ ! -d "$1" ]; then
    echo "[fail] missing directory: $1"
    exit 1
  fi
  echo "[ok] directory: $1"
}

need_cmd cmake
need_cmd g++
need_cmd python3
need_cmd find

need_file "${ROOT_DIR}/models/gemma/gemma-3-270m-it-Q4_K_S.gguf"
need_file "${ROOT_DIR}/models/all-MiniLM-L6-v2-onnx/model.onnx"
need_file "${ROOT_DIR}/models/all-MiniLM-L6-v2-onnx/tokenizer.json"
need_file "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so"

LLAMA_LIB_PATH="$(find "${LLAMA_CPP_ROOT}/build" -maxdepth 3 -type f -name 'libllama.so*' | sort | head -n 1 || true)"
if [ -z "${LLAMA_LIB_PATH}" ]; then
  echo "[fail] missing llama shared lib under ${LLAMA_CPP_ROOT}/build"
  exit 1
fi
echo "[ok] path: ${LLAMA_LIB_PATH}"

DATA_DIRS=()
if [ -d "${ROOT_DIR}/data/raw" ]; then
  DATA_DIRS+=("${ROOT_DIR}/data/raw")
fi
if [ -d "${ROOT_DIR}/data/beir" ]; then
  while IFS= read -r dir; do
    DATA_DIRS+=("${dir}")
  done < <(find "${ROOT_DIR}/data/beir" -mindepth 1 -maxdepth 1 -type d -name '*_corpus' | sort)
fi

if [ "${#DATA_DIRS[@]}" -eq 0 ]; then
  echo "[fail] no ingestible data directories found under ${ROOT_DIR}/data/raw or ${ROOT_DIR}/data/beir/*_corpus"
  exit 1
fi

TOTAL_TEXT_FILE_COUNT=0
for data_dir in "${DATA_DIRS[@]}"; do
  DIR_TEXT_FILE_COUNT="$(find "${data_dir}" -type f \( -name '*.txt' -o -name '*.md' -o -name '*.jsonl' \) | wc -l | tr -d ' ')"
  if [ "${DIR_TEXT_FILE_COUNT}" -gt 0 ]; then
    echo "[ok] ingestible files under ${data_dir#${ROOT_DIR}/}: ${DIR_TEXT_FILE_COUNT}"
    TOTAL_TEXT_FILE_COUNT=$((TOTAL_TEXT_FILE_COUNT + DIR_TEXT_FILE_COUNT))
  fi
done

if [ "${TOTAL_TEXT_FILE_COUNT}" -eq 0 ]; then
  echo "[fail] no ingestible files (.txt/.md/.jsonl) under data/raw or data/beir/*_corpus"
  exit 1
fi
echo "[ok] total ingestible files: ${TOTAL_TEXT_FILE_COUNT}"

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

echo "[check] testing header: nlohmann/json.hpp (system or llama.cpp vendor)"
TMP_JSON="$(mktemp -d)"
cat > "${TMP_JSON}/main.cpp" <<'CPP'
#include <nlohmann/json.hpp>
int main() { nlohmann::json j; j["ok"] = true; return 0; }
CPP
JSON_INCLUDE_FLAGS=()
if [ -f "${LLAMA_CPP_ROOT}/vendor/nlohmann/json.hpp" ]; then
  JSON_INCLUDE_FLAGS=(-I"${LLAMA_CPP_ROOT}/vendor")
  echo "[info] using vendored header: ${LLAMA_CPP_ROOT}/vendor/nlohmann/json.hpp"
else
  echo "[info] vendored nlohmann not found; trying system include path"
fi
g++ -std=c++17 "${JSON_INCLUDE_FLAGS[@]}" "${TMP_JSON}/main.cpp" -o "${TMP_JSON}/check_json" >/dev/null
echo "[ok] nlohmann/json.hpp"

echo "[done] environment check passed"
