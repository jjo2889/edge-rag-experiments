#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_DIR="${LLAMA_CPP_ROOT:-${ROOT_DIR}/third_party/llama.cpp}"
BUILD_DIR="${LLAMA_DIR}/build"

echo "[rebuild] ROOT_DIR=${ROOT_DIR}"
echo "[rebuild] LLAMA_DIR=${LLAMA_DIR}"

# GPU 빌드는 opt-in 이다. 타깃인 라즈베리파이에는 CUDA 가 없으므로 기본값을
# 바꾸면 거기서 빌드가 깨진다. 평가용 PC 에서만 켠다:
#
#   LLAMA_CUDA=1 CUDA_ARCH=86 scripts/rebuild_llama.sh    # RTX 3060 = 8.6
#
# 이 스위치는 속도만 바꾸고 측정 대상은 바꾸지 않는다 — docs/research_proposal.md
# §4.2 가 품질 지표(실행률·검색·밴드)를 PC 에서, 비용 지표(레이턴시·메모리)를
# 파이에서 재도록 갈라 둔 이유가 이것이다. 두 환경에 공통으로 존재하는 칸들이
# 그 분리가 유효한지 확인하는 다리 역할을 한다.
CMAKE_ARGS=(
  -DCMAKE_BUILD_TYPE=Release
  -DBUILD_SHARED_LIBS=ON
  -DLLAMA_BUILD_TESTS=OFF
  -DLLAMA_BUILD_EXAMPLES=ON
)
if [ "${LLAMA_CUDA:-0}" = "1" ]; then
  CMAKE_ARGS+=(-DGGML_CUDA=ON)
  if [ -n "${CUDA_ARCH:-}" ]; then
    CMAKE_ARGS+=(-DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}")
  fi
  echo "[rebuild] CUDA 활성 (GGML_CUDA=ON, arch=${CUDA_ARCH:-auto})"
else
  echo "[rebuild] CPU 빌드 (GPU 를 쓰려면 LLAMA_CUDA=1)"
fi

rm -rf "${BUILD_DIR}"

cmake -S "${LLAMA_DIR}" -B "${BUILD_DIR}" "${CMAKE_ARGS[@]}"

cmake --build "${BUILD_DIR}" -j"$(nproc)"

echo "[done] llama.cpp rebuild complete"
echo "[hint] quick check (optional): ${BUILD_DIR}/bin/llama-cli --version"
echo "[next] continue setup flow:"
echo "       - prepare embedding model: scripts/setup_embedding_model.sh (or place model files manually)"
echo "       - prepare data: scripts/setup_scifact_data.sh / scripts/setup_nq_data.sh (or place custom files under data/raw)"
echo "       - then run: source ${ROOT_DIR}/scripts/activate_env.sh && ${ROOT_DIR}/scripts/check_env.sh"
