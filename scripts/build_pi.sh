#!/usr/bin/env bash
# 라즈베리파이(aarch64) 빌드.
#
# 이 설정은 CMakeCache.txt 안에만 있었다. 캐시를 한 번 지우자 경로를 하나씩
# 다시 찾아야 했고 네 번 실패했다. 실패한 순서와 이유를 남겨 둔다:
#
#   1. cmake 가 PATH 에 없다        — ~/.local/edge-rag/tools/cmake/bin 에 있다
#   2. OpenMP 를 못 찾는다           — clang 을 쓸 때만. GCC 는 정상
#   3. faiss 를 못 찾는다            — CMAKE_PREFIX_PATH 에 sysroot 의 cmake 경로가 필요
#   4. faiss 링크가 깨진다           — CMAKE_BUILD_TYPE 없이 빌드하면 안 된다
#
# 파이는 x86 PC 와 달리 의존성이 전부 손수 설치된 sysroot 에 있다. 그래서
# find_package 가 기본 경로에서 아무것도 못 찾는다.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SYSROOT="${EDGE_SYSROOT:-$HOME/.local/edge-rag/sysroot}"
CMAKE_BIN="${EDGE_CMAKE:-$HOME/.local/edge-rag/tools/cmake/bin}"
ONNX_ROOT="${ROOT_DIR}/third_party/onnxruntime-linux-aarch64-1.24.3"
JOBS="${JOBS:-3}"   # 4 코어에서 4 를 주면 메모리 압박으로 느려진다

[ -d "$SYSROOT" ] || { echo "[build_pi] sysroot 없음: $SYSROOT" >&2; exit 1; }
[ -d "$ONNX_ROOT" ] || { echo "[build_pi] onnxruntime(aarch64) 없음: $ONNX_ROOT" >&2; exit 1; }
export PATH="${CMAKE_BIN}:${PATH}"

# shellcheck source=/dev/null
source "${ROOT_DIR}/scripts/activate_env.sh" >/dev/null 2>&1 || true

LIB="${SYSROOT}/usr/lib/aarch64-linux-gnu"

echo "[build_pi] ROOT=${ROOT_DIR}"
echo "[build_pi] SYSROOT=${SYSROOT}"

# Release 를 반드시 준다. 최적화 없이 빌드하면 libfaiss.a 의 아카이브 멤버
# 사이에서 미해결 참조가 남는다 (RangeSearchPartialResult 생성자).
"${CMAKE_BIN}/cmake" -S "${ROOT_DIR}" -B "${ROOT_DIR}/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="${SYSROOT}/usr;${LIB}/cmake/faiss;${LIB}/cmake/yaml-cpp;${ONNX_ROOT}" \
  -DONNXRUNTIME_ROOT="${ONNX_ROOT}"

"${CMAKE_BIN}/cmake" --build "${ROOT_DIR}/build" -j"${JOBS}"

echo "[build_pi] 완료: ${ROOT_DIR}/build/rag_query"
echo "[build_pi] 실행 전 'source scripts/activate_env.sh' 로 LD_LIBRARY_PATH 를 잡을 것"
