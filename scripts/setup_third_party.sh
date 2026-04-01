#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY_DIR="${ROOT_DIR}/third_party"
LLAMA_DIR="${THIRD_PARTY_DIR}/llama.cpp"
LLAMA_REF="${LLAMA_REF:-f5ddcd1696ec}"

ORT_VERSION="${ORT_VERSION:-1.24.3}"
ORT_ARCHIVE="onnxruntime-linux-x64-${ORT_VERSION}.tgz"
ORT_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ORT_ARCHIVE}"
ORT_DIR="${THIRD_PARTY_DIR}/onnxruntime-linux-x64-${ORT_VERSION}"
ORT_SHA256="${ORT_SHA256:-}"

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "[fail] missing command: $1"
    exit 1
  fi
}

need_cmd git
need_cmd curl
need_cmd tar

mkdir -p "${THIRD_PARTY_DIR}"

echo "[setup] third_party=${THIRD_PARTY_DIR}"

echo "[setup] llama.cpp"
if [ ! -d "${LLAMA_DIR}/.git" ]; then
  git clone https://github.com/ggml-org/llama.cpp.git "${LLAMA_DIR}"
else
  git -C "${LLAMA_DIR}" fetch --tags origin
fi

git -C "${LLAMA_DIR}" checkout "${LLAMA_REF}"

echo "[setup] onnxruntime ${ORT_VERSION}"
if [ ! -d "${ORT_DIR}" ]; then
  TMP_DIR="$(mktemp -d)"
  ORT_ARCHIVE_PATH="${TMP_DIR}/${ORT_ARCHIVE}"
  curl -fL "${ORT_URL}" -o "${ORT_ARCHIVE_PATH}"
  if [ -n "${ORT_SHA256}" ]; then
    need_cmd sha256sum
    echo "${ORT_SHA256}  ${ORT_ARCHIVE_PATH}" | sha256sum -c -
    echo "[ok] checksum verified: ${ORT_ARCHIVE}"
  else
    echo "[warn] ORT_SHA256 is empty; checksum verification skipped"
  fi
  tar -xzf "${TMP_DIR}/${ORT_ARCHIVE}" -C "${THIRD_PARTY_DIR}"
  rm -rf "${TMP_DIR}"
else
  echo "[skip] ${ORT_DIR} already exists"
fi

if [ ! -e "${ORT_DIR}/lib64" ]; then
  ln -s lib "${ORT_DIR}/lib64"
  echo "[ok] created ${ORT_DIR}/lib64 -> lib"
fi
if [ ! -e "${ORT_DIR}/include/onnxruntime" ]; then
  ln -s . "${ORT_DIR}/include/onnxruntime"
  echo "[ok] created ${ORT_DIR}/include/onnxruntime -> ."
fi

echo "[done] third_party setup complete"
echo "[next] run: scripts/rebuild_llama.sh"
