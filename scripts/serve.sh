#!/usr/bin/env bash
# Generic launcher for the RAG HTTP server against a pinned retrieval config.
#
# Prefer the dataset wrappers (scripts/serve_scifact.sh, scripts/serve_nq.sh)
# over calling this directly — they pin the BGE-reranker config so that
# `advanced=rerank` runs the cross-encoder instead of falling back to lexical
# overlap (CODE_REVIEW_FIXES.md P0-3).
#
# Every value can be overridden by environment variable:
#   RAG_RETRIEVAL_CONFIG=configs/retrieval_nq_bge_rerank.yaml scripts/serve.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

HOST="${RAG_HTTP_HOST:-0.0.0.0}"
PORT="${RAG_HTTP_PORT:-8000}"
MODEL_CONFIG="${RAG_MODEL_CONFIG:-configs/model.yaml}"
RETRIEVAL_CONFIG="${RAG_RETRIEVAL_CONFIG:-configs/retrieval_scifact_bge_rerank.yaml}"
REQUEST_LOG_DIR="${RAG_REQUEST_LOG_DIR:-results/http-requests}"
TIMEOUT_SEC="${RAG_TIMEOUT_SEC:-600}"
LOG_FILE="${RAG_HTTP_LOG:-results/logs/rag_http_server.log}"

if [ ! -x ./build/rag_http_server ]; then
  echo "[fail] ./build/rag_http_server not found; build the http server target first" >&2
  exit 1
fi
if [ ! -f "${RETRIEVAL_CONFIG}" ]; then
  echo "[fail] retrieval config not found: ${RETRIEVAL_CONFIG}" >&2
  exit 1
fi

# Fail here rather than per request: the server itself never opens the index —
# each forked rag_query does — so a config pointing at a missing index produces a
# server that starts fine and then errors on every single query.
INDEX_PATH="$(sed -n 's/^index_path:[[:space:]]*"\{0,1\}\([^"]*\)"\{0,1\}[[:space:]]*$/\1/p' "${RETRIEVAL_CONFIG}" | head -1)"
CHUNK_STORE_PATH="$(sed -n 's/^chunk_store_path:[[:space:]]*"\{0,1\}\([^"]*\)"\{0,1\}[[:space:]]*$/\1/p' "${RETRIEVAL_CONFIG}" | head -1)"
for artifact in "${INDEX_PATH}" "${CHUNK_STORE_PATH}"; do
  if [ -n "${artifact}" ] && [ ! -f "${artifact}" ]; then
    echo "[fail] ${RETRIEVAL_CONFIG} points at a missing artifact: ${artifact}" >&2
    echo "[fail] run rag_ingest with this retrieval config first" >&2
    exit 1
  fi
done

mkdir -p "$(dirname "${LOG_FILE}")" "${REQUEST_LOG_DIR}"

# Stop any existing server (and its bash wrapper) on this binary.
pkill -f "build/rag_http_server" 2>/dev/null || true
sleep 1

setsid nohup ./build/rag_http_server \
  --host "${HOST}" \
  --port "${PORT}" \
  --model_config "${MODEL_CONFIG}" \
  --retrieval_config "${RETRIEVAL_CONFIG}" \
  --request_log_dir "${REQUEST_LOG_DIR}" \
  --timeout_sec "${TIMEOUT_SEC}" \
  > "${LOG_FILE}" 2>&1 < /dev/null &

sleep 2
echo "[serve] rag_http_server started"
echo "[serve] host=${HOST} port=${PORT}"
echo "[serve] model_config=${MODEL_CONFIG}"
echo "[serve] retrieval_config=${RETRIEVAL_CONFIG}"
echo "[serve] index=${INDEX_PATH}"
echo "[serve] request_log_dir=${REQUEST_LOG_DIR} timeout_sec=${TIMEOUT_SEC}"
echo "[serve] log=${LOG_FILE}"
