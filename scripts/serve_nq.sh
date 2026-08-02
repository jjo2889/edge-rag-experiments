#!/usr/bin/env bash
# Serve NQ with the BGE cross-encoder reranker enabled.
#
# Counterpart of scripts/serve_scifact.sh. Requires the NQ index to exist:
#   ./build/rag_ingest --model_config configs/model.yaml \
#     --retrieval_config configs/retrieval_nq_bge_rerank.yaml
# serve.sh refuses to start if the configured index/chunk store is missing.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export RAG_RETRIEVAL_CONFIG="${RAG_RETRIEVAL_CONFIG:-configs/retrieval_nq_bge_rerank.yaml}"
exec "${ROOT_DIR}/scripts/serve.sh"
