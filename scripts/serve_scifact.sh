#!/usr/bin/env bash
# Serve SciFact with the BGE cross-encoder reranker enabled.
#
# The BGE config is identical to configs/retrieval_scifact.yaml except it adds
# rerank_candidate_k and the reranker block, both of which only take effect when
# advanced=rerank — the other five modes are unaffected and share the same index.
# Pinning it here is what keeps `advanced=rerank` from silently running the
# lexical fallback (CODE_REVIEW_FIXES.md P0-3).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export RAG_RETRIEVAL_CONFIG="${RAG_RETRIEVAL_CONFIG:-configs/retrieval_scifact_bge_rerank.yaml}"
exec "${ROOT_DIR}/scripts/serve.sh"
