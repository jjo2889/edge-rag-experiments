#pragma once

#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "advanced_mode.hpp"
#include "config.hpp"
#include "query_request_io.hpp"
#include "types.hpp"

namespace rag::query {

// Serialize with the "replace" UTF-8 error handler so a generated answer or
// corpus snippet that ends on a partial multi-byte token (byte-fallback BPE can
// emit a lone UTF-8 continuation byte) is emitted with U+FFFD instead of throwing
// nlohmann type_error.316 and aborting the run / batch export.
inline std::string dumpSafe(const nlohmann::json& value) {
  return value.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
}

double getResidentMemoryMb();

// Serializes a context list (rank/doc_id/score/text). Exposed so BEIR export can
// emit pre_transform_contexts in the same shape as retrieved_contexts.
nlohmann::json buildContextRows(const std::vector<RetrievedChunk>& items);

nlohmann::json buildRunConfigJson(const ModelConfig& model_cfg,
                                  const RetrievalConfig& retrieval_cfg,
                                  const std::string& model_config_path,
                                  const std::string& retrieval_config_path,
                                  AdvancedMode advanced);

// `model_load_ms` is the one-time process startup cost (index + chunk store +
// embedding model + GGUF + optional reranker load) measured in query_main before
// the pipeline runs. The HTTP server spawns a fresh rag_query per request, so
// every served query pays this cost; without it trace.latency_ms.total badly
// understates on-device query latency (cold). In batch mode one process serves
// all rows, so the same (amortized) value is echoed on every row.
nlohmann::json buildTraceJson(const RagResult& result, double memory_mb, int64_t model_load_ms);

nlohmann::json buildSingleRunSuccessJson(const QueryRequest& request,
                                         const RagResult& result,
                                         const ModelConfig& model_cfg,
                                         const RetrievalConfig& retrieval_cfg,
                                         const std::string& model_config_path,
                                         const std::string& retrieval_config_path,
                                         double memory_mb,
                                         int64_t model_load_ms);

nlohmann::json buildSingleRunErrorJson(const std::optional<QueryRequest>& request, const std::string& message);

}  // namespace rag::query
