#pragma once

#include <optional>
#include <string>

#include <nlohmann/json.hpp>

#include "config.hpp"
#include "query_request_io.hpp"
#include "types.hpp"

namespace rag::query {

double getResidentMemoryMb();

nlohmann::json buildRunConfigJson(const ModelConfig& model_cfg,
                                  const RetrievalConfig& retrieval_cfg,
                                  const std::string& model_config_path,
                                  const std::string& retrieval_config_path);

nlohmann::json buildTraceJson(const RagResult& result, double memory_mb);

nlohmann::json buildSingleRunSuccessJson(const QueryRequest& request,
                                         const RagResult& result,
                                         const ModelConfig& model_cfg,
                                         const RetrievalConfig& retrieval_cfg,
                                         const std::string& model_config_path,
                                         const std::string& retrieval_config_path,
                                         double memory_mb);

nlohmann::json buildSingleRunErrorJson(const std::optional<QueryRequest>& request, const std::string& message);

}  // namespace rag::query
