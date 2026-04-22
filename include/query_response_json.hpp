#pragma once

#include <optional>
#include <string>

#include <nlohmann/json.hpp>

#include "query_request_io.hpp"
#include "types.hpp"

namespace rag::query {

double getResidentMemoryMb();

nlohmann::json buildSingleRunSuccessJson(const QueryRequest& request,
                                         const RagResult& result,
                                         double memory_mb);

nlohmann::json buildSingleRunErrorJson(const std::optional<QueryRequest>& request, const std::string& message);

}  // namespace rag::query
