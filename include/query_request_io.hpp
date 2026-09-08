#pragma once

#include <string>
#include <vector>

#include "advanced_mode.hpp"
#include "context_source.hpp"

namespace rag::query {

struct QueryRequest {
  std::string run_id;
  std::string dataset;
  std::string split;
  std::string sample_id;
  std::string question;
  AdvancedMode advanced = AdvancedMode::None;
  ContextSource context_source = ContextSource::Retrieved;
  // Required when context_source is oracle, rejected otherwise. The server has
  // no gold labels of its own, so the caller supplies them.
  std::vector<OracleContext> oracle_contexts;
};

QueryRequest loadRequestFromJson(const std::string& path);
QueryRequest buildQuestionRequest(const std::string& question);
void validateRequest(const QueryRequest& request);

}  // namespace rag::query
