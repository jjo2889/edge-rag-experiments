#pragma once

#include <string>

#include "advanced_mode.hpp"

namespace rag::query {

struct QueryRequest {
  std::string run_id;
  std::string dataset;
  std::string split;
  std::string sample_id;
  std::string question;
  AdvancedMode advanced = AdvancedMode::None;
};

QueryRequest loadRequestFromJson(const std::string& path);
QueryRequest buildQuestionRequest(const std::string& question);
void validateRequest(const QueryRequest& request);

}  // namespace rag::query
