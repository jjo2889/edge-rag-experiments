#pragma once

#include <string>

namespace rag::query {

struct QueryRequest {
  std::string run_id;
  std::string dataset;
  std::string split;
  std::string sample_id;
  std::string question;
};

QueryRequest loadRequestFromJson(const std::string& path);
QueryRequest buildQuestionRequest(const std::string& question);
void validateRequest(const QueryRequest& request);

}  // namespace rag::query
