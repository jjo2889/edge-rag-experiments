#pragma once

#include <string>

#include "config.hpp"

namespace rag::query {

struct QueryRequest {
  std::string run_id;
  std::string dataset;
  std::string question_id;
  std::string question;
  std::string prompt_template_id;
  int retrieval_top_k = 0;
  float generation_temperature = 0.0F;
  float generation_top_p = 0.0F;
  int generation_max_tokens = 0;
};

QueryRequest loadRequestFromJson(const std::string& path);
QueryRequest buildQuestionRequest(const std::string& question,
                                  const ModelConfig& model_cfg,
                                  const RetrievalConfig& retrieval_cfg);
void validateRequest(const QueryRequest& request);
void applyRequestToConfigs(const QueryRequest& request, ModelConfig& model_cfg, RetrievalConfig& retrieval_cfg);

}  // namespace rag::query
