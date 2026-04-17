#pragma once

#include <mutex>
#include <string>

#include "config.hpp"

struct llama_context;
struct llama_model;
struct llama_vocab;

namespace rag {

struct GenerationTrace {
  std::string prompt_body;
  std::string final_prompt;
  std::string output_text;
  int prompt_tokens_before_truncation = 0;
  int prompt_tokens = 0;
  int output_tokens = 0;
  bool prompt_truncated = false;
};

class Generator {
 public:
  explicit Generator(ModelConfig config);
  ~Generator();

  Generator(const Generator&) = delete;
  Generator& operator=(const Generator&) = delete;

  std::string generate(const std::string& prompt_body) const;
  GenerationTrace generateWithTrace(const std::string& prompt_body) const;

 private:
  std::string buildFinalPrompt(const std::string& prompt_body) const;
  int computeRequiredBatchSize(int prompt_token_count) const;
  void recreateContextLocked(int requested_n_batch) const;

  ModelConfig config_;
  llama_model* model_;
  const llama_vocab* vocab_;
  std::string system_prompt_;
  int context_limit_;
  mutable int ctx_n_batch_;
  mutable llama_context* ctx_;
  mutable std::mutex ctx_mutex_;
};

}  // namespace rag
