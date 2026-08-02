#pragma once

#include <functional>
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

  // Called with each freshly generated token piece (raw, before stop-string
  // trimming). Empty by default; used to stream generation incrementally.
  using TokenCallback = std::function<void(const std::string&)>;

  std::string generate(const std::string& prompt_body) const;
  // These two use the model's configured QA system prompt (final-answer path).
  GenerationTrace generateWithTrace(const std::string& prompt_body) const;
  GenerationTrace generateWithTrace(const std::string& prompt_body,
                                    const TokenCallback& on_token) const;
  // Auxiliary generations (query rewrite, HyDE, contextual compression) must NOT
  // inherit the final-answer QA system prompt, or they emit e.g. "Label: ..." /
  // "I don't know" instead of a rewrite. Pass an explicit override ("" = none).
  GenerationTrace generateWithSystemPrompt(const std::string& prompt_body,
                                           const std::string& system_prompt_override) const;

 private:
  GenerationTrace generateImpl(const std::string& prompt_body,
                               const std::string& system_prompt,
                               const TokenCallback& on_token) const;
  std::string buildFinalPrompt(const std::string& prompt_body,
                               const std::string& system_prompt) const;
  std::string buildChatPrompt(const std::string& prompt_body,
                              const std::string& system_prompt) const;
  bool trimAtStopString(std::string& output_text) const;
  int computeRequiredBatchSize(int prompt_token_count) const;
  void recreateContextLocked(int requested_n_batch) const;

  ModelConfig config_;
  llama_model* model_;
  const llama_vocab* vocab_;
  std::string system_prompt_;
  std::string chat_template_;
  int context_limit_;
  mutable int ctx_n_batch_;
  mutable llama_context* ctx_;
  mutable std::mutex ctx_mutex_;
};

}  // namespace rag
