#pragma once

#include <functional>
#include <string>
#include <vector>

#include "advanced_mode.hpp"
#include "generator.hpp"
#include "prompt_builder.hpp"
#include "retriever.hpp"
#include "sentence_window.hpp"
#include "types.hpp"

namespace rag {

class BgeReranker;

class RagPipeline {
 public:
  RagPipeline(const Retriever& retriever,
              const PromptBuilder& prompt_builder,
              const Generator& generator,
              int top_k,
              int rerank_candidate_k,
              const BgeReranker* bge_reranker = nullptr,
              std::vector<DocumentSentence> sentences = {},
              int sentence_window_size = 1);

  // on_token, when set, streams the final answer generation token-by-token.
  // Advanced-mode transform generations (HyDE, query rewrite, etc.) are never
  // streamed through this callback.
  using TokenCallback = std::function<void(const std::string&)>;

  RagResult run(const std::string& query,
                AdvancedMode advanced = AdvancedMode::None,
                const TokenCallback& on_token = nullptr) const;

 private:
  const Retriever* retriever_;
  const PromptBuilder* prompt_builder_;
  const Generator* generator_;
  const BgeReranker* bge_reranker_;
  int top_k_;
  int rerank_candidate_k_;
  int sentence_window_size_;
  // Owns the sentence store; sentence_index_ points into it, so the vector must
  // outlive the index and must not be reallocated after construction.
  std::vector<DocumentSentence> sentences_;
  SentenceIndex sentence_index_;
};

}  // namespace rag
