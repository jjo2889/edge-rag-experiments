#include "rag_pipeline.hpp"

#include <algorithm>

namespace rag {

RagPipeline::RagPipeline(const Retriever& retriever,
                         const PromptBuilder& prompt_builder,
                         const Generator& generator,
                         const int top_k)
    : retriever_(&retriever), prompt_builder_(&prompt_builder), generator_(&generator), top_k_(std::max(1, top_k)) {}

RagResult RagPipeline::run(const std::string& query) const {
  const std::vector<RetrievedChunk> contexts = retriever_->retrieve(query, top_k_);
  const std::string prompt = prompt_builder_->buildPrompt(query, contexts);
  const std::string answer = generator_->generate(prompt);

  RagResult result;
  result.prompt = prompt;
  result.answer = answer;
  result.contexts = contexts;
  return result;
}

}  // namespace rag
