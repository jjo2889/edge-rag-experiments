#include "rag_pipeline.hpp"

#include <algorithm>
#include <chrono>

namespace rag {

RagPipeline::RagPipeline(const Retriever& retriever,
                         const PromptBuilder& prompt_builder,
                         const Generator& generator,
                         const int top_k)
    : retriever_(&retriever), prompt_builder_(&prompt_builder), generator_(&generator), top_k_(std::max(1, top_k)) {}

RagResult RagPipeline::run(const std::string& query) const {
  const auto total_start = std::chrono::steady_clock::now();

  const auto retrieve_start = std::chrono::steady_clock::now();
  const std::vector<RetrievedChunk> contexts = retriever_->retrieve(query, top_k_);
  const auto retrieve_end = std::chrono::steady_clock::now();

  const auto prompt_build_start = std::chrono::steady_clock::now();
  const std::string prompt_body = prompt_builder_->buildPrompt(query, contexts);
  const auto prompt_build_end = std::chrono::steady_clock::now();

  const auto generate_start = std::chrono::steady_clock::now();
  const GenerationTrace generation_trace = generator_->generateWithTrace(prompt_body);
  const auto generate_end = std::chrono::steady_clock::now();

  RagResult result;
  result.original_query = query;
  result.retrieval_query = query;
  result.prompt_body = generation_trace.prompt_body;
  result.final_prompt = generation_trace.final_prompt;
  result.prompt = generation_trace.final_prompt;
  result.answer = generation_trace.output_text;
  result.prompt_tokens_before_truncation = generation_trace.prompt_tokens_before_truncation;
  result.prompt_tokens = generation_trace.prompt_tokens;
  result.output_tokens = generation_trace.output_tokens;
  result.prompt_truncated = generation_trace.prompt_truncated;
  result.retrieve_ms = std::chrono::duration_cast<std::chrono::milliseconds>(retrieve_end - retrieve_start).count();
  result.prompt_build_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(prompt_build_end - prompt_build_start).count();
  result.generate_ms = std::chrono::duration_cast<std::chrono::milliseconds>(generate_end - generate_start).count();
  result.total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(generate_end - total_start).count();

  const double generate_sec = std::chrono::duration<double>(generate_end - generate_start).count();
  if (generate_sec > 0.0) {
    result.output_tokens_per_sec = static_cast<double>(result.output_tokens) / generate_sec;
  }

  result.contexts = contexts;
  return result;
}

}  // namespace rag
