#include "rag_pipeline.hpp"

#include <algorithm>
#include <chrono>

#include "advanced_query_transformer.hpp"
#include "bge_reranker.hpp"
#include "contextual_compressor.hpp"
#include "hyde_transformer.hpp"
#include "reranker.hpp"

namespace rag {

RagPipeline::RagPipeline(const Retriever& retriever,
                         const PromptBuilder& prompt_builder,
                         const Generator& generator,
                         const int top_k,
                         const int rerank_candidate_k,
                         const BgeReranker* bge_reranker)
    : retriever_(&retriever),
      prompt_builder_(&prompt_builder),
      generator_(&generator),
      bge_reranker_(bge_reranker),
      top_k_(std::max(1, top_k)),
      rerank_candidate_k_(std::max(std::max(1, top_k), rerank_candidate_k)) {}

RagResult RagPipeline::run(const std::string& query, const AdvancedMode advanced) const {
  const auto total_start = std::chrono::steady_clock::now();

  const auto query_transform_start = std::chrono::steady_clock::now();
  std::string retrieval_query = query;
  if (advanced == AdvancedMode::QueryRewrite) {
    const GenerationTrace rewrite_trace = generator_->generateWithTrace(buildQueryRewritePrompt(query));
    retrieval_query = sanitizeQueryRewrite(rewrite_trace.output_text, query).text;
  } else if (advanced == AdvancedMode::Hyde) {
    const GenerationTrace hyde_trace = generator_->generateWithTrace(buildHydePrompt(query));
    retrieval_query = sanitizeHydeDocument(hyde_trace.output_text, query).document;
  }
  const auto query_transform_end = std::chrono::steady_clock::now();

  const auto retrieve_start = std::chrono::steady_clock::now();
  const int retrieve_k = advanced == AdvancedMode::Rerank ? rerank_candidate_k_ : top_k_;
  std::vector<RetrievedChunk> contexts = retriever_->retrieve(retrieval_query, retrieve_k);
  const auto retrieve_end = std::chrono::steady_clock::now();

  const auto rerank_start = std::chrono::steady_clock::now();
  if (advanced == AdvancedMode::Rerank) {
    if (bge_reranker_) {
      contexts = bge_reranker_->rerank(query, contexts, top_k_);
    } else {
      contexts = rerankByLexicalOverlap(query, contexts, top_k_);
    }
  }
  const auto rerank_end = std::chrono::steady_clock::now();

  const auto context_transform_start = std::chrono::steady_clock::now();
  if (advanced == AdvancedMode::ContextualCompression) {
    const GenerationTrace compression_trace =
        generator_->generateWithTrace(buildContextualCompressionPrompt(query, contexts));
    contexts = buildCompressedContexts(compression_trace.output_text, contexts).contexts;
  }
  const auto context_transform_end = std::chrono::steady_clock::now();

  const auto prompt_build_start = std::chrono::steady_clock::now();
  const std::string prompt_body = prompt_builder_->buildPrompt(query, contexts);
  const auto prompt_build_end = std::chrono::steady_clock::now();

  const auto generate_start = std::chrono::steady_clock::now();
  const GenerationTrace generation_trace = generator_->generateWithTrace(prompt_body);
  const auto generate_end = std::chrono::steady_clock::now();

  RagResult result;
  result.advanced_mode = advanced;
  result.original_query = query;
  result.retrieval_query = retrieval_query;
  result.prompt_body = generation_trace.prompt_body;
  result.final_prompt = generation_trace.final_prompt;
  result.prompt = generation_trace.final_prompt;
  result.answer = generation_trace.output_text;
  result.prompt_tokens_before_truncation = generation_trace.prompt_tokens_before_truncation;
  result.prompt_tokens = generation_trace.prompt_tokens;
  result.output_tokens = generation_trace.output_tokens;
  result.prompt_truncated = generation_trace.prompt_truncated;
  result.query_transform_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(query_transform_end - query_transform_start).count();
  result.retrieve_ms = std::chrono::duration_cast<std::chrono::milliseconds>(retrieve_end - retrieve_start).count();
  result.rerank_ms = std::chrono::duration_cast<std::chrono::milliseconds>(rerank_end - rerank_start).count();
  result.context_transform_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(context_transform_end - context_transform_start).count();
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
