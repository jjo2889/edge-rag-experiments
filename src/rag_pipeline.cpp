#include "rag_pipeline.hpp"

#include <algorithm>
#include <chrono>
#include <utility>

#include "advanced_query_transformer.hpp"
#include "bge_reranker.hpp"
#include "contextual_compressor.hpp"
#include "hyde_transformer.hpp"
#include "reranker.hpp"
#include "sentence_window.hpp"

namespace rag {

RagPipeline::RagPipeline(const Retriever& retriever,
                         const PromptBuilder& prompt_builder,
                         const Generator& generator,
                         const int top_k,
                         const int rerank_candidate_k,
                         const BgeReranker* bge_reranker,
                         std::vector<DocumentSentence> sentences,
                         const int sentence_window_size)
    : retriever_(&retriever),
      prompt_builder_(&prompt_builder),
      generator_(&generator),
      bge_reranker_(bge_reranker),
      top_k_(std::max(1, top_k)),
      rerank_candidate_k_(std::max(std::max(1, top_k), rerank_candidate_k)),
      sentence_window_size_(std::max(0, sentence_window_size)),
      sentences_(std::move(sentences)),
      // Built once here rather than per query: a per-call scan of the whole
      // sentence store made context_transform_ms measure the scan, not the mode.
      sentence_index_(sentences_) {}

RagResult RagPipeline::run(const std::string& query,
                           const AdvancedMode advanced,
                           const TokenCallback& on_token) const {
  const auto total_start = std::chrono::steady_clock::now();

  const auto query_transform_start = std::chrono::steady_clock::now();
  std::string retrieval_query = query;
  bool transform_used_fallback = false;
  int aux_prompt_tokens = 0;
  int aux_output_tokens = 0;
  // Auxiliary generations pass an empty system prompt so they don't inherit the
  // final-answer QA system prompt (which would turn a rewrite into "Label: ..." /
  // "I don't know"). The final answer below still uses the configured QA prompt.
  if (advanced == AdvancedMode::QueryRewrite) {
    const GenerationTrace rewrite_trace =
        generator_->generateWithSystemPrompt(buildQueryRewritePrompt(query), "");
    const QueryTransformResult rewrite = sanitizeQueryRewrite(rewrite_trace.output_text, query);
    retrieval_query = rewrite.text;
    transform_used_fallback = rewrite.used_fallback;
    aux_prompt_tokens += rewrite_trace.prompt_tokens;
    aux_output_tokens += rewrite_trace.output_tokens;
  } else if (advanced == AdvancedMode::Hyde) {
    const GenerationTrace hyde_trace =
        generator_->generateWithSystemPrompt(buildHydePrompt(query), "");
    const HydeTransformResult hyde = sanitizeHydeDocument(hyde_trace.output_text, query);
    retrieval_query = hyde.document;
    transform_used_fallback = hyde.used_fallback;
    aux_prompt_tokens += hyde_trace.prompt_tokens;
    aux_output_tokens += hyde_trace.output_tokens;
  }
  const auto query_transform_end = std::chrono::steady_clock::now();

  const auto retrieve_start = std::chrono::steady_clock::now();
  const int retrieve_k = advanced == AdvancedMode::Rerank ? rerank_candidate_k_ : top_k_;
  std::vector<RetrievedChunk> contexts = retriever_->retrieve(retrieval_query, retrieve_k);
  const auto retrieve_end = std::chrono::steady_clock::now();

  const auto rerank_start = std::chrono::steady_clock::now();
  std::string reranker_backend_used;
  if (advanced == AdvancedMode::Rerank) {
    if (bge_reranker_) {
      contexts = bge_reranker_->rerank(query, contexts, top_k_);
      reranker_backend_used = "bge";
    } else {
      contexts = rerankByLexicalOverlap(query, contexts, top_k_);
      reranker_backend_used = "lexical";
    }
  }
  const auto rerank_end = std::chrono::steady_clock::now();

  const auto context_transform_start = std::chrono::steady_clock::now();
  std::vector<RetrievedChunk> pre_transform_contexts;
  if (advanced == AdvancedMode::SentenceWindow) {
    contexts = expandSentenceWindows(query, contexts, sentence_index_, sentence_window_size_);
  } else if (advanced == AdvancedMode::ContextualCompression) {
    const GenerationTrace compression_trace =
        generator_->generateWithSystemPrompt(buildContextualCompressionPrompt(query, contexts), "");
    const ContextCompressionResult compression =
        buildCompressedContexts(compression_trace.output_text, contexts);
    aux_prompt_tokens += compression_trace.prompt_tokens;
    aux_output_tokens += compression_trace.output_tokens;
    // Compression replaces the retrieved passages with one synthetic chunk that
    // has no doc_id of its own. Keep the originals so retrieval quality for this
    // mode stays measurable.
    if (!compression.used_fallback) {
      pre_transform_contexts = contexts;
    }
    contexts = compression.contexts;
    transform_used_fallback = compression.used_fallback;
  }
  const auto context_transform_end = std::chrono::steady_clock::now();

  const auto prompt_build_start = std::chrono::steady_clock::now();
  const std::string prompt_body = prompt_builder_->buildPrompt(query, contexts);
  const auto prompt_build_end = std::chrono::steady_clock::now();

  const auto generate_start = std::chrono::steady_clock::now();
  const GenerationTrace generation_trace =
      on_token ? generator_->generateWithTrace(prompt_body, on_token)
               : generator_->generateWithTrace(prompt_body);
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
  result.aux_prompt_tokens = aux_prompt_tokens;
  result.aux_output_tokens = aux_output_tokens;
  result.prompt_truncated = generation_trace.prompt_truncated;
  // Round to milliseconds once, from a microsecond measurement. Truncating each
  // stage with duration_cast<milliseconds> floored every sub-millisecond stage to
  // 0 and made the stage sum fall short of total for no real reason.
  const auto elapsedMs = [](const auto& from, const auto& to) -> int64_t {
    const auto us = std::chrono::duration_cast<std::chrono::microseconds>(to - from).count();
    return (us + 500) / 1000;
  };
  result.query_transform_ms = elapsedMs(query_transform_start, query_transform_end);
  result.retrieve_ms = elapsedMs(retrieve_start, retrieve_end);
  result.rerank_ms = elapsedMs(rerank_start, rerank_end);
  result.context_transform_ms = elapsedMs(context_transform_start, context_transform_end);
  result.prompt_build_ms = elapsedMs(prompt_build_start, prompt_build_end);
  result.generate_ms = elapsedMs(generate_start, generate_end);
  result.total_ms = elapsedMs(total_start, generate_end);

  const double generate_sec = std::chrono::duration<double>(generate_end - generate_start).count();
  if (generate_sec > 0.0) {
    result.output_tokens_per_sec = static_cast<double>(result.output_tokens) / generate_sec;
  }

  result.transform_used_fallback = transform_used_fallback;
  result.reranker_backend_used = reranker_backend_used;
  result.contexts = contexts;
  result.pre_transform_contexts = std::move(pre_transform_contexts);
  return result;
}

}  // namespace rag
