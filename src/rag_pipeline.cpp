#include "rag_pipeline.hpp"

#include <algorithm>
#include <chrono>
#include <optional>
#include <string>
#include <utility>

#include "advanced_query_transformer.hpp"
#include "bge_reranker.hpp"
#include "contextual_compressor.hpp"
#include "hyde_transformer.hpp"
#include "reranker.hpp"
#include "sentence_window.hpp"
#include "technique_effect.hpp"

namespace rag {

namespace {

// Mean of the query embedding and each hypothetical document's, matching
// `np.mean([query] + hypothesis_documents)` in the reference implementation.
// The query is one contributor among N+1, not a weighted anchor.
std::vector<float> meanEmbedding(const Retriever& retriever,
                                 const std::string& query,
                                 const std::vector<std::string>& documents) {
  std::vector<float> sum = retriever.embed(query);
  for (const auto& document : documents) {
    const std::vector<float> vec = retriever.embed(document);
    if (vec.size() != sum.size()) {
      throw std::runtime_error("HyDE embedding dimension mismatch.");
    }
    for (size_t i = 0; i < sum.size(); ++i) {
      sum[i] += vec[i];
    }
  }

  const float divisor = static_cast<float>(documents.size() + 1);
  for (float& value : sum) {
    value /= divisor;
  }
  return sum;
}

}  // namespace

RagPipeline::RagPipeline(const Retriever& retriever,
                         const PromptBuilder& prompt_builder,
                         const Generator& generator,
                         const int top_k,
                         const int rerank_candidate_k,
                         const BgeReranker* bge_reranker,
                         std::vector<DocumentSentence> sentences,
                         const int sentence_window_size,
                         HydeConfig hyde,
                         std::string hyde_prompt_template)
    : retriever_(&retriever),
      prompt_builder_(&prompt_builder),
      generator_(&generator),
      bge_reranker_(bge_reranker),
      top_k_(std::max(1, top_k)),
      rerank_candidate_k_(std::max(std::max(1, top_k), rerank_candidate_k)),
      sentence_window_size_(std::max(0, sentence_window_size)),
      hyde_(hyde),
      hyde_prompt_template_(std::move(hyde_prompt_template)),
      sentences_(std::move(sentences)),
      // Built once here rather than per query: a per-call scan of the whole
      // sentence store made context_transform_ms measure the scan, not the mode.
      sentence_index_(sentences_) {}

RagResult RagPipeline::run(const std::string& query,
                           const AdvancedMode advanced,
                           const TokenCallback& on_token,
                           const ContextOverride& context_override) const {
  const auto total_start = std::chrono::steady_clock::now();

  // closed_book and oracle both bypass retrieval, so query_rewrite, hyde and
  // rerank have nothing to act on. They are reported not_applicable rather than
  // silently burning an auxiliary generation on a query that never reaches an
  // index.
  const bool retrieval_active = context_override.source == ContextSource::Retrieved;

  const auto query_transform_start = std::chrono::steady_clock::now();
  std::string retrieval_query = query;
  bool transform_used_fallback = false;
  TechniqueEffect technique_effect = TechniqueEffect::NotApplicable;
  std::vector<std::string> hyde_documents;
  int aux_prompt_tokens = 0;
  int aux_output_tokens = 0;
  // Auxiliary generations pass an empty system prompt so they don't inherit the
  // final-answer QA system prompt (which would turn a rewrite into "Label: ..." /
  // "I don't know"). The final answer below still uses the configured QA prompt.
  if (retrieval_active && advanced == AdvancedMode::QueryRewrite) {
    const GenerationTrace rewrite_trace =
        generator_->generateWithSystemPrompt(buildQueryRewritePrompt(query), "");
    const QueryTransformResult rewrite = sanitizeQueryRewrite(rewrite_trace.output_text, query);
    retrieval_query = rewrite.text;
    transform_used_fallback = rewrite.used_fallback;
    technique_effect = classifyTextTransform(rewrite.used_fallback, query, rewrite.text);
    aux_prompt_tokens += rewrite_trace.prompt_tokens;
    aux_output_tokens += rewrite_trace.output_tokens;
  } else if (retrieval_active && advanced == AdvancedMode::Hyde) {
    // Gao et al. draw several hypothetical documents and search with the mean of
    // their embeddings and the query's. Replacing the query with one document —
    // what this used to do — drops the anchor that keeps the search vector near
    // the question when a draw goes wrong.
    const std::string hyde_prompt = buildHydePromptFromTemplate(hyde_prompt_template_, query);
    std::vector<std::string> documents;
    documents.reserve(static_cast<size_t>(hyde_.num_documents));

    for (int draw = 0; draw < hyde_.num_documents; ++draw) {
      Generator::SamplingOverride sampling;
      sampling.temperature = hyde_.temperature;
      // Each draw offsets the configured seed. Without this every draw would be
      // byte-identical and the mean would collapse to N=1; replacing the seed
      // instead of offsetting it would ignore model.yaml entirely.
      sampling.seed_offset = static_cast<uint32_t>(draw);
      sampling.max_new_tokens = hyde_.max_new_tokens;

      const GenerationTrace hyde_trace =
          generator_->generateWithSystemPrompt(hyde_prompt, "", sampling);
      aux_prompt_tokens += hyde_trace.prompt_tokens;
      aux_output_tokens += hyde_trace.output_tokens;

      const HydeTransformResult hyde = sanitizeHydeDocument(hyde_trace.output_text, query);
      if (!hyde.used_fallback) {
        documents.push_back(hyde.document);
      }
    }

    hyde_documents = documents;
    // Every draw unusable means there is nothing to average, so the plain query
    // embedding is used — the same search advanced=none performs.
    transform_used_fallback = documents.empty();
    technique_effect = documents.empty() ? TechniqueEffect::Fallback : TechniqueEffect::Applied;
  }

  const auto query_transform_end = std::chrono::steady_clock::now();

  const auto retrieve_start = std::chrono::steady_clock::now();
  std::vector<RetrievedChunk> contexts;
  if (retrieval_active) {
    const int retrieve_k = advanced == AdvancedMode::Rerank ? rerank_candidate_k_ : top_k_;
    if (!hyde_documents.empty()) {
      contexts = retriever_->retrieveByVector(meanEmbedding(*retriever_, query, hyde_documents), retrieve_k);
    } else {
      contexts = retriever_->retrieve(retrieval_query, retrieve_k);
    }
  } else if (context_override.source == ContextSource::Oracle) {
    contexts.reserve(context_override.oracle_contexts.size());
    for (const auto& oracle : context_override.oracle_contexts) {
      RetrievedChunk chunk;
      chunk.chunk.doc_id = oracle.doc_id.empty() ? "oracle" : oracle.doc_id;
      chunk.chunk.text = oracle.text;
      // No index produced these, so a similarity score would be fabricated.
      chunk.has_retrieval_score = false;
      contexts.push_back(std::move(chunk));
    }
  }
  const auto retrieve_end = std::chrono::steady_clock::now();

  const auto rerank_start = std::chrono::steady_clock::now();
  std::optional<std::string> reranker_backend;
  if (advanced == AdvancedMode::Rerank) {
    if (bge_reranker_) {
      contexts = bge_reranker_->rerank(query, contexts, top_k_);
      reranker_backend = "bge";
    } else {
      contexts = rerankByLexicalOverlap(query, contexts, top_k_);
      reranker_backend = "lexical";
    }
    technique_effect = classifyContextTransform(false, retrieval_order_top_k, contexts);
  }
  const auto rerank_end = std::chrono::steady_clock::now();

  const auto context_transform_start = std::chrono::steady_clock::now();
  std::vector<RetrievedChunk> pre_transform_contexts;
  if (advanced == AdvancedMode::SentenceWindow) {
    const std::vector<RetrievedChunk> before_expansion = contexts;
    contexts = expandSentenceWindows(query, contexts, sentence_index_, sentence_window_size_);
    // window_size 0, a missing sentence store, or a chunk already covering its
    // whole document all return the input untouched.
    technique_effect = classifyContextTransform(false, before_expansion, contexts);
  } else if (advanced == AdvancedMode::ContextualCompression) {
    // One call per document, as LLMChainExtractor does. Compressing all of them
    // in a single call asks a small model to synthesise across documents, which
    // is a harder task than the technique actually specifies.
    std::vector<RetrievedChunk> compressed;
    compressed.reserve(contexts.size());
    bool any_fallback = false;
    bool any_change = false;

    for (const auto& context : contexts) {
      const GenerationTrace compression_trace =
          generator_->generateWithSystemPrompt(buildContextualCompressionPrompt(query, context), "");
      aux_prompt_tokens += compression_trace.prompt_tokens;
      aux_output_tokens += compression_trace.output_tokens;

      const SingleContextCompression result =
          compressSingleContext(compression_trace.output_text, context);
      any_fallback = any_fallback || result.used_fallback;
      if (result.dropped) {
        // NO_OUTPUT: the document holds nothing relevant, so it leaves the
        // prompt. Dropping every document is a legitimate outcome, not a
        // failure — the generator then answers from an empty context.
        any_change = true;
        continue;
      }
      any_change = any_change || result.context.chunk.text != context.chunk.text;
      compressed.push_back(result.context);
    }

    contexts = std::move(compressed);
    transform_used_fallback = any_fallback;
    technique_effect = any_fallback ? TechniqueEffect::Fallback
                                    : (any_change ? TechniqueEffect::Applied : TechniqueEffect::NoOp);
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
  result.context_source = context_override.source;
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
  result.technique_effect = technique_effect;
  result.reranker_backend_used = reranker_backend_used;
  result.hyde_documents = std::move(hyde_documents);
  result.contexts = contexts;
  result.reranker_backend = reranker_backend;
  return result;
}

}  // namespace rag
