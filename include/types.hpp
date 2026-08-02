#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "advanced_mode.hpp"
#include "context_source.hpp"

namespace rag {

struct Document {
  std::string id;
  std::string title;
  std::string text;
};

struct Chunk {
  int64_t chunk_id = -1;
  int doc_chunk_index = -1;
  int sentence_begin = -1;
  int sentence_end = -1;
  std::string doc_id;
  std::string text;
};

struct DocumentSentence {
  std::string doc_id;
  int sentence_index = -1;
  std::string text;
};

struct RetrievedChunk {
  Chunk chunk;
  float score = 0.0F;
  float rerank_score = 0.0F;
  bool has_rerank_score = false;
  // False for chunks that are not a retrieved passage — currently only the
  // merged chunk produced by contextual_compression, which has no retrieval
  // score and no document of its own. Serialization emits null instead of an
  // inherited score for these.
  bool has_retrieval_score = true;
};

struct RagResult {
  AdvancedMode advanced_mode = AdvancedMode::None;
  // Which baseline this row belongs to. Mixing closed_book or oracle rows into
  // a retrieved run's mean would compare a technique against a different
  // experiment.
  ContextSource context_source = ContextSource::Retrieved;
  std::string original_query;
  std::string retrieval_query;
  std::string prompt_body;
  std::string final_prompt;
  std::string prompt;  // Backward-compatible alias of final_prompt.
  std::string answer;
  int prompt_tokens_before_truncation = 0;
  int prompt_tokens = 0;
  int output_tokens = 0;
  // Tokens spent on auxiliary LLM calls (query_rewrite / hyde /
  // contextual_compression). These are real on-device work; leaving them out of
  // the accounting understated those modes' token cost, throughput, and energy.
  int aux_prompt_tokens = 0;
  int aux_output_tokens = 0;
  bool prompt_truncated = false;
  int64_t query_transform_ms = 0;
  int64_t retrieve_ms = 0;
  int64_t rerank_ms = 0;
  int64_t context_transform_ms = 0;
  int64_t prompt_build_ms = 0;
  int64_t generate_ms = 0;
  int64_t total_ms = 0;
  double output_tokens_per_sec = 0.0;
  // True when an advanced transform (query_rewrite / hyde / contextual_compression)
  // produced empty/unusable output and the pipeline silently reverted to the
  // original query / uncompressed contexts. Lets analysis separate genuine
  // transforms from fallbacks that were averaged into a mode's results.
  bool transform_used_fallback = false;
  // Whether this mode's transform changed anything — see TechniqueEffect. A
  // mode whose effect is NoOp produced the same result as advanced=none, so
  // averaging it into that mode's score reports the baseline as the technique.
  TechniqueEffect technique_effect = TechniqueEffect::NotApplicable;
  // Which reranker actually ran for advanced=rerank: "" (not a rerank run),
  // "lexical", or "bge". Guards against a rerank run silently using the lexical
  // heuristic while the config claimed a cross-encoder.
  std::string reranker_backend_used;
  std::vector<RetrievedChunk> contexts;
  // For contextual_compression, the retrieved passages as they were BEFORE the
  // compressor merged them into a single synthetic chunk. `contexts` for that
  // mode has doc_id="compressed" and carries no usable retrieval provenance, so
  // hit@k / MRR / doc_id-based metrics must be computed from this field instead.
  // Empty for every other mode (where `contexts` is already the retrieval output).
  std::vector<RetrievedChunk> pre_transform_contexts;
};

}  // namespace rag
