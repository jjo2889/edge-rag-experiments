#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "advanced_mode.hpp"

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
};

struct RagResult {
  AdvancedMode advanced_mode = AdvancedMode::None;
  std::string original_query;
  std::string retrieval_query;
  std::string prompt_body;
  std::string final_prompt;
  std::string prompt;  // Backward-compatible alias of final_prompt.
  std::string answer;
  int prompt_tokens_before_truncation = 0;
  int prompt_tokens = 0;
  int output_tokens = 0;
  bool prompt_truncated = false;
  int64_t query_transform_ms = 0;
  int64_t retrieve_ms = 0;
  int64_t rerank_ms = 0;
  int64_t context_transform_ms = 0;
  int64_t prompt_build_ms = 0;
  int64_t generate_ms = 0;
  int64_t total_ms = 0;
  double output_tokens_per_sec = 0.0;
  std::vector<RetrievedChunk> contexts;
  // Set only when advanced_mode == Rerank: "bge" or "lexical" depending on which
  // backend RagPipeline::run() actually used. std::nullopt for every other mode.
  std::optional<std::string> reranker_backend;
};

}  // namespace rag
