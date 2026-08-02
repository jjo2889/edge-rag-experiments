#include "contextual_compressor.hpp"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <string>
#include <utility>

#include "utf8_util.hpp"

namespace rag {

namespace {

constexpr size_t kMaxCompressedChars = 1600;

std::string trim(const std::string& text) {
  size_t begin = 0;
  while (begin < text.size() && std::isspace(static_cast<unsigned char>(text[begin])) != 0) {
    ++begin;
  }

  size_t end = text.size();
  while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1])) != 0) {
    --end;
  }

  return text.substr(begin, end - begin);
}

}  // namespace

std::string buildContextualCompressionPrompt(const std::string& query, const std::vector<RetrievedChunk>& contexts) {
  std::ostringstream oss;
  oss << "[Task]\n";
  oss << "Extract only the context statements that help answer the query.\n";
  oss << "Do not add facts that are not present in the contexts.\n";
  oss << "Return a concise compressed context, not an answer.\n\n";
  oss << "[Query]\n" << query << "\n\n";
  oss << "[Contexts]\n";
  for (size_t i = 0; i < contexts.size(); ++i) {
    oss << "Context " << (i + 1) << ":\n";
    oss << contexts[i].chunk.text << "\n\n";
  }
  oss << "[Compressed Context]\n";
  return oss.str();
}

ContextCompressionResult buildCompressedContexts(const std::string& generated_text,
                                                 const std::vector<RetrievedChunk>& fallback_contexts) {
  std::string compressed = trim(generated_text);
  compressed.erase(std::remove(compressed.begin(), compressed.end(), '\r'), compressed.end());

  if (compressed.empty()) {
    return ContextCompressionResult{fallback_contexts, true};
  }
  if (compressed.size() > kMaxCompressedChars) {
    truncateUtf8(compressed, kMaxCompressedChars);
    compressed = trim(compressed);
  }
  if (compressed.empty()) {
    return ContextCompressionResult{fallback_contexts, true};
  }

  // The merged chunk is a new synthetic passage, not any retrieved document.
  // It used to inherit the top context's score and sentence range wholesale,
  // which made the inherited numbers look like real retrieval metadata for a
  // chunk that no longer corresponds to a document. Clear all of it; callers
  // that need retrieval provenance for this mode read pre_transform_contexts.
  RetrievedChunk merged;
  merged.chunk.chunk_id = -1;
  merged.chunk.doc_chunk_index = -1;
  merged.chunk.sentence_begin = -1;
  merged.chunk.sentence_end = -1;
  merged.chunk.doc_id = "compressed";
  merged.chunk.text = std::move(compressed);
  merged.score = 0.0F;
  merged.has_retrieval_score = false;
  merged.rerank_score = 0.0F;
  merged.has_rerank_score = false;

  return ContextCompressionResult{{std::move(merged)}, false};
}

}  // namespace rag
