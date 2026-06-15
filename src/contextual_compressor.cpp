#include "contextual_compressor.hpp"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <string>
#include <utility>

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
    compressed.resize(kMaxCompressedChars);
    compressed = trim(compressed);
  }
  if (compressed.empty()) {
    return ContextCompressionResult{fallback_contexts, true};
  }

  RetrievedChunk merged;
  if (!fallback_contexts.empty()) {
    merged = fallback_contexts.front();
  }
  merged.chunk.chunk_id = -1;
  merged.chunk.doc_id = "compressed";
  merged.chunk.text = std::move(compressed);

  return ContextCompressionResult{{std::move(merged)}, false};
}

}  // namespace rag
