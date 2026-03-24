#include "prompt_builder.hpp"

#include <sstream>

namespace rag {

std::string PromptBuilder::buildPrompt(const std::string& query, const std::vector<RetrievedChunk>& contexts) const {
  std::ostringstream oss;
  oss << "You are a question-answering assistant.\n";
  oss << "Answer using only the provided context.\n";
  oss << "If the context is insufficient, say you do not know.\n\n";
  oss << "[Context]\n";

  for (size_t i = 0; i < contexts.size(); ++i) {
    oss << "Chunk " << (i + 1) << " (doc=" << contexts[i].chunk.doc_id << ", score=" << contexts[i].score << "):\n";
    oss << contexts[i].chunk.text << "\n\n";
  }

  oss << "[Question]\n" << query << "\n\n";
  oss << "[Answer]\n";
  return oss.str();
}

}  // namespace rag
