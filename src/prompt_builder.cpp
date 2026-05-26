#include "prompt_builder.hpp"

#include <sstream>
#include <utility>

namespace rag {

PromptBuilder::PromptBuilder(PromptConfig config) : config_(std::move(config)) {}

std::string PromptBuilder::buildPrompt(const std::string& query, const std::vector<RetrievedChunk>& contexts) const {
  std::ostringstream oss;
  oss << "[" << config_.context_section << "]\n";

  for (size_t i = 0; i < contexts.size(); ++i) {
    oss << config_.context_item_label << " " << (i + 1) << ":\n";
    oss << contexts[i].chunk.text << "\n\n";
  }

  oss << "[" << config_.query_section << "]\n" << query << "\n\n";
  oss << "[" << config_.output_section << "]\n";

  return oss.str();
}

}  // namespace rag
