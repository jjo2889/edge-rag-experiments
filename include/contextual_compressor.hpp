#pragma once

#include <string>
#include <vector>

#include "types.hpp"

namespace rag {

struct ContextCompressionResult {
  std::vector<RetrievedChunk> contexts;
  bool used_fallback = false;
};

std::string buildContextualCompressionPrompt(const std::string& query, const std::vector<RetrievedChunk>& contexts);
ContextCompressionResult buildCompressedContexts(const std::string& generated_text,
                                                 const std::vector<RetrievedChunk>& fallback_contexts);

}  // namespace rag
