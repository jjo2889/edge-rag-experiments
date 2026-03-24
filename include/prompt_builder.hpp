#pragma once

#include <string>
#include <vector>

#include "types.hpp"

namespace rag {

class PromptBuilder {
 public:
  std::string buildPrompt(const std::string& query, const std::vector<RetrievedChunk>& contexts) const;
};

}  // namespace rag
