#pragma once

#include <string>
#include <vector>

#include "config.hpp"
#include "types.hpp"

namespace rag {

class PromptBuilder {
 public:
  explicit PromptBuilder(PromptConfig config);

  std::string buildPrompt(const std::string& query, const std::vector<RetrievedChunk>& contexts) const;

 private:
  PromptConfig config_;
};

}  // namespace rag
