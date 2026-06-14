#pragma once

#include <string>

namespace rag {

struct HydeTransformResult {
  std::string document;
  bool used_fallback = false;
};

std::string buildHydePrompt(const std::string& query);
HydeTransformResult sanitizeHydeDocument(const std::string& generated_text, const std::string& fallback_query);

}  // namespace rag
