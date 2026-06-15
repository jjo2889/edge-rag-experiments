#pragma once

#include <string>

namespace rag {

struct QueryTransformResult {
  std::string text;
  bool used_fallback = false;
};

std::string buildQueryRewritePrompt(const std::string& query);
QueryTransformResult sanitizeQueryRewrite(const std::string& generated_text, const std::string& fallback_query);

}  // namespace rag
