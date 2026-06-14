#include "advanced_query_transformer.hpp"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <string>

namespace rag {

namespace {

constexpr size_t kMaxRewriteChars = 256;

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

std::string firstNonEmptyLine(const std::string& text) {
  std::istringstream in(text);
  std::string line;
  while (std::getline(in, line)) {
    line = trim(line);
    if (line.empty()) {
      continue;
    }
    if (!line.empty() && line.front() == '"' && line.back() == '"' && line.size() >= 2) {
      line = trim(line.substr(1, line.size() - 2));
    }
    if (!line.empty()) {
      return line;
    }
  }
  return {};
}

}  // namespace

std::string buildQueryRewritePrompt(const std::string& query) {
  std::ostringstream oss;
  oss << "[Task]\n";
  oss << "Rewrite the query for semantic document retrieval.\n";
  oss << "Preserve the original meaning and entities.\n";
  oss << "Return only one rewritten query, with no explanation.\n\n";
  oss << "[Query]\n" << query << "\n\n";
  oss << "[Rewritten Query]\n";
  return oss.str();
}

QueryTransformResult sanitizeQueryRewrite(const std::string& generated_text, const std::string& fallback_query) {
  std::string rewritten = firstNonEmptyLine(generated_text);
  rewritten.erase(std::remove(rewritten.begin(), rewritten.end(), '\r'), rewritten.end());

  if (rewritten.empty()) {
    return QueryTransformResult{fallback_query, true};
  }
  if (rewritten.size() > kMaxRewriteChars) {
    rewritten.resize(kMaxRewriteChars);
    rewritten = trim(rewritten);
  }
  if (rewritten.empty()) {
    return QueryTransformResult{fallback_query, true};
  }

  return QueryTransformResult{rewritten, false};
}

}  // namespace rag
