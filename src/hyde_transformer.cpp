#include "hyde_transformer.hpp"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <string>

namespace rag {

namespace {

constexpr size_t kMaxHydeChars = 768;

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

std::string buildHydePrompt(const std::string& query) {
  std::ostringstream oss;
  oss << "[Task]\n";
  oss << "Write a short hypothetical document that would be useful for retrieving evidence for the query.\n";
  oss << "Preserve the query meaning and important entities.\n";
  oss << "Do not answer with uncertainty markers or explanations.\n";
  oss << "Return only the hypothetical document.\n\n";
  oss << "[Query]\n" << query << "\n\n";
  oss << "[Hypothetical Document]\n";
  return oss.str();
}

HydeTransformResult sanitizeHydeDocument(const std::string& generated_text, const std::string& fallback_query) {
  std::string document = trim(generated_text);
  document.erase(std::remove(document.begin(), document.end(), '\r'), document.end());

  if (document.empty()) {
    return HydeTransformResult{fallback_query, true};
  }
  if (document.size() > kMaxHydeChars) {
    document.resize(kMaxHydeChars);
    document = trim(document);
  }
  if (document.empty()) {
    return HydeTransformResult{fallback_query, true};
  }

  return HydeTransformResult{document, false};
}

}  // namespace rag
