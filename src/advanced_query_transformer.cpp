#include "advanced_query_transformer.hpp"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <string>

#include "utf8_util.hpp"

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

// Strip a leading section marker / label the model tends to echo, e.g.
// "[Rewritten Query]", "Rewritten query:", "Here is the rewritten query:".
std::string stripLeadingMarker(const std::string& line) {
  std::string out = line;
  // Whole-line bracketed marker like "[Rewritten Query]" -> empty (skip).
  if (out.size() >= 2 && out.front() == '[' && out.back() == ']') {
    return {};
  }
  // Leading "Something:" label prefix (letters/spaces before the first colon,
  // colon near the start) -> drop the label, keep the rest.
  const size_t colon = out.find(':');
  if (colon != std::string::npos && colon <= 40) {
    bool label_like = true;
    for (size_t i = 0; i < colon; ++i) {
      const char c = out[i];
      if (std::isalpha(static_cast<unsigned char>(c)) == 0 && c != ' ') {
        label_like = false;
        break;
      }
    }
    if (label_like && colon + 1 < out.size()) {
      out = trim(out.substr(colon + 1));
    } else if (label_like) {
      out.clear();
    }
  }
  return out;
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
    line = stripLeadingMarker(line);
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
    truncateUtf8(rewritten, kMaxRewriteChars);
    rewritten = trim(rewritten);
  }
  if (rewritten.empty()) {
    return QueryTransformResult{fallback_query, true};
  }

  return QueryTransformResult{rewritten, false};
}

}  // namespace rag
