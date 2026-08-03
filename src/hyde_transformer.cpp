#include "hyde_transformer.hpp"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <stdexcept>
#include <string>

#include "utf8_util.hpp"

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

// Remove markdown decoration that carries no retrieval signal but does get
// embedded: bold/italic runs and leading bullet or heading markers. The model
// reliably formats its hypothetical document as markdown, and "**", "*   ", "###"
// are pure noise in the vector that goes to the retriever.
std::string stripMarkdownDecoration(const std::string& line) {
  std::string out;
  out.reserve(line.size());
  for (size_t i = 0; i < line.size();) {
    if (line[i] == '*' || line[i] == '_') {
      const char marker = line[i];
      size_t run = 0;
      while (i + run < line.size() && line[i + run] == marker) {
        ++run;
      }
      i += run;
      continue;
    }
    out.push_back(line[i]);
    ++i;
  }

  out = trim(out);
  // Leading heading hashes and ordered-list numbering.
  size_t pos = 0;
  while (pos < out.size() && out[pos] == '#') {
    ++pos;
  }
  if (pos > 0 && pos < out.size() && out[pos] == ' ') {
    out = trim(out.substr(pos));
  }
  size_t digits = 0;
  while (digits < out.size() && std::isdigit(static_cast<unsigned char>(out[digits])) != 0) {
    ++digits;
  }
  if (digits > 0 && digits + 1 < out.size() && out[digits] == '.' && out[digits + 1] == ' ') {
    out = trim(out.substr(digits + 2));
  }
  return out;
}

// True for a line that is only a section marker and no content, e.g.
// "[Hypothetical Document]", "Hypothetical Document:", "Document:".
// Such a line becomes retrieval-query text otherwise.
bool isMarkerOnlyLine(const std::string& line) {
  if (line.empty()) {
    return true;
  }
  if (line.front() == '[' && line.back() == ']') {
    return true;
  }
  if (line.back() != ':') {
    return false;
  }
  // "Label:" with nothing after it, where the label is plain words.
  for (size_t i = 0; i + 1 < line.size(); ++i) {
    const char c = line[i];
    if (std::isalpha(static_cast<unsigned char>(c)) == 0 && c != ' ' && c != '-') {
      return false;
    }
  }
  return true;
}

}  // namespace

std::string buildHydePromptFromTemplate(const std::string& tmpl, const std::string& query) {
  if (tmpl.empty()) {
    return buildHydePrompt(query);
  }

  static const std::string kPlaceholder = "{query}";
  std::string out = tmpl;
  size_t pos = out.find(kPlaceholder);
  if (pos == std::string::npos) {
    // A template without the placeholder would send the same prompt for every
    // claim and every hypothetical document would be identical.
    throw std::runtime_error("HyDE prompt template must contain {query}.");
  }
  while (pos != std::string::npos) {
    out.replace(pos, kPlaceholder.size(), query);
    pos = out.find(kPlaceholder, pos + query.size());
  }
  return out;
}

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
  std::string cleaned = generated_text;
  cleaned.erase(std::remove(cleaned.begin(), cleaned.end(), '\r'), cleaned.end());

  // Drop the model's own section headers ("**Hypothetical Document:**") and
  // markdown decoration before the text becomes the retrieval query. Without
  // this the embedded "query" led with boilerplate instead of the hypothetical
  // document, which is the whole point of HyDE.
  std::istringstream in(cleaned);
  std::string line;
  std::ostringstream body;
  bool seen_content = false;
  while (std::getline(in, line)) {
    const std::string stripped = stripMarkdownDecoration(line);
    if (!seen_content && isMarkerOnlyLine(stripped)) {
      continue;
    }
    if (!seen_content && stripped.empty()) {
      continue;
    }
    if (seen_content) {
      body << '\n';
    }
    body << stripped;
    seen_content = true;
  }

  std::string document = trim(body.str());

  if (document.empty()) {
    return HydeTransformResult{fallback_query, true};
  }
  if (document.size() > kMaxHydeChars) {
    truncateUtf8(document, kMaxHydeChars);
    document = trim(document);
  }
  if (document.empty()) {
    return HydeTransformResult{fallback_query, true};
  }

  return HydeTransformResult{document, false};
}

}  // namespace rag
