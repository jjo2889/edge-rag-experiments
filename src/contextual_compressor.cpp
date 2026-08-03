#include "contextual_compressor.hpp"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <string>
#include <utility>

#include "utf8_util.hpp"

namespace rag {

namespace {

constexpr size_t kMaxCompressedChars = 1600;

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

namespace {

bool containsIgnoreCase(const std::string& haystack, const std::string& needle) {
  const auto it = std::search(haystack.begin(), haystack.end(), needle.begin(), needle.end(),
                              [](const unsigned char a, const unsigned char b) {
                                return std::tolower(a) == std::tolower(b);
                              });
  return it != haystack.end();
}

}  // namespace

const char* const kCompressionNoOutput = "NO_OUTPUT";

std::string buildContextualCompressionPrompt(const std::string& query, const RetrievedChunk& context) {
  // Wording follows LangChain's LLMChainExtractor prompt. The emphasis markers
  // are load-bearing: without *AS IS* and *DO NOT* edit the model summarises,
  // and a summary can state things the passage does not.
  std::ostringstream oss;
  oss << "Given the following question and context, extract any part of the context *AS IS* that is "
         "relevant to answer the question. If none of the context is relevant return "
      << kCompressionNoOutput << ". Remember, *DO NOT* edit the extracted parts of the context.\n\n";
  oss << "> Question: " << query << "\n";
  oss << "> Context:\n";
  oss << ">>>\n" << context.chunk.text << "\n>>>\n";
  oss << "Extracted relevant parts:";
  return oss.str();
}

SingleContextCompression compressSingleContext(const std::string& generated_text,
                                               const RetrievedChunk& original) {
  std::string extracted = trim(generated_text);
  extracted.erase(std::remove(extracted.begin(), extracted.end(), '\r'), extracted.end());

  if (extracted.empty()) {
    return SingleContextCompression{original, false, true};
  }
  // Matched case-insensitively: a 270M model asked for NO_OUTPUT writes
  // "No_OUTPUT" or "no_output" often enough that an exact match would file the
  // token itself as the extracted evidence.
  if (containsIgnoreCase(extracted, kCompressionNoOutput)) {
    return SingleContextCompression{original, true, false};
  }
  if (extracted.size() > kMaxCompressedChars) {
    truncateUtf8(extracted, kMaxCompressedChars);
    extracted = trim(extracted);
  }
  if (extracted.empty()) {
    return SingleContextCompression{original, false, true};
  }

  // Only the text changes. doc_id, scores and sentence range stay, so this mode
  // needs no provenance side-channel to keep retrieval metrics computable.
  RetrievedChunk compressed = original;
  compressed.chunk.text = std::move(extracted);
  return SingleContextCompression{std::move(compressed), false, false};
}

}  // namespace rag
