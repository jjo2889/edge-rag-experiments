#include "text_chunker.hpp"

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

namespace rag {

namespace {

std::vector<std::string> splitByWhitespace(const std::string& text) {
  std::istringstream iss(text);
  std::vector<std::string> words;
  std::string word;
  while (iss >> word) {
    words.push_back(word);
  }
  return words;
}

std::string joinWords(const std::vector<std::string>& words, size_t begin, size_t end) {
  std::ostringstream oss;
  for (size_t i = begin; i < end; ++i) {
    if (i > begin) {
      oss << ' ';
    }
    oss << words[i];
  }
  return oss.str();
}

}  // namespace

TextChunker::TextChunker(const int chunk_size, const int chunk_overlap)
    : chunk_size_(std::max(1, chunk_size)), chunk_overlap_(std::max(0, chunk_overlap)) {
  if (chunk_overlap_ >= chunk_size_) {
    chunk_overlap_ = chunk_size_ - 1;
  }
}

std::vector<Chunk> TextChunker::chunkDocuments(const std::vector<Document>& documents) const {
  std::vector<Chunk> chunks;
  int64_t next_id = 0;

  const int step = std::max(1, chunk_size_ - chunk_overlap_);

  for (const auto& doc : documents) {
    const std::vector<std::string> words = splitByWhitespace(doc.text);
    if (words.empty()) {
      continue;
    }

    for (size_t start = 0; start < words.size(); start += static_cast<size_t>(step)) {
      const size_t end = std::min(start + static_cast<size_t>(chunk_size_), words.size());
      if (end <= start) {
        break;
      }

      Chunk chunk;
      chunk.chunk_id = next_id++;
      chunk.doc_id = doc.id;
      chunk.text = joinWords(words, start, end);
      chunks.push_back(std::move(chunk));

      if (end == words.size()) {
        break;
      }
    }
  }

  return chunks;
}

}  // namespace rag
