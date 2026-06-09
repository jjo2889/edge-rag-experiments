#include "text_chunker.hpp"

#include <algorithm>
#include <sstream>
#include <string>
#include <utility>
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

bool endsSentence(const std::string& word) {
  for (auto it = word.rbegin(); it != word.rend(); ++it) {
    const char ch = *it;
    if (ch == '.' || ch == '!' || ch == '?') {
      return true;
    }
    if ((ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9')) {
      return false;
    }
  }
  return false;
}

struct SentenceBuildResult {
  std::vector<DocumentSentence> sentences;
  std::vector<int> word_sentence_indices;
};

SentenceBuildResult buildSentences(const std::string& doc_id, const std::vector<std::string>& words) {
  SentenceBuildResult result;
  result.word_sentence_indices.reserve(words.size());

  std::vector<std::string> current_words;
  int sentence_index = 0;
  for (size_t i = 0; i < words.size(); ++i) {
    current_words.push_back(words[i]);
    result.word_sentence_indices.push_back(sentence_index);

    if (!endsSentence(words[i]) && i + 1 < words.size()) {
      continue;
    }

    DocumentSentence sentence;
    sentence.doc_id = doc_id;
    sentence.sentence_index = sentence_index;
    sentence.text = joinWords(current_words, 0, current_words.size());
    result.sentences.push_back(std::move(sentence));
    current_words.clear();
    ++sentence_index;
  }

  return result;
}

}  // namespace

TextChunker::TextChunker(const int chunk_size, const int chunk_overlap)
    : chunk_size_(std::max(1, chunk_size)), chunk_overlap_(std::max(0, chunk_overlap)) {
  if (chunk_overlap_ >= chunk_size_) {
    chunk_overlap_ = chunk_size_ - 1;
  }
}

std::vector<Chunk> TextChunker::chunkDocuments(const std::vector<Document>& documents) const {
  return chunkDocumentsWithSentences(documents).chunks;
}

ChunkingResult TextChunker::chunkDocumentsWithSentences(const std::vector<Document>& documents) const {
  ChunkingResult result;
  int64_t next_id = 0;

  const int step = std::max(1, chunk_size_ - chunk_overlap_);

  for (const auto& doc : documents) {
    const std::vector<std::string> words = splitByWhitespace(doc.text);
    if (words.empty()) {
      continue;
    }

    const SentenceBuildResult sentence_result = buildSentences(doc.id, words);
    result.sentences.insert(result.sentences.end(), sentence_result.sentences.begin(), sentence_result.sentences.end());

    int doc_chunk_index = 0;
    for (size_t start = 0; start < words.size(); start += static_cast<size_t>(step)) {
      const size_t end = std::min(start + static_cast<size_t>(chunk_size_), words.size());
      if (end <= start) {
        break;
      }

      Chunk chunk;
      chunk.chunk_id = next_id++;
      chunk.doc_chunk_index = doc_chunk_index++;
      chunk.sentence_begin = sentence_result.word_sentence_indices[start];
      chunk.sentence_end = sentence_result.word_sentence_indices[end - 1] + 1;
      chunk.doc_id = doc.id;
      chunk.text = joinWords(words, start, end);
      result.chunks.push_back(std::move(chunk));

      if (end == words.size()) {
        break;
      }
    }
  }

  return result;
}

}  // namespace rag
