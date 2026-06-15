#include "sentence_window.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace rag {

namespace {

bool hasSentenceRange(const Chunk& chunk) {
  return chunk.sentence_begin >= 0 && chunk.sentence_end > chunk.sentence_begin;
}

std::unordered_set<std::string> tokenizeTerms(const std::string& text) {
  std::unordered_set<std::string> terms;
  std::string current;

  auto flush = [&]() {
    if (!current.empty()) {
      terms.insert(current);
      current.clear();
    }
  };

  for (const char raw : text) {
    const unsigned char ch = static_cast<unsigned char>(raw);
    if (std::isalnum(ch) != 0) {
      current.push_back(static_cast<char>(std::tolower(ch)));
    } else {
      flush();
    }
  }
  flush();

  return terms;
}

float lexicalOverlapScore(const std::unordered_set<std::string>& query_terms,
                          const std::unordered_set<std::string>& sentence_terms) {
  if (query_terms.empty() || sentence_terms.empty()) {
    return 0.0F;
  }

  int overlap = 0;
  for (const auto& term : query_terms) {
    if (sentence_terms.count(term) != 0) {
      ++overlap;
    }
  }

  const float denom = std::sqrt(static_cast<float>(query_terms.size() * sentence_terms.size()));
  if (denom <= 0.0F) {
    return 0.0F;
  }
  return static_cast<float>(overlap) / denom;
}

std::vector<const DocumentSentence*> collectSentences(const std::vector<DocumentSentence>& sentences,
                                                      const std::string& doc_id,
                                                      const int begin,
                                                      const int end) {
  std::vector<const DocumentSentence*> out;
  for (const auto& sentence : sentences) {
    if (sentence.doc_id != doc_id || sentence.sentence_index < begin || sentence.sentence_index >= end) {
      continue;
    }
    out.push_back(&sentence);
  }

  std::sort(out.begin(), out.end(), [](const DocumentSentence* a, const DocumentSentence* b) {
    return a->sentence_index < b->sentence_index;
  });
  return out;
}

std::string joinSentences(const std::vector<const DocumentSentence*>& sentences) {
  std::ostringstream oss;
  for (size_t i = 0; i < sentences.size(); ++i) {
    if (i > 0) {
      oss << ' ';
    }
    oss << sentences[i]->text;
  }
  return oss.str();
}

}  // namespace

int selectRepresentativeSentence(const std::string& query,
                                 const RetrievedChunk& context,
                                 const std::vector<DocumentSentence>& sentences) {
  if (!hasSentenceRange(context.chunk)) {
    return -1;
  }

  const auto query_terms = tokenizeTerms(query);
  const std::vector<const DocumentSentence*> candidates =
      collectSentences(sentences, context.chunk.doc_id, context.chunk.sentence_begin, context.chunk.sentence_end);
  if (candidates.empty()) {
    return -1;
  }

  int best_index = candidates.front()->sentence_index;
  float best_score = -1.0F;
  for (const auto* sentence : candidates) {
    const float score = lexicalOverlapScore(query_terms, tokenizeTerms(sentence->text));
    if (score > best_score) {
      best_score = score;
      best_index = sentence->sentence_index;
    }
  }

  return best_index;
}

std::vector<RetrievedChunk> expandSentenceWindows(const std::string& query,
                                                  const std::vector<RetrievedChunk>& contexts,
                                                  const std::vector<DocumentSentence>& sentences,
                                                  const int window_size) {
  if (contexts.empty() || sentences.empty()) {
    return contexts;
  }

  const int safe_window_size = std::max(0, window_size);
  std::vector<RetrievedChunk> expanded;
  expanded.reserve(contexts.size());

  for (const auto& context : contexts) {
    const int selected = selectRepresentativeSentence(query, context, sentences);
    if (selected < 0) {
      expanded.push_back(context);
      continue;
    }

    const int begin = std::max(0, selected - safe_window_size);
    const int end = selected + safe_window_size + 1;
    const std::vector<const DocumentSentence*> window =
        collectSentences(sentences, context.chunk.doc_id, begin, end);
    if (window.empty()) {
      expanded.push_back(context);
      continue;
    }

    RetrievedChunk out = context;
    out.chunk.sentence_begin = window.front()->sentence_index;
    out.chunk.sentence_end = window.back()->sentence_index + 1;
    out.chunk.text = joinSentences(window);
    expanded.push_back(std::move(out));
  }

  return expanded;
}

}  // namespace rag
