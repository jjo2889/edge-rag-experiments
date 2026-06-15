#include "reranker.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace rag {

namespace {

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
                          const std::unordered_set<std::string>& context_terms) {
  if (query_terms.empty() || context_terms.empty()) {
    return 0.0F;
  }

  int overlap = 0;
  for (const auto& term : query_terms) {
    if (context_terms.count(term) != 0) {
      ++overlap;
    }
  }

  const float denom = std::sqrt(static_cast<float>(query_terms.size() * context_terms.size()));
  if (denom <= 0.0F) {
    return 0.0F;
  }
  return static_cast<float>(overlap) / denom;
}

}  // namespace

std::vector<RetrievedChunk> rerankByLexicalOverlap(const std::string& query,
                                                   const std::vector<RetrievedChunk>& candidates,
                                                   const int top_k) {
  if (top_k <= 0 || candidates.empty()) {
    return {};
  }

  const auto query_terms = tokenizeTerms(query);

  struct ScoredCandidate {
    RetrievedChunk chunk;
    float rerank_score = 0.0F;
    size_t original_index = 0;
  };

  std::vector<ScoredCandidate> scored;
  scored.reserve(candidates.size());
  for (size_t i = 0; i < candidates.size(); ++i) {
    ScoredCandidate item;
    item.chunk = candidates[i];
    item.rerank_score = lexicalOverlapScore(query_terms, tokenizeTerms(candidates[i].chunk.text));
    item.original_index = i;
    scored.push_back(std::move(item));
  }

  std::stable_sort(scored.begin(), scored.end(), [](const ScoredCandidate& a, const ScoredCandidate& b) {
    if (a.rerank_score != b.rerank_score) {
      return a.rerank_score > b.rerank_score;
    }
    if (a.chunk.score != b.chunk.score) {
      return a.chunk.score > b.chunk.score;
    }
    return a.original_index < b.original_index;
  });

  std::vector<RetrievedChunk> out;
  const size_t limit = std::min(scored.size(), static_cast<size_t>(top_k));
  out.reserve(limit);
  for (size_t i = 0; i < limit; ++i) {
    scored[i].chunk.rerank_score = scored[i].rerank_score;
    scored[i].chunk.has_rerank_score = true;
    out.push_back(std::move(scored[i].chunk));
  }
  return out;
}

}  // namespace rag
