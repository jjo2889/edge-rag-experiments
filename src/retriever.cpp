#include "retriever.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace rag {

Retriever::Retriever(const EmbeddingModel& embedding_model, FaissIndex index, std::vector<Chunk> chunks)
    : embedding_model_(&embedding_model), index_(std::move(index)), chunks_(std::move(chunks)) {}

std::vector<RetrievedChunk> Retriever::retrieve(const std::string& query, const int top_k) const {
  if (!embedding_model_) {
    throw std::runtime_error("Retriever has no embedding model.");
  }
  if (top_k <= 0) {
    return {};
  }

  const std::vector<float> qvec = embedding_model_->embed(query);
  const auto hits = index_.search(qvec, top_k);

  std::vector<RetrievedChunk> out;
  out.reserve(hits.size());

  for (const auto& [idx, score] : hits) {
    if (idx < 0 || idx >= static_cast<int64_t>(chunks_.size())) {
      continue;
    }
    RetrievedChunk rc;
    rc.chunk = chunks_[static_cast<size_t>(idx)];
    rc.score = score;
    out.push_back(std::move(rc));
  }

  std::sort(out.begin(), out.end(), [](const RetrievedChunk& a, const RetrievedChunk& b) {
    return a.score > b.score;
  });

  return out;
}

}  // namespace rag
