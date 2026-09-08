#pragma once

#include <string>
#include <vector>

#include "embedding_model.hpp"
#include "faiss_index.hpp"
#include "types.hpp"

namespace rag {

class Retriever {
 public:
  Retriever(const EmbeddingModel& embedding_model, FaissIndex index, std::vector<Chunk> chunks);

  std::vector<RetrievedChunk> retrieve(const std::string& query, int top_k) const;

  // HyDE searches with the mean of the query embedding and its hypothetical
  // documents' embeddings, so it needs to hand in a vector rather than text.
  std::vector<RetrievedChunk> retrieveByVector(const std::vector<float>& embedding, int top_k) const;

  // Exposed so callers can build that mean. Returns the same vector retrieve()
  // would compute internally for the same text.
  std::vector<float> embed(const std::string& text) const;

 private:
  const EmbeddingModel* embedding_model_;
  FaissIndex index_;
  std::vector<Chunk> chunks_;
};

}  // namespace rag
