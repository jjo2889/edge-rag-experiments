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

 private:
  const EmbeddingModel* embedding_model_;
  FaissIndex index_;
  std::vector<Chunk> chunks_;
};

}  // namespace rag
