#pragma once

#include <vector>

#include "types.hpp"

namespace rag {

struct ChunkingResult {
  std::vector<Chunk> chunks;
  std::vector<DocumentSentence> sentences;
};

class TextChunker {
 public:
  TextChunker(int chunk_size, int chunk_overlap);

  std::vector<Chunk> chunkDocuments(const std::vector<Document>& documents) const;
  ChunkingResult chunkDocumentsWithSentences(const std::vector<Document>& documents) const;

 private:
  int chunk_size_;
  int chunk_overlap_;
};

}  // namespace rag
