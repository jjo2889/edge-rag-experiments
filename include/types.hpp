#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace rag {

struct Document {
  std::string id;
  std::string title;
  std::string text;
};

struct Chunk {
  int64_t chunk_id = -1;
  std::string doc_id;
  std::string text;
};

struct RetrievedChunk {
  Chunk chunk;
  float score = 0.0F;
};

struct RagResult {
  std::string prompt;
  std::string answer;
  std::vector<RetrievedChunk> contexts;
};

}  // namespace rag
