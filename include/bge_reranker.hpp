#pragma once

#include <memory>
#include <string>
#include <vector>

#include "config.hpp"
#include "types.hpp"

namespace rag {

class BgeReranker {
 public:
  explicit BgeReranker(const RerankerConfig& config);
  ~BgeReranker();

  BgeReranker(const BgeReranker&) = delete;
  BgeReranker& operator=(const BgeReranker&) = delete;

  std::vector<RetrievedChunk> rerank(const std::string& query,
                                     const std::vector<RetrievedChunk>& candidates,
                                     int top_k) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace rag
