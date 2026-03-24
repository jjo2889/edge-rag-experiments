#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <faiss/Index.h>

namespace rag {

class FaissIndex {
 public:
  FaissIndex();
  explicit FaissIndex(int dimension);

  FaissIndex(FaissIndex&&) noexcept = default;
  FaissIndex& operator=(FaissIndex&&) noexcept = default;

  FaissIndex(const FaissIndex&) = delete;
  FaissIndex& operator=(const FaissIndex&) = delete;

  void add(const std::vector<std::vector<float>>& vectors);
  std::vector<std::pair<int64_t, float>> search(const std::vector<float>& query, int top_k) const;

  void save(const std::string& path) const;
  static FaissIndex load(const std::string& path);

  int dimension() const;
  int64_t size() const;
  bool empty() const;

 private:
  std::unique_ptr<faiss::Index> index_;
};

}  // namespace rag
