#include "faiss_index.hpp"

#include <filesystem>
#include <stdexcept>

#include <faiss/IndexFlat.h>
#include <faiss/index_io.h>

namespace rag {

FaissIndex::FaissIndex() = default;

FaissIndex::FaissIndex(const int dimension) {
  if (dimension <= 0) {
    throw std::runtime_error("FAISS dimension must be positive.");
  }
  index_ = std::make_unique<faiss::IndexFlatIP>(dimension);
}

void FaissIndex::add(const std::vector<std::vector<float>>& vectors) {
  if (vectors.empty()) {
    return;
  }

  const int dim = static_cast<int>(vectors.front().size());
  if (dim <= 0) {
    throw std::runtime_error("Embedding dimension must be positive.");
  }

  if (!index_) {
    index_ = std::make_unique<faiss::IndexFlatIP>(dim);
  }

  if (index_->d != dim) {
    throw std::runtime_error("Embedding dimension mismatch for FAISS add().");
  }

  std::vector<float> flat;
  flat.reserve(vectors.size() * static_cast<size_t>(dim));
  for (const auto& v : vectors) {
    if (static_cast<int>(v.size()) != dim) {
      throw std::runtime_error("Inconsistent embedding dimensions in batch.");
    }
    flat.insert(flat.end(), v.begin(), v.end());
  }

  index_->add(static_cast<faiss::idx_t>(vectors.size()), flat.data());
}

std::vector<std::pair<int64_t, float>> FaissIndex::search(const std::vector<float>& query, const int top_k) const {
  if (!index_ || query.empty() || top_k <= 0) {
    return {};
  }
  if (static_cast<int>(query.size()) != index_->d) {
    throw std::runtime_error("Query embedding dimension mismatch.");
  }

  std::vector<float> distances(top_k);
  std::vector<faiss::idx_t> indices(top_k);

  index_->search(1, query.data(), top_k, distances.data(), indices.data());

  std::vector<std::pair<int64_t, float>> out;
  out.reserve(top_k);
  for (int i = 0; i < top_k; ++i) {
    if (indices[i] < 0) {
      continue;
    }
    out.emplace_back(static_cast<int64_t>(indices[i]), distances[i]);
  }
  return out;
}

void FaissIndex::save(const std::string& path) const {
  if (!index_) {
    throw std::runtime_error("Cannot save empty FAISS index.");
  }
  const std::filesystem::path p(path);
  if (p.has_parent_path()) {
    std::filesystem::create_directories(p.parent_path());
  }
  faiss::write_index(index_.get(), path.c_str());
}

FaissIndex FaissIndex::load(const std::string& path) {
  FaissIndex out;
  out.index_.reset(faiss::read_index(path.c_str()));
  if (!out.index_) {
    throw std::runtime_error("Failed to load FAISS index: " + path);
  }
  return out;
}

int FaissIndex::dimension() const {
  return index_ ? index_->d : 0;
}

int64_t FaissIndex::size() const {
  return index_ ? static_cast<int64_t>(index_->ntotal) : 0;
}

bool FaissIndex::empty() const {
  return size() == 0;
}

}  // namespace rag
