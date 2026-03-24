#pragma once

#include <string>

#include "config.hpp"

struct llama_model;
struct llama_vocab;

namespace rag {

class Generator {
 public:
  explicit Generator(ModelConfig config);
  ~Generator();

  Generator(const Generator&) = delete;
  Generator& operator=(const Generator&) = delete;

  std::string generate(const std::string& prompt) const;

 private:
  ModelConfig config_;
  llama_model* model_;
  const llama_vocab* vocab_;
};

}  // namespace rag
