#pragma once

#include <cstdint>
#include <string>

namespace rag {

struct ModelConfig {
  std::string llama_model_path;
  std::string onnx_model_path;
  std::string tokenizer_json_path;

  int n_gpu_layers = 0;
  int n_ctx = 2048;
  int n_batch = 512;
  int n_threads = 4;

  int max_new_tokens = 160;
  float temperature = 0.2F;
  int top_k = 40;
  float top_p = 0.9F;
  uint32_t seed = 42;
};

struct RetrievalConfig {
  std::string data_dir = "data/raw";
  std::string index_path = "data/indexes/faiss.index";
  std::string chunk_store_path = "data/indexes/chunks.jsonl";

  int chunk_size = 256;
  int chunk_overlap = 32;
  int top_k = 3;
};

class ConfigLoader {
 public:
  static ModelConfig loadModelConfig(const std::string& path);
  static RetrievalConfig loadRetrievalConfig(const std::string& path);
};

}  // namespace rag
