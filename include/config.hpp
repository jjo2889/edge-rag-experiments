#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace rag {

struct ModelConfig {
  std::string llama_model_path;
  std::string onnx_model_path;
  std::string tokenizer_json_path;
  std::string prompt_system_path;

  int n_gpu_layers = 0;
  int n_ctx = 2048;
  int n_batch = 512;
  int n_threads = 4;

  // Generation length control key in YAML: max_new_tokens (max_tokens is unsupported).
  int max_new_tokens = 160;
  float temperature = 0.2F;
  int top_k = 40;
  float top_p = 0.9F;
  uint32_t seed = 42;
  float repetition_penalty = 1.1F;
  int repetition_penalty_last_n = 64;

  // Optional generation guards for instruction/chat models.
  bool use_model_chat_template = false;
  std::vector<std::string> stop_strings;
};

struct PromptConfig {
  std::string system_prompt_path;
  std::string context_section = "Context";
  std::string context_item_label = "Chunk";
  std::string query_section = "Question";
  std::string output_section = "Answer";
};

struct RetrievalConfig {
  std::string data_dir = "data/raw";
  std::string index_path = "data/indexes/faiss.index";
  std::string chunk_store_path = "data/indexes/chunks.jsonl";

  int chunk_size = 256;
  int chunk_overlap = 32;
  int top_k = 3;
  PromptConfig prompt;
};

class ConfigLoader {
 public:
  static ModelConfig loadModelConfig(const std::string& path);
  static RetrievalConfig loadRetrievalConfig(const std::string& path);
};

}  // namespace rag
