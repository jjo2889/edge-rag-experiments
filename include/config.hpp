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

  int n_gpu_layers;
  int n_ctx;
  int n_batch;
  int n_threads;

  // Generation length control key in YAML: max_new_tokens (max_tokens is unsupported).
  int max_new_tokens;
  float temperature;
  int top_k;
  float top_p;
  uint32_t seed;
  float repetition_penalty;
  int repetition_penalty_last_n;

  // Optional generation guards for instruction/chat models.
  bool use_model_chat_template;
  std::vector<std::string> stop_strings;
};

struct PromptConfig {
  std::string system_prompt_path;
  std::string context_section;
  std::string context_item_label;
  std::string query_section;
  std::string output_section;
};

struct RetrievalConfig {
  std::string data_dir;
  std::string index_path;
  std::string chunk_store_path;

  int chunk_size;
  int chunk_overlap;
  int top_k;
  PromptConfig prompt;
};

class ConfigLoader {
 public:
  static ModelConfig loadModelConfig(const std::string& path);
  static RetrievalConfig loadRetrievalConfig(const std::string& path);
};

}  // namespace rag
