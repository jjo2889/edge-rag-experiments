#include "config.hpp"

#include <stdexcept>

#include <yaml-cpp/yaml.h>

namespace rag {

namespace {

template <typename T>
T readOrDefault(const YAML::Node& node, const char* key, const T& default_value) {
  if (!node[key]) {
    return default_value;
  }
  return node[key].as<T>();
}

template <typename T>
T readRequired(const YAML::Node& node, const char* key) {
  if (!node[key]) {
    throw std::runtime_error(std::string("Missing required config key: ") + key);
  }
  return node[key].as<T>();
}

void rejectUnsupportedModelConfigKeys(const YAML::Node& node) {
  if (node["max_tokens"]) {
    throw std::runtime_error(
        "Unsupported config key: max_tokens. This project only supports max_new_tokens for generation length control.");
  }
}

PromptConfig readPromptConfig(const YAML::Node& node, const PromptConfig& default_value) {
  PromptConfig cfg = default_value;
  if (!node) {
    return cfg;
  }

  cfg.system_prompt_path = readOrDefault<std::string>(node, "system_prompt_path", cfg.system_prompt_path);
  cfg.context_section = readOrDefault<std::string>(node, "context_section", cfg.context_section);
  cfg.context_item_label = readOrDefault<std::string>(node, "context_item_label", cfg.context_item_label);
  cfg.query_section = readOrDefault<std::string>(node, "query_section", cfg.query_section);
  cfg.output_section = readOrDefault<std::string>(node, "output_section", cfg.output_section);
  return cfg;
}

}  // namespace

ModelConfig ConfigLoader::loadModelConfig(const std::string& path) {
  const YAML::Node root = YAML::LoadFile(path);
  rejectUnsupportedModelConfigKeys(root);

  ModelConfig cfg;
  cfg.llama_model_path = readRequired<std::string>(root, "llama_model_path");
  cfg.onnx_model_path = readRequired<std::string>(root, "onnx_model_path");
  cfg.tokenizer_json_path = readRequired<std::string>(root, "tokenizer_json_path");
  cfg.prompt_system_path = readOrDefault<std::string>(root, "prompt_system_path", cfg.prompt_system_path);

  cfg.n_gpu_layers = readOrDefault<int>(root, "n_gpu_layers", cfg.n_gpu_layers);
  cfg.n_ctx = readOrDefault<int>(root, "n_ctx", cfg.n_ctx);
  cfg.n_batch = readOrDefault<int>(root, "n_batch", cfg.n_batch);
  cfg.n_threads = readOrDefault<int>(root, "n_threads", cfg.n_threads);
  cfg.max_new_tokens = readOrDefault<int>(root, "max_new_tokens", cfg.max_new_tokens);
  cfg.temperature = readOrDefault<float>(root, "temperature", cfg.temperature);
  cfg.top_k = readOrDefault<int>(root, "top_k", cfg.top_k);
  cfg.top_p = readOrDefault<float>(root, "top_p", cfg.top_p);
  cfg.seed = readOrDefault<uint32_t>(root, "seed", cfg.seed);
  cfg.repetition_penalty = readOrDefault<float>(root, "repetition_penalty", cfg.repetition_penalty);
  cfg.repetition_penalty_last_n =
      readOrDefault<int>(root, "repetition_penalty_last_n", cfg.repetition_penalty_last_n);
  cfg.use_model_chat_template = readOrDefault<bool>(root, "use_model_chat_template", cfg.use_model_chat_template);
  if (root["stop_strings"]) {
    cfg.stop_strings = root["stop_strings"].as<std::vector<std::string>>();
  }

  return cfg;
}

RetrievalConfig ConfigLoader::loadRetrievalConfig(const std::string& path) {
  const YAML::Node root = YAML::LoadFile(path);

  RetrievalConfig cfg;
  cfg.data_dir = readOrDefault<std::string>(root, "data_dir", cfg.data_dir);
  cfg.index_path = readOrDefault<std::string>(root, "index_path", cfg.index_path);
  cfg.chunk_store_path = readOrDefault<std::string>(root, "chunk_store_path", cfg.chunk_store_path);

  cfg.chunk_size = readOrDefault<int>(root, "chunk_size", cfg.chunk_size);
  cfg.chunk_overlap = readOrDefault<int>(root, "chunk_overlap", cfg.chunk_overlap);
  cfg.top_k = readOrDefault<int>(root, "top_k", cfg.top_k);
  cfg.prompt = readPromptConfig(root["prompt"], cfg.prompt);

  return cfg;
}

}  // namespace rag
