#include "config.hpp"

#include <stdexcept>

#include <yaml-cpp/yaml.h>

namespace rag {

namespace {

template <typename T>
T readRequired(const YAML::Node& node, const char* key) {
  if (!node[key]) {
    throw std::runtime_error(std::string("Missing required config key: ") + key);
  }
  return node[key].as<T>();
}

template <typename T>
T readOptional(const YAML::Node& node, const char* key, const T& fallback) {
  if (!node[key]) {
    return fallback;
  }
  return node[key].as<T>();
}

void rejectUnsupportedModelConfigKeys(const YAML::Node& node) {
  if (node["max_tokens"]) {
    throw std::runtime_error(
        "Unsupported config key: max_tokens. This project only supports max_new_tokens for generation length control.");
  }
  if (node["prompt_system_path"]) {
    throw std::runtime_error(
        "Unsupported config key: prompt_system_path. Set prompt.system_prompt_path in retrieval config.");
  }
}

PromptConfig readPromptConfig(const YAML::Node& node) {
  if (!node) {
    throw std::runtime_error("Missing required config key: prompt");
  }

  PromptConfig cfg;
  cfg.system_prompt_path = readRequired<std::string>(node, "system_prompt_path");
  cfg.context_section = readRequired<std::string>(node, "context_section");
  cfg.context_item_label = readRequired<std::string>(node, "context_item_label");
  cfg.query_section = readRequired<std::string>(node, "query_section");
  cfg.output_section = readRequired<std::string>(node, "output_section");
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

  cfg.n_gpu_layers = readRequired<int>(root, "n_gpu_layers");
  cfg.n_ctx = readRequired<int>(root, "n_ctx");
  cfg.n_batch = readRequired<int>(root, "n_batch");
  cfg.n_threads = readRequired<int>(root, "n_threads");
  cfg.max_new_tokens = readRequired<int>(root, "max_new_tokens");
  cfg.temperature = readRequired<float>(root, "temperature");
  cfg.top_k = readRequired<int>(root, "top_k");
  cfg.top_p = readRequired<float>(root, "top_p");
  cfg.seed = readRequired<uint32_t>(root, "seed");
  cfg.repetition_penalty = readRequired<float>(root, "repetition_penalty");
  cfg.repetition_penalty_last_n = readRequired<int>(root, "repetition_penalty_last_n");
  cfg.use_model_chat_template = readRequired<bool>(root, "use_model_chat_template");
  cfg.stop_strings = readRequired<std::vector<std::string>>(root, "stop_strings");

  return cfg;
}

RetrievalConfig ConfigLoader::loadRetrievalConfig(const std::string& path) {
  const YAML::Node root = YAML::LoadFile(path);

  RetrievalConfig cfg;
  cfg.data_dir = readRequired<std::string>(root, "data_dir");
  cfg.index_path = readRequired<std::string>(root, "index_path");
  cfg.chunk_store_path = readRequired<std::string>(root, "chunk_store_path");

  cfg.chunk_size = readRequired<int>(root, "chunk_size");
  cfg.chunk_overlap = readRequired<int>(root, "chunk_overlap");
  cfg.top_k = readRequired<int>(root, "top_k");
  cfg.rerank_candidate_k = readOptional<int>(root, "rerank_candidate_k", cfg.top_k * 3);
  cfg.prompt = readPromptConfig(root["prompt"]);

  return cfg;
}

}  // namespace rag
