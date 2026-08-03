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
  // Appended to every prompt this model sees, main and auxiliary alike.
  //
  // Hybrid-reasoning models need a switch ("/no_think" for Qwen3) or they spend
  // the whole token budget on chain of thought and return no answer. It belongs
  // here rather than in the dataset prompt files for two reasons: the auxiliary
  // generations pass an empty system prompt and would never see it, and the
  // dataset prompts should stay model-agnostic. Empty for models without a
  // reasoning mode.
  std::string reasoning_suppress_marker;
  std::vector<std::string> stop_strings;
};

struct PromptConfig {
  std::string system_prompt_path;
  std::string context_section;
  std::string context_item_label;
  std::string query_section;
  std::string output_section;
  // HyDE prompts are task-specific in the original work: SciFact asks for a
  // "scientific paper passage", which aims the hypothetical document at the
  // region of embedding space where paper abstracts live. A generic "write a
  // hypothetical document" loses that targeting. The file holds one template
  // with a {query} placeholder; empty falls back to the built-in generic one.
  std::string hyde_prompt_path;
};

// HyDE draws several hypothetical documents and searches with the mean of their
// embeddings and the query's. The defaults reproduce Gao et al. (arXiv:2212.10496)
// and its reference implementation: n=8, temperature 0.7, max_tokens 512.
//
// The temperature matters as much as the count. This project pins temperature
// 0.1 with a fixed seed for reproducibility, and under those settings eight
// draws are identical — the mean would equal a single document and the technique
// would quietly become N=1. Each draw therefore uses its own seed
// (base_seed + draw index) so the set stays diverse and still reproducible.
struct HydeConfig {
  int num_documents = 8;
  float temperature = 0.7F;
  int max_new_tokens = 512;
};

struct RerankerConfig {
  std::string backend;
  std::string model_path;
  std::string sentencepiece_model_path;
  int max_length;
};

struct RetrievalConfig {
  std::string data_dir;
  std::string index_path;
  std::string chunk_store_path;

  int chunk_size;
  int chunk_overlap;
  int top_k;
  int rerank_candidate_k;
  // Sentences kept on each side of the representative sentence in
  // advanced=sentence_window. Was hard-coded to 1; it is the single knob that
  // defines the technique, so it belongs in the config and in the echoed
  // run_config rather than in the source.
  int sentence_window_size;
  RerankerConfig reranker;
  PromptConfig prompt;
  HydeConfig hyde;
};

class ConfigLoader {
 public:
  static ModelConfig loadModelConfig(const std::string& path);
  static RetrievalConfig loadRetrievalConfig(const std::string& path);
};

}  // namespace rag
