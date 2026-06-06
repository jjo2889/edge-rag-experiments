#include "query_response_json.hpp"

#include <fstream>
#include <sstream>
#include <utility>

namespace rag::query {

namespace {

nlohmann::json nullableString(const std::string& value) {
  return value.empty() ? nlohmann::json(nullptr) : nlohmann::json(value);
}

nlohmann::json buildRetrievedContexts(const RagResult& result) {
  nlohmann::json contexts = nlohmann::json::array();
  for (size_t i = 0; i < result.contexts.size(); ++i) {
    const auto& item = result.contexts[i];
    nlohmann::json row;
    row["rank"] = static_cast<int>(i + 1);
    row["doc_id"] = item.chunk.doc_id;
    row["chunk_id"] = item.chunk.chunk_id;
    row["score"] = item.score;
    row["text"] = item.chunk.text;
    contexts.push_back(std::move(row));
  }
  return contexts;
}

nlohmann::json buildGenerationConfigJson(const ModelConfig& model_cfg) {
  nlohmann::json generation;
  generation["max_new_tokens"] = model_cfg.max_new_tokens;
  generation["temperature"] = model_cfg.temperature;
  generation["top_p"] = model_cfg.top_p;
  generation["top_k"] = model_cfg.top_k;
  generation["seed"] = model_cfg.seed;
  generation["repetition_penalty"] = model_cfg.repetition_penalty;
  generation["repetition_penalty_last_n"] = model_cfg.repetition_penalty_last_n;
  generation["stop_strings"] = model_cfg.stop_strings;
  return generation;
}

nlohmann::json buildRetrievalConfigJson(const RetrievalConfig& retrieval_cfg) {
  nlohmann::json retrieval;
  retrieval["top_k"] = retrieval_cfg.top_k;
  retrieval["chunk_size"] = retrieval_cfg.chunk_size;
  retrieval["chunk_overlap"] = retrieval_cfg.chunk_overlap;
  retrieval["data_dir"] = retrieval_cfg.data_dir;
  retrieval["index_path"] = retrieval_cfg.index_path;
  retrieval["chunk_store_path"] = retrieval_cfg.chunk_store_path;
  return retrieval;
}

nlohmann::json buildPromptConfigJson(const ModelConfig& model_cfg, const RetrievalConfig& retrieval_cfg) {
  nlohmann::json prompt;
  prompt["system_prompt_path"] = nullableString(model_cfg.prompt_system_path);
  prompt["use_model_chat_template"] = model_cfg.use_model_chat_template;
  prompt["prompt_builder"] = "configurable_sectioned";
  prompt["context_section"] = retrieval_cfg.prompt.context_section;
  prompt["context_item_label"] = retrieval_cfg.prompt.context_item_label;
  prompt["query_section"] = retrieval_cfg.prompt.query_section;
  prompt["output_section"] = retrieval_cfg.prompt.output_section;
  return prompt;
}

nlohmann::json buildLatencyJson(const RagResult& result) {
  nlohmann::json latency_ms;
  latency_ms["total"] = result.total_ms;
  latency_ms["retrieval"] = result.retrieve_ms;
  latency_ms["prompt_build"] = result.prompt_build_ms;
  latency_ms["generation"] = result.generate_ms;
  return latency_ms;
}

nlohmann::json buildEmptyLatencyJson() {
  nlohmann::json latency_ms;
  latency_ms["total"] = 0;
  latency_ms["retrieval"] = 0;
  latency_ms["prompt_build"] = 0;
  latency_ms["generation"] = 0;
  return latency_ms;
}

nlohmann::json buildEmptyTraceJson() {
  nlohmann::json trace;
  trace["latency_ms"] = buildEmptyLatencyJson();
  trace["output_tokens_per_sec"] = 0.0;
  trace["memory_mb"] = nullptr;
  trace["prompt_tokens_before_truncation"] = 0;
  trace["prompt_tokens"] = 0;
  trace["output_tokens"] = 0;
  trace["prompt_truncated"] = false;
  return trace;
}

}  // namespace

double getResidentMemoryMb() {
  std::ifstream in("/proc/self/status");
  if (!in) {
    return -1.0;
  }

  std::string line;
  while (std::getline(in, line)) {
    if (line.rfind("VmRSS:", 0) != 0) {
      continue;
    }

    std::istringstream iss(line.substr(6));
    long rss_kb = 0;
    iss >> rss_kb;
    if (rss_kb <= 0) {
      return -1.0;
    }
    return static_cast<double>(rss_kb) / 1024.0;
  }

  return -1.0;
}

nlohmann::json buildTraceJson(const RagResult& result, const double memory_mb) {
  nlohmann::json trace;
  trace["latency_ms"] = buildLatencyJson(result);
  trace["output_tokens_per_sec"] = result.output_tokens_per_sec;
  if (memory_mb >= 0.0) {
    trace["memory_mb"] = memory_mb;
  } else {
    trace["memory_mb"] = nullptr;
  }
  trace["prompt_tokens_before_truncation"] = result.prompt_tokens_before_truncation;
  trace["prompt_tokens"] = result.prompt_tokens;
  trace["output_tokens"] = result.output_tokens;
  trace["prompt_truncated"] = result.prompt_truncated;
  return trace;
}

nlohmann::json buildRunConfigJson(const ModelConfig& model_cfg,
                                  const RetrievalConfig& retrieval_cfg,
                                  const std::string& model_config_path,
                                  const std::string& retrieval_config_path,
                                  const AdvancedMode advanced) {
  nlohmann::json run_config;
  run_config["model_config_path"] = nullableString(model_config_path);
  run_config["retrieval_config_path"] = nullableString(retrieval_config_path);
  run_config["advanced"] = advancedModeToString(advanced);
  run_config["generation"] = buildGenerationConfigJson(model_cfg);
  run_config["retrieval"] = buildRetrievalConfigJson(retrieval_cfg);
  run_config["prompt"] = buildPromptConfigJson(model_cfg, retrieval_cfg);
  return run_config;
}

nlohmann::json buildSingleRunSuccessJson(const QueryRequest& request,
                                         const RagResult& result,
                                         const ModelConfig& model_cfg,
                                         const RetrievalConfig& retrieval_cfg,
                                         const std::string& model_config_path,
                                         const std::string& retrieval_config_path,
                                         const double memory_mb) {
  nlohmann::json output;
  output["status"] = "ok";
  output["error"] = nullptr;
  output["mode"] = "naive_rag";
  output["advanced"] = advancedModeToString(request.advanced);
  output["request"] = nlohmann::json{{"run_id", nullableString(request.run_id)},
                                     {"dataset", nullableString(request.dataset)},
                                     {"split", nullableString(request.split)},
                                     {"sample_id", nullableString(request.sample_id)},
                                     {"advanced", advancedModeToString(request.advanced)},
                                     {"question", request.question}};
  output["answer"] = result.answer;
  output["retrieved_contexts"] = buildRetrievedContexts(result);
  output["run_config"] =
      buildRunConfigJson(model_cfg, retrieval_cfg, model_config_path, retrieval_config_path, request.advanced);
  output["trace"] = buildTraceJson(result, memory_mb);
  return output;
}

nlohmann::json buildSingleRunErrorJson(const std::optional<QueryRequest>& request, const std::string& message) {
  nlohmann::json output;
  output["status"] = "error";
  output["error"] = nlohmann::json{{"message", message}};
  output["mode"] = "naive_rag";
  output["advanced"] = request ? advancedModeToString(request->advanced) : "none";
  output["request"] = nlohmann::json{
      {"run_id", request ? nullableString(request->run_id) : nlohmann::json(nullptr)},
      {"dataset", request ? nullableString(request->dataset) : nlohmann::json(nullptr)},
      {"split", request ? nullableString(request->split) : nlohmann::json(nullptr)},
      {"sample_id", request ? nullableString(request->sample_id) : nlohmann::json(nullptr)},
      {"advanced", request ? nlohmann::json(advancedModeToString(request->advanced)) : nlohmann::json("none")},
      {"question", request ? nlohmann::json(request->question) : nlohmann::json(nullptr)}};
  output["answer"] = "";
  output["retrieved_contexts"] = nlohmann::json::array();
  output["run_config"] = nullptr;
  output["trace"] = buildEmptyTraceJson();
  return output;
}

}  // namespace rag::query
