#include "query_response_json.hpp"

#include <fstream>
#include <sstream>

namespace rag::query {

namespace {

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

nlohmann::json buildSingleRunSuccessJson(const QueryRequest& request, const RagResult& result, const double memory_mb) {
  nlohmann::json output;
  output["run_id"] = request.run_id;
  output["mode"] = "naive_rag";
  output["dataset"] = request.dataset;
  output["question_id"] = request.question_id;
  output["question"] = request.question;
  output["answer"] = result.answer;
  output["retrieved_contexts"] = buildRetrievedContexts(result);
  output["prompt_template_id"] = request.prompt_template_id;

  nlohmann::json generation;
  generation["temperature"] = request.generation_temperature;
  generation["top_p"] = request.generation_top_p;
  generation["max_tokens"] = request.generation_max_tokens;
  output["generation"] = generation;

  nlohmann::json latency_ms;
  latency_ms["total"] = result.total_ms;
  latency_ms["retrieval"] = result.retrieve_ms;
  latency_ms["prompt_build"] = result.prompt_build_ms;
  latency_ms["generation"] = result.generate_ms;

  nlohmann::json metrics;
  metrics["latency_ms"] = latency_ms;
  metrics["throughput_tokens_per_sec"] = result.output_tokens_per_sec;
  if (memory_mb >= 0.0) {
    metrics["memory_mb"] = memory_mb;
  } else {
    metrics["memory_mb"] = nullptr;
  }
  output["metrics"] = metrics;

  output["status"] = "ok";
  output["error"] = nullptr;
  return output;
}

nlohmann::json buildSingleRunErrorJson(const std::optional<QueryRequest>& request, const std::string& message) {
  nlohmann::json output;

  output["run_id"] = request ? nlohmann::json(request->run_id) : nlohmann::json(nullptr);
  output["mode"] = "naive_rag";
  output["dataset"] = request ? nlohmann::json(request->dataset) : nlohmann::json(nullptr);
  output["question_id"] = request ? nlohmann::json(request->question_id) : nlohmann::json(nullptr);
  output["question"] = request ? nlohmann::json(request->question) : nlohmann::json(nullptr);
  output["answer"] = "";
  output["retrieved_contexts"] = nlohmann::json::array();
  output["prompt_template_id"] = request ? nlohmann::json(request->prompt_template_id) : nlohmann::json(nullptr);

  nlohmann::json generation;
  generation["temperature"] = request ? nlohmann::json(request->generation_temperature) : nlohmann::json(nullptr);
  generation["top_p"] = request ? nlohmann::json(request->generation_top_p) : nlohmann::json(nullptr);
  generation["max_tokens"] = request ? nlohmann::json(request->generation_max_tokens) : nlohmann::json(nullptr);
  output["generation"] = generation;

  nlohmann::json latency_ms;
  latency_ms["total"] = 0;
  latency_ms["retrieval"] = 0;
  latency_ms["prompt_build"] = 0;
  latency_ms["generation"] = 0;

  nlohmann::json metrics;
  metrics["latency_ms"] = latency_ms;
  metrics["throughput_tokens_per_sec"] = 0.0;
  metrics["memory_mb"] = nullptr;
  output["metrics"] = metrics;

  output["status"] = "error";
  output["error"] = nlohmann::json{{"message", message}};
  return output;
}

}  // namespace rag::query
