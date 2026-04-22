#include "query_request_io.hpp"

#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace rag::query {

namespace {

std::string readRequiredString(const nlohmann::json& root, const char* key) {
  if (!root.contains(key) || !root.at(key).is_string()) {
    throw std::runtime_error(std::string("Missing or invalid string field: ") + key);
  }
  return root.at(key).get<std::string>();
}

std::optional<std::string> readOptionalString(const nlohmann::json& root, const char* key) {
  if (!root.contains(key)) {
    return std::nullopt;
  }
  if (!root.at(key).is_string()) {
    throw std::runtime_error(std::string("Invalid string field: ") + key);
  }
  return root.at(key).get<std::string>();
}

int readRequiredInt(const nlohmann::json& root, const char* key) {
  if (!root.contains(key) || !root.at(key).is_number_integer()) {
    throw std::runtime_error(std::string("Missing or invalid integer field: ") + key);
  }
  return root.at(key).get<int>();
}

float readRequiredFloat(const nlohmann::json& root, const char* key) {
  if (!root.contains(key) || !root.at(key).is_number()) {
    throw std::runtime_error(std::string("Missing or invalid number field: ") + key);
  }
  return root.at(key).get<float>();
}

bool isBlank(const std::string& s) {
  for (const unsigned char ch : s) {
    if (!std::isspace(ch)) {
      return false;
    }
  }
  return true;
}

std::string sanitizeIdPart(const std::string& value) {
  std::string out;
  out.reserve(value.size());
  for (const unsigned char ch : value) {
    if (std::isalnum(ch) || ch == '_' || ch == '-') {
      out.push_back(static_cast<char>(ch));
    } else {
      out.push_back('-');
    }
  }
  if (out.empty()) {
    return "na";
  }
  return out;
}

std::string buildAutoRunId(const QueryRequest& request) {
  const int temperature_milli = static_cast<int>(std::lround(static_cast<double>(request.generation_temperature) * 1000.0));
  const int top_p_milli = static_cast<int>(std::lround(static_cast<double>(request.generation_top_p) * 1000.0));

  std::ostringstream oss;
  oss << "auto_" << sanitizeIdPart(request.dataset) << "_" << sanitizeIdPart(request.question_id) << "_k"
      << request.retrieval_top_k << "_t" << temperature_milli << "_p" << top_p_milli << "_m"
      << request.generation_max_tokens;
  return oss.str();
}

std::string inferPromptTemplateId(const ModelConfig& model_cfg) {
  const std::filesystem::path prompt_path(model_cfg.prompt_system_path);
  if (!prompt_path.stem().empty()) {
    return prompt_path.stem().string();
  }
  if (!prompt_path.filename().empty()) {
    return prompt_path.filename().string();
  }
  return "prompt_template_unknown";
}

}  // namespace

QueryRequest loadRequestFromJson(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("Failed to open request JSON: " + path);
  }

  nlohmann::json root;
  in >> root;

  QueryRequest request;
  request.run_id = readOptionalString(root, "run_id").value_or("");
  request.dataset = readRequiredString(root, "dataset");
  request.question_id = readRequiredString(root, "question_id");
  request.question = readRequiredString(root, "question");
  request.prompt_template_id = readRequiredString(root, "prompt_template_id");

  if (!root.contains("retrieval") || !root.at("retrieval").is_object()) {
    throw std::runtime_error("Missing or invalid object field: retrieval");
  }
  if (!root.contains("generation") || !root.at("generation").is_object()) {
    throw std::runtime_error("Missing or invalid object field: generation");
  }

  const auto& retrieval = root.at("retrieval");
  const auto& generation = root.at("generation");

  request.retrieval_top_k = readRequiredInt(retrieval, "top_k");
  request.generation_temperature = readRequiredFloat(generation, "temperature");
  request.generation_top_p = readRequiredFloat(generation, "top_p");
  request.generation_max_tokens = readRequiredInt(generation, "max_tokens");
  if (isBlank(request.run_id)) {
    request.run_id = buildAutoRunId(request);
  }

  return request;
}

QueryRequest buildQuestionRequest(const std::string& question,
                                  const ModelConfig& model_cfg,
                                  const RetrievalConfig& retrieval_cfg) {
  QueryRequest request;
  request.run_id = "adhoc_single_run";
  request.dataset = "unknown";
  request.question_id = "single";
  request.question = question;
  request.prompt_template_id = inferPromptTemplateId(model_cfg);
  request.retrieval_top_k = retrieval_cfg.top_k;
  request.generation_temperature = model_cfg.temperature;
  request.generation_top_p = model_cfg.top_p;
  request.generation_max_tokens = model_cfg.max_new_tokens;
  return request;
}

void validateRequest(const QueryRequest& request) {
  if (request.question.empty()) {
    throw std::runtime_error("Request field question must not be empty.");
  }
  if (request.retrieval_top_k <= 0) {
    throw std::runtime_error("Request field retrieval.top_k must be > 0.");
  }
  if (request.generation_max_tokens <= 0) {
    throw std::runtime_error("Request field generation.max_tokens must be > 0.");
  }
  if (request.generation_temperature < 0.0F) {
    throw std::runtime_error("Request field generation.temperature must be >= 0.");
  }
  if (request.generation_top_p <= 0.0F || request.generation_top_p > 1.0F) {
    throw std::runtime_error("Request field generation.top_p must be in (0, 1].");
  }
}

void applyRequestToConfigs(const QueryRequest& request, ModelConfig& model_cfg, RetrievalConfig& retrieval_cfg) {
  model_cfg.temperature = request.generation_temperature;
  model_cfg.top_p = request.generation_top_p;
  model_cfg.max_new_tokens = request.generation_max_tokens;
  retrieval_cfg.top_k = request.retrieval_top_k;
}

}  // namespace rag::query
