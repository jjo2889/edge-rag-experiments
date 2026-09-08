#include "query_request_io.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

namespace rag::query {

namespace {

bool isBlank(const std::string& s) {
  for (const unsigned char ch : s) {
    if (!std::isspace(ch)) {
      return false;
    }
  }
  return true;
}

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
  const std::string value = root.at(key).get<std::string>();
  if (isBlank(value)) {
    throw std::runtime_error(std::string("Request field ") + key + " must not be blank when provided.");
  }
  return value;
}

void rejectUnsupportedRequestFields(const nlohmann::json& root) {
  static constexpr std::array<const char*, 9> allowed = {
      "run_id", "dataset", "split", "sample_id",       "question",
      "advanced", "stream", "context_source", "oracle_contexts"};

  for (auto it = root.begin(); it != root.end(); ++it) {
    const auto found = std::find_if(allowed.begin(), allowed.end(), [&](const char* key) {
      return it.key() == key;
    });
    if (found == allowed.end()) {
      throw std::runtime_error("Unsupported request field: " + it.key());
    }
  }
}

// Accepts either a bare string (text only) or {"doc_id": ..., "text": ...}.
// doc_id is optional: it only carries provenance so an oracle run still scores
// retrieval metrics, which is how you confirm gold actually reached the prompt.
std::vector<OracleContext> readOracleContexts(const nlohmann::json& root) {
  if (!root.contains("oracle_contexts")) {
    return {};
  }
  const auto& node = root.at("oracle_contexts");
  if (!node.is_array()) {
    throw std::runtime_error("Request field oracle_contexts must be an array.");
  }

  std::vector<OracleContext> contexts;
  contexts.reserve(node.size());
  for (const auto& item : node) {
    OracleContext context;
    if (item.is_string()) {
      context.text = item.get<std::string>();
    } else if (item.is_object()) {
      if (!item.contains("text") || !item.at("text").is_string()) {
        throw std::runtime_error("Each oracle_contexts object needs a string text field.");
      }
      context.text = item.at("text").get<std::string>();
      if (item.contains("doc_id")) {
        if (!item.at("doc_id").is_string()) {
          throw std::runtime_error("oracle_contexts doc_id must be a string.");
        }
        context.doc_id = item.at("doc_id").get<std::string>();
      }
    } else {
      throw std::runtime_error("Each oracle_contexts entry must be a string or an object.");
    }
    if (isBlank(context.text)) {
      throw std::runtime_error("oracle_contexts entries must not be blank.");
    }
    contexts.push_back(std::move(context));
  }
  return contexts;
}

}  // namespace

QueryRequest loadRequestFromJson(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("Failed to open request JSON: " + path);
  }

  nlohmann::json root;
  in >> root;
  if (!root.is_object()) {
    throw std::runtime_error("Request JSON must be an object.");
  }
  rejectUnsupportedRequestFields(root);

  QueryRequest request;
  request.run_id = readOptionalString(root, "run_id").value_or("");
  request.dataset = readOptionalString(root, "dataset").value_or("");
  request.split = readOptionalString(root, "split").value_or("");
  request.sample_id = readOptionalString(root, "sample_id").value_or("");
  request.question = readRequiredString(root, "question");
  request.advanced = advancedModeFromString(readOptionalString(root, "advanced").value_or(""));
  request.context_source = contextSourceFromString(readOptionalString(root, "context_source").value_or(""));
  request.oracle_contexts = readOracleContexts(root);
  return request;
}

QueryRequest buildQuestionRequest(const std::string& question) {
  QueryRequest request;
  request.question = question;
  return request;
}

void validateRequest(const QueryRequest& request) {
  if (isBlank(request.question)) {
    throw std::runtime_error("Request field question must not be blank.");
  }
  // Silently retrieving because the caller forgot the gold passages would turn
  // the ceiling condition back into an ordinary run and nothing downstream
  // could tell.
  if (request.context_source == ContextSource::Oracle && request.oracle_contexts.empty()) {
    throw std::runtime_error("context_source=oracle requires a non-empty oracle_contexts array.");
  }
  if (request.context_source != ContextSource::Oracle && !request.oracle_contexts.empty()) {
    throw std::runtime_error("oracle_contexts is only valid with context_source=oracle.");
  }
}

}  // namespace rag::query
