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
  static constexpr std::array<const char*, 7> allowed = {"run_id",   "dataset",  "split", "sample_id",
                                                         "question", "advanced", "stream"};

  for (auto it = root.begin(); it != root.end(); ++it) {
    const auto found = std::find_if(allowed.begin(), allowed.end(), [&](const char* key) {
      return it.key() == key;
    });
    if (found == allowed.end()) {
      throw std::runtime_error("Unsupported request field: " + it.key());
    }
  }
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
}

}  // namespace rag::query
