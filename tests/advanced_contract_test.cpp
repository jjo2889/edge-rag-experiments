#include <cassert>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

#include "advanced_mode.hpp"
#include "hyde_transformer.hpp"
#include "query_request_io.hpp"

namespace {

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void expectUnsupportedAdvancedMode() {
  bool threw = false;
  try {
    (void)rag::advancedModeFromString("unknown_mode");
  } catch (const std::runtime_error&) {
    threw = true;
  }
  require(threw, "unknown advanced mode must throw");
}

std::filesystem::path writeTempJson(const std::string& name, const std::string& body) {
  const std::filesystem::path path = std::filesystem::temp_directory_path() / name;
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("failed to write temp request json");
  }
  out << body;
  return path;
}

void testAdvancedModeParser() {
  require(rag::advancedModeFromString("") == rag::AdvancedMode::None, "empty advanced mode must be none");
  require(rag::advancedModeFromString("none") == rag::AdvancedMode::None, "none mode parse failed");
  require(rag::advancedModeFromString("query_rewrite") == rag::AdvancedMode::QueryRewrite,
          "query_rewrite mode parse failed");
  require(rag::advancedModeFromString("hyde") == rag::AdvancedMode::Hyde, "hyde mode parse failed");
  require(rag::advancedModeFromString("rerank") == rag::AdvancedMode::Rerank, "rerank mode parse failed");
  require(rag::advancedModeFromString("sentence_window") == rag::AdvancedMode::SentenceWindow,
          "sentence_window mode parse failed");
  require(rag::advancedModeFromString("contextual_compression") == rag::AdvancedMode::ContextualCompression,
          "contextual_compression mode parse failed");
  require(rag::advancedModeToString(rag::AdvancedMode::Hyde) == "hyde", "hyde mode stringify failed");
  expectUnsupportedAdvancedMode();
}

void testRequestDefaultAdvancedMode() {
  const auto path = writeTempJson("rag_contract_default_request.json", R"({"question":"What is RAG?"})");
  const rag::query::QueryRequest request = rag::query::loadRequestFromJson(path.string());
  std::filesystem::remove(path);

  require(request.question == "What is RAG?", "question parse failed");
  require(request.advanced == rag::AdvancedMode::None, "missing advanced field must default to none");
}

void testRequestAdvancedMode() {
  const auto path =
      writeTempJson("rag_contract_hyde_request.json", R"({"question":"What is RAG?","advanced":"hyde"})");
  const rag::query::QueryRequest request = rag::query::loadRequestFromJson(path.string());
  std::filesystem::remove(path);

  require(request.question == "What is RAG?", "question parse failed");
  require(request.advanced == rag::AdvancedMode::Hyde, "advanced field parse failed");
}

void testRequestRejectsInvalidAdvancedMode() {
  const auto path =
      writeTempJson("rag_contract_invalid_request.json", R"({"question":"What is RAG?","advanced":"invalid"})");

  bool threw = false;
  try {
    (void)rag::query::loadRequestFromJson(path.string());
  } catch (const std::runtime_error&) {
    threw = true;
  }
  std::filesystem::remove(path);

  require(threw, "invalid advanced request field must throw");
}

void testHydePrompt() {
  const std::string query = "Who invented the frisbee?";
  const std::string prompt = rag::buildHydePrompt(query);
  require(prompt.find(query) != std::string::npos, "HyDE prompt must include query");
  require(prompt.find("[Hypothetical Document]") != std::string::npos, "HyDE prompt must request document");
}

void testHydeSanitizer() {
  const rag::HydeTransformResult result = rag::sanitizeHydeDocument("  A short retrieval document.\n", "fallback");
  require(!result.used_fallback, "valid HyDE document must not fallback");
  require(result.document == "A short retrieval document.", "HyDE sanitizer trim failed");
}

void testHydeFallback() {
  const rag::HydeTransformResult result = rag::sanitizeHydeDocument("\n \t", "fallback query");
  require(result.used_fallback, "blank HyDE document must fallback");
  require(result.document == "fallback query", "HyDE fallback text mismatch");
}

}  // namespace

int main() {
  testAdvancedModeParser();
  testRequestDefaultAdvancedMode();
  testRequestAdvancedMode();
  testRequestRejectsInvalidAdvancedMode();
  testHydePrompt();
  testHydeSanitizer();
  testHydeFallback();
  return 0;
}
