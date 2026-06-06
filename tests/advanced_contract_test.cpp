#include <cassert>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "advanced_mode.hpp"
#include "contextual_compressor.hpp"
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

std::vector<rag::RetrievedChunk> sampleContexts() {
  rag::RetrievedChunk context;
  context.chunk.chunk_id = 7;
  context.chunk.doc_id = "doc-1";
  context.chunk.text = "The retrieved evidence says the material is inductive.";
  context.score = 0.75F;
  return {context};
}

void testCompressionPrompt() {
  const auto contexts = sampleContexts();
  const std::string prompt = rag::buildContextualCompressionPrompt("Is the material inductive?", contexts);
  require(prompt.find("Is the material inductive?") != std::string::npos, "compression prompt must include query");
  require(prompt.find(contexts.front().chunk.text) != std::string::npos, "compression prompt must include context");
}

void testCompressedContext() {
  const auto contexts = sampleContexts();
  const rag::ContextCompressionResult result = rag::buildCompressedContexts("  Relevant evidence only.  ", contexts);
  require(!result.used_fallback, "valid compression must not fallback");
  require(result.contexts.size() == 1, "compressed context should be a single context");
  require(result.contexts.front().chunk.doc_id == "compressed", "compressed context doc_id mismatch");
  require(result.contexts.front().chunk.chunk_id == -1, "compressed context chunk_id mismatch");
  require(result.contexts.front().chunk.text == "Relevant evidence only.", "compressed context trim failed");
}

void testCompressionFallback() {
  const auto contexts = sampleContexts();
  const rag::ContextCompressionResult result = rag::buildCompressedContexts(" \n\t", contexts);
  require(result.used_fallback, "blank compression must fallback");
  require(result.contexts.size() == contexts.size(), "fallback contexts size mismatch");
  require(result.contexts.front().chunk.doc_id == "doc-1", "fallback context metadata mismatch");
}

}  // namespace

int main() {
  testAdvancedModeParser();
  testRequestDefaultAdvancedMode();
  testRequestAdvancedMode();
  testRequestRejectsInvalidAdvancedMode();
  testCompressionPrompt();
  testCompressedContext();
  testCompressionFallback();
  return 0;
}
