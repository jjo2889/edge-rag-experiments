#include <cassert>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "advanced_mode.hpp"
#include "query_request_io.hpp"
#include "sentence_window.hpp"
#include "text_chunker.hpp"

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

rag::RetrievedChunk makeHit(const std::string& doc_id,
                            const int sentence_begin,
                            const int sentence_end,
                            const std::string& text) {
  rag::RetrievedChunk hit;
  hit.chunk.chunk_id = 1;
  hit.chunk.doc_id = doc_id;
  hit.chunk.sentence_begin = sentence_begin;
  hit.chunk.sentence_end = sentence_end;
  hit.chunk.text = text;
  hit.score = 0.9F;
  return hit;
}

std::vector<rag::DocumentSentence> sampleSentences() {
  return {
      rag::DocumentSentence{"doc-1", 0, "Cats sleep quietly."},
      rag::DocumentSentence{"doc-1", 1, "Quantum biomaterials induce bone growth."},
      rag::DocumentSentence{"doc-1", 2, "Dogs bark loudly."},
      rag::DocumentSentence{"doc-2", 0, "Biomaterials appear in another document."},
  };
}

void testChunkerBuildsSentenceMetadata() {
  const std::vector<rag::Document> docs = {
      rag::Document{"doc-1", "", "Alpha beta. Gamma delta. Epsilon zeta."},
  };

  rag::TextChunker chunker(4, 0);
  const rag::ChunkingResult result = chunker.chunkDocumentsWithSentences(docs);

  require(result.sentences.size() == 3, "chunker must split document sentences");
  require(result.chunks.size() == 2, "chunker must preserve chunking cadence");
  require(result.chunks.front().doc_chunk_index == 0, "first chunk doc index mismatch");
  require(result.chunks.front().sentence_begin == 0, "first chunk sentence_begin mismatch");
  require(result.chunks.front().sentence_end == 2, "first chunk sentence_end mismatch");
  require(result.chunks.back().sentence_begin == 2, "second chunk sentence_begin mismatch");
  require(result.chunks.back().sentence_end == 3, "second chunk sentence_end mismatch");
}

void testRepresentativeSentenceSelection() {
  const rag::RetrievedChunk hit = makeHit("doc-1", 0, 3, "full hit text");
  const int selected = rag::selectRepresentativeSentence("biomaterials bone", hit, sampleSentences());
  require(selected == 1, "representative sentence should maximize lexical overlap inside hit range");
}

void testSentenceWindowExpansion() {
  const rag::RetrievedChunk hit = makeHit("doc-1", 0, 3, "full hit text");
  const std::vector<rag::RetrievedChunk> expanded =
      rag::expandSentenceWindows("biomaterials bone", {hit}, sampleSentences(), 1);

  require(expanded.size() == 1, "sentence window must preserve hit count");
  require(expanded.front().chunk.doc_id == "doc-1", "sentence window must preserve doc_id");
  require(expanded.front().chunk.sentence_begin == 0, "sentence window begin mismatch");
  require(expanded.front().chunk.sentence_end == 3, "sentence window end mismatch");
  require(expanded.front().chunk.text ==
              "Cats sleep quietly. Quantum biomaterials induce bone growth. Dogs bark loudly.",
          "sentence window text mismatch");
}

void testSentenceWindowSizeZero() {
  const rag::RetrievedChunk hit = makeHit("doc-1", 0, 3, "full hit text");
  const std::vector<rag::RetrievedChunk> expanded =
      rag::expandSentenceWindows("biomaterials bone", {hit}, sampleSentences(), 0);

  require(expanded.size() == 1, "window_size=0 must preserve hit count");
  require(expanded.front().chunk.text == "Quantum biomaterials induce bone growth.",
          "window_size=0 must keep selected sentence only");
  require(expanded.front().chunk.sentence_begin == 1, "window_size=0 begin mismatch");
  require(expanded.front().chunk.sentence_end == 2, "window_size=0 end mismatch");
}

void testSentenceWindowLegacyFallback() {
  const rag::RetrievedChunk hit = makeHit("doc-1", -1, -1, "legacy chunk text");
  const std::vector<rag::RetrievedChunk> expanded =
      rag::expandSentenceWindows("biomaterials bone", {hit}, sampleSentences(), 1);

  require(expanded.size() == 1, "legacy fallback must preserve hit count");
  require(expanded.front().chunk.text == "legacy chunk text", "legacy fallback must preserve text");
}

void testSentenceWindowCrossDocIsolation() {
  const rag::RetrievedChunk hit = makeHit("doc-2", 0, 1, "doc two hit");
  const std::vector<rag::RetrievedChunk> expanded =
      rag::expandSentenceWindows("biomaterials", {hit}, sampleSentences(), 1);

  require(expanded.size() == 1, "cross-doc test must preserve hit count");
  require(expanded.front().chunk.doc_id == "doc-2", "cross-doc test must preserve doc_id");
  require(expanded.front().chunk.text == "Biomaterials appear in another document.",
          "sentence window must not include other documents");
}

}  // namespace

int main() {
  testAdvancedModeParser();
  testRequestDefaultAdvancedMode();
  testRequestAdvancedMode();
  testRequestRejectsInvalidAdvancedMode();
  testChunkerBuildsSentenceMetadata();
  testRepresentativeSentenceSelection();
  testSentenceWindowExpansion();
  testSentenceWindowSizeZero();
  testSentenceWindowLegacyFallback();
  testSentenceWindowCrossDocIsolation();
  return 0;
}
