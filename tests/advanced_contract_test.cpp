#include <cassert>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "advanced_mode.hpp"
#include "advanced_query_transformer.hpp"
#include "contextual_compressor.hpp"
#include "hyde_transformer.hpp"
#include "query_request_io.hpp"
#include "reranker.hpp"
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

void testQueryRewritePrompt() {
  const std::string query = "0-dimensional biomaterials show inductive properties.";
  const std::string prompt = rag::buildQueryRewritePrompt(query);
  require(prompt.find(query) != std::string::npos, "rewrite prompt must include query");
  require(prompt.find("[Rewritten Query]") != std::string::npos, "rewrite prompt must request rewritten query");
}

void testQueryRewriteSanitizer() {
  const rag::QueryTransformResult rewritten =
      rag::sanitizeQueryRewrite("\n  \"biomaterials inductive properties\"  \nextra line", "fallback query");
  require(!rewritten.used_fallback, "valid rewrite must not fallback");
  require(rewritten.text == "biomaterials inductive properties", "rewrite sanitizer should keep first clean line");
}

void testQueryRewriteFallback() {
  const rag::QueryTransformResult rewritten = rag::sanitizeQueryRewrite(" \n\t\n", "fallback query");
  require(rewritten.used_fallback, "blank rewrite must fallback");
  require(rewritten.text == "fallback query", "blank rewrite fallback text mismatch");
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

rag::RetrievedChunk makeCandidate(int64_t chunk_id, const std::string& text, float dense_score) {
  rag::RetrievedChunk candidate;
  candidate.chunk.chunk_id = chunk_id;
  candidate.chunk.doc_id = "doc-" + std::to_string(chunk_id);
  candidate.chunk.text = text;
  candidate.score = dense_score;
  return candidate;
}

void testRerankByLexicalOverlap() {
  const std::vector<rag::RetrievedChunk> candidates = {
      makeCandidate(1, "unrelated cooking recipe", 0.95F),
      makeCandidate(2, "biomaterials show inductive properties", 0.60F),
      makeCandidate(3, "inductive biomaterials evidence", 0.50F),
  };

  const std::vector<rag::RetrievedChunk> reranked =
      rag::rerankByLexicalOverlap("Do biomaterials show inductive properties?", candidates, 2);

  require(reranked.size() == 2, "rerank must return top_k candidates");
  require(reranked.front().chunk.chunk_id == 2, "rerank should prefer strongest lexical overlap");
  require(reranked.front().has_rerank_score, "reranked result must mark rerank score");
  require(reranked.front().rerank_score > reranked.back().rerank_score, "rerank scores must be ordered");
}

void testRerankZeroTopK() {
  const std::vector<rag::RetrievedChunk> candidates = {makeCandidate(1, "text", 0.1F)};
  require(rag::rerankByLexicalOverlap("text", candidates, 0).empty(), "top_k <= 0 must return empty rerank");
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
  testQueryRewritePrompt();
  testQueryRewriteSanitizer();
  testQueryRewriteFallback();
  testHydePrompt();
  testHydeSanitizer();
  testHydeFallback();
  testCompressionPrompt();
  testCompressedContext();
  testCompressionFallback();
  testRerankByLexicalOverlap();
  testRerankZeroTopK();
  testChunkerBuildsSentenceMetadata();
  testRepresentativeSentenceSelection();
  testSentenceWindowExpansion();
  testSentenceWindowSizeZero();
  testSentenceWindowLegacyFallback();
  testSentenceWindowCrossDocIsolation();
  return 0;
}
