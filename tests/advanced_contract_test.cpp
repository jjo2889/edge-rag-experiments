#include <cassert>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "advanced_mode.hpp"
#include "advanced_query_transformer.hpp"
#include "bert_basic_tokenizer.hpp"
#include "contextual_compressor.hpp"
#include "hyde_transformer.hpp"
#include "query_request_io.hpp"
#include "reranker.hpp"
#include "sentence_window.hpp"
#include "context_source.hpp"
#include "reasoning_output.hpp"
#include "technique_effect.hpp"
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

void testCompressedContextDropsInheritedProvenance() {
  const auto contexts = sampleContexts();
  const rag::ContextCompressionResult result = rag::buildCompressedContexts("Relevant evidence only.", contexts);
  const auto& merged = result.contexts.front();
  require(!merged.has_retrieval_score, "compressed chunk must not claim a retrieval score");
  require(!merged.has_rerank_score, "compressed chunk must not inherit a rerank score");
  require(merged.chunk.sentence_begin == -1 && merged.chunk.sentence_end == -1,
          "compressed chunk must not inherit a sentence range");
  require(merged.chunk.doc_chunk_index == -1, "compressed chunk must not inherit doc_chunk_index");
}

void testSentenceWindowDeduplicatesIdenticalWindows() {
  // Two overlapping chunks of the same document that expand to the same window.
  const std::vector<rag::RetrievedChunk> hits = {makeHit("doc-1", 0, 3, "chunk a"),
                                                 makeHit("doc-1", 1, 3, "chunk b")};
  const std::vector<rag::RetrievedChunk> expanded =
      rag::expandSentenceWindows("biomaterials bone", hits, sampleSentences(), 1);

  require(expanded.size() == 1, "identical sentence windows must be deduplicated");
  require(expanded.front().chunk.sentence_begin == 0, "surviving window begin mismatch");
  require(expanded.front().chunk.sentence_end == 3, "surviving window end mismatch");
}

void testSentenceIndexRangeIsScopedToDocument() {
  const auto sentences = sampleSentences();
  const rag::SentenceIndex index(sentences);

  const auto doc1 = index.range("doc-1", 0, 3);
  require(doc1.size() == 3, "doc-1 window size mismatch");
  require(doc1.front()->sentence_index == 0 && doc1.back()->sentence_index == 2, "doc-1 window ordering mismatch");

  const auto doc2 = index.range("doc-2", 0, 3);
  require(doc2.size() == 1, "doc-2 must not pull in other documents");

  require(index.range("missing-doc", 0, 3).empty(), "unknown doc must yield an empty range");
  require(index.range("doc-1", 5, 9).empty(), "out-of-range window must be empty");
}

void testBasicTokenizerHandlesUtf8() {
  // Byte-wise tokenization used to turn each continuation byte into its own
  // "word", so `café` produced caf + two unmatchable fragments.
  const auto accented = rag::bertBasicTokenize("Caf\xC3\xA9 au lait", true);
  require(accented.size() == 3, "accented input token count mismatch");
  require(accented[0] == "cafe", "lowercase + accent strip failed for cafe");

  // Combining-mark form must normalize to the same token as the precomposed one.
  const auto decomposed = rag::bertBasicTokenize("Cafe\xCC\x81", true);
  require(decomposed.size() == 1 && decomposed[0] == "cafe", "combining mark must be stripped");

  // Curly quotes / en dash are punctuation: their own tokens, word left intact.
  const auto punctuation = rag::bertBasicTokenize("don\xE2\x80\x99t\xE2\x80\x94stop", true);
  require(punctuation.size() == 5, "unicode punctuation must split into standalone tokens");
  require(punctuation[0] == "don" && punctuation[2] == "t" && punctuation[4] == "stop",
          "unicode punctuation split mismatch");

  // CJK ideographs are one token each.
  const auto cjk = rag::bertBasicTokenize("\xE6\x97\xA5\xE6\x9C\xAC", true);
  require(cjk.size() == 2, "CJK characters must each be their own token");

  // Characters without a canonical decomposition survive unchanged, as in HF.
  const auto no_decomp = rag::bertBasicTokenize("\xC3\xB8l", true);
  require(no_decomp.size() == 1 && no_decomp[0] == "\xC3\xB8l", "non-decomposable letter must be preserved");

  // Truncated UTF-8 must not hang or emit garbage words.
  const auto truncated = rag::bertBasicTokenize("abc\xC3", true);
  require(truncated.size() == 1 && truncated[0] == "abc", "invalid trailing byte must be dropped");

  require(rag::utf8Length("caf\xC3\xA9") == 4, "utf8Length must count code points");
}

void testHydeSanitizerStripsMarkersAndMarkdown() {
  // Shape the model actually produces on the Pi (observed 2026-08-01): a bold
  // section header followed by markdown-bulleted prose.
  const std::string generated =
      "**Hypothetical Document:**\n"
      "\n"
      "*   **Question:** Does cardiac injury increase mortality?\n"
      "Cardiac troponin elevation predicted in-hospital death.";
  const rag::HydeTransformResult result = rag::sanitizeHydeDocument(generated, "fallback query");

  require(!result.used_fallback, "hyde sanitizer must not fall back on valid markdown output");
  require(result.document.rfind("Hypothetical Document", 0) != 0,
          "hyde document must not start with the model's section header");
  require(result.document.find('*') == std::string::npos, "hyde document must not keep markdown emphasis");
  require(result.document.rfind("Question: Does cardiac injury", 0) == 0,
          "hyde document must start with the generated content");
  require(result.document.find("troponin") != std::string::npos, "hyde document must keep the body text");
}

void testHydeSanitizerKeepsPlainProse() {
  const rag::HydeTransformResult result =
      rag::sanitizeHydeDocument("Cardiac injury is associated with higher mortality.", "fallback");
  require(!result.used_fallback, "plain prose must not fall back");
  require(result.document == "Cardiac injury is associated with higher mortality.",
          "plain prose must pass through unchanged");
}

void testHydeSanitizerFallsBackOnMarkerOnlyOutput() {
  const rag::HydeTransformResult result = rag::sanitizeHydeDocument("[Hypothetical Document]\n\n", "fallback query");
  require(result.used_fallback, "marker-only output must fall back to the original query");
  require(result.document == "fallback query", "fallback document mismatch");
}

void testBasicTokenizerPreservesCaseWhenNotLowercasing() {
  const auto tokens = rag::bertBasicTokenize("Caf\xC3\xA9", false);
  require(tokens.size() == 1, "no-lowercase token count mismatch");
  require(tokens[0] == "Caf\xC3\xA9", "accents must be kept when do_lower_case is false");
}


rag::RetrievedChunk chunkWithText(const std::string& doc_id, const std::string& text) {
  rag::RetrievedChunk c;
  c.chunk.doc_id = doc_id;
  c.chunk.text = text;
  return c;
}

// The defect this instrumentation exists for: sanitizeQueryRewrite returns the
// success path when the model echoes the query back, so a run reported 300
// rewrites while 252 of them retrieved with the original string.
void testEchoedRewriteIsNoOpNotSuccess() {
  const std::string query = "ALDH1 expression is associated with poorer prognosis.";
  const rag::QueryTransformResult echoed = rag::sanitizeQueryRewrite(query, query);
  require(!echoed.used_fallback, "echoing the query is not the fallback path");
  require(rag::classifyTextTransform(echoed.used_fallback, query, echoed.text) ==
              rag::TechniqueEffect::NoOp,
          "an unchanged rewrite must classify as noop, not applied");
}

void testChangedRewriteIsApplied() {
  require(rag::classifyTextTransform(false, "who wrote Dune", "Dune novel author") ==
              rag::TechniqueEffect::Applied,
          "a changed query must classify as applied");
}

// A no-op and a fallback both retrieve with the original query, so they are
// indistinguishable downstream unless the fallback flag wins here.
void testFallbackOutranksNoOp() {
  require(rag::classifyTextTransform(true, "same", "same") == rag::TechniqueEffect::Fallback,
          "fallback must not be reported as noop");
}

// Transforms round-trip through the generator; a stray newline is not a change.
void testWhitespaceOnlyDifferenceIsNoOp() {
  require(rag::classifyTextTransform(false, "a b c", "  a   b  c\n") == rag::TechniqueEffect::NoOp,
          "whitespace-only differences must not count as applied");
}

void testRerankKeepingRetrievalOrderIsNoOp() {
  const std::vector<rag::RetrievedChunk> before = {chunkWithText("d1", "alpha"), chunkWithText("d2", "beta")};
  require(rag::classifyContextTransform(false, before, before) == rag::TechniqueEffect::NoOp,
          "a reranker returning the retrieval order unchanged is a noop");

  const std::vector<rag::RetrievedChunk> reordered = {chunkWithText("d2", "beta"), chunkWithText("d1", "alpha")};
  require(rag::classifyContextTransform(false, before, reordered) == rag::TechniqueEffect::Applied,
          "a reordered selection is applied");
}

void testContextTransformDetectsSizeChange() {
  const std::vector<rag::RetrievedChunk> before = {chunkWithText("d1", "alpha")};
  const std::vector<rag::RetrievedChunk> expanded = {chunkWithText("d1", "alpha"), chunkWithText("d1", "gamma")};
  require(rag::classifyContextTransform(false, before, expanded) == rag::TechniqueEffect::Applied,
          "a sentence window that added a chunk is applied");
}

// Compression replaces N chunks with one, so a selection comparison would call
// every run "applied". Only the concatenated text can tell an echo apart.
void testCompressionEchoIsNoOp() {
  const std::vector<rag::RetrievedChunk> before = {chunkWithText("d1", "alpha"), chunkWithText("d2", "beta")};
  const std::vector<rag::RetrievedChunk> echoed = {chunkWithText("compressed", "alpha beta")};
  require(rag::classifyCompression(false, before, echoed) == rag::TechniqueEffect::NoOp,
          "a compressor echoing its input is a noop");

  const std::vector<rag::RetrievedChunk> shortened = {chunkWithText("compressed", "alpha")};
  require(rag::classifyCompression(false, before, shortened) == rag::TechniqueEffect::Applied,
          "a compressor that dropped text is applied");
}

void testTechniqueEffectStrings() {
  require(rag::techniqueEffectToString(rag::TechniqueEffect::NotApplicable) == "not_applicable", "n/a string");
  require(rag::techniqueEffectToString(rag::TechniqueEffect::Applied) == "applied", "applied string");
  require(rag::techniqueEffectToString(rag::TechniqueEffect::NoOp) == "noop", "noop string");
  require(rag::techniqueEffectToString(rag::TechniqueEffect::Fallback) == "fallback", "fallback string");
}


rag::query::QueryRequest parseRequest(const char* name, const std::string& body) {
  const auto path = writeTempJson(name, body);
  const rag::query::QueryRequest request = rag::query::loadRequestFromJson(path.string());
  std::filesystem::remove(path);
  return request;
}

bool requestRejects(const char* name, const std::string& body) {
  const auto path = writeTempJson(name, body);
  bool threw = false;
  try {
    const rag::query::QueryRequest request = rag::query::loadRequestFromJson(path.string());
    rag::query::validateRequest(request);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  std::filesystem::remove(path);
  return threw;
}

void testContextSourceDefaultsToRetrieved() {
  const auto request = parseRequest("rag_ctx_default.json", R"({"question":"q"})");
  require(request.context_source == rag::ContextSource::Retrieved,
          "a request without context_source must run the normal pipeline");
  require(request.oracle_contexts.empty(), "no oracle contexts by default");
}

void testClosedBookParses() {
  const auto request =
      parseRequest("rag_ctx_closed.json", R"({"question":"q","context_source":"closed_book"})");
  require(request.context_source == rag::ContextSource::ClosedBook, "closed_book parse failed");
}

void testOracleAcceptsStringsAndObjects() {
  const auto request = parseRequest(
      "rag_ctx_oracle.json",
      R"({"question":"q","context_source":"oracle","oracle_contexts":["plain",{"doc_id":"d7","text":"rich"}]})");
  require(request.oracle_contexts.size() == 2, "both entry shapes must parse");
  require(request.oracle_contexts[0].doc_id.empty(), "a bare string carries no provenance");
  require(request.oracle_contexts[0].text == "plain", "string entry text");
  require(request.oracle_contexts[1].doc_id == "d7", "object entry keeps doc_id");
  require(request.oracle_contexts[1].text == "rich", "object entry text");
}

// Retrieving because the caller forgot the gold passages would turn the ceiling
// condition back into an ordinary run, and no downstream column would show it.
void testOracleWithoutContextsIsRejected() {
  require(requestRejects("rag_ctx_oracle_empty.json", R"({"question":"q","context_source":"oracle"})"),
          "oracle without oracle_contexts must be rejected, not silently retrieved");
}

void testOracleContextsWithoutOracleSourceIsRejected() {
  require(requestRejects("rag_ctx_stray_oracle.json", R"({"question":"q","oracle_contexts":["x"]})"),
          "gold passages sent to a retrieved run must be rejected as contradictory");
}

void testContextSourceStrings() {
  require(rag::contextSourceToString(rag::ContextSource::Retrieved) == "retrieved", "retrieved string");
  require(rag::contextSourceToString(rag::ContextSource::ClosedBook) == "closed_book", "closed_book string");
  require(rag::contextSourceToString(rag::ContextSource::Oracle) == "oracle", "oracle string");
  require(rag::contextSourceFromString("") == rag::ContextSource::Retrieved, "empty defaults to retrieved");

  bool threw = false;
  try {
    (void)rag::contextSourceFromString("gold");
  } catch (const std::runtime_error&) {
    threw = true;
  }
  require(threw, "unknown context source must throw");
}


std::string stripped(std::string text) {
  rag::stripReasoningBlock(text);
  return text;
}

void testEmptyReasoningBlockIsRemoved() {
  require(stripped("<think>\n\n</think>\n\nSUPPORTS") == "SUPPORTS",
          "/no_think still emits the markers; the answer is what follows them");
}

// The failure that motivated this: the auxiliary generations take the first
// non-empty line, so an unstripped reply made "<think>" the retrieval query —
// and the hygiene counter called it `applied`, because it did differ.
void testReasoningMarkerNeverBecomesTheFirstLine() {
  const std::string out = stripped("<think>\n\n</think>\n\nVitamin B12 deficiency raises homocysteine.");
  require(out.rfind("<think>", 0) != 0, "reply must not start with the reasoning marker");
  require(out == "Vitamin B12 deficiency raises homocysteine.", "answer text must survive intact");
}

void testFilledReasoningBlockIsRemoved() {
  require(stripped("<think>\nWeighing the evidence...\n</think>\nNo.") == "No.",
          "a non-empty reasoning block is dropped too");
}

// Reasoning ran past max_new_tokens: there is no answer in the reply at all.
// Empty is the honest result — the sanitizers then take their fallback path.
void testUnclosedReasoningBlockLeavesNothing() {
  require(stripped("<think>\nOkay, let me think about ALDH1 and whether") == "",
          "a truncated thought must not be handed downstream as the answer");
}

void testTextWithoutReasoningIsUntouched() {
  require(stripped("The claim is supported.") == "The claim is supported.",
          "models without a reasoning mode must be unaffected");
  require(stripped("") == "", "empty stays empty");
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
  testCompressedContextDropsInheritedProvenance();
  testSentenceWindowDeduplicatesIdenticalWindows();
  testSentenceIndexRangeIsScopedToDocument();
  testBasicTokenizerHandlesUtf8();
  testBasicTokenizerPreservesCaseWhenNotLowercasing();
  testHydeSanitizerStripsMarkersAndMarkdown();
  testHydeSanitizerKeepsPlainProse();
  testHydeSanitizerFallsBackOnMarkerOnlyOutput();
  testEchoedRewriteIsNoOpNotSuccess();
  testChangedRewriteIsApplied();
  testFallbackOutranksNoOp();
  testWhitespaceOnlyDifferenceIsNoOp();
  testRerankKeepingRetrievalOrderIsNoOp();
  testContextTransformDetectsSizeChange();
  testCompressionEchoIsNoOp();
  testTechniqueEffectStrings();
  testContextSourceDefaultsToRetrieved();
  testClosedBookParses();
  testOracleAcceptsStringsAndObjects();
  testOracleWithoutContextsIsRejected();
  testOracleContextsWithoutOracleSourceIsRejected();
  testContextSourceStrings();
  testEmptyReasoningBlockIsRemoved();
  testReasoningMarkerNeverBecomesTheFirstLine();
  testFilledReasoningBlockIsRemoved();
  testUnclosedReasoningBlockLeavesNothing();
  testTextWithoutReasoningIsUntouched();
  return 0;
}
