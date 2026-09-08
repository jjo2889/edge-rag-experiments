#pragma once

#include <string>
#include <vector>

#include "types.hpp"

namespace rag {

struct ContextCompressionResult {
  std::vector<RetrievedChunk> contexts;
  bool used_fallback = false;
};

// LangChain's LLMChainExtractor, which this mode reproduces, runs once per
// retrieved document and asks for a verbatim extract — not once over all of
// them asking for a summary. Two reasons the distinction matters here:
// synthesising across documents is a much harder task for a small model than
// extracting from one, and a summary invites invented text where `*AS IS*`
// forbids it. Compressing per document also keeps each doc_id, so retrieval
// metrics stay computable without a provenance side-channel.
std::string buildContextualCompressionPrompt(const std::string& query, const RetrievedChunk& context);

// Signals that the document holds nothing relevant. LangChain's prompt asks for
// this token by name; without it a small model has no defined way to say "none
// of this helps" and pads instead.
extern const char* const kCompressionNoOutput;

// `generated_text` is the model's reply for one document. The returned context
// carries the extract in place of the original text; `dropped` marks a
// NO_OUTPUT answer (remove the document from the prompt) and `used_fallback`
// marks unusable output, where the original text is kept untouched.
struct SingleContextCompression {
  RetrievedChunk context;
  bool dropped = false;
  bool used_fallback = false;
};

SingleContextCompression compressSingleContext(const std::string& generated_text, const RetrievedChunk& original);

}  // namespace rag
