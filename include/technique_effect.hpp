#pragma once

#include <string>
#include <vector>

#include "advanced_mode.hpp"
#include "types.hpp"

namespace rag {

// Classifiers for TechniqueEffect (see advanced_mode.hpp). Kept out of
// rag_pipeline.cpp so they can be tested without a model, an index, or a Pi —
// this is instrumentation whose only value is being trustworthy.
//
// Comparison is whitespace-insensitive throughout. Transforms round-trip
// through the generator, so an otherwise unchanged string can return with a
// different trailing newline or a collapsed run of spaces; treating that as a
// change would hide exactly the no-ops these exist to catch.

// query_rewrite, hyde: did the retrieval query actually change?
TechniqueEffect classifyTextTransform(bool used_fallback, const std::string& before, const std::string& after);

// rerank, sentence_window: same chunks in the same order means no change.
TechniqueEffect classifyContextTransform(bool used_fallback,
                                         const std::vector<RetrievedChunk>& before,
                                         const std::vector<RetrievedChunk>& after);

// contextual_compression: the compressor replaces N chunks with one synthetic
// chunk, so compare the concatenated text rather than the selection. A
// compressor that echoes its input kept every token — the work of
// advanced=none plus one extra generation.
TechniqueEffect classifyCompression(bool used_fallback,
                                    const std::vector<RetrievedChunk>& before,
                                    const std::vector<RetrievedChunk>& after);

}  // namespace rag
