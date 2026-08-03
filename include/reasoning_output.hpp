#pragma once

#include <string>

namespace rag {

// Removes the chain-of-thought block hybrid-reasoning models emit before their
// answer. Qwen3 wraps it in <think>...</think>; `/no_think` empties the block
// but still writes the markers, so raw output is literally
// "<think>\n\n</think>\n\nYes".
//
// Leaving the markers in is not cosmetic. The auxiliary generations for
// query_rewrite, hyde and contextual_compression take the first non-empty line
// of the output, which would be "<think>" — the retrieval query becomes the tag
// and the hygiene counter reports `applied`, because it did change. Downstream,
// RAGAS would score the markup as part of the answer.
//
// When reasoning runs past max_new_tokens the closing tag never arrives and the
// reply is entirely chain of thought. That yields an empty string rather than a
// truncated thought, which the sanitizers correctly treat as a fallback.
void stripReasoningBlock(std::string& text);

}  // namespace rag
