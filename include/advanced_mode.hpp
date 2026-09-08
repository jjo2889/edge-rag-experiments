#pragma once

#include <string>

namespace rag {

enum class AdvancedMode {
  None,
  QueryRewrite,
  Hyde,
  Rerank,
  SentenceWindow,
  ContextualCompression,
};

AdvancedMode advancedModeFromString(const std::string& value);
std::string advancedModeToString(AdvancedMode mode);

// Did the mode's transform actually change anything on this query?
//
// `transform_used_fallback` only reports the defensive path — empty or otherwise
// unusable model output. A transform that hands its input straight back takes
// the success path, so a run could report 300 successful rewrites while 252 of
// them retrieved with a byte-identical query (SciFact, 2026-08-01). NoOp
// separates "ran, changed nothing" from both "ran, changed something" and
// "failed, reverted" — which matters because a no-op and a fallback produce the
// same output for opposite reasons.
enum class TechniqueEffect {
  NotApplicable,  // advanced=none, or the mode did not reach its transform
  Applied,
  NoOp,
  Fallback,
};

std::string techniqueEffectToString(TechniqueEffect effect);

}  // namespace rag
