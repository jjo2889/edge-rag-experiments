#pragma once

#include <string>
#include <vector>

namespace rag {

// Where the generator's context comes from. The advanced modes all try to
// improve that context, so their effect can only be read against the range the
// context can move in: `ClosedBook` is the floor, `Oracle` the ceiling, and
// `Retrieved` the system as built. A delta between two retrieved runs means
// nothing without knowing that range — if closed-book already scores what
// retrieved scores, no technique can help, and if retrieved already scores what
// oracle scores, there is nothing left to improve.
enum class ContextSource {
  Retrieved,   // live retrieval — the normal pipeline
  ClosedBook,  // no context at all
  Oracle,      // gold context supplied by the caller, retrieval skipped
};

ContextSource contextSourceFromString(const std::string& value);
std::string contextSourceToString(ContextSource source);

// Gold passage supplied with an Oracle request. `doc_id` is optional and only
// carries provenance so retrieval metrics stay computable — an oracle run
// should score perfect recall, which is a useful check that the path really
// delivered gold rather than silently falling back to retrieval.
struct OracleContext {
  std::string doc_id;
  std::string text;
};

}  // namespace rag
