#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "types.hpp"

namespace rag {

// doc_id -> that document's sentences, ordered by sentence_index.
//
// Without this, every window lookup scanned the entire corpus sentence vector
// comparing doc_id strings, twice per retrieved context. At NQ scale that scan —
// not the technique — dominated context_transform_ms, so the mode's reported cost
// measured the implementation rather than sentence windowing.
class SentenceIndex {
 public:
  SentenceIndex() = default;
  explicit SentenceIndex(const std::vector<DocumentSentence>& sentences);

  // Sentences of `doc_id` whose sentence_index is in [begin, end), ordered.
  // Empty when the document is unknown or the range selects nothing.
  std::vector<const DocumentSentence*> range(const std::string& doc_id, int begin, int end) const;

  bool empty() const { return by_doc_.empty(); }

 private:
  std::unordered_map<std::string, std::vector<const DocumentSentence*>> by_doc_;
};

int selectRepresentativeSentence(const std::string& query,
                                 const RetrievedChunk& context,
                                 const SentenceIndex& sentences);

std::vector<RetrievedChunk> expandSentenceWindows(const std::string& query,
                                                  const std::vector<RetrievedChunk>& contexts,
                                                  const SentenceIndex& sentences,
                                                  int window_size);

// Convenience overloads that build a SentenceIndex on the fly. Intended for
// tests and one-off calls; hot paths should hold a SentenceIndex.
int selectRepresentativeSentence(const std::string& query,
                                 const RetrievedChunk& context,
                                 const std::vector<DocumentSentence>& sentences);

std::vector<RetrievedChunk> expandSentenceWindows(const std::string& query,
                                                  const std::vector<RetrievedChunk>& contexts,
                                                  const std::vector<DocumentSentence>& sentences,
                                                  int window_size);

}  // namespace rag
