#pragma once

#include <string>
#include <vector>

#include "types.hpp"

namespace rag {

int selectRepresentativeSentence(const std::string& query,
                                 const RetrievedChunk& context,
                                 const std::vector<DocumentSentence>& sentences);

std::vector<RetrievedChunk> expandSentenceWindows(const std::string& query,
                                                  const std::vector<RetrievedChunk>& contexts,
                                                  const std::vector<DocumentSentence>& sentences,
                                                  int window_size);

}  // namespace rag
