#pragma once

#include <string>
#include <vector>

#include "types.hpp"

namespace rag {

std::vector<RetrievedChunk> rerankByLexicalOverlap(const std::string& query,
                                                   const std::vector<RetrievedChunk>& candidates,
                                                   int top_k);

}  // namespace rag
