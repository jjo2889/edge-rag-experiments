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

}  // namespace rag
