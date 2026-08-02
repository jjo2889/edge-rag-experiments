#include "advanced_mode.hpp"

#include <stdexcept>

namespace rag {

AdvancedMode advancedModeFromString(const std::string& value) {
  if (value.empty() || value == "none") {
    return AdvancedMode::None;
  }
  if (value == "query_rewrite") {
    return AdvancedMode::QueryRewrite;
  }
  if (value == "hyde") {
    return AdvancedMode::Hyde;
  }
  if (value == "rerank") {
    return AdvancedMode::Rerank;
  }
  if (value == "sentence_window") {
    return AdvancedMode::SentenceWindow;
  }
  if (value == "contextual_compression") {
    return AdvancedMode::ContextualCompression;
  }

  throw std::runtime_error("Unsupported advanced mode: " + value);
}

std::string advancedModeToString(const AdvancedMode mode) {
  switch (mode) {
    case AdvancedMode::None:
      return "none";
    case AdvancedMode::QueryRewrite:
      return "query_rewrite";
    case AdvancedMode::Hyde:
      return "hyde";
    case AdvancedMode::Rerank:
      return "rerank";
    case AdvancedMode::SentenceWindow:
      return "sentence_window";
    case AdvancedMode::ContextualCompression:
      return "contextual_compression";
  }

  throw std::runtime_error("Unknown advanced mode enum value.");
}

std::string techniqueEffectToString(const TechniqueEffect effect) {
  switch (effect) {
    case TechniqueEffect::NotApplicable:
      return "not_applicable";
    case TechniqueEffect::Applied:
      return "applied";
    case TechniqueEffect::NoOp:
      return "noop";
    case TechniqueEffect::Fallback:
      return "fallback";
  }

  throw std::runtime_error("Unknown technique effect enum value.");
}

}  // namespace rag
