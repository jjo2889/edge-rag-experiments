#include "technique_effect.hpp"

#include <cctype>

namespace rag {

namespace {

std::string normalizeForComparison(const std::string& text) {
  std::string out;
  out.reserve(text.size());
  bool pending_space = false;
  for (const unsigned char ch : text) {
    if (std::isspace(ch) != 0) {
      pending_space = !out.empty();
      continue;
    }
    if (pending_space) {
      out.push_back(' ');
      pending_space = false;
    }
    out.push_back(static_cast<char>(ch));
  }
  return out;
}

bool sameText(const std::string& a, const std::string& b) {
  return normalizeForComparison(a) == normalizeForComparison(b);
}

std::string joinContextTexts(const std::vector<RetrievedChunk>& contexts) {
  std::string joined;
  for (const auto& c : contexts) {
    joined.append(c.chunk.text);
    joined.push_back(' ');
  }
  return joined;
}

TechniqueEffect effectOf(const bool used_fallback, const bool changed) {
  if (used_fallback) {
    return TechniqueEffect::Fallback;
  }
  return changed ? TechniqueEffect::Applied : TechniqueEffect::NoOp;
}

}  // namespace

TechniqueEffect classifyTextTransform(const bool used_fallback,
                                      const std::string& before,
                                      const std::string& after) {
  return effectOf(used_fallback, !sameText(before, after));
}

TechniqueEffect classifyContextTransform(const bool used_fallback,
                                         const std::vector<RetrievedChunk>& before,
                                         const std::vector<RetrievedChunk>& after) {
  if (before.size() != after.size()) {
    return effectOf(used_fallback, true);
  }
  for (size_t i = 0; i < before.size(); ++i) {
    if (!sameText(before[i].chunk.text, after[i].chunk.text)) {
      return effectOf(used_fallback, true);
    }
  }
  return effectOf(used_fallback, false);
}

TechniqueEffect classifyCompression(const bool used_fallback,
                                    const std::vector<RetrievedChunk>& before,
                                    const std::vector<RetrievedChunk>& after) {
  return effectOf(used_fallback, !sameText(joinContextTexts(before), joinContextTexts(after)));
}

}  // namespace rag
