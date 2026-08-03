#pragma once

#include <string>

namespace rag {

struct HydeTransformResult {
  std::string document;
  bool used_fallback = false;
};

std::string buildHydePrompt(const std::string& query);

// Renders a task-specific template, substituting every "{query}" occurrence.
// HyDE's own prompts name the document genre per dataset — SciFact asks for a
// "scientific paper passage" — which is what aims the hypothetical document at
// the right neighbourhood of the embedding space. An empty template falls back
// to the generic builder above.
std::string buildHydePromptFromTemplate(const std::string& tmpl, const std::string& query);
HydeTransformResult sanitizeHydeDocument(const std::string& generated_text, const std::string& fallback_query);

}  // namespace rag
