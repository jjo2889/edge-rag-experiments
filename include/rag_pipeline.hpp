#pragma once

#include <string>

#include "generator.hpp"
#include "prompt_builder.hpp"
#include "retriever.hpp"
#include "types.hpp"

namespace rag {

class RagPipeline {
 public:
  RagPipeline(const Retriever& retriever, const PromptBuilder& prompt_builder, const Generator& generator, int top_k);

  RagResult run(const std::string& query) const;

 private:
  const Retriever* retriever_;
  const PromptBuilder* prompt_builder_;
  const Generator* generator_;
  int top_k_;
};

}  // namespace rag
