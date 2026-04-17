#pragma once

#include <string>

#include "config.hpp"
#include "rag_pipeline.hpp"

namespace rag {

struct ScifactExportOptions {
  std::string questions_path;
  std::string qrels_path;
  std::string corpus_path;
  std::string output_jsonl_path;
  int limit = -1;
};

void exportScifactPredictionsJsonl(const RagPipeline& pipeline,
                                   const ModelConfig& model_cfg,
                                   const RetrievalConfig& retrieval_cfg,
                                   const ScifactExportOptions& options);

}  // namespace rag
