#pragma once

#include <cstddef>
#include <string>

#include "config.hpp"
#include "rag_pipeline.hpp"

namespace rag {

struct BeirExportOptions {
  std::string dataset;
  std::string split;
  std::string questions_path;
  std::string qrels_path;
  std::string corpus_path;
  std::string output_jsonl_path;
  int limit = -1;
};

std::string buildBeirManifestJsonPath(const std::string& output_jsonl_path);

size_t exportBeirPredictionsJsonl(const RagPipeline& pipeline,
                                  const ModelConfig& model_cfg,
                                  const RetrievalConfig& retrieval_cfg,
                                  const BeirExportOptions& options,
                                  const std::string& model_config_path,
                                  const std::string& retrieval_config_path);

}  // namespace rag
