#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "advanced_mode.hpp"
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
  AdvancedMode advanced = AdvancedMode::None;
  int limit = -1;
};

std::string buildBeirManifestJsonPath(const std::string& output_jsonl_path);

// `model_load_ms`: one-time startup (index/embedding/GGUF load) measured by the
// caller; echoed into every row's trace.latency_ms.model_load (amortized over
// the batch since a single process serves all rows).
size_t exportBeirPredictionsJsonl(const RagPipeline& pipeline,
                                  const ModelConfig& model_cfg,
                                  const RetrievalConfig& retrieval_cfg,
                                  const BeirExportOptions& options,
                                  const std::string& model_config_path,
                                  const std::string& retrieval_config_path,
                                  int64_t model_load_ms);

}  // namespace rag
