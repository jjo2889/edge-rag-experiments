#pragma once

#include <stdexcept>
#include <string>

namespace rag::app {

struct CommonCliOptions {
  std::string model_config_path = "configs/model.yaml";
  std::string retrieval_config_path = "configs/retrieval.yaml";
};

inline bool tryParseCommonConfigArg(const std::string& arg, int argc, char** argv, int& i, CommonCliOptions& options) {
  if (arg == "--model_config") {
    if (i + 1 >= argc) {
      throw std::runtime_error("Missing value for --model_config");
    }
    options.model_config_path = argv[++i];
    return true;
  }
  if (arg == "--retrieval_config") {
    if (i + 1 >= argc) {
      throw std::runtime_error("Missing value for --retrieval_config");
    }
    options.retrieval_config_path = argv[++i];
    return true;
  }

  return false;
}

}  // namespace rag::app
