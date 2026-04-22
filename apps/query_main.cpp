#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "app_cli.hpp"
#include "config.hpp"
#include "embedding_model.hpp"
#include "faiss_index.hpp"
#include "generator.hpp"
#include "prompt_builder.hpp"
#include "query_request_io.hpp"
#include "query_response_json.hpp"
#include "rag_pipeline.hpp"
#include "retriever.hpp"
#include "scifact_export.hpp"
#include "types.hpp"

namespace {

struct QueryCliOptions {
  rag::app::CommonCliOptions common;
  std::string question;
  std::string request_json_path;
  rag::ScifactExportOptions scifact_export;
  bool show_help = false;
};

void printUsage() {
  std::cout << "Usage: rag_query [--model_config path] [--retrieval_config path] --request_json path\n";
  std::cout << "       rag_query [--model_config path] [--retrieval_config path] --question \"...\"\n";
  std::cout << "       rag_query --questions path --qrels path --corpus path --output_jsonl path [--limit N]\n";
}

std::vector<rag::Chunk> loadChunkStore(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("Failed to open chunk store: " + path);
  }

  std::vector<rag::Chunk> chunks;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) {
      continue;
    }
    auto j = nlohmann::json::parse(line);
    rag::Chunk chunk;
    chunk.chunk_id = j.at("chunk_id").get<int64_t>();
    chunk.doc_id = j.at("doc_id").get<std::string>();
    chunk.text = j.at("text").get<std::string>();
    chunks.push_back(std::move(chunk));
  }
  return chunks;
}

QueryCliOptions parseQueryCliOptions(int argc, char** argv) {
  QueryCliOptions options;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (rag::app::tryParseCommonConfigArg(arg, argc, argv, i, options.common)) {
      continue;
    }

    if (arg == "--question") {
      if (i + 1 >= argc) {
        throw std::runtime_error("Missing value for --question");
      }
      options.question = argv[++i];
    } else if (arg == "--request_json") {
      if (i + 1 >= argc) {
        throw std::runtime_error("Missing value for --request_json");
      }
      options.request_json_path = argv[++i];
    } else if (arg == "--questions") {
      if (i + 1 >= argc) {
        throw std::runtime_error("Missing value for --questions");
      }
      options.scifact_export.questions_path = argv[++i];
    } else if (arg == "--qrels") {
      if (i + 1 >= argc) {
        throw std::runtime_error("Missing value for --qrels");
      }
      options.scifact_export.qrels_path = argv[++i];
    } else if (arg == "--corpus") {
      if (i + 1 >= argc) {
        throw std::runtime_error("Missing value for --corpus");
      }
      options.scifact_export.corpus_path = argv[++i];
    } else if (arg == "--output_jsonl") {
      if (i + 1 >= argc) {
        throw std::runtime_error("Missing value for --output_jsonl");
      }
      options.scifact_export.output_jsonl_path = argv[++i];
    } else if (arg == "--limit") {
      if (i + 1 >= argc) {
        throw std::runtime_error("Missing value for --limit");
      }
      options.scifact_export.limit = std::stoi(argv[++i]);
    } else if (arg == "--help" || arg == "-h") {
      options.show_help = true;
      return options;
    } else {
      throw std::runtime_error("Unknown argument: " + arg);
    }
  }

  return options;
}

bool isBatchJsonlMode(const QueryCliOptions& options) {
  return !options.scifact_export.output_jsonl_path.empty();
}

void validateQueryCliOptions(const QueryCliOptions& options) {
  const bool batch_mode = isBatchJsonlMode(options);

  if (batch_mode) {
    if (options.scifact_export.questions_path.empty() || options.scifact_export.qrels_path.empty() ||
        options.scifact_export.corpus_path.empty()) {
      throw std::runtime_error("Batch JSONL mode requires --questions, --qrels, --corpus, and --output_jsonl.");
    }
    if (!options.question.empty() || !options.request_json_path.empty()) {
      throw std::runtime_error("Batch JSONL mode cannot be combined with --question or --request_json.");
    }
    return;
  }

  if (!options.question.empty() && !options.request_json_path.empty()) {
    throw std::runtime_error("Use either --question or --request_json, not both.");
  }
  if (options.question.empty() && options.request_json_path.empty()) {
    throw std::runtime_error("Single query mode requires --request_json or --question.");
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::optional<rag::query::QueryRequest> request_for_error;

  try {
    const QueryCliOptions cli = parseQueryCliOptions(argc, argv);
    if (cli.show_help) {
      printUsage();
      return 0;
    }
    validateQueryCliOptions(cli);

    rag::ModelConfig model_cfg = rag::ConfigLoader::loadModelConfig(cli.common.model_config_path);
    rag::RetrievalConfig retrieval_cfg = rag::ConfigLoader::loadRetrievalConfig(cli.common.retrieval_config_path);

    std::cerr << "[trace] model_config_path=" << cli.common.model_config_path << '\n';
    std::cerr << "[trace] retrieval_config_path=" << cli.common.retrieval_config_path << '\n';
    std::cerr << "[trace] prompt_system_path=" << model_cfg.prompt_system_path << '\n';

    if (!isBatchJsonlMode(cli)) {
      rag::query::QueryRequest request =
          cli.request_json_path.empty() ? rag::query::buildQuestionRequest(cli.question, model_cfg, retrieval_cfg)
                                        : rag::query::loadRequestFromJson(cli.request_json_path);
      rag::query::validateRequest(request);
      request_for_error = request;
      rag::query::applyRequestToConfigs(request, model_cfg, retrieval_cfg);
    }

    std::vector<rag::Chunk> chunks = loadChunkStore(retrieval_cfg.chunk_store_path);
    rag::FaissIndex index = rag::FaissIndex::load(retrieval_cfg.index_path);

    rag::EmbeddingModel embedding_model(model_cfg.onnx_model_path, model_cfg.tokenizer_json_path);
    rag::Retriever retriever(embedding_model, std::move(index), std::move(chunks));
    rag::PromptBuilder prompt_builder;
    rag::Generator generator(model_cfg);
    rag::RagPipeline pipeline(retriever, prompt_builder, generator, retrieval_cfg.top_k);

    if (isBatchJsonlMode(cli)) {
      rag::exportScifactPredictionsJsonl(pipeline, model_cfg, retrieval_cfg, cli.scifact_export);
      nlohmann::json batch_result;
      batch_result["mode"] = "batch_export_jsonl";
      batch_result["status"] = "ok";
      batch_result["error"] = nullptr;
      batch_result["output_jsonl"] = cli.scifact_export.output_jsonl_path;
      std::cout << batch_result.dump() << '\n';
      return 0;
    }

    const rag::RagResult result = pipeline.run(request_for_error->question);
    const double memory_mb = rag::query::getResidentMemoryMb();

    const nlohmann::json output = rag::query::buildSingleRunSuccessJson(*request_for_error, result, memory_mb);
    std::cout << output.dump() << '\n';
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[query][error] " << e.what() << '\n';
    std::cout << rag::query::buildSingleRunErrorJson(request_for_error, e.what()).dump() << '\n';
    return 1;
  }
}
