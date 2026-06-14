#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "advanced_mode.hpp"
#include "app_cli.hpp"
#include "beir_export.hpp"
#include "bge_reranker.hpp"
#include "config.hpp"
#include "embedding_model.hpp"
#include "faiss_index.hpp"
#include "generator.hpp"
#include "prompt_builder.hpp"
#include "query_request_io.hpp"
#include "query_response_json.hpp"
#include "rag_pipeline.hpp"
#include "retriever.hpp"
#include "types.hpp"

namespace {

struct QueryCliOptions {
  rag::app::CommonCliOptions common;
  std::string question;
  std::string request_json_path;
  rag::BeirExportOptions beir_export;
  rag::AdvancedMode advanced = rag::AdvancedMode::None;
  bool question_provided = false;
  bool advanced_provided = false;
  bool show_help = false;
};

void printUsage() {
  std::cout << "Usage: rag_query [--model_config path] [--retrieval_config path] --request_json path\n";
  std::cout << "       rag_query [--model_config path] [--retrieval_config path] [--advanced mode] --question \"...\"\n";
  std::cout << "       rag_query --dataset name --split name --questions path --qrels path --corpus path "
               "--output_jsonl path [--advanced mode] [--limit N]\n";
}

std::string buildSentenceStorePath(const std::string& chunk_store_path) {
  return chunk_store_path + ".sentences.jsonl";
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
    if (j.contains("doc_chunk_index") && j.at("doc_chunk_index").is_number_integer()) {
      chunk.doc_chunk_index = j.at("doc_chunk_index").get<int>();
    }
    if (j.contains("sentence_begin") && j.at("sentence_begin").is_number_integer()) {
      chunk.sentence_begin = j.at("sentence_begin").get<int>();
    }
    if (j.contains("sentence_end") && j.at("sentence_end").is_number_integer()) {
      chunk.sentence_end = j.at("sentence_end").get<int>();
    }
    chunk.doc_id = j.at("doc_id").get<std::string>();
    chunk.text = j.at("text").get<std::string>();
    chunks.push_back(std::move(chunk));
  }
  return chunks;
}

std::vector<rag::DocumentSentence> loadSentenceStoreIfExists(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    return {};
  }

  std::vector<rag::DocumentSentence> sentences;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) {
      continue;
    }
    auto j = nlohmann::json::parse(line);
    rag::DocumentSentence sentence;
    sentence.doc_id = j.at("doc_id").get<std::string>();
    sentence.sentence_index = j.at("sentence_index").get<int>();
    sentence.text = j.at("text").get<std::string>();
    sentences.push_back(std::move(sentence));
  }
  return sentences;
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
      options.question_provided = true;
    } else if (arg == "--advanced") {
      if (i + 1 >= argc) {
        throw std::runtime_error("Missing value for --advanced");
      }
      options.advanced = rag::advancedModeFromString(argv[++i]);
      options.advanced_provided = true;
      options.beir_export.advanced = options.advanced;
    } else if (arg == "--request_json") {
      if (i + 1 >= argc) {
        throw std::runtime_error("Missing value for --request_json");
      }
      options.request_json_path = argv[++i];
    } else if (arg == "--dataset") {
      if (i + 1 >= argc) {
        throw std::runtime_error("Missing value for --dataset");
      }
      options.beir_export.dataset = argv[++i];
    } else if (arg == "--split") {
      if (i + 1 >= argc) {
        throw std::runtime_error("Missing value for --split");
      }
      options.beir_export.split = argv[++i];
    } else if (arg == "--questions") {
      if (i + 1 >= argc) {
        throw std::runtime_error("Missing value for --questions");
      }
      options.beir_export.questions_path = argv[++i];
    } else if (arg == "--qrels") {
      if (i + 1 >= argc) {
        throw std::runtime_error("Missing value for --qrels");
      }
      options.beir_export.qrels_path = argv[++i];
    } else if (arg == "--corpus") {
      if (i + 1 >= argc) {
        throw std::runtime_error("Missing value for --corpus");
      }
      options.beir_export.corpus_path = argv[++i];
    } else if (arg == "--output_jsonl") {
      if (i + 1 >= argc) {
        throw std::runtime_error("Missing value for --output_jsonl");
      }
      options.beir_export.output_jsonl_path = argv[++i];
    } else if (arg == "--limit") {
      if (i + 1 >= argc) {
        throw std::runtime_error("Missing value for --limit");
      }
      options.beir_export.limit = std::stoi(argv[++i]);
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
  return !options.beir_export.output_jsonl_path.empty();
}

void validateQueryCliOptions(const QueryCliOptions& options) {
  const bool batch_mode = isBatchJsonlMode(options);

  if (batch_mode) {
    if (options.beir_export.dataset.empty() || options.beir_export.split.empty() ||
        options.beir_export.questions_path.empty() || options.beir_export.qrels_path.empty() ||
        options.beir_export.corpus_path.empty()) {
      throw std::runtime_error(
          "Batch JSONL mode requires --dataset, --split, --questions, --qrels, --corpus, and --output_jsonl.");
    }
    if (options.question_provided || !options.request_json_path.empty()) {
      throw std::runtime_error("Batch JSONL mode cannot be combined with --question or --request_json.");
    }
    return;
  }

  if (!options.beir_export.dataset.empty() || !options.beir_export.split.empty() ||
      !options.beir_export.questions_path.empty() || !options.beir_export.qrels_path.empty() ||
      !options.beir_export.corpus_path.empty()) {
    throw std::runtime_error(
        "Batch JSONL options require --output_jsonl and cannot be used with single query mode.");
  }

  if (options.question_provided && !options.request_json_path.empty()) {
    throw std::runtime_error("Use either --question or --request_json, not both.");
  }
  if (options.advanced_provided && !options.request_json_path.empty()) {
    throw std::runtime_error("Use advanced in request JSON when using --request_json.");
  }
  if (!options.question_provided && options.request_json_path.empty()) {
    throw std::runtime_error("Single query mode requires --request_json or --question.");
  }
}

nlohmann::json buildBatchExportErrorJson(const std::string& output_jsonl_path, const std::string& message) {
  nlohmann::json output;
  const nlohmann::json output_jsonl =
      output_jsonl_path.empty() ? nlohmann::json(nullptr) : nlohmann::json(output_jsonl_path);
  output["mode"] = "batch_export_jsonl";
  output["status"] = "error";
  output["error"] = nlohmann::json{{"message", message}};
  output["output_jsonl"] = output_jsonl;
  output["manifest_json"] = output_jsonl_path.empty()
                                ? nlohmann::json(nullptr)
                                : nlohmann::json(rag::buildBeirManifestJsonPath(output_jsonl_path));
  output["rows_written"] = 0;
  return output;
}

}  // namespace

int main(int argc, char** argv) {
  std::optional<rag::query::QueryRequest> request_for_error;
  std::string error_mode = "naive_rag";
  std::string output_jsonl_for_error;

  try {
    const QueryCliOptions cli = parseQueryCliOptions(argc, argv);
    if (cli.show_help) {
      printUsage();
      return 0;
    }
    const bool batch_mode = isBatchJsonlMode(cli);
    if (batch_mode) {
      error_mode = "batch_export_jsonl";
      output_jsonl_for_error = cli.beir_export.output_jsonl_path;
    }
    validateQueryCliOptions(cli);

    rag::ModelConfig model_cfg = rag::ConfigLoader::loadModelConfig(cli.common.model_config_path);
    rag::RetrievalConfig retrieval_cfg = rag::ConfigLoader::loadRetrievalConfig(cli.common.retrieval_config_path);
    model_cfg.prompt_system_path = retrieval_cfg.prompt.system_prompt_path;
    if (model_cfg.prompt_system_path.empty()) {
      throw std::runtime_error("Missing required config key: prompt.system_prompt_path");
    }

    std::cerr << "[trace] model_config_path=" << cli.common.model_config_path << '\n';
    std::cerr << "[trace] retrieval_config_path=" << cli.common.retrieval_config_path << '\n';
    std::cerr << "[trace] prompt_system_path=" << model_cfg.prompt_system_path << '\n';

    if (!batch_mode) {
      rag::query::QueryRequest request =
          cli.request_json_path.empty() ? rag::query::buildQuestionRequest(cli.question)
                                        : rag::query::loadRequestFromJson(cli.request_json_path);
      if (cli.request_json_path.empty()) {
        request.advanced = cli.advanced;
      }
      request_for_error = request;
      rag::query::validateRequest(*request_for_error);
    }

    std::vector<rag::Chunk> chunks = loadChunkStore(retrieval_cfg.chunk_store_path);
    std::vector<rag::DocumentSentence> sentences =
        loadSentenceStoreIfExists(buildSentenceStorePath(retrieval_cfg.chunk_store_path));
    rag::FaissIndex index = rag::FaissIndex::load(retrieval_cfg.index_path);

    rag::EmbeddingModel embedding_model(model_cfg.onnx_model_path, model_cfg.tokenizer_json_path);
    rag::Retriever retriever(embedding_model, std::move(index), std::move(chunks));
    rag::PromptBuilder prompt_builder(retrieval_cfg.prompt);
    rag::Generator generator(model_cfg);
    std::unique_ptr<rag::BgeReranker> bge_reranker;
    const bool run_uses_rerank =
        batch_mode ? cli.beir_export.advanced == rag::AdvancedMode::Rerank
                   : request_for_error && request_for_error->advanced == rag::AdvancedMode::Rerank;
    if (run_uses_rerank && retrieval_cfg.reranker.backend == "bge") {
      bge_reranker = std::make_unique<rag::BgeReranker>(retrieval_cfg.reranker);
    }
    rag::RagPipeline pipeline(retriever,
                              prompt_builder,
                              generator,
                              retrieval_cfg.top_k,
                              retrieval_cfg.rerank_candidate_k,
                              bge_reranker.get(),
                              std::move(sentences));

    if (batch_mode) {
      const size_t rows_written =
          rag::exportBeirPredictionsJsonl(pipeline,
                                          model_cfg,
                                          retrieval_cfg,
                                          cli.beir_export,
                                          cli.common.model_config_path,
                                          cli.common.retrieval_config_path);
      nlohmann::json batch_result;
      batch_result["mode"] = "batch_export_jsonl";
      batch_result["status"] = "ok";
      batch_result["error"] = nullptr;
      batch_result["dataset"] = cli.beir_export.dataset;
      batch_result["split"] = cli.beir_export.split;
      batch_result["output_jsonl"] = cli.beir_export.output_jsonl_path;
      batch_result["manifest_json"] = rag::buildBeirManifestJsonPath(cli.beir_export.output_jsonl_path);
      batch_result["rows_written"] = rows_written;
      std::cout << batch_result.dump() << '\n';
      return 0;
    }

    const rag::RagResult result = pipeline.run(request_for_error->question, request_for_error->advanced);
    const double memory_mb = rag::query::getResidentMemoryMb();

    const nlohmann::json output = rag::query::buildSingleRunSuccessJson(*request_for_error,
                                                                         result,
                                                                         model_cfg,
                                                                         retrieval_cfg,
                                                                         cli.common.model_config_path,
                                                                         cli.common.retrieval_config_path,
                                                                         memory_mb);
    std::cout << output.dump() << '\n';
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[query][error] " << e.what() << '\n';
    if (error_mode == "batch_export_jsonl") {
      std::cout << buildBatchExportErrorJson(output_jsonl_for_error, e.what()).dump() << '\n';
    } else {
      std::cout << rag::query::buildSingleRunErrorJson(request_for_error, e.what()).dump() << '\n';
    }
    return 1;
  }
}
