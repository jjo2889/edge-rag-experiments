#include <fstream>
#include <iomanip>
#include <iostream>
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
#include "rag_pipeline.hpp"
#include "retriever.hpp"
#include "scifact_export.hpp"
#include "types.hpp"

namespace {

struct QueryCliOptions {
  rag::app::CommonCliOptions common;
  std::string question;
  rag::ScifactExportOptions scifact_export;
  bool show_help = false;
};

void printUsage() {
  std::cout << "Usage: rag_query [--model_config path] [--retrieval_config path] [--question \"...\"]\n";
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
    if (!options.question.empty()) {
      throw std::runtime_error("Use either --question or --output_jsonl batch mode, not both.");
    }
  }
}

void printSingleQueryTrace(const rag::RagResult& result) {
  std::cerr << "[trace] original_query=" << result.original_query << '\n';
  std::cerr << "[trace] retrieval_query=" << result.retrieval_query << '\n';
  std::cerr << "[trace] prompt_tokens_before_truncation=" << result.prompt_tokens_before_truncation << '\n';
  std::cerr << "[trace] prompt_tokens_used=" << result.prompt_tokens << '\n';
  std::cerr << "[trace] prompt_truncated=" << (result.prompt_truncated ? "true" : "false") << '\n';
  std::cerr << "[trace] output_tokens=" << result.output_tokens << '\n';
  std::cerr << "[trace] retrieve_ms=" << result.retrieve_ms << '\n';
  std::cerr << "[trace] generate_ms=" << result.generate_ms << '\n';
  std::cerr << "[trace] total_ms=" << result.total_ms << '\n';
  std::cerr << "[trace] output_tokens_per_sec=" << std::fixed << std::setprecision(2) << result.output_tokens_per_sec
            << '\n';
}

void runSingleQuery(const rag::RagPipeline& pipeline, const std::string& question) {
  const rag::RagResult result = pipeline.run(question);
  printSingleQueryTrace(result);

  std::cout << "\n=== Retrieved Contexts ===\n";
  for (size_t i = 0; i < result.contexts.size(); ++i) {
    const auto& c = result.contexts[i];
    std::cout << "[" << i + 1 << "] doc=" << c.chunk.doc_id << " score=" << c.score << '\n';
    std::cout << c.chunk.text << "\n\n";
  }

  std::cout << "=== Answer ===\n";
  std::cout << result.answer << "\n\n";
}

void runInteractive(const rag::RagPipeline& pipeline) {
  std::cout << "Interactive mode. Type 'exit' to quit.\n";
  while (true) {
    std::cout << "\nquestion> " << std::flush;
    std::string line;
    if (!std::getline(std::cin, line)) {
      break;
    }
    if (line == "exit" || line == "quit") {
      break;
    }
    if (line.empty()) {
      continue;
    }
    runSingleQuery(pipeline, line);
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const QueryCliOptions cli = parseQueryCliOptions(argc, argv);
    if (cli.show_help) {
      printUsage();
      return 0;
    }
    validateQueryCliOptions(cli);

    const rag::ModelConfig model_cfg = rag::ConfigLoader::loadModelConfig(cli.common.model_config_path);
    const rag::RetrievalConfig retrieval_cfg = rag::ConfigLoader::loadRetrievalConfig(cli.common.retrieval_config_path);
    std::cerr << "[trace] model_config_path=" << cli.common.model_config_path << '\n';
    std::cerr << "[trace] retrieval_config_path=" << cli.common.retrieval_config_path << '\n';
    std::cerr << "[trace] prompt_system_path=" << model_cfg.prompt_system_path << '\n';

    std::vector<rag::Chunk> chunks = loadChunkStore(retrieval_cfg.chunk_store_path);
    rag::FaissIndex index = rag::FaissIndex::load(retrieval_cfg.index_path);

    rag::EmbeddingModel embedding_model(model_cfg.onnx_model_path, model_cfg.tokenizer_json_path);
    rag::Retriever retriever(embedding_model, std::move(index), std::move(chunks));
    rag::PromptBuilder prompt_builder;
    rag::Generator generator(model_cfg);
    rag::RagPipeline pipeline(retriever, prompt_builder, generator, retrieval_cfg.top_k);

    if (isBatchJsonlMode(cli)) {
      rag::exportScifactPredictionsJsonl(pipeline, model_cfg, retrieval_cfg, cli.scifact_export);
      return 0;
    }

    if (!cli.question.empty()) {
      runSingleQuery(pipeline, cli.question);
      return 0;
    }

    runInteractive(pipeline);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[query][error] " << e.what() << '\n';
    return 1;
  }
}
