#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "config.hpp"
#include "embedding_model.hpp"
#include "faiss_index.hpp"
#include "generator.hpp"
#include "prompt_builder.hpp"
#include "rag_pipeline.hpp"
#include "retriever.hpp"
#include "types.hpp"

namespace {

void printUsage() {
  std::cout << "Usage: rag_query [--model_config path] [--retrieval_config path] [--question \"...\"]\n";
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

void runSingleQuery(const rag::RagPipeline& pipeline, const std::string& question) {
  const rag::RagResult result = pipeline.run(question);

  std::cout << "\n=== Retrieved Contexts ===\n";
  for (size_t i = 0; i < result.contexts.size(); ++i) {
    const auto& c = result.contexts[i];
    std::cout << "[" << i + 1 << "] doc=" << c.chunk.doc_id << " score=" << c.score << '\n';
    std::cout << c.chunk.text << "\n\n";
  }

  std::cout << "=== Answer ===\n";
  std::cout << result.answer << "\n\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::string model_config_path = "configs/model.yaml";
    std::string retrieval_config_path = "configs/retrieval.yaml";
    std::string question;

    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--model_config" && i + 1 < argc) {
        model_config_path = argv[++i];
      } else if (arg == "--retrieval_config" && i + 1 < argc) {
        retrieval_config_path = argv[++i];
      } else if (arg == "--question" && i + 1 < argc) {
        question = argv[++i];
      } else if (arg == "--help" || arg == "-h") {
        printUsage();
        return 0;
      } else {
        throw std::runtime_error("Unknown argument: " + arg);
      }
    }

    const rag::ModelConfig model_cfg = rag::ConfigLoader::loadModelConfig(model_config_path);
    const rag::RetrievalConfig retrieval_cfg = rag::ConfigLoader::loadRetrievalConfig(retrieval_config_path);

    std::vector<rag::Chunk> chunks = loadChunkStore(retrieval_cfg.chunk_store_path);
    rag::FaissIndex index = rag::FaissIndex::load(retrieval_cfg.index_path);

    rag::EmbeddingModel embedding_model(model_cfg.onnx_model_path, model_cfg.tokenizer_json_path);
    rag::Retriever retriever(embedding_model, std::move(index), std::move(chunks));
    rag::PromptBuilder prompt_builder;
    rag::Generator generator(model_cfg);
    rag::RagPipeline pipeline(retriever, prompt_builder, generator, retrieval_cfg.top_k);

    if (!question.empty()) {
      runSingleQuery(pipeline, question);
      return 0;
    }

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

    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[query][error] " << e.what() << '\n';
    return 1;
  }
}
