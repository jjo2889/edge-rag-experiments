#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "config.hpp"
#include "document_loader.hpp"
#include "embedding_model.hpp"
#include "faiss_index.hpp"
#include "text_chunker.hpp"
#include "types.hpp"

namespace fs = std::filesystem;

namespace {

void printUsage() {
  std::cout << "Usage: rag_ingest [--model_config path] [--retrieval_config path]\n";
}

std::vector<rag::Chunk> saveChunkStore(const std::vector<rag::Chunk>& chunks, const std::string& path) {
  const fs::path out_path(path);
  if (out_path.has_parent_path()) {
    fs::create_directories(out_path.parent_path());
  }

  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("Failed to open chunk store path for write: " + path);
  }

  for (const auto& chunk : chunks) {
    nlohmann::json j;
    j["chunk_id"] = chunk.chunk_id;
    j["doc_id"] = chunk.doc_id;
    j["text"] = chunk.text;
    out << j.dump() << '\n';
  }
  return chunks;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::string model_config_path = "configs/model.yaml";
    std::string retrieval_config_path = "configs/retrieval.yaml";

    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--model_config" && i + 1 < argc) {
        model_config_path = argv[++i];
      } else if (arg == "--retrieval_config" && i + 1 < argc) {
        retrieval_config_path = argv[++i];
      } else if (arg == "--help" || arg == "-h") {
        printUsage();
        return 0;
      } else {
        throw std::runtime_error("Unknown argument: " + arg);
      }
    }

    const rag::ModelConfig model_cfg = rag::ConfigLoader::loadModelConfig(model_config_path);
    const rag::RetrievalConfig retrieval_cfg = rag::ConfigLoader::loadRetrievalConfig(retrieval_config_path);

    std::cout << "[ingest] loading documents from: " << retrieval_cfg.data_dir << '\n';
    rag::DocumentLoader loader;
    const std::vector<rag::Document> docs = loader.loadFromDirectory(retrieval_cfg.data_dir);
    if (docs.empty()) {
      throw std::runtime_error("No documents found in data directory.");
    }
    std::cout << "[ingest] documents: " << docs.size() << '\n';

    rag::TextChunker chunker(retrieval_cfg.chunk_size, retrieval_cfg.chunk_overlap);
    const std::vector<rag::Chunk> chunks = chunker.chunkDocuments(docs);
    if (chunks.empty()) {
      throw std::runtime_error("No chunks produced.");
    }
    std::cout << "[ingest] chunks: " << chunks.size() << '\n';

    rag::EmbeddingModel embedding_model(model_cfg.onnx_model_path, model_cfg.tokenizer_json_path);

    std::vector<std::vector<float>> embeddings;
    embeddings.reserve(chunks.size());

    for (size_t i = 0; i < chunks.size(); ++i) {
      embeddings.push_back(embedding_model.embed(chunks[i].text));
      if ((i + 1) % 50 == 0 || i + 1 == chunks.size()) {
        std::cout << "[ingest] embedded " << (i + 1) << "/" << chunks.size() << '\r' << std::flush;
      }
    }
    std::cout << '\n';

    rag::FaissIndex index;
    index.add(embeddings);
    index.save(retrieval_cfg.index_path);
    std::cout << "[ingest] faiss index saved: " << retrieval_cfg.index_path << '\n';

    saveChunkStore(chunks, retrieval_cfg.chunk_store_path);
    std::cout << "[ingest] chunk store saved: " << retrieval_cfg.chunk_store_path << '\n';
    std::cout << "[ingest] done.\n";

    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[ingest][error] " << e.what() << '\n';
    return 1;
  }
}
