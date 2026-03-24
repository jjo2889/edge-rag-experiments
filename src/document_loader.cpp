#include "document_loader.hpp"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace rag {

namespace fs = std::filesystem;

bool DocumentLoader::isTextLikeFile(const std::string& extension) {
  return extension == ".txt" || extension == ".md" || extension == ".jsonl";
}

std::vector<Document> DocumentLoader::loadTextFile(const std::string& path, const std::string& base_dir) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("Failed to open file: " + path);
  }

  std::ostringstream oss;
  oss << in.rdbuf();
  const std::string content = oss.str();
  if (content.empty()) {
    return {};
  }

  const fs::path file_path(path);
  const fs::path rel = fs::relative(file_path, fs::path(base_dir));

  Document doc;
  doc.id = rel.string();
  doc.title = file_path.stem().string();
  doc.text = content;

  return {doc};
}

std::vector<Document> DocumentLoader::loadJsonlFile(const std::string& path, const std::string& base_dir) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("Failed to open file: " + path);
  }

  const fs::path file_path(path);
  const fs::path rel = fs::relative(file_path, fs::path(base_dir));

  std::vector<Document> docs;
  std::string line;
  size_t line_no = 0;
  while (std::getline(in, line)) {
    ++line_no;
    if (line.empty()) {
      continue;
    }

    nlohmann::json j;
    try {
      j = nlohmann::json::parse(line);
    } catch (const std::exception&) {
      continue;
    }

    std::string text;
    if (j.contains("text") && j["text"].is_string()) {
      text = j["text"].get<std::string>();
    } else if (j.contains("contents") && j["contents"].is_string()) {
      text = j["contents"].get<std::string>();
    } else if (j.contains("body") && j["body"].is_string()) {
      text = j["body"].get<std::string>();
    }

    if (text.empty()) {
      continue;
    }

    Document doc;
    if (j.contains("id") && j["id"].is_string()) {
      doc.id = j["id"].get<std::string>();
    } else {
      doc.id = rel.string() + "#" + std::to_string(line_no);
    }

    if (j.contains("title") && j["title"].is_string()) {
      doc.title = j["title"].get<std::string>();
    } else {
      doc.title = file_path.stem().string();
    }

    doc.text = std::move(text);
    docs.push_back(std::move(doc));
  }

  return docs;
}

std::vector<Document> DocumentLoader::loadFromDirectory(const std::string& directory) const {
  if (!fs::exists(directory)) {
    throw std::runtime_error("Data directory not found: " + directory);
  }

  std::vector<Document> out;

  for (const auto& entry : fs::recursive_directory_iterator(directory)) {
    if (!entry.is_regular_file()) {
      continue;
    }

    const fs::path path = entry.path();
    const std::string ext = path.extension().string();
    if (!isTextLikeFile(ext)) {
      continue;
    }

    std::vector<Document> docs;
    if (ext == ".jsonl") {
      docs = loadJsonlFile(path.string(), directory);
    } else {
      docs = loadTextFile(path.string(), directory);
    }

    for (auto& d : docs) {
      if (!d.text.empty()) {
        out.push_back(std::move(d));
      }
    }
  }

  return out;
}

}  // namespace rag
