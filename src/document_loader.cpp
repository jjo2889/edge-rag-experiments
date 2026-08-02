#include "document_loader.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <utility>

#include <nlohmann/json.hpp>

namespace rag {

namespace fs = std::filesystem;

namespace {

std::string jsonValueToString(const nlohmann::json& value) {
  if (value.is_string()) {
    return value.get<std::string>();
  }
  if (value.is_number_integer()) {
    return std::to_string(value.get<int64_t>());
  }
  if (value.is_number_unsigned()) {
    return std::to_string(value.get<uint64_t>());
  }
  if (value.is_number_float()) {
    std::ostringstream oss;
    oss << value.get<double>();
    return oss.str();
  }
  return {};
}

}  // namespace

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
  // Skipping malformed/empty lines silently meant an index could be built from a
  // fraction of the corpus and look complete. Count every reason a line was
  // dropped and report it, so an ingest against a broken file is visible.
  size_t malformed_json = 0;
  size_t missing_text = 0;
  std::unordered_set<std::string> seen_ids;

  while (std::getline(in, line)) {
    ++line_no;
    if (line.empty()) {
      continue;
    }

    nlohmann::json j;
    try {
      j = nlohmann::json::parse(line);
    } catch (const std::exception&) {
      ++malformed_json;
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
      ++missing_text;
      continue;
    }

    Document doc;
    if (j.contains("_id")) {
      doc.id = jsonValueToString(j["_id"]);
    } else if (j.contains("id")) {
      doc.id = jsonValueToString(j["id"]);
    }
    if (doc.id.empty()) {
      doc.id = rel.string() + "#" + std::to_string(line_no);
    }

    // Duplicate ids would make retrieval ambiguous and break qrels matching, so
    // reject rather than silently keeping whichever copy landed last.
    if (!seen_ids.insert(doc.id).second) {
      throw std::runtime_error("Duplicate document id '" + doc.id + "' at " + path + ":" +
                               std::to_string(line_no) +
                               ". Document ids must be unique within a corpus file.");
    }

    if (j.contains("title") && j["title"].is_string()) {
      doc.title = j["title"].get<std::string>();
    } else {
      doc.title = file_path.stem().string();
    }

    doc.text = std::move(text);
    docs.push_back(std::move(doc));
  }

  const size_t skipped = malformed_json + missing_text;
  std::cerr << "[loader] " << path << ": loaded=" << docs.size() << " skipped=" << skipped;
  if (skipped > 0) {
    std::cerr << " (malformed_json=" << malformed_json << " missing_text=" << missing_text << ")";
  }
  std::cerr << '\n';

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
