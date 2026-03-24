#pragma once

#include <string>
#include <vector>

#include "types.hpp"

namespace rag {

class DocumentLoader {
 public:
  std::vector<Document> loadFromDirectory(const std::string& directory) const;

 private:
  static bool isTextLikeFile(const std::string& extension);
  static std::vector<Document> loadTextFile(const std::string& path, const std::string& base_dir);
  static std::vector<Document> loadJsonlFile(const std::string& path, const std::string& base_dir);
};

}  // namespace rag
