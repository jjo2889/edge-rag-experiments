#include "reasoning_output.hpp"

#include <cctype>

namespace rag {

void stripReasoningBlock(std::string& text) {
  static constexpr const char* kOpen = "<think>";
  static constexpr const char* kClose = "</think>";

  const size_t open = text.find(kOpen);
  const size_t close = text.find(kClose);

  if (close == std::string::npos) {
    if (open == std::string::npos) {
      return;
    }
    text.erase(open);
  } else {
    // A truncated reply can carry the closing tag without the opening one, so
    // cut from the start rather than requiring a matched pair.
    const size_t begin = (open != std::string::npos && open < close) ? open : 0;
    text.erase(begin, close + std::char_traits<char>::length(kClose) - begin);
  }

  size_t first = 0;
  while (first < text.size() && std::isspace(static_cast<unsigned char>(text[first])) != 0) {
    ++first;
  }
  text.erase(0, first);
}

}  // namespace rag
