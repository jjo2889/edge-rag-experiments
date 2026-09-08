#pragma once

#include <string>

namespace rag {

// Truncate `text` to at most `max_bytes` bytes WITHOUT splitting a UTF-8
// sequence. If the byte at the cut point is in the middle of a multi-byte
// character, the whole incomplete trailing sequence is dropped. This prevents
// invalid UTF-8 from reaching nlohmann::json::dump(), which throws type_error.316
// on malformed input and would otherwise abort a whole batch export.
inline void truncateUtf8(std::string& text, const std::size_t max_bytes) {
  if (text.size() <= max_bytes) {
    return;
  }
  std::size_t cut = max_bytes;
  // Back up over UTF-8 continuation bytes (10xxxxxx) so we don't leave a partial
  // sequence; then drop the lead byte of that incomplete sequence too.
  while (cut > 0 && (static_cast<unsigned char>(text[cut]) & 0xC0) == 0x80) {
    --cut;
  }
  text.resize(cut);
}

}  // namespace rag
