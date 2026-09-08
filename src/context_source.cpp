#include "context_source.hpp"

#include <stdexcept>

namespace rag {

ContextSource contextSourceFromString(const std::string& value) {
  if (value.empty() || value == "retrieved") {
    return ContextSource::Retrieved;
  }
  if (value == "closed_book") {
    return ContextSource::ClosedBook;
  }
  if (value == "oracle") {
    return ContextSource::Oracle;
  }

  throw std::runtime_error("Unsupported context source: " + value);
}

std::string contextSourceToString(const ContextSource source) {
  switch (source) {
    case ContextSource::Retrieved:
      return "retrieved";
    case ContextSource::ClosedBook:
      return "closed_book";
    case ContextSource::Oracle:
      return "oracle";
  }

  throw std::runtime_error("Unknown context source enum value.");
}

}  // namespace rag
