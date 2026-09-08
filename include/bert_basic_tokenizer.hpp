#pragma once

#include <string>
#include <vector>

namespace rag {

// BERT "basic tokenization" (the whitespace/punctuation/CJK split that runs
// before WordPiece), operating on UTF-8 code points rather than raw bytes.
//
// Byte-wise iteration was the previous implementation and silently corrupted any
// non-ASCII input: every continuation byte (0x80-0xBF) failed isalnum/isspace and
// became its own single-byte "word", which WordPiece could not match, so `café`
// tokenized as `caf` + [UNK] + [UNK]. That noise went straight into the document
// and query embeddings.
//
// Behaviour follows HuggingFace `BertTokenizer`'s BasicTokenizer:
//   - control characters are dropped, Unicode whitespace splits words
//   - punctuation and CJK ideographs each become their own token
//   - when `do_lower_case` is set, text is lowercased and accents are stripped
//     (NFD + drop combining marks), so `Café` -> `cafe`
//
// Verified against the project's real all-MiniLM-L6-v2 tokenizer.json (normalizer
// + pre_tokenizer via the `tokenizers` library): 0 divergences over the whole
// SciFact corpus (5,729 sentences, 2,729 of them non-ASCII) and over 50,000
// randomized strings drawn from ASCII, Latin-1, Latin Extended-A, Greek,
// Cyrillic, punctuation, combining marks, CJK, and fullwidth forms.
//
// Known limitation: case folding is table-driven over those blocks. Uppercase
// letters from historic/rare scripts not covered there (Latin Extended-B beyond
// the pairs listed, Coptic, Armenian, Deseret, ...) pass through un-lowercased.
// None occur in the BEIR corpora this project uses; extending the tables is the
// fix if that changes.
std::vector<std::string> bertBasicTokenize(const std::string& text, bool do_lower_case);

// Number of UTF-8 code points in `text` (invalid bytes count as one each).
// WordPiece's max_input_chars_per_word is a character limit, not a byte limit.
size_t utf8Length(const std::string& text);

}  // namespace rag
