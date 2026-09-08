#include "bert_basic_tokenizer.hpp"

#include <array>
#include <cctype>
#include <cstdint>
#include <string>
#include <vector>

namespace rag {

namespace {

constexpr uint32_t kReplacementChar = 0xFFFD;

// Decode one UTF-8 code point starting at `i`, advancing `i` past it. Malformed
// sequences consume a single byte and yield U+FFFD, so a corrupt input can never
// cause an infinite loop or read past the end.
uint32_t decodeUtf8(const std::string& text, size_t& i) {
  const size_t n = text.size();
  const auto byte = [&](const size_t k) { return static_cast<unsigned char>(text[k]); };

  const unsigned char lead = byte(i);
  if (lead < 0x80) {
    ++i;
    return lead;
  }

  size_t extra = 0;
  uint32_t cp = 0;
  if ((lead & 0xE0) == 0xC0) {
    extra = 1;
    cp = lead & 0x1FU;
  } else if ((lead & 0xF0) == 0xE0) {
    extra = 2;
    cp = lead & 0x0FU;
  } else if ((lead & 0xF8) == 0xF0) {
    extra = 3;
    cp = lead & 0x07U;
  } else {
    ++i;
    return kReplacementChar;
  }

  if (i + extra >= n) {  // truncated sequence at end of input
    ++i;
    return kReplacementChar;
  }

  for (size_t k = 1; k <= extra; ++k) {
    const unsigned char cont = byte(i + k);
    if ((cont & 0xC0) != 0x80) {
      ++i;
      return kReplacementChar;
    }
    cp = (cp << 6) | (cont & 0x3FU);
  }

  i += extra + 1;
  return cp;
}

void appendUtf8(std::string& out, const uint32_t cp) {
  if (cp < 0x80) {
    out.push_back(static_cast<char>(cp));
  } else if (cp < 0x800) {
    out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else if (cp < 0x10000) {
    out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else {
    out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  }
}

bool isWhitespaceCp(const uint32_t cp) {
  switch (cp) {
    case ' ':
    case '\t':
    case '\n':
    case '\r':
    case 0x0B:
    case 0x0C:
    case 0x85:
    case 0xA0:
    case 0x1680:
    case 0x2028:
    case 0x2029:
    case 0x202F:
    case 0x205F:
    case 0x3000:
      return true;
    default:
      return cp >= 0x2000 && cp <= 0x200A;
  }
}

// Cc/Cf minus tab/newline/return, which are handled as whitespace above.
// U+00AD (soft hyphen) belongs here: it is Cf, so the reference tokenizer drops
// it, and keeping it split words like "pro<SHY>tein" into unmatchable pieces.
bool isControlCp(const uint32_t cp) {
  if (cp == '\t' || cp == '\n' || cp == '\r') {
    return false;
  }
  if (cp < 0x20 || cp == 0x7F) {
    return true;
  }
  if (cp >= 0x80 && cp <= 0x9F) {
    return true;
  }
  if (cp == 0xAD) {
    return true;
  }
  if (cp >= 0x600 && cp <= 0x605) {
    return true;
  }
  if (cp == 0x61C || cp == 0x6DD || cp == 0x70F || cp == 0x180E) {
    return true;
  }
  if (cp >= 0x200B && cp <= 0x200F) {
    return true;
  }
  if (cp >= 0x202A && cp <= 0x202E) {
    return true;
  }
  if (cp >= 0x2060 && cp <= 0x2064) {
    return true;
  }
  if (cp >= 0x2066 && cp <= 0x206F) {
    return true;
  }
  if (cp >= 0xFFF9 && cp <= 0xFFFB) {
    return true;
  }
  // Private Use Areas (category Co). The reference tokenizer's clean_text drops
  // these; keeping them glued characters into words that could never match vocab.
  if (cp >= 0xE000 && cp <= 0xF8FF) {
    return true;
  }
  if (cp >= 0xF0000 && cp <= 0xFFFFD) {
    return true;
  }
  if (cp >= 0x100000 && cp <= 0x10FFFD) {
    return true;
  }
  return cp == 0xFEFF;
}

// HuggingFace treats every ASCII non-alphanumeric printable as punctuation, plus
// anything in Unicode category P*. The ranges below cover the P* code points that
// realistically appear in English corpora (curly quotes, dashes, ellipsis, CJK
// and fullwidth punctuation).
bool isPunctuationCp(const uint32_t cp) {
  if ((cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) || (cp >= 91 && cp <= 96) ||
      (cp >= 123 && cp <= 126)) {
    return true;
  }
  switch (cp) {
    case 0xA1:  // ¡
    case 0xA7:  // §
    case 0xAB:  // «
    case 0xB6:  // ¶
    case 0xB7:  // ·
    case 0xBB:  // »
    case 0xBF:  // ¿
    case 0x37E:  // Greek question mark
    case 0x387:  // Greek ano teleia
      return true;
    default:
      break;
  }
  if (cp >= 0x2010 && cp <= 0x2027) {  // dashes, quotes, daggers, ellipsis
    return true;
  }
  // Per-mille, primes, brackets. U+2044 (fraction slash) and U+2052 (commercial
  // minus) sit inside this range but are category Sm, not P.
  if (cp >= 0x2030 && cp <= 0x205E && cp != 0x2044 && cp != 0x2052) {
    return true;
  }
  if (cp >= 0x2E00 && cp <= 0x2E7F) {  // supplemental punctuation
    return true;
  }
  if (cp >= 0x3001 && cp <= 0x3003) {  // 、。〃
    return true;
  }
  if (cp >= 0x3008 && cp <= 0x3011) {  // CJK brackets
    return true;
  }
  if (cp >= 0x3014 && cp <= 0x301F) {
    return true;
  }
  if (cp == 0x3030 || cp == 0x303D || cp == 0x30A0 || cp == 0x30FB) {
    return true;
  }
  // Fullwidth forms: only the category-P members. ＋ ＜ ＝ ＞ ｜ ～ are Sm, ＾ ｀
  // are Sk, ＄ is Sc — the reference does not treat those as punctuation, and
  // splitting on them produced tokens the reference kept joined.
  if ((cp >= 0xFF01 && cp <= 0xFF03) || (cp >= 0xFF05 && cp <= 0xFF0A) ||
      (cp >= 0xFF0C && cp <= 0xFF0F) || (cp >= 0xFF1A && cp <= 0xFF1B) ||
      (cp >= 0xFF1F && cp <= 0xFF20) || (cp >= 0xFF3B && cp <= 0xFF3D) || cp == 0xFF3F ||
      cp == 0xFF5B || cp == 0xFF5D || (cp >= 0xFF5F && cp <= 0xFF65)) {
    return true;
  }
  return false;
}

// Same ranges HuggingFace's BasicTokenizer._is_chinese_char uses.
bool isCjkCp(const uint32_t cp) {
  return (cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0x3400 && cp <= 0x4DBF) ||
         (cp >= 0x20000 && cp <= 0x2A6DF) || (cp >= 0x2A700 && cp <= 0x2B73F) ||
         (cp >= 0x2B740 && cp <= 0x2B81F) || (cp >= 0x2B820 && cp <= 0x2CEAF) ||
         (cp >= 0xF900 && cp <= 0xFAFF) || (cp >= 0x2F800 && cp <= 0x2FA1F);
}

// Base letter for the accented Latin code points that have a canonical NFD
// decomposition; '\0' means "no decomposition, keep the character as-is"
// (e.g. ø, ß, æ, đ, ł — HuggingFace leaves these intact too).
char latin1BaseLetter(const uint32_t cp) {
  static constexpr std::array<char, 64> kUpper = {
      'a', 'a', 'a', 'a', 'a', 'a', '\0', 'c',   // C0-C7  À Á Â Ã Ä Å Æ Ç
      'e', 'e', 'e', 'e', 'i', 'i', 'i',  'i',   // C8-CF  È É Ê Ë Ì Í Î Ï
      '\0', 'n', 'o', 'o', 'o', 'o', 'o', '\0',  // D0-D7  Ð Ñ Ò Ó Ô Õ Ö ×
      '\0', 'u', 'u', 'u', 'u', 'y', '\0', '\0', // D8-DF  Ø Ù Ú Û Ü Ý Þ ß
      'a', 'a', 'a', 'a', 'a', 'a', '\0', 'c',   // E0-E7  à á â ã ä å æ ç
      'e', 'e', 'e', 'e', 'i', 'i', 'i',  'i',   // E8-EF  è é ê ë ì í î ï
      '\0', 'n', 'o', 'o', 'o', 'o', 'o', '\0',  // F0-F7  ð ñ ò ó ô õ ö ÷
      '\0', 'u', 'u', 'u', 'u', 'y', '\0', 'y'}; // F8-FF  ø ù ú û ü ý þ ÿ
  if (cp < 0xC0 || cp > 0xFF) {
    return '\0';
  }
  return kUpper[cp - 0xC0];
}

// Latin Extended-A (U+0100-U+017F). '\0' entries are the code points without a
// canonical decomposition: Đ đ Ħ ħ ı Ĳ ĳ ĸ Ŀ ŀ Ł ł ŉ Ŋ ŋ Œ œ Ŧ ŧ ſ.
char latinExtendedABaseLetter(const uint32_t cp) {
  static constexpr std::array<char, 128> kBase = {
      'a', 'a', 'a', 'a', 'a', 'a', 'c', 'c',      // 100-107
      'c', 'c', 'c', 'c', 'c', 'c', 'd', 'd',      // 108-10F
      '\0', '\0', 'e', 'e', 'e', 'e', 'e', 'e',    // 110-117
      'e', 'e', 'e', 'e', 'g', 'g', 'g', 'g',      // 118-11F
      'g', 'g', 'g', 'g', 'h', 'h', '\0', '\0',    // 120-127
      'i', 'i', 'i', 'i', 'i', 'i', 'i', 'i',      // 128-12F
      'i', '\0', '\0', '\0', 'j', 'j', 'k', 'k',   // 130-137
      '\0', 'l', 'l', 'l', 'l', 'l', 'l', '\0',    // 138-13F
      '\0', '\0', '\0', 'n', 'n', 'n', 'n', 'n',   // 140-147
      'n', '\0', '\0', '\0', 'o', 'o', 'o', 'o',   // 148-14F
      'o', 'o', '\0', '\0', 'r', 'r', 'r', 'r',    // 150-157
      'r', 'r', 's', 's', 's', 's', 's', 's',      // 158-15F
      's', 's', 't', 't', 't', 't', '\0', '\0',    // 160-167
      'u', 'u', 'u', 'u', 'u', 'u', 'u', 'u',      // 168-16F
      'u', 'u', 'u', 'u', 'w', 'w', 'y', 'y',      // 170-177
      'y', 'z', 'z', 'z', 'z', 'z', 'z', '\0'};    // 178-17F
  if (cp < 0x100 || cp > 0x17F) {
    return '\0';
  }
  return kBase[cp - 0x100];
}

// Category Mn ranges. strip_accents runs NFD then drops every combining mark, so
// any of these present in the input is dropped regardless of where it came from.
bool isCombiningMark(const uint32_t cp) {
  return (cp >= 0x300 && cp <= 0x36F) || (cp >= 0x1AB0 && cp <= 0x1AFF) ||
         (cp >= 0x1DC0 && cp <= 0x1DFF) || (cp >= 0x20D0 && cp <= 0x20F0) ||
         // 0x302A-0x302D only: U+302E/U+302F are Mc (spacing), which NFD keeps.
         (cp >= 0x302A && cp <= 0x302D) || (cp >= 0x3099 && cp <= 0x309A) ||
         (cp >= 0xFE20 && cp <= 0xFE2F);
}

// Latin letters that have NO canonical decomposition but DO have a lowercase
// form: Ø, Þ, Æ, Ð, Đ, Ł, Œ, ... Stripping accents leaves them unchanged, but the
// reference still lowercases them, so returning them untouched produced tokens
// like "Øst" where the reference produced "øst". Returns 0 when not applicable.
uint32_t latinNonDecomposableLower(const uint32_t cp) {
  switch (cp) {
    case 0xC6: return 0xE6;    // Æ -> æ
    case 0xD0: return 0xF0;    // Ð -> ð
    case 0xD8: return 0xF8;    // Ø -> ø
    case 0xDE: return 0xFE;    // Þ -> þ
    case 0x110: return 0x111;  // Đ -> đ
    case 0x126: return 0x127;  // Ħ -> ħ
    case 0x132: return 0x133;  // Ĳ -> ĳ
    case 0x13F: return 0x140;  // Ŀ -> ŀ
    case 0x141: return 0x142;  // Ł -> ł
    case 0x14A: return 0x14B;  // Ŋ -> ŋ
    case 0x152: return 0x153;  // Œ -> œ
    case 0x166: return 0x167;  // Ŧ -> ŧ
    default: return 0;
  }
}

// Greek lowercasing with accent stripping. Scientific abstracts are full of
// Δ/Ψ/Ω, and leaving them uppercase made every token containing one differ from
// the reference tokenizer (e.g. "Δage" instead of "δage"), costing a vocab match.
// Returns 0 when `cp` is not Greek.
uint32_t greekFold(const uint32_t cp) {
  switch (cp) {
    case 0x386: return 0x3B1;  // Ά -> α
    case 0x388: return 0x3B5;  // Έ -> ε
    case 0x389: return 0x3B7;  // Ή -> η
    case 0x38A: return 0x3B9;  // Ί -> ι
    case 0x38C: return 0x3BF;  // Ό -> ο
    case 0x38E: return 0x3C5;  // Ύ -> υ
    case 0x38F: return 0x3C9;  // Ώ -> ω
    case 0x390: return 0x3B9;  // ΐ -> ι
    case 0x3AA: return 0x3B9;  // Ϊ -> ι
    case 0x3AB: return 0x3C5;  // Ϋ -> υ
    case 0x3AC: return 0x3B1;  // ά -> α
    case 0x3AD: return 0x3B5;  // έ -> ε
    case 0x3AE: return 0x3B7;  // ή -> η
    case 0x3AF: return 0x3B9;  // ί -> ι
    case 0x3B0: return 0x3C5;  // ΰ -> υ
    case 0x3CA: return 0x3B9;  // ϊ -> ι
    case 0x3CB: return 0x3C5;  // ϋ -> υ
    case 0x3CC: return 0x3BF;  // ό -> ο
    case 0x3CD: return 0x3C5;  // ύ -> υ
    case 0x3CE: return 0x3C9;  // ώ -> ω
    default: break;
  }
  // Α-Ρ and Σ-Ω lowercase by a fixed +0x20 (U+03A2 is unassigned).
  if ((cp >= 0x391 && cp <= 0x3A1) || (cp >= 0x3A3 && cp <= 0x3A9)) {
    return cp + 0x20;
  }
  return 0;
}

// Cyrillic lowercasing, with the decomposable letters folded to their base
// (Ё -> е, Й -> и) exactly as NFD + drop-combining-marks would. Returns 0 when
// `cp` is not Cyrillic.
uint32_t cyrillicFold(const uint32_t cp) {
  switch (cp) {
    case 0x400:
    case 0x401:
    case 0x450:
    case 0x451: return 0x435;  // Ѐ Ё ѐ ё -> е
    case 0x403:
    case 0x453: return 0x433;  // Ѓ ѓ -> г
    case 0x407:
    case 0x457: return 0x456;  // Ї ї -> і
    case 0x40C:
    case 0x45C: return 0x43A;  // Ќ ќ -> к
    case 0x40D:
    case 0x45D:
    case 0x419:
    case 0x439: return 0x438;  // Ѝ ѝ Й й -> и
    case 0x40E:
    case 0x45E: return 0x443;  // Ў ў -> у
    default: break;
  }
  if (cp >= 0x410 && cp <= 0x42F) {
    return cp + 0x20;
  }
  if (cp >= 0x402 && cp <= 0x40F) {
    return cp + 0x50;
  }
  return 0;
}

// Letterlike symbols with a canonical singleton decomposition. U+212B (ANGSTROM
// SIGN) is common in structural-biology text and is a distinct code point from
// U+00C5, so the Latin-1 table alone does not catch it. Returns 0 otherwise.
uint32_t letterlikeFold(const uint32_t cp) {
  switch (cp) {
    case 0x387: return 0xB7;    // · GREEK ANO TELEIA -> · MIDDLE DOT (singleton)
    case 0x3D2:
    case 0x3D3:
    case 0x3D4: return 0x3C5;   // ϒ ϓ ϔ upsilon-with-hook symbols -> υ
    case 0x212B: return 'a';    // Å ANGSTROM SIGN -> A + ring -> a
    case 0x2126: return 0x3C9;  // Ω OHM SIGN -> Ω -> ω
    case 0x212A: return 'k';    // K KELVIN SIGN -> K -> k
    case 0x2132: return 0x214E; // Ⅎ TURNED CAPITAL F -> ⅎ
    default: break;
  }
  if (cp >= 0x2160 && cp <= 0x216F) {  // Roman numerals Ⅰ-Ⅿ -> ⅰ-ⅿ
    return cp + 0x10;
  }
  if (cp == 0x2183) {  // Ↄ ROMAN NUMERAL REVERSED HUNDRED -> ↄ
    return 0x2184;
  }
  if (cp >= 0x24B6 && cp <= 0x24CF) {  // Circled Ⓐ-Ⓩ -> ⓐ-ⓩ
    return cp + 0x1A;
  }
  if (cp >= 0xFF21 && cp <= 0xFF3A) {  // Fullwidth Ａ-Ｚ -> ａ-ｚ
    return cp + 0x20;
  }
  return 0;
}

}  // namespace

size_t utf8Length(const std::string& text) {
  size_t i = 0;
  size_t count = 0;
  while (i < text.size()) {
    decodeUtf8(text, i);
    ++count;
  }
  return count;
}

std::vector<std::string> bertBasicTokenize(const std::string& text, const bool do_lower_case) {
  std::vector<std::string> tokens;
  std::string current;

  const auto flush = [&]() {
    if (!current.empty()) {
      tokens.push_back(current);
      current.clear();
    }
  };
  const auto emitStandalone = [&](const uint32_t cp) {
    flush();
    std::string token;
    appendUtf8(token, cp);
    tokens.push_back(std::move(token));
  };

  size_t i = 0;
  while (i < text.size()) {
    uint32_t cp = decodeUtf8(text, i);

    if (cp == kReplacementChar || isControlCp(cp)) {
      continue;
    }
    if (isWhitespaceCp(cp)) {
      flush();
      continue;
    }

    if (do_lower_case) {
      // NFD + drop combining marks + lowercase, matching BertNormalizer.
      if (isCombiningMark(cp)) {
        continue;
      }
      if (cp < 0x80) {
        cp = static_cast<uint32_t>(std::tolower(static_cast<unsigned char>(cp)));
      } else if (const char base = latin1BaseLetter(cp); base != '\0') {
        cp = static_cast<uint32_t>(base);
      } else if (const char base_ext = latinExtendedABaseLetter(cp); base_ext != '\0') {
        cp = static_cast<uint32_t>(base_ext);
      } else if (const uint32_t latin_lower = latinNonDecomposableLower(cp); latin_lower != 0) {
        cp = latin_lower;
      } else if (const uint32_t greek = greekFold(cp); greek != 0) {
        cp = greek;
      } else if (const uint32_t cyrillic = cyrillicFold(cp); cyrillic != 0) {
        cp = cyrillic;
      } else if (const uint32_t letterlike = letterlikeFold(cp); letterlike != 0) {
        cp = letterlike;
      }
    }

    if (isPunctuationCp(cp) || isCjkCp(cp)) {
      emitStandalone(cp);
      continue;
    }

    appendUtf8(current, cp);
  }
  flush();

  return tokens;
}

}  // namespace rag
