#include "embedding_model.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

namespace rag {

class BertWordPieceTokenizer {
 public:
  BertWordPieceTokenizer(const std::string& tokenizer_json_path, const size_t max_length) : max_length_(max_length) {
    std::ifstream in(tokenizer_json_path);
    if (!in) {
      throw std::runtime_error("Failed to open tokenizer.json: " + tokenizer_json_path);
    }

    nlohmann::json j;
    in >> j;

    const auto& model = j.at("model");
    const auto& vocab = model.at("vocab");
    for (auto it = vocab.begin(); it != vocab.end(); ++it) {
      vocab_[it.key()] = it.value().get<int64_t>();
    }

    unk_id_ = findTokenId("[UNK]", 100);
    cls_id_ = findTokenId("[CLS]", 101);
    sep_id_ = findTokenId("[SEP]", 102);
    pad_id_ = findTokenId("[PAD]", 0);

    if (model.contains("continuing_subword_prefix") && model["continuing_subword_prefix"].is_string()) {
      continuation_prefix_ = model["continuing_subword_prefix"].get<std::string>();
    }
    if (model.contains("max_input_chars_per_word") && model["max_input_chars_per_word"].is_number_integer()) {
      max_input_chars_per_word_ = model["max_input_chars_per_word"].get<int>();
    }

    if (j.contains("normalizer") && j["normalizer"].is_object()) {
      const auto& normalizer = j["normalizer"];
      if (normalizer.contains("lowercase") && normalizer["lowercase"].is_boolean()) {
        do_lower_case_ = normalizer["lowercase"].get<bool>();
      }
    }
  }

  void encode(const std::string& text,
              std::vector<int64_t>& input_ids,
              std::vector<int64_t>& attention_mask,
              std::vector<int64_t>& token_type_ids) const {
    input_ids.clear();
    attention_mask.clear();
    token_type_ids.clear();

    input_ids.push_back(cls_id_);

    const std::vector<std::string> basic_tokens = basicTokenize(text);
    for (const auto& token : basic_tokens) {
      const std::vector<int64_t> wp = wordpieceTokenize(token);
      input_ids.insert(input_ids.end(), wp.begin(), wp.end());
    }

    input_ids.push_back(sep_id_);

    if (input_ids.size() > max_length_) {
      input_ids.resize(max_length_);
      input_ids.back() = sep_id_;
    }

    attention_mask.assign(input_ids.size(), 1);
    token_type_ids.assign(input_ids.size(), 0);
  }

 private:
  int64_t findTokenId(const std::string& token, const int64_t fallback) const {
    const auto it = vocab_.find(token);
    return it == vocab_.end() ? fallback : it->second;
  }

  std::vector<std::string> basicTokenize(const std::string& text) const {
    std::vector<std::string> tokens;
    std::string current;

    auto flush_current = [&]() {
      if (!current.empty()) {
        tokens.push_back(current);
        current.clear();
      }
    };

    for (char raw_ch : text) {
      unsigned char uch = static_cast<unsigned char>(raw_ch);
      if (std::isspace(uch) != 0) {
        flush_current();
        continue;
      }

      char ch = raw_ch;
      if (do_lower_case_) {
        ch = static_cast<char>(std::tolower(uch));
        uch = static_cast<unsigned char>(ch);
      }

      if (std::isalnum(uch) != 0) {
        current.push_back(ch);
      } else {
        flush_current();
        tokens.emplace_back(1, ch);
      }
    }
    flush_current();

    return tokens;
  }

  std::vector<int64_t> wordpieceTokenize(const std::string& token) const {
    if (token.empty()) {
      return {};
    }

    if (static_cast<int>(token.size()) > max_input_chars_per_word_) {
      return {unk_id_};
    }

    std::vector<int64_t> output;
    size_t start = 0;
    bool is_bad = false;
    while (start < token.size()) {
      size_t end = token.size();
      std::string matched_piece;
      while (start < end) {
        std::string piece = token.substr(start, end - start);
        if (start > 0) {
          piece = continuation_prefix_ + piece;
        }
        if (vocab_.find(piece) != vocab_.end()) {
          matched_piece = std::move(piece);
          break;
        }
        --end;
      }

      if (matched_piece.empty()) {
        is_bad = true;
        break;
      }

      output.push_back(vocab_.at(matched_piece));
      start = end;
    }

    if (is_bad) {
      return {unk_id_};
    }
    return output;
  }

  std::unordered_map<std::string, int64_t> vocab_;
  int64_t unk_id_ = 100;
  int64_t cls_id_ = 101;
  int64_t sep_id_ = 102;
  int64_t pad_id_ = 0;
  bool do_lower_case_ = true;
  int max_input_chars_per_word_ = 100;
  std::string continuation_prefix_ = "##";
  size_t max_length_ = 256;
};

std::vector<float> l2Normalize(std::vector<float> vec) {
  float norm_sq = 0.0F;
  for (const float v : vec) {
    norm_sq += v * v;
  }
  const float norm = std::sqrt(norm_sq);
  if (norm > 1e-12F) {
    for (float& v : vec) {
      v /= norm;
    }
  }
  return vec;
}

EmbeddingModel::EmbeddingModel(const std::string& onnx_model_path,
                               const std::string& tokenizer_json_path,
                               const size_t max_length)
    : tokenizer_(std::make_unique<BertWordPieceTokenizer>(tokenizer_json_path, max_length)),
      env_(ORT_LOGGING_LEVEL_WARNING, "rag-embedding"),
      memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
      embedding_dim_(0) {
  session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
  session_options_.SetIntraOpNumThreads(1);
  session_ = std::make_unique<Ort::Session>(env_, onnx_model_path.c_str(), session_options_);

  Ort::AllocatorWithDefaultOptions allocator;

  const size_t n_inputs = session_->GetInputCount();
  input_names_.reserve(n_inputs);
  for (size_t i = 0; i < n_inputs; ++i) {
    auto name = session_->GetInputNameAllocated(i, allocator);
    input_names_.push_back(name.get());
  }

  const size_t n_outputs = session_->GetOutputCount();
  output_names_.reserve(n_outputs);
  for (size_t i = 0; i < n_outputs; ++i) {
    auto name = session_->GetOutputNameAllocated(i, allocator);
    output_names_.push_back(name.get());
  }

  if (output_names_.empty()) {
    throw std::runtime_error("ONNX model has no outputs.");
  }

  sentence_output_name_ = output_names_.front();
  for (const auto& name : output_names_) {
    if (name == "sentence_embedding") {
      sentence_output_name_ = name;
      break;
    }
  }
}

EmbeddingModel::~EmbeddingModel() = default;

std::vector<float> EmbeddingModel::embed(const std::string& text) const {
  std::vector<int64_t> input_ids;
  std::vector<int64_t> attention_mask;
  std::vector<int64_t> token_type_ids;
  tokenizer_->encode(text, input_ids, attention_mask, token_type_ids);

  if (input_ids.empty()) {
    throw std::runtime_error("Tokenization produced empty input.");
  }

  const std::array<int64_t, 2> shape = {1, static_cast<int64_t>(input_ids.size())};

  auto createTensor = [&](std::vector<int64_t>& values) {
    return Ort::Value::CreateTensor<int64_t>(
        memory_info_, values.data(), values.size(), shape.data(), shape.size());
  };

  std::vector<const char*> input_name_ptrs;
  std::vector<Ort::Value> input_tensors;
  input_name_ptrs.reserve(input_names_.size());
  input_tensors.reserve(input_names_.size());

  std::vector<int64_t> zeros(shape[1], 0);

  for (const auto& name : input_names_) {
    input_name_ptrs.push_back(name.c_str());
    if (name == "input_ids") {
      input_tensors.push_back(createTensor(input_ids));
    } else if (name == "attention_mask") {
      input_tensors.push_back(createTensor(attention_mask));
    } else if (name == "token_type_ids") {
      input_tensors.push_back(createTensor(token_type_ids));
    } else {
      input_tensors.push_back(createTensor(zeros));
    }
  }

  std::vector<const char*> output_name_ptrs;
  output_name_ptrs.reserve(output_names_.size());
  for (const auto& name : output_names_) {
    output_name_ptrs.push_back(name.c_str());
  }

  Ort::RunOptions run_options;
  auto outputs = session_->Run(run_options,
                               input_name_ptrs.data(),
                               input_tensors.data(),
                               input_tensors.size(),
                               output_name_ptrs.data(),
                               output_name_ptrs.size());

  size_t output_index = 0;
  for (size_t i = 0; i < output_names_.size(); ++i) {
    if (output_names_[i] == sentence_output_name_) {
      output_index = i;
      break;
    }
  }

  auto& output = outputs.at(output_index);
  const auto tensor_info = output.GetTensorTypeAndShapeInfo();
  const std::vector<int64_t> out_shape = tensor_info.GetShape();
  const float* data = output.GetTensorData<float>();

  std::vector<float> embedding;

  if (out_shape.size() == 2) {
    const int64_t dim = out_shape[1];
    embedding.assign(data, data + dim);
  } else if (out_shape.size() == 3) {
    const int64_t seq_len = out_shape[1];
    const int64_t dim = out_shape[2];
    embedding.assign(dim, 0.0F);
    float weight_sum = 0.0F;
    for (int64_t i = 0; i < seq_len; ++i) {
      const float w = static_cast<float>(i < static_cast<int64_t>(attention_mask.size()) ? attention_mask[i] : 0);
      if (w <= 0.0F) {
        continue;
      }
      const float* row = data + i * dim;
      for (int64_t d = 0; d < dim; ++d) {
        embedding[d] += row[d] * w;
      }
      weight_sum += w;
    }
    if (weight_sum > 0.0F) {
      for (float& v : embedding) {
        v /= weight_sum;
      }
    }
  } else {
    throw std::runtime_error("Unsupported ONNX output shape rank.");
  }

  embedding = l2Normalize(std::move(embedding));
  embedding_dim_ = static_cast<int>(embedding.size());
  return embedding;
}

std::vector<std::vector<float>> EmbeddingModel::embedBatch(const std::vector<std::string>& texts) const {
  std::vector<std::vector<float>> out;
  out.reserve(texts.size());
  for (const auto& text : texts) {
    out.push_back(embed(text));
  }
  return out;
}

int EmbeddingModel::dimension() const {
  return embedding_dim_;
}

}  // namespace rag
