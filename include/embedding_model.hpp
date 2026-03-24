#pragma once

#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

namespace rag {

class BertWordPieceTokenizer;

class EmbeddingModel {
 public:
  EmbeddingModel(const std::string& onnx_model_path, const std::string& tokenizer_json_path, size_t max_length = 256);
  ~EmbeddingModel();

  std::vector<float> embed(const std::string& text) const;
  std::vector<std::vector<float>> embedBatch(const std::vector<std::string>& texts) const;

  int dimension() const;

 private:
  std::unique_ptr<BertWordPieceTokenizer> tokenizer_;
  Ort::Env env_;
  Ort::SessionOptions session_options_;
  std::unique_ptr<Ort::Session> session_;
  Ort::MemoryInfo memory_info_;

  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::string sentence_output_name_;

  mutable int embedding_dim_;
};

}  // namespace rag
