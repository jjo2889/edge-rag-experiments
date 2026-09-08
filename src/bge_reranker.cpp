#include "bge_reranker.hpp"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <onnxruntime_cxx_api.h>

#ifndef RAG_HAS_SENTENCEPIECE
#define RAG_HAS_SENTENCEPIECE 0
#endif

#if RAG_HAS_SENTENCEPIECE
#include <sentencepiece_processor.h>
#endif

namespace rag {

namespace {

constexpr int64_t kXlmRobertaBosId = 0;
constexpr int64_t kXlmRobertaEosId = 2;
constexpr int64_t kXlmRobertaUnkId = 3;

}  // namespace

struct BgeReranker::Impl {
#if RAG_HAS_SENTENCEPIECE
  explicit Impl(const RerankerConfig& config)
      : max_length_(std::max(8, config.max_length)),
        env_(ORT_LOGGING_LEVEL_WARNING, "rag-bge-reranker"),
        memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
    const auto status = processor_.Load(config.sentencepiece_model_path);
    if (!status.ok()) {
      throw std::runtime_error("Failed to load reranker sentencepiece model: " + status.ToString());
    }

    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    session_options_.SetIntraOpNumThreads(1);
    session_ = std::make_unique<Ort::Session>(env_, config.model_path.c_str(), session_options_);

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
      throw std::runtime_error("BGE reranker ONNX model has no outputs.");
    }
  }

  int64_t toXlmRobertaId(const int sentencepiece_id) const {
    if (sentencepiece_id == processor_.unk_id()) {
      return kXlmRobertaUnkId;
    }
    return static_cast<int64_t>(sentencepiece_id) + 1;
  }

  std::vector<int64_t> encodeText(const std::string& text) const {
    std::vector<int> piece_ids;
    const auto status = processor_.Encode(text, &piece_ids);
    if (!status.ok()) {
      throw std::runtime_error("BGE reranker tokenization failed: " + status.ToString());
    }

    std::vector<int64_t> token_ids;
    token_ids.reserve(piece_ids.size());
    for (const int id : piece_ids) {
      token_ids.push_back(toXlmRobertaId(id));
    }
    return token_ids;
  }

  std::vector<int64_t> encodePair(const std::string& query, const std::string& passage) const {
    std::vector<int64_t> query_ids = encodeText(query);
    std::vector<int64_t> passage_ids = encodeText(passage);

    const size_t max_body = static_cast<size_t>(std::max(0, max_length_ - 4));
    while (query_ids.size() + passage_ids.size() > max_body) {
      if (!passage_ids.empty()) {
        passage_ids.pop_back();
      } else if (!query_ids.empty()) {
        query_ids.pop_back();
      } else {
        break;
      }
    }

    std::vector<int64_t> input_ids;
    input_ids.reserve(4 + query_ids.size() + passage_ids.size());
    input_ids.push_back(kXlmRobertaBosId);
    input_ids.insert(input_ids.end(), query_ids.begin(), query_ids.end());
    input_ids.push_back(kXlmRobertaEosId);
    input_ids.push_back(kXlmRobertaEosId);
    input_ids.insert(input_ids.end(), passage_ids.begin(), passage_ids.end());
    input_ids.push_back(kXlmRobertaEosId);
    return input_ids;
  }

  float score(const std::string& query, const std::string& passage) const {
    std::vector<int64_t> input_ids = encodePair(query, passage);
    if (input_ids.empty()) {
      throw std::runtime_error("BGE reranker tokenization produced empty input.");
    }

    std::vector<int64_t> attention_mask(input_ids.size(), 1);
    std::vector<int64_t> token_type_ids(input_ids.size(), 0);
    const std::array<int64_t, 2> shape = {1, static_cast<int64_t>(input_ids.size())};

    auto createTensor = [&](std::vector<int64_t>& values) {
      return Ort::Value::CreateTensor<int64_t>(
          memory_info_, values.data(), values.size(), shape.data(), shape.size());
    };

    std::vector<const char*> input_name_ptrs;
    std::vector<Ort::Value> input_tensors;
    input_name_ptrs.reserve(input_names_.size());
    input_tensors.reserve(input_names_.size());

    for (const auto& name : input_names_) {
      input_name_ptrs.push_back(name.c_str());
      if (name == "input_ids") {
        input_tensors.push_back(createTensor(input_ids));
      } else if (name == "attention_mask") {
        input_tensors.push_back(createTensor(attention_mask));
      } else if (name == "token_type_ids") {
        input_tensors.push_back(createTensor(token_type_ids));
      } else {
        // Feeding token_type_ids to an input we do not recognise runs fine and
        // returns quietly wrong scores, which would make rerank look like a
        // technique that does nothing. Swapping the ONNX file for one with a
        // different signature must fail at the first query, not silently
        // randomise the ranking.
        throw std::runtime_error("Unexpected reranker ONNX input: " + name);
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

    auto& logits = outputs.front();
    const auto tensor_info = logits.GetTensorTypeAndShapeInfo();
    const size_t count = tensor_info.GetElementCount();
    if (count == 0) {
      throw std::runtime_error("BGE reranker output tensor is empty.");
    }

    const float* data = logits.GetTensorData<float>();
    return data[0];
  }

  std::vector<RetrievedChunk> rerank(const std::string& query,
                                     const std::vector<RetrievedChunk>& candidates,
                                     const int top_k) const {
    if (top_k <= 0 || candidates.empty()) {
      return {};
    }

    struct ScoredCandidate {
      RetrievedChunk chunk;
      size_t original_index = 0;
    };

    std::vector<ScoredCandidate> scored;
    scored.reserve(candidates.size());
    for (size_t i = 0; i < candidates.size(); ++i) {
      ScoredCandidate item;
      item.chunk = candidates[i];
      item.chunk.rerank_score = score(query, candidates[i].chunk.text);
      item.chunk.has_rerank_score = true;
      item.original_index = i;
      scored.push_back(std::move(item));
    }

    std::stable_sort(scored.begin(), scored.end(), [](const ScoredCandidate& a, const ScoredCandidate& b) {
      if (a.chunk.rerank_score != b.chunk.rerank_score) {
        return a.chunk.rerank_score > b.chunk.rerank_score;
      }
      if (a.chunk.score != b.chunk.score) {
        return a.chunk.score > b.chunk.score;
      }
      return a.original_index < b.original_index;
    });

    std::vector<RetrievedChunk> out;
    const size_t limit = std::min(scored.size(), static_cast<size_t>(top_k));
    out.reserve(limit);
    for (size_t i = 0; i < limit; ++i) {
      out.push_back(std::move(scored[i].chunk));
    }
    return out;
  }

  int max_length_;
  sentencepiece::SentencePieceProcessor processor_;
  Ort::Env env_;
  Ort::SessionOptions session_options_;
  std::unique_ptr<Ort::Session> session_;
  Ort::MemoryInfo memory_info_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
#else
  explicit Impl(const RerankerConfig&) {
    throw std::runtime_error(
        "BGE reranker requires sentencepiece C++ headers/library. Run scripts/setup_third_party.sh or install "
        "sentencepiece, then rebuild.");
  }

  std::vector<RetrievedChunk> rerank(const std::string&,
                                     const std::vector<RetrievedChunk>&,
                                     int) const {
    return {};
  }
#endif
};

BgeReranker::BgeReranker(const RerankerConfig& config) : impl_(std::make_unique<Impl>(config)) {}

BgeReranker::~BgeReranker() = default;

std::vector<RetrievedChunk> BgeReranker::rerank(const std::string& query,
                                                const std::vector<RetrievedChunk>& candidates,
                                                const int top_k) const {
  return impl_->rerank(query, candidates, top_k);
}

}  // namespace rag
