#include "generator.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <llama.h>

#include "reasoning_output.hpp"

namespace rag {

namespace {

void silentLlamaLog(enum ggml_log_level /* level */, const char* /* text */, void* /* user_data */) {}

struct SamplerDeleter {
  void operator()(llama_sampler* sampler) const {
    if (sampler != nullptr) {
      llama_sampler_free(sampler);
    }
  }
};

std::string loadTextFile(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("Failed to open system prompt file: " + path);
  }

  std::ostringstream oss;
  oss << in.rdbuf();
  const std::string text = oss.str();
  if (text.empty()) {
    throw std::runtime_error("System prompt file is empty: " + path);
  }
  return text;
}

// PromptBuilder marks sections with brackets ("[Evidence]") and individual items
// with a bare label followed by a space ("Evidence 1:"). Both are structure the
// model must never emit, so both are safe to cut at the very start of a reply.
bool isStructuralMarker(const std::string& stop) {
  if (stop.size() < 2 || stop.front() != '\n') {
    return false;
  }
  const char first = stop[1];
  return first == '[' || (stop.back() == ' ' && std::isupper(static_cast<unsigned char>(first)));
}

}  // namespace

Generator::Generator(ModelConfig config)
    : config_(std::move(config)),
      model_(nullptr),
      vocab_(nullptr),
      system_prompt_(),
      chat_template_(),
      context_limit_(std::max(128, config_.n_ctx)),
      ctx_n_batch_(0),
      ctx_(nullptr),
      ctx_mutex_() {
  llama_log_set(silentLlamaLog, nullptr);
  llama_backend_init();
  system_prompt_ = loadTextFile(config_.prompt_system_path);

  llama_model_params model_params = llama_model_default_params();
  model_params.n_gpu_layers = config_.n_gpu_layers;

  model_ = llama_model_load_from_file(config_.llama_model_path.c_str(), model_params);
  if (model_ == nullptr) {
    throw std::runtime_error("Failed to load GGUF model: " + config_.llama_model_path);
  }

  vocab_ = llama_model_get_vocab(model_);
  if (vocab_ == nullptr) {
    throw std::runtime_error("Failed to get model vocabulary.");
  }

  if (config_.use_model_chat_template) {
    const char* tmpl = llama_model_chat_template(model_, nullptr);
    if (tmpl != nullptr) {
      chat_template_ = tmpl;
    }
  }

  const int initial_n_batch = std::max(1, std::min(context_limit_, config_.n_batch));
  recreateContextLocked(initial_n_batch);
}

Generator::~Generator() {
  {
    std::lock_guard<std::mutex> lock(ctx_mutex_);
    if (ctx_ != nullptr) {
      llama_free(ctx_);
      ctx_ = nullptr;
      ctx_n_batch_ = 0;
    }
  }

  if (model_ != nullptr) {
    llama_model_free(model_);
    model_ = nullptr;
  }
}

std::string Generator::buildFinalPrompt(const std::string& prompt_body,
                                        const std::string& system_prompt) const {
  if (!chat_template_.empty()) {
    return buildChatPrompt(prompt_body, system_prompt);
  }

  if (system_prompt.empty()) {
    return prompt_body;
  }

  std::string final_prompt = system_prompt;
  if (!final_prompt.empty() && final_prompt.back() != '\n') {
    final_prompt.push_back('\n');
  }
  final_prompt.push_back('\n');
  final_prompt.append(prompt_body);
  return final_prompt;
}

std::string Generator::buildChatPrompt(const std::string& prompt_body,
                                       const std::string& system_prompt) const {
  std::vector<llama_chat_message> chat;
  chat.reserve(system_prompt.empty() ? 1 : 2);

  if (!system_prompt.empty()) {
    chat.push_back({"system", system_prompt.c_str()});
  }
  chat.push_back({"user", prompt_body.c_str()});

  const size_t approx_chars = system_prompt.size() + prompt_body.size();
  std::vector<char> buf(std::max<size_t>(512, approx_chars * 4 + 256), '\0');
  int32_t needed =
      llama_chat_apply_template(chat_template_.c_str(), chat.data(), chat.size(), true, buf.data(), buf.size());
  if (needed < 0) {
    return system_prompt.empty() ? prompt_body : (system_prompt + "\n\n" + prompt_body);
  }

  if (needed >= static_cast<int32_t>(buf.size())) {
    buf.assign(static_cast<size_t>(needed) + 1, '\0');
    needed = llama_chat_apply_template(chat_template_.c_str(), chat.data(), chat.size(), true, buf.data(), buf.size());
    if (needed < 0) {
      return system_prompt.empty() ? prompt_body : (system_prompt + "\n\n" + prompt_body);
    }
  }

  return std::string(buf.data(), static_cast<size_t>(needed));
}

bool Generator::trimAtStopString(std::string& output_text) const {
  size_t first_match = std::string::npos;
  for (const auto& stop : config_.stop_strings) {
    if (stop.empty()) {
      continue;
    }

    const size_t pos = output_text.find(stop);
    if (pos != std::string::npos && (first_match == std::string::npos || pos < first_match)) {
      first_match = pos;
    }

    // Every configured stop string requires a leading newline so that the same
    // word mid-sentence does not trigger a cut. That guard also blinds them to
    // output that opens with the marker: an answer starting with "Evidence 4:"
    // matched nothing and ran to max_new_tokens. Treat position 0 as a match,
    // but only for the structural markers — a reply that legitimately opens
    // with "I don't know" must survive.
    if (isStructuralMarker(stop)) {
      const std::string at_start = stop.substr(1);
      if (output_text.compare(0, at_start.size(), at_start) == 0) {
        first_match = 0;
      }
    }
  }

  if (first_match == std::string::npos) {
    return false;
  }

  output_text.erase(first_match);
  return true;
}

std::string Generator::generate(const std::string& prompt_body) const {
  return generateWithTrace(prompt_body).output_text;
}

int Generator::computeRequiredBatchSize(const int prompt_token_count) const {
  return std::max(1, std::min(context_limit_, std::max(config_.n_batch, prompt_token_count)));
}

void Generator::recreateContextLocked(const int requested_n_batch) const {
  const int safe_n_batch = std::max(1, std::min(context_limit_, requested_n_batch));

  if (ctx_ != nullptr) {
    llama_free(ctx_);
    ctx_ = nullptr;
  }

  llama_context_params ctx_params = llama_context_default_params();
  ctx_params.n_ctx = static_cast<uint32_t>(context_limit_);
  ctx_params.n_batch = static_cast<uint32_t>(safe_n_batch);
  ctx_params.n_threads = std::max(1, config_.n_threads);
  ctx_params.n_threads_batch = std::max(1, config_.n_threads);
  ctx_params.no_perf = false;

  ctx_ = llama_init_from_model(model_, ctx_params);
  if (ctx_ == nullptr) {
    throw std::runtime_error("Failed to create llama context.");
  }
  ctx_n_batch_ = safe_n_batch;
}

GenerationTrace Generator::generateWithTrace(const std::string& prompt_body) const {
  return generateImpl(prompt_body, system_prompt_, nullptr);
}

GenerationTrace Generator::generateWithTrace(const std::string& prompt_body,
                                             const TokenCallback& on_token) const {
  return generateImpl(prompt_body, system_prompt_, on_token);
}

GenerationTrace Generator::generateWithSystemPrompt(const std::string& prompt_body,
                                                    const std::string& system_prompt_override) const {
  return generateImpl(prompt_body, system_prompt_override, nullptr);
}

GenerationTrace Generator::generateImpl(const std::string& prompt_body,
                                        const std::string& system_prompt,
                                        const TokenCallback& on_token) const {
  if (model_ == nullptr || vocab_ == nullptr) {
    throw std::runtime_error("Generator model is not initialized.");
  }

  GenerationTrace trace;
  trace.prompt_body = config_.reasoning_suppress_marker.empty()
                          ? prompt_body
                          : prompt_body + config_.reasoning_suppress_marker + "\n";
  trace.final_prompt = buildFinalPrompt(trace.prompt_body, system_prompt);

  const int n_prompt =
      -llama_tokenize(vocab_, trace.final_prompt.c_str(), static_cast<int32_t>(trace.final_prompt.size()), nullptr, 0, true, true);
  if (n_prompt <= 0) {
    throw std::runtime_error("Failed to tokenize prompt.");
  }
  trace.prompt_tokens_before_truncation = n_prompt;

  std::vector<llama_token> prompt_tokens(static_cast<size_t>(n_prompt));
  if (llama_tokenize(vocab_,
                     trace.final_prompt.c_str(),
                     static_cast<int32_t>(trace.final_prompt.size()),
                     prompt_tokens.data(),
                     static_cast<int32_t>(prompt_tokens.size()),
                     true,
                     true) < 0) {
    throw std::runtime_error("Failed to tokenize prompt into token buffer.");
  }

  const int generation_limit = std::max(1, config_.max_new_tokens);
  const int max_prompt_tokens = std::max(16, context_limit_ - generation_limit);
  if (static_cast<int>(prompt_tokens.size()) > max_prompt_tokens) {
    trace.prompt_truncated = true;
    // Drop from the MIDDLE (the overflowing retrieved context), preserving the
    // head (BOS + system prompt / template preamble) and the tail (the query and
    // output marker, which carry the actual question). Dropping the head instead
    // — as this did before — deletes BOS and the system instructions, so a
    // truncated run measures a structurally broken prompt rather than the RAG
    // condition under test.
    int head_keep = 0;
    if (!system_prompt.empty()) {
      const int system_tokens =
          -llama_tokenize(vocab_, system_prompt.c_str(),
                          static_cast<int32_t>(system_prompt.size()), nullptr, 0, true, true);
      if (system_tokens > 0) {
        head_keep = system_tokens;
      }
    }
    // Never let the head consume the whole budget; always leave room for the tail.
    head_keep = std::max(0, std::min(head_keep, max_prompt_tokens / 2));
    const int total = static_cast<int>(prompt_tokens.size());
    const int tail_keep = max_prompt_tokens - head_keep;
    prompt_tokens.erase(prompt_tokens.begin() + head_keep,
                        prompt_tokens.begin() + (total - tail_keep));
  }
  trace.prompt_tokens = static_cast<int>(prompt_tokens.size());
  const int required_n_batch = computeRequiredBatchSize(static_cast<int>(prompt_tokens.size()));

  auto sampler_params = llama_sampler_chain_default_params();
  std::unique_ptr<llama_sampler, SamplerDeleter> sampler(llama_sampler_chain_init(sampler_params));
  if (!sampler) {
    throw std::runtime_error("Failed to create llama sampler chain.");
  }

  if (config_.repetition_penalty_last_n != 0 && config_.repetition_penalty != 1.0F) {
    llama_sampler_chain_add(sampler.get(),
                            llama_sampler_init_penalties(
                                config_.repetition_penalty_last_n, config_.repetition_penalty, 0.0F, 0.0F));
  }

  if (config_.temperature <= 0.0F) {
    llama_sampler_chain_add(sampler.get(), llama_sampler_init_greedy());
  } else {
    if (config_.top_k > 0) {
      llama_sampler_chain_add(sampler.get(), llama_sampler_init_top_k(config_.top_k));
    }
    if (config_.top_p > 0.0F && config_.top_p <= 1.0F) {
      llama_sampler_chain_add(sampler.get(), llama_sampler_init_top_p(config_.top_p, 1));
    }
    llama_sampler_chain_add(sampler.get(), llama_sampler_init_temp(config_.temperature));
    llama_sampler_chain_add(sampler.get(), llama_sampler_init_dist(config_.seed));
  }

  std::lock_guard<std::mutex> lock(ctx_mutex_);
  if (ctx_ == nullptr || required_n_batch > ctx_n_batch_) {
    recreateContextLocked(required_n_batch);
  }
  llama_memory_clear(llama_get_memory(ctx_), false);

  llama_batch batch = llama_batch_get_one(prompt_tokens.data(), static_cast<int32_t>(prompt_tokens.size()));

  if (llama_model_has_encoder(model_)) {
    if (llama_encode(ctx_, batch) != 0) {
      throw std::runtime_error("llama_encode failed.");
    }
    llama_token decoder_start = llama_model_decoder_start_token(model_);
    if (decoder_start == LLAMA_TOKEN_NULL) {
      decoder_start = llama_vocab_bos(vocab_);
    }
    batch = llama_batch_get_one(&decoder_start, 1);
  }

  trace.output_text.reserve(1024);
  int output_tokens = 0;

  for (int generated = 0; generated < generation_limit; ++generated) {
    if (llama_decode(ctx_, batch) != 0) {
      throw std::runtime_error("llama_decode failed.");
    }

    llama_token token = llama_sampler_sample(sampler.get(), ctx_, -1);
    if (llama_vocab_is_eog(vocab_, token)) {
      break;
    }

    char piece[256];
    const int n = llama_token_to_piece(vocab_, token, piece, sizeof(piece), 0, true);
    if (n < 0) {
      throw std::runtime_error("llama_token_to_piece failed.");
    }
    trace.output_text.append(piece, piece + n);
    ++output_tokens;
    if (on_token && n > 0) {
      on_token(std::string(piece, piece + n));
    }
    if (trimAtStopString(trace.output_text)) {
      break;
    }

    // NOTE: do not call llama_sampler_accept here. llama_sampler_sample()
    // already accepts the sampled token internally; a second accept double-counts
    // every token in the repetition-penalty ring buffer, halving the effective
    // penalty window.
    batch = llama_batch_get_one(&token, 1);
  }
  trace.output_tokens = output_tokens;
  stripReasoningBlock(trace.output_text);

  return trace;
}

}  // namespace rag
