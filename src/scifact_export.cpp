#include "scifact_export.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

namespace rag {

namespace {
namespace fs = std::filesystem;

struct QueryEntry {
  std::string id;
  std::string text;
};

struct CorpusEntry {
  std::string id;
  std::string title;
  std::string text;
};

std::string jsonValueToString(const nlohmann::json& value) {
  if (value.is_string()) {
    return value.get<std::string>();
  }
  if (value.is_number_integer()) {
    return std::to_string(value.get<int64_t>());
  }
  if (value.is_number_unsigned()) {
    return std::to_string(value.get<uint64_t>());
  }
  if (value.is_number_float()) {
    std::ostringstream oss;
    oss << value.get<double>();
    return oss.str();
  }
  return {};
}

std::vector<QueryEntry> loadQueriesJsonl(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("Failed to open queries file: " + path);
  }

  std::vector<QueryEntry> queries;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) {
      continue;
    }

    const auto j = nlohmann::json::parse(line);
    std::string id;
    if (j.contains("_id")) {
      id = jsonValueToString(j.at("_id"));
    } else if (j.contains("id")) {
      id = jsonValueToString(j.at("id"));
    }

    std::string text;
    if (j.contains("text") && j.at("text").is_string()) {
      text = j.at("text").get<std::string>();
    } else if (j.contains("query") && j.at("query").is_string()) {
      text = j.at("query").get<std::string>();
    }

    if (!id.empty() && !text.empty()) {
      queries.push_back(QueryEntry{std::move(id), std::move(text)});
    }
  }

  return queries;
}

std::unordered_map<std::string, CorpusEntry> loadCorpusJsonl(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("Failed to open corpus file: " + path);
  }

  std::unordered_map<std::string, CorpusEntry> corpus_by_id;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) {
      continue;
    }

    const auto j = nlohmann::json::parse(line);
    std::string id;
    if (j.contains("_id")) {
      id = jsonValueToString(j.at("_id"));
    } else if (j.contains("id")) {
      id = jsonValueToString(j.at("id"));
    }
    if (id.empty()) {
      continue;
    }

    CorpusEntry entry;
    entry.id = id;
    if (j.contains("title") && j.at("title").is_string()) {
      entry.title = j.at("title").get<std::string>();
    }
    if (j.contains("text") && j.at("text").is_string()) {
      entry.text = j.at("text").get<std::string>();
    } else if (j.contains("contents") && j.at("contents").is_string()) {
      entry.text = j.at("contents").get<std::string>();
    }

    corpus_by_id[id] = std::move(entry);
  }

  return corpus_by_id;
}

std::unordered_map<std::string, std::vector<std::string>> loadPositiveQrels(const std::string& path,
                                                                             std::vector<std::string>& query_order) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("Failed to open qrels file: " + path);
  }

  std::unordered_map<std::string, std::vector<std::string>> qrels;
  std::unordered_set<std::string> seen_query;

  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) {
      continue;
    }

    std::stringstream ss(line);
    std::string qid;
    std::string docid;
    std::string score_str;
    if (!std::getline(ss, qid, '\t') || !std::getline(ss, docid, '\t') || !std::getline(ss, score_str, '\t')) {
      continue;
    }
    if (qid == "query-id") {
      continue;
    }

    int score = 0;
    try {
      score = std::stoi(score_str);
    } catch (const std::exception&) {
      continue;
    }
    if (score <= 0) {
      continue;
    }

    if (!seen_query.count(qid)) {
      query_order.push_back(qid);
      seen_query.insert(qid);
    }
    auto& docs = qrels[qid];
    if (std::find(docs.begin(), docs.end(), docid) == docs.end()) {
      docs.push_back(docid);
    }
  }

  return qrels;
}

// Reference policy for SciFact export:
// - qrels(test) positive corpus-id docs only
// - each doc contributes "title + '\n' + text"
// - docs are joined with "\n\n"
std::string buildReferenceFromQrels(const std::vector<std::string>& qrel_doc_ids,
                                    const std::unordered_map<std::string, CorpusEntry>& corpus_by_id) {
  std::ostringstream oss;
  bool first = true;
  for (const auto& doc_id : qrel_doc_ids) {
    const auto it = corpus_by_id.find(doc_id);
    if (it == corpus_by_id.end()) {
      continue;
    }

    if (!first) {
      oss << "\n\n";
    }
    first = false;

    if (!it->second.title.empty()) {
      oss << it->second.title << '\n';
    }
    oss << it->second.text;
  }

  return oss.str();
}

bool isBlank(const std::string& s) {
  for (const unsigned char ch : s) {
    if (!std::isspace(ch)) {
      return false;
    }
  }
  return true;
}

nlohmann::json buildPredictionRow(const std::string& sample_id,
                                  const std::string& user_input,
                                  const RagResult& result,
                                  const std::vector<std::string>& qrel_doc_ids,
                                  const std::string& reference,
                                  const ModelConfig& model_cfg,
                                  const RetrievalConfig& retrieval_cfg) {
  std::vector<std::string> retrieved_contexts;
  std::vector<std::string> retrieved_doc_ids;
  std::vector<float> retrieved_scores;
  retrieved_contexts.reserve(result.contexts.size());
  retrieved_doc_ids.reserve(result.contexts.size());
  retrieved_scores.reserve(result.contexts.size());

  for (const auto& c : result.contexts) {
    retrieved_contexts.push_back(c.chunk.text);
    retrieved_doc_ids.push_back(c.chunk.doc_id);
    retrieved_scores.push_back(c.score);
  }

  nlohmann::json row;
  row["sample_id"] = sample_id;
  row["dataset"] = "scifact";
  row["split"] = "test";
  row["user_input"] = user_input;
  row["response"] = result.answer;
  row["retrieved_contexts"] = retrieved_contexts;
  row["reference"] = reference;
  row["qrel_doc_ids"] = qrel_doc_ids;
  row["retrieved_doc_ids"] = retrieved_doc_ids;
  row["retrieved_scores"] = retrieved_scores;

  nlohmann::json generation_config;
  generation_config["temperature"] = model_cfg.temperature;
  generation_config["top_p"] = model_cfg.top_p;
  generation_config["max_new_tokens"] = model_cfg.max_new_tokens;
  generation_config["seed"] = model_cfg.seed;
  row["generation_config"] = generation_config;

  nlohmann::json retrieval_config;
  retrieval_config["top_k"] = retrieval_cfg.top_k;
  retrieval_config["chunk_size"] = retrieval_cfg.chunk_size;
  retrieval_config["chunk_overlap"] = retrieval_cfg.chunk_overlap;
  row["retrieval_config"] = retrieval_config;

  nlohmann::json trace;
  trace["prompt_tokens_before_truncation"] = result.prompt_tokens_before_truncation;
  trace["prompt_tokens"] = result.prompt_tokens;
  trace["output_tokens"] = result.output_tokens;
  trace["prompt_truncated"] = result.prompt_truncated;
  trace["retrieve_ms"] = result.retrieve_ms;
  trace["prompt_build_ms"] = result.prompt_build_ms;
  trace["generate_ms"] = result.generate_ms;
  trace["total_ms"] = result.total_ms;
  trace["output_tokens_per_sec"] = result.output_tokens_per_sec;
  row["trace"] = trace;

  return row;
}

}  // namespace

void exportScifactPredictionsJsonl(const RagPipeline& pipeline,
                                   const ModelConfig& model_cfg,
                                   const RetrievalConfig& retrieval_cfg,
                                   const ScifactExportOptions& options) {
  const auto queries = loadQueriesJsonl(options.questions_path);
  std::unordered_map<std::string, std::string> query_text_by_id;
  query_text_by_id.reserve(queries.size());
  for (const auto& q : queries) {
    query_text_by_id[q.id] = q.text;
  }

  std::vector<std::string> qrel_query_order;
  const auto qrels = loadPositiveQrels(options.qrels_path, qrel_query_order);
  const auto corpus_by_id = loadCorpusJsonl(options.corpus_path);

  const fs::path out_path(options.output_jsonl_path);
  if (out_path.has_parent_path()) {
    fs::create_directories(out_path.parent_path());
  }

  std::ofstream out(options.output_jsonl_path);
  if (!out) {
    throw std::runtime_error("Failed to open output_jsonl for write: " + options.output_jsonl_path);
  }

  std::cerr << "[info] reference_policy=qrels_positive_docs_title_plus_text_concat\n";
  std::cerr << "[info] qrels_split=test\n";

  size_t written = 0;
  for (const auto& sample_id : qrel_query_order) {
    if (options.limit > 0 && static_cast<int>(written) >= options.limit) {
      break;
    }

    const auto query_it = query_text_by_id.find(sample_id);
    if (query_it == query_text_by_id.end()) {
      std::cerr << "[warn] query id not found in queries.jsonl: " << sample_id << '\n';
      continue;
    }

    const auto qrel_it = qrels.find(sample_id);
    const std::vector<std::string> qrel_doc_ids =
        (qrel_it == qrels.end()) ? std::vector<std::string>() : qrel_it->second;
    const std::string reference = buildReferenceFromQrels(qrel_doc_ids, corpus_by_id);

    const RagResult result = pipeline.run(query_it->second);
    if (isBlank(result.answer)) {
      throw std::runtime_error("Empty response generated for sample_id=" + sample_id);
    }

    nlohmann::json row =
        buildPredictionRow(sample_id, query_it->second, result, qrel_doc_ids, reference, model_cfg, retrieval_cfg);
    out << row.dump() << '\n';
    ++written;

    std::cerr << "[batch] wrote sample_id=" << sample_id << '\n';
  }

  std::cerr << "[batch] output_jsonl=" << options.output_jsonl_path << " rows=" << written << '\n';
}

}  // namespace rag
