#include "beir_export.hpp"

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

#include "query_response_json.hpp"

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
    } else if (j.contains("body") && j.at("body").is_string()) {
      entry.text = j.at("body").get<std::string>();
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

// Reference policy for BEIR export:
// - qrels positive corpus-id docs only
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
                                  const std::string& dataset,
                                  const std::string& split,
                                  const std::string& user_input,
                                  const RagResult& result,
                                  const double memory_mb,
                                  const std::vector<std::string>& qrel_doc_ids,
                                  const std::string& reference) {
  std::vector<std::string> retrieved_contexts;
  std::vector<int> retrieved_ranks;
  std::vector<std::string> retrieved_doc_ids;
  std::vector<float> retrieved_scores;
  retrieved_contexts.reserve(result.contexts.size());
  retrieved_ranks.reserve(result.contexts.size());
  retrieved_doc_ids.reserve(result.contexts.size());
  retrieved_scores.reserve(result.contexts.size());

  for (size_t i = 0; i < result.contexts.size(); ++i) {
    const auto& c = result.contexts[i];
    retrieved_contexts.push_back(c.chunk.text);
    retrieved_ranks.push_back(static_cast<int>(i + 1));
    retrieved_doc_ids.push_back(c.chunk.doc_id);
    retrieved_scores.push_back(c.score);
  }

  nlohmann::json row;
  row["sample_id"] = sample_id;
  row["dataset"] = dataset;
  row["split"] = split;
  row["advanced"] = advancedModeToString(result.advanced_mode);
  row["user_input"] = user_input;
  row["response"] = result.answer;
  row["retrieved_contexts"] = retrieved_contexts;
  row["reference"] = reference;
  row["qrel_doc_ids"] = qrel_doc_ids;
  row["retrieved_ranks"] = retrieved_ranks;
  row["retrieved_doc_ids"] = retrieved_doc_ids;
  row["retrieved_scores"] = retrieved_scores;
  row["trace"] = query::buildTraceJson(result, memory_mb);

  return row;
}

void writeManifestJson(const BeirExportOptions& options,
                       const ModelConfig& model_cfg,
                       const RetrievalConfig& retrieval_cfg,
                       const std::string& model_config_path,
                       const std::string& retrieval_config_path,
                       const size_t rows_written) {
  const std::string manifest_path = buildBeirManifestJsonPath(options.output_jsonl_path);
  const fs::path out_path(manifest_path);
  if (out_path.has_parent_path()) {
    fs::create_directories(out_path.parent_path());
  }

  nlohmann::json manifest;
  manifest["mode"] = "batch_export_jsonl";
  manifest["status"] = "ok";
  manifest["dataset"] = options.dataset;
  manifest["split"] = options.split;
  manifest["advanced"] = advancedModeToString(options.advanced);
  manifest["output_jsonl"] = options.output_jsonl_path;
  manifest["rows_written"] = rows_written;
  manifest["inputs"] =
      nlohmann::json{{"questions", options.questions_path},
                     {"qrels", options.qrels_path},
                     {"corpus", options.corpus_path}};
  manifest["reference_policy"] = "qrels_positive_docs_title_plus_text_concat";
  manifest["run_config"] =
      query::buildRunConfigJson(model_cfg, retrieval_cfg, model_config_path, retrieval_config_path, options.advanced);

  std::ofstream out(manifest_path);
  if (!out) {
    throw std::runtime_error("Failed to open manifest_json for write: " + manifest_path);
  }
  out << manifest.dump() << '\n';
}

}  // namespace

std::string buildBeirManifestJsonPath(const std::string& output_jsonl_path) {
  return output_jsonl_path + ".manifest.json";
}

size_t exportBeirPredictionsJsonl(const RagPipeline& pipeline,
                                  const ModelConfig& model_cfg,
                                  const RetrievalConfig& retrieval_cfg,
                                  const BeirExportOptions& options,
                                  const std::string& model_config_path,
                                  const std::string& retrieval_config_path) {
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
  std::cerr << "[info] dataset=" << options.dataset << '\n';
  std::cerr << "[info] split=" << options.split << '\n';

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

    const RagResult result = pipeline.run(query_it->second, options.advanced);
    const double memory_mb = query::getResidentMemoryMb();
    if (isBlank(result.answer)) {
      throw std::runtime_error("Empty response generated for sample_id=" + sample_id);
    }

    nlohmann::json row =
        buildPredictionRow(sample_id,
                           options.dataset,
                           options.split,
                           query_it->second,
                           result,
                           memory_mb,
                           qrel_doc_ids,
                           reference);
    out << row.dump() << '\n';
    ++written;

    std::cerr << "[batch] wrote sample_id=" << sample_id << '\n';
  }

  std::cerr << "[batch] output_jsonl=" << options.output_jsonl_path << " rows=" << written << '\n';
  writeManifestJson(options, model_cfg, retrieval_cfg, model_config_path, retrieval_config_path, written);
  return written;
}

}  // namespace rag
