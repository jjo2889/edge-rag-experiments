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
                                  const int64_t model_load_ms,
                                  const std::vector<std::string>& qrel_doc_ids,
                                  const std::string& reference) {
  std::vector<std::string> retrieved_contexts;
  std::vector<int> retrieved_ranks;
  std::vector<std::string> retrieved_doc_ids;
  std::vector<nlohmann::json> retrieved_scores;
  std::vector<nlohmann::json> retrieved_rerank_scores;
  std::vector<int> retrieved_sentence_begins;
  std::vector<int> retrieved_sentence_ends;
  retrieved_contexts.reserve(result.contexts.size());
  retrieved_ranks.reserve(result.contexts.size());
  retrieved_doc_ids.reserve(result.contexts.size());
  retrieved_scores.reserve(result.contexts.size());
  retrieved_rerank_scores.reserve(result.contexts.size());
  retrieved_sentence_begins.reserve(result.contexts.size());
  retrieved_sentence_ends.reserve(result.contexts.size());

  // For contextual_compression the prompt contexts are one synthetic chunk with
  // no doc_id, so retrieval metrics must be scored against what was actually
  // retrieved. Use pre_transform_contexts for the retrieval columns whenever the
  // mode recorded them; `retrieved_contexts` still holds the texts the generator
  // saw.
  const std::vector<RetrievedChunk>& provenance =
      result.pre_transform_contexts.empty() ? result.contexts : result.pre_transform_contexts;

  for (const auto& c : result.contexts) {
    retrieved_contexts.push_back(c.chunk.text);
  }
  for (size_t i = 0; i < provenance.size(); ++i) {
    const auto& c = provenance[i];
    retrieved_ranks.push_back(static_cast<int>(i + 1));
    retrieved_doc_ids.push_back(c.chunk.doc_id);
    retrieved_scores.push_back(c.has_retrieval_score ? nlohmann::json(c.score) : nlohmann::json(nullptr));
    retrieved_rerank_scores.push_back(c.has_rerank_score ? nlohmann::json(c.rerank_score) : nlohmann::json(nullptr));
    retrieved_sentence_begins.push_back(c.chunk.sentence_begin);
    retrieved_sentence_ends.push_back(c.chunk.sentence_end);
  }

  nlohmann::json row;
  row["sample_id"] = sample_id;
  row["dataset"] = dataset;
  row["split"] = split;
  row["advanced"] = advancedModeToString(result.advanced_mode);
  row["user_input"] = user_input;
  row["retrieval_query"] = result.retrieval_query;
  row["response"] = result.answer;
  row["retrieved_contexts"] = retrieved_contexts;
  row["reference"] = reference;
  row["qrel_doc_ids"] = qrel_doc_ids;
  row["retrieved_ranks"] = retrieved_ranks;
  row["retrieved_doc_ids"] = retrieved_doc_ids;
  row["retrieved_scores"] = retrieved_scores;
  row["retrieved_rerank_scores"] = retrieved_rerank_scores;
  row["retrieved_sentence_begins"] = retrieved_sentence_begins;
  row["retrieved_sentence_ends"] = retrieved_sentence_ends;
  // Present only for modes that replace the retrieved passages (compression);
  // null elsewhere, where retrieved_contexts is already the retrieval output.
  row["pre_transform_contexts"] = result.pre_transform_contexts.empty()
                                      ? nlohmann::json(nullptr)
                                      : query::buildContextRows(result.pre_transform_contexts);
  row["trace"] = query::buildTraceJson(result, memory_mb, model_load_ms);

  return row;
}

void writeManifestJson(const BeirExportOptions& options,
                       const ModelConfig& model_cfg,
                       const RetrievalConfig& retrieval_cfg,
                       const std::string& model_config_path,
                       const std::string& retrieval_config_path,
                       const size_t rows_written,
                       const std::vector<std::string>& skipped_sample_ids) {
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
  manifest["rows_skipped"] = skipped_sample_ids.size();
  manifest["skipped_sample_ids"] = skipped_sample_ids;
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
                                  const std::string& retrieval_config_path,
                                  const int64_t model_load_ms) {
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

  // Write to a temp path and rename on success so a crash mid-batch never
  // leaves a partial JSONL at the final path masquerading as a complete run.
  // The manifest is written only after the rename, so its existence is the
  // completion signal for downstream consumers.
  const std::string tmp_jsonl_path = options.output_jsonl_path + ".tmp";
  std::ofstream out(tmp_jsonl_path, std::ios::trunc);
  if (!out) {
    throw std::runtime_error("Failed to open output_jsonl for write: " + tmp_jsonl_path);
  }

  std::cerr << "[info] reference_policy=qrels_positive_docs_title_plus_text_concat\n";
  std::cerr << "[info] dataset=" << options.dataset << '\n';
  std::cerr << "[info] split=" << options.split << '\n';

  size_t written = 0;
  std::vector<std::string> skipped_sample_ids;
  for (const auto& sample_id : qrel_query_order) {
    // Bound on rows ATTEMPTED, not rows written: now that a failing row is
    // skipped rather than fatal, a limit on writes would let a run where every
    // row fails walk the entire split instead of stopping after N queries.
    const size_t attempted = written + skipped_sample_ids.size();
    if (options.limit > 0 && static_cast<int>(attempted) >= options.limit) {
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
    if (!qrel_doc_ids.empty() && reference.empty()) {
      std::cerr << "[warn] sample_id=" << sample_id << " has " << qrel_doc_ids.size()
                << " positive qrel doc(s) but none matched the corpus; reference is EMPTY "
                   "(RAGAS reference-based metrics will be meaningless for this row). "
                   "Check corpus file / id scheme.\n";
    }

    // One bad row (pipeline exception, empty generation) must not abort a
    // multi-hour batch: skip it with a warning, record it in the manifest, and
    // keep going. Downstream sees the skip via rows_skipped/skipped_sample_ids.
    try {
      const RagResult result = pipeline.run(query_it->second, options.advanced);
      const double memory_mb = query::getResidentMemoryMb();
      if (isBlank(result.answer)) {
        throw std::runtime_error("empty generated response");
      }

      nlohmann::json row =
          buildPredictionRow(sample_id,
                             options.dataset,
                             options.split,
                             query_it->second,
                             result,
                             memory_mb,
                             model_load_ms,
                             qrel_doc_ids,
                             reference);
      out << query::dumpSafe(row) << '\n';
      ++written;

      std::cerr << "[batch] wrote sample_id=" << sample_id << '\n';
    } catch (const std::exception& e) {
      skipped_sample_ids.push_back(sample_id);
      std::cerr << "[batch][warn] SKIPPED sample_id=" << sample_id << ": " << e.what() << '\n';
    }
  }

  out.flush();
  if (!out) {
    throw std::runtime_error("Failed while writing output_jsonl: " + tmp_jsonl_path);
  }
  out.close();

  // A run where nothing survived is a failed run, not an empty success: leave
  // the temp file in place (no rename, no manifest) so no downstream step can
  // read a zero-row export as a completed one.
  if (written == 0 && !skipped_sample_ids.empty()) {
    throw std::runtime_error("Batch export produced 0 rows; all " +
                             std::to_string(skipped_sample_ids.size()) +
                             " attempted sample(s) failed. Partial output left at " + tmp_jsonl_path);
  }
  if (!skipped_sample_ids.empty()) {
    std::cerr << "[batch][warn] " << skipped_sample_ids.size() << " of "
              << (written + skipped_sample_ids.size())
              << " attempted sample(s) were SKIPPED; per-mode metrics are computed over the "
                 "written rows only (see manifest skipped_sample_ids).\n";
  }

  fs::rename(tmp_jsonl_path, options.output_jsonl_path);

  std::cerr << "[batch] output_jsonl=" << options.output_jsonl_path << " rows=" << written
            << " skipped=" << skipped_sample_ids.size() << '\n';
  writeManifestJson(options, model_cfg, retrieval_cfg, model_config_path, retrieval_config_path, written,
                    skipped_sample_ids);
  return written;
}

}  // namespace rag
