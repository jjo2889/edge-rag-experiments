#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_URL="${DATASET_URL:-https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nq.zip}"
CORPUS_OUT="${ROOT_DIR}/data/beir/nq_corpus/corpus.jsonl"
QUERIES_OUT="${ROOT_DIR}/data/beir/nq/queries.jsonl"
QRELS_OUT="${ROOT_DIR}/data/beir/nq/qrels/test.tsv"
FORCE=0
FULL=0

# Default subset policy:
# - first 500 queries in qrels order
# - all positive docs for those queries
# - first 20000 non-gold docs as distractors
SUBSET_QUERY_LIMIT=500
SUBSET_DISTRACTOR_DOCS=20000

usage() {
  cat <<'USAGE'
Usage: scripts/setup_nq_data.sh [--force] [--full]

Downloads BEIR nq and prepares files into:
  data/beir/nq_corpus/corpus.jsonl
  data/beir/nq/queries.jsonl
  data/beir/nq/qrels/test.tsv

Default behavior prepares a fixed-size subset.
Use --full to extract the full BEIR nq files instead.
USAGE
}

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "[fail] missing command: $1"
    exit 1
  fi
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --force)
      FORCE=1
      shift
      ;;
    --full)
      FULL=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[fail] unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

need_cmd curl
need_cmd python3
need_cmd mktemp

if [ -f "${CORPUS_OUT}" ] && [ -f "${QUERIES_OUT}" ] && [ -f "${QRELS_OUT}" ] && [ "${FORCE}" -ne 1 ]; then
  echo "[skip] already exists:"
  echo "  ${CORPUS_OUT}"
  echo "  ${QUERIES_OUT}"
  echo "  ${QRELS_OUT}"
  echo "[hint] rerun with --force to refresh"
  exit 0
fi

TMP_DIR="$(mktemp -d)"
ARCHIVE_PATH="${TMP_DIR}/nq.zip"

cleanup() {
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

if [ -f "${DATASET_URL}" ]; then
  echo "[copy] ${DATASET_URL}"
  cp "${DATASET_URL}" "${ARCHIVE_PATH}"
else
  echo "[download] ${DATASET_URL}"
  curl -fL "${DATASET_URL}" -o "${ARCHIVE_PATH}"
fi

python3 - "${ARCHIVE_PATH}" "${CORPUS_OUT}" "${QUERIES_OUT}" "${QRELS_OUT}" "${FULL}" "${SUBSET_QUERY_LIMIT}" "${SUBSET_DISTRACTOR_DOCS}" <<'PY'
import csv
import json
import pathlib
import shutil
import sys
import zipfile

archive_path = pathlib.Path(sys.argv[1])
corpus_out = pathlib.Path(sys.argv[2])
queries_out = pathlib.Path(sys.argv[3])
qrels_out = pathlib.Path(sys.argv[4])
full = sys.argv[5] == "1"
subset_query_limit = int(sys.argv[6])
subset_distractor_docs = int(sys.argv[7])


def pick_member(names, suffixes):
    matches = []
    for name in names:
        for suffix in suffixes:
            if name.endswith(suffix):
                matches.append(name)
                break
    if not matches:
        return None
    return sorted(matches, key=len)[0]


def json_value_to_str(value):
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return str(value)
    return ""


with zipfile.ZipFile(archive_path) as zf:
    names = zf.namelist()
    corpus_member = pick_member(names, ["/corpus.jsonl", "corpus.jsonl"])
    queries_member = pick_member(names, ["/queries.jsonl", "queries.jsonl"])
    qrels_member = pick_member(names, ["/qrels/test.tsv", "qrels/test.tsv", "/qrels.tsv", "qrels.tsv"])

    if corpus_member is None:
        raise SystemExit("[fail] corpus.jsonl not found in nq archive")
    if queries_member is None:
        raise SystemExit("[fail] queries.jsonl not found in nq archive")
    if qrels_member is None:
        raise SystemExit("[fail] qrels/test.tsv not found in nq archive")

    if full:
        for member, out_path in (
            (corpus_member, corpus_out),
            (queries_member, queries_out),
            (qrels_member, qrels_out),
        ):
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, out_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            print(f"[ok] extracted {member} -> {out_path}")
        raise SystemExit(0)

    selected_query_ids = []
    selected_query_id_set = set()
    selected_qrels = []
    selected_gold_doc_ids = set()

    with zf.open(qrels_member) as src:
        reader = csv.reader((line.decode("utf-8") for line in src), delimiter="\t")
        for row in reader:
            if not row or row[0] == "query-id" or len(row) < 3:
                continue
            qid, docid, score = row[:3]
            try:
                relevance = int(score)
            except ValueError:
                continue
            if relevance <= 0:
                continue

            if qid not in selected_query_id_set:
                if len(selected_query_ids) >= subset_query_limit:
                    continue
                selected_query_ids.append(qid)
                selected_query_id_set.add(qid)

            if qid in selected_query_id_set:
                selected_qrels.append((qid, docid, relevance))
                selected_gold_doc_ids.add(docid)

    if not selected_query_ids:
        raise SystemExit("[fail] no positive qrels found in nq archive")

    selected_query_lines = []
    with zf.open(queries_member) as src:
        for raw_line in src:
            line = raw_line.decode("utf-8")
            if not line.strip():
                continue
            row = json.loads(line)
            qid = json_value_to_str(row.get("_id", row.get("id")))
            if qid in selected_query_id_set:
                selected_query_lines.append(line if line.endswith("\n") else line + "\n")

    if len(selected_query_lines) != len(selected_query_ids):
        raise SystemExit("[fail] some selected queries were not found in queries.jsonl")

    gold_doc_lines = []
    found_gold_doc_ids = set()
    distractor_lines = []

    with zf.open(corpus_member) as src:
        for raw_line in src:
            line = raw_line.decode("utf-8")
            if not line.strip():
                continue
            row = json.loads(line)
            doc_id = json_value_to_str(row.get("_id", row.get("id")))
            if not doc_id:
                continue
            normalized_line = line if line.endswith("\n") else line + "\n"

            if doc_id in selected_gold_doc_ids:
                if doc_id not in found_gold_doc_ids:
                    gold_doc_lines.append(normalized_line)
                    found_gold_doc_ids.add(doc_id)
                continue

            if len(distractor_lines) < subset_distractor_docs:
                distractor_lines.append(normalized_line)

    missing_gold = sorted(selected_gold_doc_ids.difference(found_gold_doc_ids))
    if missing_gold:
        preview = ", ".join(missing_gold[:5])
        raise SystemExit(f"[fail] missing gold docs in corpus.jsonl: {preview}")

    corpus_out.parent.mkdir(parents=True, exist_ok=True)
    with corpus_out.open("w", encoding="utf-8") as dst:
        for line in gold_doc_lines:
            dst.write(line)
        for line in distractor_lines:
            dst.write(line)
    print(f"[ok] wrote subset corpus -> {corpus_out}")

    queries_out.parent.mkdir(parents=True, exist_ok=True)
    with queries_out.open("w", encoding="utf-8") as dst:
        for line in selected_query_lines:
            dst.write(line)
    print(f"[ok] wrote subset queries -> {queries_out}")

    qrels_out.parent.mkdir(parents=True, exist_ok=True)
    with qrels_out.open("w", encoding="utf-8", newline="") as dst:
        writer = csv.writer(dst, delimiter="\t")
        writer.writerow(["query-id", "corpus-id", "score"])
        for row in selected_qrels:
            writer.writerow(row)
    print(f"[ok] wrote subset qrels -> {qrels_out}")
PY

if [ "${FULL}" -eq 1 ]; then
  echo "[done] full NQ data ready"
else
  echo "[done] subset NQ data ready"
  echo "  queries=${SUBSET_QUERY_LIMIT}"
  echo "  distractor_docs=${SUBSET_DISTRACTOR_DOCS}"
fi
echo "  corpus:  data/beir/nq_corpus/corpus.jsonl"
echo "  queries: data/beir/nq/queries.jsonl"
echo "  qrels:   data/beir/nq/qrels/test.tsv"
