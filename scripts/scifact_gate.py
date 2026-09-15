#!/usr/bin/env python3
"""SciFact 게이트 — 무맥락/정답근거 밴드를 재는 소형 실험.

`docs/research_proposal.md` §4.5의 기준선을 실제로 측정한다. 세 조건 중
closed_book 과 oracle 은 검색을 건너뛰므로, 이 스크립트는 인덱스도 임베딩
모델도 쓰지 않는다 — 생성기 하나만 있으면 돈다.

밴드 = oracle 정확도 − closed_book 정확도. 그 폭이 0에 가까우면 그 모델은
근거를 읽지 않는 것이고, 위에 어떤 검색 기법을 얹어도 생성 지표가 움직이지
않는다.

사용:
    python3 scripts/scifact_gate.py \
        --model-config configs/model_qwen3_1.7b.yaml \
        --context-source oracle --limit 50 \
        --out reports/gate/qwen3-1.7b.oracle.jsonl

출력 JSONL 한 줄이 한 문항이고, 마지막에 요약을 stderr 로 찍는다. 같은
--out 으로 다시 돌리면 이미 있는 문항은 건너뛴다(중단 후 재개).
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import subprocess
import sys
import tempfile
import time

# 데이터셋 라벨(SUPPORT/CONTRADICT)과 프롬프트가 요구하는 토큰
# (SUPPORTS/CONTRADICTS/NOT_ENOUGH_INFO)이 다르다. 채점은 후자로 통일한다.
LABELS = ("SUPPORTS", "CONTRADICTS", "NOT_ENOUGH_INFO")
_FROM_DATASET = {"SUPPORT": "SUPPORTS", "CONTRADICT": "CONTRADICTS"}

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _path(rel: str) -> str:
    return rel if os.path.isabs(rel) else os.path.join(REPO_ROOT, rel)


def load_dataset(queries_path: str, qrels_path: str, corpus_path: str):
    """test split 문항을 (claim, gold 라벨, gold 문서 본문)으로 만들어 돌려준다."""
    qrels = collections.defaultdict(list)
    with open(_path(qrels_path), encoding="utf-8") as handle:
        next(handle)  # header
        for line in handle:
            query_id, doc_id, _score = line.rstrip("\n").split("\t")
            qrels[query_id].append(doc_id)

    wanted_docs = {doc for docs in qrels.values() for doc in docs}
    corpus = {}
    with open(_path(corpus_path), encoding="utf-8") as handle:
        for line in handle:
            doc = json.loads(line)
            if doc["_id"] in wanted_docs:
                # BEIR batch export 와 같은 결합 정책 (README 실험 고정 정책).
                corpus[doc["_id"]] = f"{doc['title']}\n{doc['text']}"

    items = []
    with open(_path(queries_path), encoding="utf-8") as handle:
        for line in handle:
            query = json.loads(line)
            query_id = query["_id"]
            if query_id not in qrels:
                continue
            # 근거 문서가 없거나 라벨이 안 붙은 클레임이 NOT_ENOUGH_INFO 다.
            # qrels 에는 문서가 있다 — "이 초록을 봤지만 판단이 안 된다"는 뜻이라
            # oracle 컨텍스트는 그 문서를 그대로 넣어야 과제가 보존된다.
            labels = {
                event["label"]
                for events in (query.get("metadata") or {}).values()
                for event in events
            }
            if len(labels) > 1:
                continue  # 한 클레임에 SUPPORT/CONTRADICT 가 섞인 건은 제외
            gold = _FROM_DATASET[labels.pop()] if labels else "NOT_ENOUGH_INFO"
            items.append(
                {
                    "sample_id": query_id,
                    "question": query["text"],
                    "gold": gold,
                    "oracle_contexts": [
                        {"doc_id": doc_id, "text": corpus[doc_id]}
                        for doc_id in qrels[query_id]
                        if doc_id in corpus
                    ],
                }
            )
    items.sort(key=lambda item: int(item["sample_id"]))
    return items


def parse_verdict(answer: str):
    """모델 답변에서 판정을 뽑는다. 실패하면 None.

    프롬프트는 JSON 을 요구하지만 작은 모델은 형식을 자주 어긴다. JSON 을
    먼저 시도하고, 실패하면 원문에서 라벨 토큰을 찾는다. 두 경로를 따로
    세어서 형식 준수율 자체를 지표로 남긴다.
    """
    text = (answer or "").strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            verdict = json.loads(match.group(0)).get("verdict", "")
            if verdict in LABELS:
                return verdict, "json"
        except (ValueError, AttributeError):
            pass
    # NOT_ENOUGH_INFO 가 먼저 걸려야 한다. SUPPORTS 를 먼저 찾으면
    # "does not SUPPORT" 같은 서술에서 오검출된다.
    for label in ("NOT_ENOUGH_INFO", "CONTRADICTS", "SUPPORTS"):
        if label in text:
            return label, "regex"
    return None, "fail"


def run_one(binary, model_config, retrieval_config, item, context_source):
    request = {
        "dataset": "scifact",
        "split": "test",
        "sample_id": item["sample_id"],
        "question": item["question"],
        "context_source": context_source,
    }
    if context_source == "oracle":
        # top_k 를 넘는 gold 문서는 자른다. 검색 조건이 상위 3개만 보므로
        # oracle 에만 더 많이 주면 천장이 과대평가된다.
        request["oracle_contexts"] = item["oracle_contexts"][:3]

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(request, handle)
        request_path = handle.name
    try:
        started = time.monotonic()
        proc = subprocess.run(
            [
                binary,
                "--model_config", model_config,
                "--retrieval_config", retrieval_config,
                "--request_json", request_path,
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        elapsed_ms = int((time.monotonic() - started) * 1000)
    finally:
        os.unlink(request_path)

    if proc.returncode != 0:
        return {"error": proc.stderr.strip()[-400:], "elapsed_ms": elapsed_ms}
    try:
        response = json.loads(proc.stdout)
    except ValueError:
        return {"error": "stdout is not JSON: " + proc.stdout[:200], "elapsed_ms": elapsed_ms}

    answer = response.get("answer", "")
    verdict, how = parse_verdict(answer)
    trace = response.get("trace") or {}
    return {
        "pred": verdict,
        "parsed_by": how,
        "answer": answer,
        "output_tokens": trace.get("output_tokens"),
        "elapsed_ms": elapsed_ms,
    }


def summarise(rows):
    """정확도·macro-F1·형식 준수율을 낸다.

    전체 정확도와 별개로 SUPPORTS/CONTRADICTS 만 따로 센다. NOT_ENOUGH_INFO
    는 근거를 줘도 정답이 그대로라 밴드가 거의 안 벌어지고, 전체 정확도에
    섞이면 "근거를 쓰는가"라는 질문을 희석한다.
    """
    total = len(rows)
    scored = [r for r in rows if r.get("pred")]
    correct = sum(1 for r in scored if r["pred"] == r["gold"])
    decidable = [r for r in scored if r["gold"] != "NOT_ENOUGH_INFO"]
    decidable_correct = sum(1 for r in decidable if r["pred"] == r["gold"])

    f1s = []
    for label in LABELS:
        tp = sum(1 for r in scored if r["pred"] == label and r["gold"] == label)
        fp = sum(1 for r in scored if r["pred"] == label and r["gold"] != label)
        fn = sum(1 for r in scored if r["pred"] != label and r["gold"] == label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)

    return {
        "n": total,
        "parsed": len(scored),
        "parse_fail": total - len(scored),
        "json_rate": sum(1 for r in rows if r.get("parsed_by") == "json") / total if total else 0.0,
        "accuracy": correct / len(scored) if scored else None,
        "accuracy_decidable": (
            decidable_correct / len(decidable) if decidable else None
        ),
        "n_decidable": len(decidable),
        "macro_f1": sum(f1s) / len(LABELS),
        "pred_dist": dict(collections.Counter(r.get("pred") or "(fail)" for r in rows)),
        "gold_dist": dict(collections.Counter(r["gold"] for r in rows)),
        "mean_elapsed_ms": (
            sum(r.get("elapsed_ms") or 0 for r in rows) // total if total else 0
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--retrieval-config", default="configs/retrieval_scifact.yaml")
    parser.add_argument("--context-source", required=True, choices=["closed_book", "oracle", "retrieved"])
    parser.add_argument("--binary", default="./build/rag_query")
    parser.add_argument("--queries", default="data/beir/scifact/queries.jsonl")
    parser.add_argument("--qrels", default="data/beir/scifact/qrels/test.tsv")
    parser.add_argument("--corpus", default="data/beir/scifact_corpus/corpus.jsonl")
    parser.add_argument("--limit", type=int, default=0, help="0 이면 test split 전체(300)")
    parser.add_argument("--out", required=True, help="문항별 결과 JSONL")
    args = parser.parse_args()

    items = load_dataset(args.queries, args.qrels, args.corpus)
    if args.limit:
        items = items[: args.limit]

    out_path = _path(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # 재개: 이미 기록된 문항은 건너뛴다. 중단돼도 이어서 돌릴 수 있어야
    # 한 칸에 수십 분씩 걸리는 사다리를 관리할 수 있다.
    done, rows = set(), []
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                done.add(row["sample_id"])
                rows.append(row)
        print(f"[재개] 기존 {len(done)}건 건너뜀", file=sys.stderr)

    with open(out_path, "a", encoding="utf-8") as handle:
        for index, item in enumerate(items, 1):
            if item["sample_id"] in done:
                continue
            result = run_one(
                args.binary, args.model_config, args.retrieval_config, item, args.context_source
            )
            row = {
                "sample_id": item["sample_id"],
                "gold": item["gold"],
                "context_source": args.context_source,
                **result,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            rows.append(row)
            mark = "O" if row.get("pred") == row["gold"] else "X"
            print(
                f"[{index}/{len(items)}] {item['sample_id']:>5} "
                f"gold={row['gold']:<16} pred={str(row.get('pred')):<16} {mark} "
                f"{row.get('elapsed_ms')}ms",
                file=sys.stderr,
            )

    summary = summarise(rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2), file=sys.stderr)
    summary_path = out_path.rsplit(".jsonl", 1)[0] + ".summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(
            {"model_config": args.model_config, "context_source": args.context_source, **summary},
            handle,
            ensure_ascii=False,
            indent=2,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
