#!/usr/bin/env python3
"""게이트 결과를 밴드 표로 집계한다.

`scripts/run_gate.sh` 가 남긴 JSONL 들을 읽어 칸마다 다음을 낸다.

    밴드 = oracle 정확도 − closed_book 정확도

두 정확도를 따로 보면 안 되는 이유가 있다. oracle 점수만 높은 것과 밴드가
넓은 것은 다르다 — 무맥락에서 이미 높으면 그 모델은 근거를 안 쓰고 아는 것으로
답하는 것이고, 그런 모델을 본 실험 생성기로 고르면 다섯 기법이 컨텍스트를
아무리 개선해도 답이 바뀌지 않는다. 고르는 것은 "좋은 모델"이 아니라
"근거를 주면 답이 달라지는 모델"이다 (docs/research_proposal.md §10.1).

NOT_ENOUGH_INFO 문항은 근거를 줘도 정답이 그대로라 밴드가 거의 벌어지지
않는다. test 300 중 112건(37%)이라 전체 정확도에 섞으면 신호가 눌린다.
그래서 `밴드(판정가능)` 열을 따로 낸다 — 게이트 판정은 이 열로 한다.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 사다리 순서. 파일명에서 읽은 이름을 이 순서로 정렬한다.
RUNG_ORDER = ["0.6b", "1.7b", "4b", "8b", "14b", "32b"]

# 항상 다수 라벨만 찍는 상수 분류기의 정확도. SciFact test 300 의 라벨 분포
# (SUPPORTS 124 / NOINFO 112 / CONTRADICTS 64) 에서 나온다. 무맥락 점수가
# 이 근처면 그 모델은 아무 일도 하지 않는 것이다.
MAJORITY_BASELINE = 0.413


def load(directory: str):
    """{칸: {조건: 요약}} 으로 모은다."""
    rows = {}
    pattern = os.path.join(directory, "qwen3-*.*.jsonl")
    for path in sorted(glob.glob(pattern)):
        if path.endswith(".summary.json"):
            continue
        base = os.path.basename(path)[len("qwen3-"):-len(".jsonl")]
        rung, _, source = base.rpartition(".")
        if not rung:
            continue
        records = []
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        if records:
            rows.setdefault(rung, {})[source] = summarise(records)
    return rows


def summarise(records):
    scored = [r for r in records if r.get("pred")]
    decidable = [r for r in scored if r["gold"] != "NOT_ENOUGH_INFO"]
    return {
        "n": len(records),
        "parse_fail": len(records) - len(scored),
        "json_rate": sum(1 for r in records if r.get("parsed_by") == "json") / len(records),
        "acc": (sum(1 for r in scored if r["pred"] == r["gold"]) / len(scored)) if scored else None,
        "acc_dec": (
            sum(1 for r in decidable if r["pred"] == r["gold"]) / len(decidable)
        ) if decidable else None,
        "n_dec": len(decidable),
        "ms": sum(r.get("elapsed_ms") or 0 for r in records) // max(1, len(records)),
    }


def pct(value):
    return "  —  " if value is None else f"{value * 100:5.1f}%"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", default="reports/gate")
    args = parser.parse_args()

    directory = args.dir if os.path.isabs(args.dir) else os.path.join(REPO_ROOT, args.dir)
    rows = load(directory)
    if not rows:
        print(f"결과 없음: {directory}", file=sys.stderr)
        return 1

    order = [r for r in RUNG_ORDER if r in rows] + sorted(set(rows) - set(RUNG_ORDER))

    print(f"{'칸':<8} {'n':>4} {'무맥락':>7} {'oracle':>7} {'밴드':>7} "
          f"{'밴드(판정가능)':>14} {'JSON':>6} {'초/건':>7}")
    print("─" * 72)
    for rung in order:
        cb = rows[rung].get("closed_book")
        orc = rows[rung].get("oracle")
        band = band_dec = None
        if cb and orc and cb["acc"] is not None and orc["acc"] is not None:
            band = orc["acc"] - cb["acc"]
        if cb and orc and cb["acc_dec"] is not None and orc["acc_dec"] is not None:
            band_dec = orc["acc_dec"] - cb["acc_dec"]
        n = (orc or cb or {}).get("n", 0)
        json_rate = (orc or cb or {}).get("json_rate")
        ms = max((orc or {}).get("ms", 0), (cb or {}).get("ms", 0))
        print(f"{rung:<8} {n:>4} {pct(cb['acc'] if cb else None)} "
              f"{pct(orc['acc'] if orc else None)} {pct(band)} "
              f"{pct(band_dec):>14} {pct(json_rate):>6} {ms / 1000:>6.1f}s")

    print("─" * 72)
    print(f"참고: 상수 분류기(항상 다수 라벨) 정확도 = {MAJORITY_BASELINE * 100:.1f}%")
    print("      무맥락이 이 근처면 그 칸은 과제를 수행하지 않는 것이다.")
    print()
    print("판정: 밴드(판정가능)가 넓은 칸 = 근거를 읽는 모델 → 본 실험 생성기 후보")
    print("      전 칸이 0 에 가까우면 생성 층위를 접고 검색 층위로 간다 (§8.2)")

    missing = [f"{r}/{s}" for r in order for s in ("closed_book", "oracle") if s not in rows[r]]
    if missing:
        print(f"\n미완료: {', '.join(missing)}")

    fails = [(r, s, d["parse_fail"]) for r in order for s, d in rows[r].items() if d["parse_fail"]]
    if fails:
        print("\n파싱 실패 (형식 미준수):")
        for rung, source, count in fails:
            print(f"  {rung}/{source}: {count}건")
    return 0


if __name__ == "__main__":
    sys.exit(main())
