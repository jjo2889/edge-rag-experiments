#!/usr/bin/env bash
# 게이트 실행 — 크기 사다리 4칸 × {무맥락, 정답근거}.
#
# docs/research_proposal.md §4.5 의 기준선을 잰다. 두 조건 모두 검색을
# 건너뛰므로 인덱스도 임베딩 모델도 쓰지 않는다. 생성기만 있으면 돈다.
#
#   밴드 = oracle 정확도 − closed_book 정확도
#
# 밴드가 0 에 가까우면 그 칸은 근거를 읽지 않는 것이고, 위에 어떤 검색 기법을
# 얹어도 생성 지표가 움직이지 않는다.
#
# 사용:
#   scripts/run_gate.sh                      # 파이(CPU), 전체 300문항
#   LIMIT=50 scripts/run_gate.sh             # 빠른 확인
#   SUFFIX=.pc scripts/run_gate.sh           # PC(GPU) 설정으로
#   RUNGS="1.7b 4b" scripts/run_gate.sh      # 일부 칸만
#
# 중단해도 같은 명령을 다시 실행하면 이어서 돈다 — scifact_gate.py 가 문항
# 단위로 재개한다. 없는 모델 파일은 건너뛰고 계속한다.
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

RUNGS="${RUNGS:-0.6b 1.7b 4b 8b}"
SOURCES="${SOURCES:-oracle closed_book}"
SUFFIX="${SUFFIX:-}"            # 빈 값 = 파이(CPU), ".pc" = PC(GPU)
LIMIT="${LIMIT:-0}"             # 0 = test split 전체(300)
OUT_ROOT="${OUT_ROOT:-reports/gate}"

# 이전 설계(v1/v2)의 결과와 섞이면 안 된다. --skip-existing 류가 파일 존재만
# 보고 깨진 결과를 재사용한 전례가 있어 출력 루트를 분리해 둔다.
mkdir -p "${OUT_ROOT}"

echo "[gate] 칸: ${RUNGS}"
echo "[gate] 조건: ${SOURCES}"
echo "[gate] 설정 접미사: '${SUFFIX}'  (빈 값이면 파이 CPU)"
echo "[gate] 문항 수: $([ "${LIMIT}" = 0 ] && echo "전체(300)" || echo "${LIMIT}")"
echo "[gate] 출력: ${OUT_ROOT}"
echo

started_all=$(date +%s)
for rung in ${RUNGS}; do
  cfg="configs/model_qwen3_${rung}${SUFFIX}.yaml"
  if [ ! -f "${cfg}" ]; then
    echo "[건너뜀] ${rung}: 설정 없음 (${cfg})"
    continue
  fi
  gguf=$(grep -m1 '^llama_model_path:' "${cfg}" | sed -E 's/.*"(.*)".*/\1/')
  if [ ! -f "${gguf}" ]; then
    echo "[건너뜀] ${rung}: 모델 파일 없음 (${gguf})"
    continue
  fi

  for src in ${SOURCES}; do
    out="${OUT_ROOT}/qwen3-${rung}.${src}.jsonl"
    echo "──── ${rung} / ${src}"
    started=$(date +%s)
    python3 scripts/scifact_gate.py \
      --model-config "${cfg}" \
      --context-source "${src}" \
      --limit "${LIMIT}" \
      --out "${out}" 2>&1 | grep -vE '^\[[0-9]+/' || true
    echo "     ($(( ( $(date +%s) - started ) / 60 ))분)"
  done
done

echo
echo "[gate] 전체 소요 $(( ( $(date +%s) - started_all ) / 60 ))분"
echo
python3 scripts/gate_report.py --dir "${OUT_ROOT}"
