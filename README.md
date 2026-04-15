# edge-rag-experiments

C++ 기반 최소 RAG 파이프라인 (`llama.cpp + ONNX 임베딩 + FAISS`)

## 문서

- 환경 설정: `docs/SETUP.md`
- 형상관리 기준: `docs/VERSION_CONTROL.md`

## 생성 길이 설정 정책

- 모델 설정 키는 `max_new_tokens`만 지원
- `max_tokens`는 지원하지 않으며, 설정 시 실행 초기에 오류로 종료됨

## 프롬프트 설정 정책

- 시스템 프롬프트는 `configs/prompt_system.txt`에서 로드
- `configs/model.yaml`의 `prompt_system_path`로 경로를 고정/관리

## 빠른 시작 (fresh clone)

이 저장소는 재현성을 위해 대용량/외부 산출물을 Git에서 제외한다.
아래 경로는 clone 직후 비어있는 것이 정상이다.

- `third_party/`
- `models/`
- `data/`

준비 방식 요약:

- 스크립트로 준비: `third_party` (`scripts/setup_third_party.sh`, `scripts/rebuild_llama.sh`)
- 직접 준비(필수): 생성 모델 1개 파일
  - `models/gemma/gemma-3-270m-it-Q4_K_S.gguf`
- 임베딩 모델 준비(택1)
  - 직접 준비: `models/all-MiniLM-L6-v2-onnx/model.onnx`, `models/all-MiniLM-L6-v2-onnx/tokenizer.json`
  - 스크립트 준비: `scripts/setup_embedding_model.sh` (기본은 고정 revision, 필요 시 `EMBEDDING_REVISION` override)
- 데이터 준비(택1)
  - 직접 준비: `data/raw` 아래에 `.txt`/`.md`/`.jsonl` 배치
  - 스크립트 준비: `scripts/setup_scifact_data.sh` (출력: `data/raw/scifact_corpus.jsonl`)

자세한 준비 절차는 `docs/SETUP.md`.

scifact 자동 데이터 기준 실행:

```bash
scripts/setup_third_party.sh
scripts/rebuild_llama.sh
scripts/setup_embedding_model.sh
scripts/setup_scifact_data.sh

source scripts/activate_env.sh
scripts/check_env.sh

cmake -S . -B build
cmake --build build -j

./build/rag_ingest --model_config configs/model.yaml --retrieval_config configs/retrieval.yaml
./build/rag_query  --model_config configs/model.yaml --retrieval_config configs/retrieval.yaml
```

직접 데이터 배치 기준 실행:

```bash
scripts/setup_third_party.sh
scripts/rebuild_llama.sh

source scripts/activate_env.sh
scripts/check_env.sh

cmake -S . -B build
cmake --build build -j

./build/rag_ingest --model_config configs/model.yaml --retrieval_config configs/retrieval.yaml
./build/rag_query  --model_config configs/model.yaml --retrieval_config configs/retrieval.yaml
```
