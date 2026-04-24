# edge-rag-experiments

C++ 기반 최소 RAG 실험 코드베이스 (`llama.cpp + ONNX 임베딩 + FAISS`)

## 문서

- 환경 설정: `docs/SETUP.md`
- 입출력 계약(API 명세): `docs/IO_CONTRACT.md`
- 형상관리 기준: `docs/VERSION_CONTROL.md`

## 빠른 시작

1. 의존성/모델/데이터 준비
2. 빌드
3. ingest
4. query 실행

예시:

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
./build/rag_query --model_config configs/model.yaml --retrieval_config configs/retrieval.yaml --question "What is RAG?"
```

참고:
- clone 직후 `third_party/`, `models/`, `data/`가 비어있는 것은 정상
- 준비 경로 상세는 `docs/SETUP.md` 참고

## Query 실행

### 단일 질의 (권장)

구조화 입력(JSON):

```bash
./build/rag_query \
  --model_config configs/model.yaml \
  --retrieval_config configs/retrieval_scifact.yaml \
  --request_json /path/to/request.json
```

간편 입력(수동 테스트용):

```bash
./build/rag_query \
  --model_config configs/model.yaml \
  --retrieval_config configs/retrieval_scifact.yaml \
  --question "0-dimensional biomaterials show inductive properties."
```

### 배치 JSONL export (RAGAS 입력 생성)

```bash
./build/rag_query \
  --model_config configs/model.yaml \
  --retrieval_config configs/retrieval_scifact.yaml \
  --questions data/beir/scifact/queries.jsonl \
  --qrels data/beir/scifact/qrels/test.tsv \
  --corpus data/beir/scifact/corpus.jsonl \
  --output_jsonl results/runs/scifact_naive_rag/predictions.jsonl \
  --limit 5
```

규칙:
- `--output_jsonl` 배치 모드와 `--request_json`/`--question` 단일 모드는 동시에 사용할 수 없음

## 출력 정책

- `stdout`: 기계 파싱용 JSON 1개만 출력
- `stderr`: trace/warn/error 운영 로그만 출력
- 벤치마크 수치 source-of-truth:
  - 단일 질의: `stdout.metrics`
  - 배치 export: JSONL row의 `trace`

## 실험 고정 정책

- 생성 길이 제어 키는 `max_new_tokens`만 지원 (`max_tokens` 미지원)
- 시스템 프롬프트는 `configs/model.yaml`의 `prompt_system_path`로 로드
- `use_model_chat_template`는 기본 비활성화. 일부 GGUF instruction 모델은 켰을 때 빈 응답(EOG) 회귀가 생길 수 있어 opt-in 사용 권장
- 필요 시 `stop_strings`로 프롬프트 재생산/반복 출력을 조기 종료
- SciFact export의 `reference`는 qrels positive 문서(`title + "\n" + text`) 결합 정책 사용

입출력 필드/필수값/null 허용 여부는 `docs/IO_CONTRACT.md`를 기준으로 사용.
