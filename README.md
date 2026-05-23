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
# or: scripts/setup_nq_data.sh

source scripts/activate_env.sh
scripts/check_env.sh

cmake -S . -B build
cmake --build build -j

./build/rag_ingest --model_config configs/model.yaml --retrieval_config configs/retrieval_scifact.yaml
./build/rag_query --model_config configs/model.yaml --retrieval_config configs/retrieval_scifact.yaml --question "0-dimensional biomaterials show inductive properties."
# NQ는 /retrieval_scifact.yaml 대신 /retrieval_nq.yaml 사용
```

참고:
- clone 직후 `third_party/`, `models/`, `data/`가 비어있는 것은 정상
- 준비 경로 상세는 `docs/SETUP.md` 참고

## Query 실행
- NQ는 /retrieval_scifact.yaml 대신 /retrieval_nq.yaml 사용

### 단일 질의

구조화 입력(JSON):

```bash
./build/rag_query \
  --model_config configs/model.yaml \
  --retrieval_config configs/retrieval_scifact.yaml \
  --request_json /path/to/request.json
```

`request.json` 형식은 `docs/IO_CONTRACT.md`를 따른다.

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
  --dataset scifact \
  --split test \
  --questions data/beir/scifact/queries.jsonl \
  --qrels data/beir/scifact/qrels/test.tsv \
  --corpus data/beir/scifact_corpus/corpus.jsonl \
  --output_jsonl results/runs/scifact_naive_rag/predictions.jsonl \
  --limit 5
```

규칙:
- `--output_jsonl` 배치 모드와 `--request_json`/`--question` 단일 모드는 동시에 사용할 수 없음

### HTTP 서버

HTTP 서버는 선택 빌드 타깃이다.

```bash
cmake -S . -B build -DBUILD_RAG_HTTP_SERVER=ON
cmake --build build --target rag_http_server -j

./build/rag_http_server \
  --host 127.0.0.1 \
  --port 8080 \
  --rag_query ./build/rag_query \
  --workdir . \
  --model_config configs/model.yaml \
  --retrieval_config configs/retrieval_nq.yaml
```

요청 예시:

```bash
curl -X POST http://127.0.0.1:8080/query \
  -H "Content-Type: application/json" \
  -d '{"question":"Result of 2017 American League Championship Series?"}'
```

서버는 요청마다 내부적으로 `rag_query --request_json ...`를 실행한다.

### 데이터/코퍼스

- 준비 절차와 경로 정책은 `docs/SETUP.md` 기준
- BEIR 루트는 ingest하지 말고 `<dataset>_corpus` 경로를 사용
- `--output_jsonl` batch export는 BEIR `queries.jsonl`, `qrels/*.tsv`, `corpus.jsonl` 입력을 사용

## 출력 정책

- `stdout`: 기계 파싱용 JSON 1개만 출력
- `stderr`: trace/warn/error 운영 로그만 출력
- 벤치마크 수치 기준:
  - 단일 질의: `stdout.trace`
  - 배치 export: JSONL row의 `trace`
- 배치 export는 prediction JSONL과 manifest JSON을 생성한다.

## 실험 고정 정책

- 생성 길이 제어 키는 `max_new_tokens`만 지원 (`max_tokens` 미지원)
- 시스템 프롬프트는 보통 `configs/retrieval_*.yaml`의 `prompt.system_prompt_path`로 선택
- `configs/model.yaml`의 `prompt_system_path`는 retrieval config에 prompt 경로가 없을 때만 fallback으로 사용
- prompt body 형식은 `configs/retrieval_*.yaml`의 `prompt` 블록으로 설정
- `prompt_system_path` 파일만 바꾸면 재빌드 없이 다음 실행부터 반영
- `src/prompt_builder.cpp`의 조립 로직 자체를 바꾸면 `rag_query` 재빌드 필요
- `use_model_chat_template`는 기본 비활성화. 일부 GGUF instruction 모델은 켰을 때 빈 응답(EOG) 회귀가 생길 수 있어 opt-in 사용 권장
- 필요 시 `stop_strings`로 프롬프트 재생산/반복 출력을 조기 종료
- BEIR batch export의 `reference`는 qrels positive 문서(`title + "\n" + text`) 결합 정책 사용

상세 입출력 명세는 `docs/IO_CONTRACT.md`를 기준으로 사용.
