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

### 단일 질의 (권장)

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

### 다른 코퍼스 추가

- `rag_ingest`는 `retrieval_config.data_dir` 아래의 `.txt`, `.md`, `.jsonl`을 읽음
- BEIR처럼 `queries.jsonl`, `qrels/`가 같이 있는 루트는 그대로 ingest하지 말고 corpus-only 디렉터리를 사용
- 새 retrieval config는 기존 `configs/retrieval_scifact.yaml` 또는 `configs/retrieval_nq.yaml`를 복사해서 시작
- JSONL 문서는 `_id`/`id`, `title`(선택), `text`/`contents`/`body` 중 하나를 가지면 됨

### BEIR 데이터셋 예시

```bash
scripts/setup_scifact_data.sh
scripts/setup_nq_data.sh
```

- SciFact ingest: `data/beir/scifact_corpus/corpus.jsonl`
- SciFact eval: `data/beir/scifact/queries.jsonl`, `data/beir/scifact/qrels/test.tsv`
- NQ ingest: `data/beir/nq_corpus/corpus.jsonl`
- NQ eval: `data/beir/nq/queries.jsonl`, `data/beir/nq/qrels/test.tsv`

```bash
scripts/setup_nq_data.sh --full --force
```

주의:
- `rag_ingest`는 `configs/retrieval_<dataset>.yaml`의 `data_dir`를 읽음
- BEIR 루트는 ingest하지 말고 `<dataset>_corpus` 경로를 사용
- 기본 `setup_nq_data.sh`는 고정 subset 생성
- `--full`일 때만 전체 추출
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
- 시스템 프롬프트는 `configs/model.yaml`의 `prompt_system_path`로 로드
- `use_model_chat_template`는 기본 비활성화. 일부 GGUF instruction 모델은 켰을 때 빈 응답(EOG) 회귀가 생길 수 있어 opt-in 사용 권장
- 필요 시 `stop_strings`로 프롬프트 재생산/반복 출력을 조기 종료
- BEIR batch export의 `reference`는 qrels positive 문서(`title + "\n" + text`) 결합 정책 사용

상세 입출력 명세는 `docs/IO_CONTRACT.md`를 기준으로 사용.
