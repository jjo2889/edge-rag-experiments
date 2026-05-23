# 환경 설정 가이드

공유 저장소 기준 최소 설정 절차

## 0) 준비 항목 구분

- 직접 준비(필수): 생성 모델 파일 1개
  - `models/gemma/gemma-3-270m-it-Q4_K_S.gguf`
- 임베딩 모델 준비(택1)
  - 직접 준비: `models/all-MiniLM-L6-v2-onnx/model.onnx`, `models/all-MiniLM-L6-v2-onnx/tokenizer.json`
  - 스크립트 준비: `scripts/setup_embedding_model.sh` 실행
- 스크립트 준비(권장): `third_party` (`scripts/setup_third_party.sh`, `scripts/rebuild_llama.sh`)
- 데이터 준비(택1)
  - 직접 준비: `data/raw` 아래에 `.txt`/`.md`/`.jsonl` 배치
  - 스크립트 준비: `scripts/setup_scifact_data.sh` 또는 `scripts/setup_nq_data.sh` 실행

## 1) 필수 의존성

- `cmake` 3.20+
- `g++` (C++17 지원)
- `python3`
- `git`
- `curl`
- `tar`
- `yaml-cpp`
- `OpenMP`
- `FAISS` (CMake package 탐지 가능 상태)

Ubuntu 예시

```bash
sudo apt update
sudo apt install -y \
  build-essential cmake git curl tar pkg-config \
  libyaml-cpp-dev
```

`nlohmann/json.hpp`는 `scripts/setup_third_party.sh`로 내려받는
`third_party/llama.cpp/vendor/nlohmann/json.hpp`를 기본 사용한다.
시스템 헤더를 쓰고 싶다면 `nlohmann-json3-dev`를 추가 설치해도 된다.

## 2) 기본 경로 정책

- FAISS 기본 prefix: `/usr/local` (이미 설치된 경우)
- 커스텀 prefix 사용 시: `FAISS_ROOT` 지정

FAISS가 없다면 먼저 설치한다. (예: 로컬 prefix 설치)

```bash
git clone https://github.com/facebookresearch/faiss.git third_party/faiss
cmake -S third_party/faiss -B third_party/faiss/build \
  -DFAISS_ENABLE_GPU=OFF \
  -DFAISS_ENABLE_PYTHON=OFF \
  -DBUILD_TESTING=OFF \
  -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PWD/third_party/faiss-install"
cmake --build third_party/faiss/build -j
cmake --install third_party/faiss/build
export FAISS_ROOT="$PWD/third_party/faiss-install"
```

## 3) `third_party` 준비

- 재현 스크립트 기준 관리
- 기본 절차

```bash
scripts/setup_third_party.sh
scripts/rebuild_llama.sh
```

`setup_third_party.sh`는 ONNX Runtime 패키지 구조 이슈 우회를 위해
`lib64 -> lib`, `include/onnxruntime -> .` 심볼릭 링크를 만든다.
필요하면 `ORT_SHA256`를 지정해 아카이브 해시 검증을 강제할 수 있다.

## 4) 모델 파일 배치

- 생성 모델(직접 준비): `models/gemma/gemma-3-270m-it-Q4_K_S.gguf`
- 임베딩 모델(택1)
  - 직접 준비: `models/all-MiniLM-L6-v2-onnx/model.onnx`, `models/all-MiniLM-L6-v2-onnx/tokenizer.json`
  - 스크립트 준비:

```bash
scripts/setup_embedding_model.sh
```

`setup_embedding_model.sh`는 같은 Hugging Face repo/revision에서
`model.onnx`와 `tokenizer.json`을 함께 내려받아
`models/all-MiniLM-L6-v2-onnx/`에 배치한다.
기본 revision은 고정값이며, 필요 시 `EMBEDDING_REVISION`으로 override할 수 있다.

## 5) 데이터 배치 (택1)

- `rag_ingest`는 `retrieval_config.data_dir` 아래의 `.txt`, `.md`, `.jsonl`을 읽는다.
- 직접 준비: 원하는 문서 파일을 `data/raw` 아래에 배치하고 `configs/retrieval.yaml` 사용
- BEIR 데이터셋은 공통 규칙을 사용:
  - ingest: `data/beir/<dataset>_corpus/corpus.jsonl`
  - eval: `data/beir/<dataset>/queries.jsonl`, `data/beir/<dataset>/qrels/test.tsv`

SciFact:

```bash
scripts/setup_scifact_data.sh
```

- `data/beir/scifact_corpus/corpus.jsonl`
- `data/beir/scifact/queries.jsonl`
- `data/beir/scifact/qrels/test.tsv`

NQ:

```bash
scripts/setup_nq_data.sh
```

- `data/beir/nq_corpus/corpus.jsonl`
- `data/beir/nq/queries.jsonl`
- `data/beir/nq/qrels/test.tsv`

기본 subset 정책:
- qrels 순서 기준 앞 500 query
- 해당 query의 positive doc 전부 포함
- non-gold doc 20000개 추가

전체가 필요할 때만:

```bash
scripts/setup_nq_data.sh --full
```

- BEIR 루트 전체는 ingest하지 않고, `<dataset>_corpus` 경로를 `retrieval_config.data_dir`로 사용한다.

## 6) 환경 활성화/점검

```bash
source scripts/activate_env.sh
scripts/check_env.sh
```

`FAISS_ROOT` 변경 예시

```bash
export FAISS_ROOT=/path/to/faiss/prefix
source scripts/activate_env.sh
scripts/check_env.sh
```

## 7) 빌드와 기본 확인

```bash
cmake -S . -B build
cmake --build build -j

./build/rag_ingest --model_config configs/model.yaml --retrieval_config configs/retrieval.yaml
./build/rag_query  --model_config configs/model.yaml --retrieval_config configs/retrieval.yaml --question "What is this corpus about?"
```

SciFact/NQ/HTTP 실행 예시는 `README.md`를 본다.
HTTP 서버는 선택 빌드 타깃이므로 필요할 때 `-DBUILD_RAG_HTTP_SERVER=ON`을 지정한다.

## 8) 프롬프트, 재빌드, ingest 재사용

- 시스템 프롬프트는 `configs/model.yaml`의 `prompt_system_path`에서 읽는다.
- `configs/prompt_system.txt`만 바꾸면 재빌드가 필요 없다. 다음 `rag_query` 실행부터 반영된다.
- `src/prompt_builder.cpp`를 바꾸면 C++ 코드 변경이므로 `rag_query` 재빌드가 필요하다.
- SciFact와 QA/NQ는 시스템 프롬프트와 prompt body 형식을 각각 맞춰야 한다.
- 프롬프트 변경은 ingest 결과를 바꾸지 않는다.
- corpus, `retrieval_config.data_dir`, chunk 설정, 임베딩 모델이 그대로면 기존 `index_path`와 `chunk_store_path`를 재사용해도 된다.
- 실험 기록에는 config 경로, 프롬프트 내용/해시, prompt body 형식, ingest 재사용 여부를 남긴다.

## 9) 생성 길이 설정 주의사항

- `model.yaml`에서 생성 길이는 `max_new_tokens`만 사용
- `max_tokens`는 의도적으로 미지원이며, 설정 시 프로그램이 시작 단계에서 오류를 발생시킴
