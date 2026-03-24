# 환경 설정 가이드

공유 저장소 기준 최소 설정 절차

## 1) 필수 의존성

- `cmake` 3.20+
- `g++` (C++17 지원)
- `python3`
- `yaml-cpp`
- `OpenMP`
- `FAISS` (CMake package 탐지 가능 상태)

## 2) 기본 경로 정책

- FAISS 기본 prefix: `/usr/local`
- 커스텀 prefix 사용 시: `FAISS_ROOT` 지정

## 3) `third_party` 준비

- 재현 스크립트 기준 관리
- 기본 절차

```bash
scripts/setup_third_party.sh
scripts/rebuild_llama.sh
```

## 4) 모델 파일 배치

- 생성 모델: `models/gemma/gemma-3-270m-it-Q4_K_S.gguf`
- 임베딩 모델: `models/all-MiniLM-L6-v2-onnx/model.onnx`
- 토크나이저: `models/all-MiniLM-L6-v2-onnx/tokenizer.json`

## 5) 환경 활성화/점검

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

## 6) 빌드/실행

```bash
cmake -S . -B build
cmake --build build -j

./build/rag_ingest --model_config configs/model.yaml --retrieval_config configs/retrieval.yaml
./build/rag_query  --model_config configs/model.yaml --retrieval_config configs/retrieval.yaml
```
