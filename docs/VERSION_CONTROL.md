# 형상관리 기준

## 1) 추적 대상

- `src/`, `include/`, `apps/`
- `configs/`
- `scripts/`
- `docs/`
- `CMakeLists.txt`

## 2) 비추적 대상

- 모델 파일 (`models/*.gguf`, `models/*.onnx` 등)
- 데이터셋 원본/압축 (`data/beir/*`, `data/raw/*`)
- 생성 인덱스 (`data/indexes/*`)
- 빌드 산출물 (`build/*`)
- 외부 의존성 폴더 (`third_party/llama.cpp`, `third_party/onnxruntime-*`)

## 3) 전역설치 의존성 처리

- FAISS 기본 prefix: `/usr/local`
- 커스텀 prefix 사용 시: `FAISS_ROOT=/custom/prefix`
- `activate_env.sh`, `check_env.sh`에서 탐지 경로 반영
