# edge-rag-experiments

C++ 기반 최소 RAG 파이프라인 (`llama.cpp + ONNX 임베딩 + FAISS`)

## 문서

- 환경 설정: `docs/SETUP.md`
- 형상관리 기준: `docs/VERSION_CONTROL.md`

## 빠른 시작

```bash
source scripts/activate_env.sh
scripts/check_env.sh
cmake -S . -B build
cmake --build build -j

./build/rag_ingest --model_config configs/model.yaml --retrieval_config configs/retrieval.yaml
./build/rag_query  --model_config configs/model.yaml --retrieval_config configs/retrieval.yaml
```
