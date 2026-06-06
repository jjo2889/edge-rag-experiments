# Advanced RAG C++ Implementation Plan

이 문서는 Hybrid Search를 제외한 Advanced RAG 기법을 현재 C++ RAG 코드베이스에 어떤 레이어로 적용할지, 그리고 브랜치/커밋/테스트 단위로 어떻게 구현할지 고정한다.

## 전제

```text
1. Hybrid Search는 본 구현 범위에서 제외한다.
2. 한 run에서는 advanced mode 하나만 적용한다.
3. Naive RAG는 advanced=none baseline으로 유지한다.
4. 입력은 최상단 request/CLI에서 받되, 실제 기법은 필요한 pipeline stage에서만 동작한다.
5. 최종 답변 prompt는 원 query를 기준으로 만들고, retrieval query만 기법에 따라 바꿀 수 있다.
6. 각 기법은 독립 브랜치에서 작업할 수 있어야 한다.
7. 각 기법 브랜치는 최소 3개 이상의 중간 커밋으로 구현 trace를 남긴다.
8. PR 직전에는 자체 테스트 코드를 추가하고 실행 결과로 동작을 확인한다.
```

## 현재 코드 구조 요약

현재 query 실행 경로는 다음과 같다.

```text
apps/query_main.cpp
  -> QueryRequest 로드
  -> chunk store / FAISS index / embedding / generator 생성
  -> RagPipeline::run(query)
  -> query_response_json 또는 beir_export JSON 출력
```

핵심 클래스와 책임:

```text
QueryRequest          include/query_request_io.hpp     request JSON 입력
ConfigLoader          include/config.hpp               model/retrieval config 로드
Retriever             include/retriever.hpp            query embedding + FAISS search
FaissIndex            include/faiss_index.hpp          IndexFlatIP search
PromptBuilder         include/prompt_builder.hpp       context + query prompt body 조립
Generator             include/generator.hpp            llama.cpp generation
RagPipeline           include/rag_pipeline.hpp         retrieve -> prompt -> generate orchestration
RagResult             include/types.hpp                answer, contexts, trace metadata
beir_export           src/beir_export.cpp              batch JSONL/RAGAS input 생성
query_response_json   src/query_response_json.cpp      single-run JSON 출력
```

현재 `RagPipeline::run()`은 다음 직선 흐름이다.

```text
query
  -> retriever.retrieve(query, top_k)
  -> prompt_builder.buildPrompt(query, contexts)
  -> generator.generateWithTrace(prompt_body)
```

Advanced RAG는 이 흐름을 다음 stage hook으로 확장한다.

```text
original_query
  -> [query_transform]       HyDE / Query Rewriting
  -> [retrieve_candidates]   Dense retrieval
  -> [post_retrieval]        Re-ranking / Sentence Window
  -> [context_transform]     Contextual Compression
  -> [prompt_build]
  -> [final_generation]
```

## 적용 레이어

| 기법 | 입력 위치 | 실제 동작 레이어 | 핵심 변경 |
|---|---|---|---|
| `none` | request/CLI | 없음 | 기존 Naive RAG 유지 |
| `query_rewrite` | request/CLI | retrieval 전 query transform | LLM으로 검색용 query 재작성 |
| `hyde` | request/CLI | retrieval 전 query transform | LLM으로 hypothetical document 생성 후 검색 |
| `rerank` | request/CLI | dense retrieval 후 post retrieval | candidate_k 검색 후 reranker 점수로 top_k 재정렬 |
| `sentence_window` | request/CLI | ingest metadata + post retrieval | hit chunk 주변 문장/window를 context로 확장 |
| `contextual_compression` | request/CLI | retrieval 후 prompt 전 context transform | query 관련 context만 압축 후 prompt 투입 |

## 공통 기반 브랜치

브랜치:

```text
feature/advanced-rag-contract
```

목표:

```text
advanced mode를 request, CLI, batch export, trace, run_config에 통과시키되 advanced=none에서는 기존 동작을 보존한다.
```

### Commit 1: Add advanced mode contract

포함 변경:

```text
include/types.hpp
  AdvancedMode enum 추가
  advancedModeFromString / advancedModeToString helper 선언 또는 별도 header 추가

include/query_request_io.hpp
src/query_request_io.cpp
  QueryRequest.advanced 추가
  request JSON 허용 필드에 advanced 추가
  unknown advanced mode는 명시적 error

docs/IO_CONTRACT.md
  Request JSON과 output JSON에 advanced 필드 명시
```

검증:

```bash
cmake -S . -B build
cmake --build build -j
```

성공 기준:

```text
advanced 없는 기존 request가 계속 성공한다.
advanced=none request가 파싱된다.
advanced=hyde 같은 known value가 파싱된다.
advanced=invalid는 명시적으로 실패한다.
```

### Commit 2: Thread advanced mode through CLI and batch export

포함 변경:

```text
apps/query_main.cpp
  --advanced 옵션 추가
  --request_json advanced와 --advanced가 동시에 있으면 충돌 정책 정의
  권장: request_json 값 우선, CLI --advanced는 --question/batch용

include/beir_export.hpp
src/beir_export.cpp
  BeirExportOptions.advanced 추가
  batch prediction row와 manifest에 advanced 기록

src/query_response_json.cpp
  request, run_config, top-level mode 또는 advanced 필드에 기록
```

검증:

```bash
cmake --build build -j
./build/rag_query --help
```

성공 기준:

```text
--advanced none은 기존 결과와 같은 경로를 탄다.
batch export manifest에 advanced가 기록된다.
single-run JSON request/run_config에 advanced가 기록된다.
```

### Commit 3: Add advanced trace fields without behavior changes

포함 변경:

```text
include/types.hpp
  RagResult에 advanced_mode, retrieval_query 추가
  query_transform_ms, rerank_ms, context_transform_ms 등 세부 latency 추가

src/rag_pipeline.cpp
  advanced=none에서 새 trace 필드를 0으로 채움
  retrieval_query=query 유지

src/query_response_json.cpp
src/beir_export.cpp
  trace와 prediction row에 retrieval_query/advanced latency 기록
```

검증:

```bash
cmake --build build -j
```

성공 기준:

```text
none mode의 retrieval_query는 original query와 같다.
추가 latency 필드는 0으로 기록된다.
기존 retrieved_contexts 형식은 깨지지 않는다.
```

### Commit 4: Add advanced contract tests

포함 변경:

```text
CMakeLists.txt
  BUILD_RAG_TESTS 옵션 추가
  ctest 활성화

tests/
  advanced_mode_test.cpp
  query_request_io_test.cpp
```

검증:

```bash
cmake -S . -B build -DBUILD_RAG_TESTS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

성공 기준:

```text
advanced parser 테스트 통과
request JSON backward compatibility 테스트 통과
invalid advanced mode 테스트 통과
```

## 기법별 브랜치 계획

각 기법 브랜치는 `feature/advanced-rag-contract`에서 분기한다.

## 1. Query Rewriting

브랜치:

```text
feature/advanced-query-rewrite
```

작동 레이어:

```text
retrieval 전 query_transform
```

흐름:

```text
original_query
  -> Generator로 검색용 query 재작성
  -> rewritten query로 retrieve
  -> original_query로 final prompt build
  -> final answer 생성
```

구현 계획:

### Commit 1: Add query rewrite prompt helper

포함 변경:

```text
include/advanced_query_transformer.hpp
src/advanced_query_transformer.cpp
  buildQueryRewritePrompt(query)
  sanitizeGeneratedQuery(text)
```

검증:

```text
rewrite prompt가 query를 포함한다.
빈/공백 rewrite는 fallback 대상이 된다.
```

### Commit 2: Apply query rewrite in RagPipeline

포함 변경:

```text
RagPipeline::run(query, mode)
  mode=query_rewrite이면 generator로 rewritten query 생성
  retrieval_query=rewritten query
  final prompt는 original_query 사용
```

검증:

```text
retrieval_query가 result에 기록된다.
query_transform_ms가 0보다 크다.
contexts는 rewritten query 검색 결과다.
```

### Commit 3: Add output and fallback behavior

포함 변경:

```text
빈 rewrite, prompt echo, 지나치게 긴 rewrite에 대한 fallback
prediction row에 retrieval_query 기록
```

검증:

```text
rewrite 실패 시 original query로 검색한다.
run은 status=ok를 유지한다.
```

### Commit 4: Add tests and smoke script

포함 변경:

```text
tests/advanced_query_transformer_test.cpp
scripts/smoke_advanced_query_rewrite.sh 또는 tests 내 fixture
```

검증:

```bash
cmake -S . -B build -DBUILD_RAG_TESTS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

모델/인덱스가 있는 환경:

```bash
./build/rag_query \
  --model_config configs/model.yaml \
  --retrieval_config configs/retrieval_scifact.yaml \
  --advanced query_rewrite \
  --question "0-dimensional biomaterials show inductive properties."
```

## 2. HyDE

브랜치:

```text
feature/advanced-hyde
```

작동 레이어:

```text
retrieval 전 query_transform
```

흐름:

```text
original_query
  -> Generator로 hypothetical document 생성
  -> hypothetical document를 embedding/retrieval query로 사용
  -> original_query로 final prompt build
  -> final answer 생성
```

구현 계획:

### Commit 1: Add HyDE prompt helper

포함 변경:

```text
AdvancedQueryTransformer에 buildHydePrompt(query) 추가
sanitizeGeneratedDocument(text) 추가
```

검증:

```text
HyDE prompt는 답변이 아니라 검색용 문서 생성을 요구한다.
생성 결과가 빈 문자열이면 fallback 대상이 된다.
```

### Commit 2: Apply HyDE retrieval query

포함 변경:

```text
RagPipeline mode=hyde 분기 추가
retrieval_query=hyde_document
result.original_query는 원 query 유지
```

검증:

```text
HyDE mode에서 generation 호출이 final answer 전 1회 추가된다.
query_transform_ms 기록
retrieval_query 기록
```

### Commit 3: Add HyDE fallback and trace

포함 변경:

```text
빈 HyDE 문서 fallback
너무 긴 HyDE 문서 제한
trace에 advanced_generation_tokens 또는 query_transform_ms 기록
```

검증:

```text
HyDE 실패가 전체 query 실패로 전파되지 않는다.
fallback 시 retrieval_query=original_query로 기록한다.
```

### Commit 4: Add tests and smoke

검증:

```bash
cmake -S . -B build -DBUILD_RAG_TESTS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

모델/인덱스가 있는 환경:

```bash
./build/rag_query \
  --model_config configs/model.yaml \
  --retrieval_config configs/retrieval_scifact.yaml \
  --advanced hyde \
  --question "0-dimensional biomaterials show inductive properties."
```

## 3. Re-ranking

브랜치:

```text
feature/advanced-rerank
```

작동 레이어:

```text
dense retrieval 후 post_retrieval
```

흐름:

```text
original_query
  -> dense retrieve candidate_k
  -> reranker(query, context) score 계산
  -> rerank score 기준 top_k 선택
  -> final prompt 생성
```

주의:

```text
bge-reranker 계열 tokenizer가 현재 WordPiece tokenizer와 다를 수 있다.
SentencePiece/XLM-R 기반이면 별도 tokenizer 구현 또는 ONNX 모델 선택 재검토가 필요하다.
```

구현 계획:

### Commit 1: Add rerank config and candidate retrieval flow

포함 변경:

```text
RetrievalConfig에 advanced.rerank_candidate_k 또는 rerank_candidate_k 추가
RagPipeline mode=rerank이면 retrieve top_k 대신 candidate_k 검색
```

검증:

```text
candidate_k > top_k이면 더 많은 후보를 검색한다.
final contexts는 top_k개로 잘린다.
```

### Commit 2: Add Reranker interface

포함 변경:

```text
include/reranker.hpp
src/reranker.cpp
  score(query, context)
  rerank(query, candidates, top_k)
```

검증:

```text
초기에는 deterministic test reranker 또는 simple lexical fallback으로 순서 테스트 가능
실제 ONNX reranker는 다음 커밋에서 연결
```

### Commit 3: Add ONNX reranker backend

포함 변경:

```text
reranker_model_path
reranker_tokenizer_json_path
ONNX Runtime session
pair input tokenization
```

검증:

```text
모델 파일이 없으면 명시적 config error
모델 파일이 있으면 query/context pair score 생성
```

### Commit 4: Add rerank output and tests

포함 변경:

```text
RetrievedChunk에 rerank_score 추가 또는 별도 metadata 추가
JSON output에 dense score와 rerank score 기록
tests/reranker_test.cpp
```

검증:

```bash
cmake -S . -B build -DBUILD_RAG_TESTS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## 4. Sentence Window Retrieval

브랜치:

```text
feature/advanced-sentence-window
```

작동 레이어:

```text
ingest metadata + retrieval 후 post_retrieval
```

주의:

현재 chunk store row는 다음 필드만 가진다.

```json
{"chunk_id": 0, "doc_id": "...", "text": "..."}
```

따라서 정확한 Sentence Window Retrieval을 구현하려면 ingest 시점에 문장 또는 위치 metadata를 저장해야 한다. 단순히 같은 문서의 인접 chunk를 붙이는 방식은 `Neighbor Chunk Window`에 가깝다. 본 연구에서 `Sentence Window Retrieval`이라고 부르려면 sentence-level metadata 확장이 필요하다.

구현 계획:

### Commit 1: Extend chunk metadata schema

포함 변경:

```text
Chunk에 doc_chunk_index, start_word, end_word 또는 sentence_start, sentence_end 추가
TextChunker가 문서 내 위치를 기록
chunk store JSONL에 새 필드 저장
loadChunkStore는 기존 JSONL도 읽을 수 있게 optional fallback
```

검증:

```text
기존 chunk store도 계속 로드된다.
새 ingest 결과에는 위치 metadata가 들어간다.
```

### Commit 2: Add sentence/window expander

포함 변경:

```text
include/sentence_window.hpp
src/sentence_window.cpp
  expand(contexts, all_chunks, window_size)
```

검증:

```text
같은 doc_id 안에서 hit 주변 window만 붙인다.
다른 문서 chunk는 섞지 않는다.
중복 window는 제거한다.
```

### Commit 3: Apply sentence_window mode in pipeline

포함 변경:

```text
RagPipeline mode=sentence_window 분기
retrieved hit는 유지하되 prompt context text만 window 확장
context_transform_ms 기록
```

검증:

```text
top_k는 유지된다.
각 context text 길이는 확장될 수 있다.
retrieved_doc_ids/ranks는 원 hit 기준으로 유지된다.
```

### Commit 4: Add tests and ingest compatibility smoke

검증:

```bash
cmake -S . -B build -DBUILD_RAG_TESTS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

추가 smoke:

```bash
./build/rag_ingest \
  --model_config configs/model.yaml \
  --retrieval_config configs/retrieval_scifact.yaml
```

## 5. Contextual Compression

브랜치:

```text
feature/advanced-contextual-compression
```

작동 레이어:

```text
retrieval 후 prompt 전 context_transform
```

흐름:

```text
original_query
  -> dense retrieve top_k
  -> Generator로 query 관련 context만 압축
  -> compressed contexts로 final prompt 생성
  -> final answer 생성
```

구현 계획:

### Commit 1: Add compression prompt helper

포함 변경:

```text
include/contextual_compressor.hpp
src/contextual_compressor.cpp
  buildCompressionPrompt(query, contexts)
  parseCompressedContexts(output)
```

검증:

```text
compression prompt는 원문에 없는 내용을 추가하지 말라고 지시한다.
출력 parsing 실패는 fallback 대상이 된다.
```

### Commit 2: Apply compression in pipeline

포함 변경:

```text
RagPipeline mode=contextual_compression 분기
retrieved contexts를 prompt build 전에 압축
context_transform_ms 기록
```

검증:

```text
compressed context가 final prompt에 들어간다.
retrieved doc_id는 가능한 한 원 context metadata를 보존한다.
```

### Commit 3: Add fallback and trace

포함 변경:

```text
빈 compression 결과 fallback
compressed_context_tokens 또는 prompt token 감소 여부 기록 검토
```

검증:

```text
compression 실패 시 원 context로 final answer 생성
prompt_tokens_before_truncation 변화 확인 가능
```

### Commit 4: Add tests and smoke

검증:

```bash
cmake -S . -B build -DBUILD_RAG_TESTS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

모델/인덱스가 있는 환경:

```bash
./build/rag_query \
  --model_config configs/model.yaml \
  --retrieval_config configs/retrieval_scifact.yaml \
  --advanced contextual_compression \
  --question "0-dimensional biomaterials show inductive properties."
```

## Pre-eval 확장 계획

본 실험에서는 advanced mode가 추가 실험 축이 된다.

권장 구조:

```yaml
advanced:
  - none
  - rerank
  - sentence_window
  - contextual_compression
  - hyde
  - query_rewrite
```

단일 조합만 실행할 때는 CLI filter를 둔다.

```bash
cd experiment/pre-eval
uv run python -m pre_eval.run_grid \
  --grid configs/grids/advanced.yaml \
  --dataset scifact_full \
  --chunk-size 256 \
  --retrieval-top-k 3 \
  --advanced hyde \
  --limit 1 \
  --skip-ragas
```

변경 대상:

```text
experiment/pre-eval/src/pre_eval/grid.py
  RunSpec.advanced 추가
  run_id에 advanced 포함: adv_<mode>__cs_<chunk_size>__k_<top_k>

experiment/pre-eval/src/pre_eval/config_writer.py
  manifest에 advanced 기록

experiment/pre-eval/src/pre_eval/subprocess_runner.py
  rag_query 호출에 --advanced 전달

experiment/pre-eval/src/pre_eval/report.py
  summary.csv에 advanced 컬럼 추가
```

## PR 직전 공통 검증

모든 브랜치는 PR 직전에 다음을 통과해야 한다.

```bash
cmake -S . -B build -DBUILD_RAG_TESTS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

모델/데이터가 준비된 환경에서는 최소 smoke를 수행한다.

```bash
./build/rag_query \
  --model_config configs/model.yaml \
  --retrieval_config configs/retrieval_scifact.yaml \
  --advanced <mode> \
  --question "0-dimensional biomaterials show inductive properties."
```

batch export smoke:

```bash
./build/rag_query \
  --model_config configs/model.yaml \
  --retrieval_config configs/retrieval_scifact.yaml \
  --advanced <mode> \
  --dataset scifact \
  --split test \
  --questions data/beir/scifact/queries.jsonl \
  --qrels data/beir/scifact/qrels/test.tsv \
  --corpus data/beir/scifact_corpus/corpus.jsonl \
  --output_jsonl /tmp/rag_<mode>_predictions.jsonl \
  --limit 1
```

성공 기준:

```text
1. status=ok
2. advanced mode가 output/manifest에 기록된다.
3. retrieval_query가 원 query와 분리되어 기록된다. 단, none/sentence_window/rerank/contextual_compression은 같을 수 있다.
4. trace에 advanced stage latency가 기록된다.
5. RAGAS 입력 필드 user_input, response, retrieved_contexts, reference가 유지된다.
6. advanced=none은 기존 Naive RAG 동작을 보존한다.
```
