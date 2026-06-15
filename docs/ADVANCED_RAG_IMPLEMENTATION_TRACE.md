# Advanced RAG Implementation Trace

이 문서는 `feature/advanced-rag-all` 브랜치를 `main`에 단일 PR로 올리기 위한 구현 추적 문서다. 내용은 구현된 코드 기준이며, 논문 방법론 섹션을 작성할 때 실험 조건, 파이프라인 처리 순서, 구현 가정, 측정 항목을 재구성할 수 있는 수준으로 남긴다.

## PR 브랜치 상태

```text
PR 후보 브랜치: feature/advanced-rag-all
PR 대상 브랜치: main
main 직접 merge 여부: 하지 않음
```

통합에 포함한 브랜치:

```text
feature/advanced-rag-contract
feature/advanced-query-rewrite
feature/advanced-hyde
feature/advanced-contextual-compression
feature/advanced-rerank
feature/advanced-rerank-bge
feature/advanced-sentence-window-v2
```

통합에서 제외한 브랜치:

```text
feature/advanced-sentence-window
add/http
add/pre-eval
```

`feature/advanced-sentence-window`는 `doc_chunk_index +/- N` 기반 neighbor chunk window MVP라서 논문용 `sentence_window` 구현에서 제외했다. 최종 통합에는 sentence metadata 기반 `feature/advanced-sentence-window-v2`만 포함했다. `add/http`, `add/pre-eval`은 Advanced RAG 구현 PR의 직접 범위가 아니라 포함하지 않았다.

## 구현 범위

Hybrid Search는 제외했다. 한 번의 query run에서는 advanced mode 하나만 선택한다. 지원 mode는 다음 여섯 개다.

```text
none
query_rewrite
hyde
rerank
sentence_window
contextual_compression
```

공통 실행 흐름은 아래와 같다.

```text
original_query
  -> query_transform: query_rewrite, hyde
  -> retrieve: FAISS dense retrieval
  -> rerank: rerank
  -> context_transform: sentence_window, contextual_compression
  -> prompt_build
  -> final_generation
```

최종 답변 prompt에는 항상 원 질의 `original_query`를 사용한다. `query_rewrite`와 `hyde`만 검색 질의 `retrieval_query`를 바꾼다. `rerank`, `sentence_window`, `contextual_compression`, `none`은 기본적으로 `retrieval_query == original_query`가 된다.

## 커밋 트레이스

공통 계약 브랜치:

```text
56c4e91 Document advanced RAG implementation plan
13eea97 Add advanced mode request contract
88fc45c Thread advanced mode through query execution
bd003f0 Add advanced trace metadata
c8fb20a Add advanced contract tests
ec5078e Document sentence window v2 plan
```

Query Rewriting 브랜치:

```text
fd344f5 Add query rewrite transformer helper
6a21d70 Apply query rewrite before retrieval
979bb89 Test query rewrite transformer contract
```

HyDE 브랜치:

```text
6eadbda Add HyDE transformer helper
bcb5d8c Apply HyDE before retrieval
f0bf0bc Test HyDE transformer contract
```

Contextual Compression 브랜치:

```text
634ed50 Add contextual compression helper
7a2bbe9 Apply contextual compression before prompting
515ac1b Test contextual compression contract
```

Native rerank 브랜치:

```text
3a43e04 Add native reranker helper
077cde5 Apply rerank candidate flow
3f20a51 Test native rerank contract
```

BGE reranker 브랜치:

```text
00f50b4 Add BGE reranker config
6df85e6 Add BGE reranker backend
0b2d257 Wire BGE reranker into pipeline
cab93a4 Load BGE reranker only for rerank runs
bce279a Add BGE reranker setup assets
```

Sentence Window v2 브랜치:

```text
ffab908 Add sentence metadata to ingest schema
8a1b8dc Add sentence window expander
b5a0bf8 Wire sentence window into query pipeline
ee0426e Test sentence window metadata flow
```

통합 브랜치 merge commit:

```text
b772e89 Merge advanced RAG contract
7b8b423 Merge query rewrite advanced mode
8345e4b Merge HyDE advanced mode
5468d81 Merge contextual compression advanced mode
814bce4 Merge sentence window advanced mode
```

## 변경 파일 전체 목록

`main...feature/advanced-rag-all` 기준 변경 파일은 다음과 같다.

```text
M  .gitignore
M  CMakeLists.txt
M  apps/ingest_main.cpp
M  apps/query_main.cpp
A  configs/retrieval_scifact_bge_rerank.yaml
A  docs/ADVANCED_RAG_IMPLEMENTATION_PLAN.md
M  docs/IO_CONTRACT.md
A  include/advanced_mode.hpp
A  include/advanced_query_transformer.hpp
M  include/beir_export.hpp
A  include/bge_reranker.hpp
M  include/config.hpp
A  include/contextual_compressor.hpp
A  include/hyde_transformer.hpp
M  include/query_request_io.hpp
M  include/query_response_json.hpp
M  include/rag_pipeline.hpp
A  include/reranker.hpp
A  include/sentence_window.hpp
M  include/text_chunker.hpp
M  include/types.hpp
A  scripts/setup_bge_reranker.sh
M  scripts/setup_third_party.sh
A  src/advanced_mode.cpp
A  src/advanced_query_transformer.cpp
M  src/beir_export.cpp
A  src/bge_reranker.cpp
M  src/config.cpp
A  src/contextual_compressor.cpp
A  src/hyde_transformer.cpp
M  src/query_request_io.cpp
M  src/query_response_json.cpp
M  src/rag_pipeline.cpp
A  src/reranker.cpp
A  src/sentence_window.cpp
M  src/text_chunker.cpp
A  tests/advanced_contract_test.cpp
```

## 공통 계약 변경

`include/advanced_mode.hpp`, `src/advanced_mode.cpp`를 추가했다. `AdvancedMode` enum은 `None`, `QueryRewrite`, `Hyde`, `Rerank`, `SentenceWindow`, `ContextualCompression`을 가진다. 문자열 parser는 빈 문자열과 `none`을 `None`으로 해석하고, 알 수 없는 mode는 `std::runtime_error`를 던진다.

`include/query_request_io.hpp`, `src/query_request_io.cpp`의 `QueryRequest`에 `advanced` 필드를 추가했다. request JSON 허용 필드에 `advanced`를 추가했고, 누락 시 `none`으로 동작한다. `--request_json` 사용 시 CLI `--advanced`와 혼용하지 않도록 `apps/query_main.cpp`에서 금지했다. 단일 질문 CLI(`--question`)와 batch export에서는 `--advanced`를 사용할 수 있다.

`include/types.hpp`의 `RagResult`에 다음 trace 필드를 추가했다.

```text
advanced_mode
original_query
retrieval_query
query_transform_ms
rerank_ms
context_transform_ms
```

기존 `retrieve_ms`, `prompt_build_ms`, `generate_ms`, `total_ms`, token 수, truncation 여부, throughput 측정은 유지했다.

`src/query_response_json.cpp`는 single-run output top-level, request, run_config, trace에 advanced 정보를 기록한다. `retrieval_query`는 top-level과 trace에 모두 기록한다. `retrieved_contexts[]`에는 기존 `rank`, `doc_id`, `chunk_id`, `score`, `text`를 유지하면서 `doc_chunk_index`, `sentence_begin`, `sentence_end`, `rerank_score`를 조건부로 추가한다.

`src/beir_export.cpp`는 batch JSONL row와 manifest에 advanced mode를 기록한다. prediction row에는 RAGAS 입력 필드인 `user_input`, `response`, `retrieved_contexts`, `reference`를 보존하고, 분석용 필드로 `retrieval_query`, `retrieved_ranks`, `retrieved_doc_ids`, `retrieved_scores`, `retrieved_rerank_scores`, `retrieved_sentence_begins`, `retrieved_sentence_ends`, `trace`를 추가했다.

## RagPipeline 통합 동작

`include/rag_pipeline.hpp`, `src/rag_pipeline.cpp`의 생성자 인자는 다음 책임을 갖는다.

```text
Retriever: dense retrieval
PromptBuilder: final prompt body 생성
Generator: query transform, compression, final answer 생성
top_k: 최종 context 수
rerank_candidate_k: rerank mode에서 1차 dense candidate 수
BgeReranker*: BGE backend 선택 시 사용, 없으면 lexical fallback
DocumentSentence vector: sentence_window mode에서 context 확장에 사용
```

`RagPipeline::run(query, advanced)`의 구현 순서는 다음과 같다.

```text
1. retrieval_query = query로 초기화한다.
2. query_rewrite이면 Generator로 rewrite prompt를 생성하고 sanitize 결과를 retrieval_query로 쓴다.
3. hyde이면 Generator로 hypothetical document를 생성하고 sanitize 결과를 retrieval_query로 쓴다.
4. rerank이면 dense retrieval 수를 rerank_candidate_k로 늘리고, 그 외 mode는 top_k로 검색한다.
5. rerank이면 BGE reranker가 있으면 BGE score로, 없으면 lexical overlap score로 top_k를 재정렬한다.
6. sentence_window이면 dense hit의 chunk text를 대표 문장 기준 window text로 교체한다.
7. contextual_compression이면 Generator로 retrieved contexts를 압축하고 압축 context를 prompt에 넣는다.
8. PromptBuilder는 항상 original query와 최종 contexts로 prompt body를 만든다.
9. Generator가 final answer를 생성한다.
10. 모든 stage latency와 output token throughput을 RagResult에 기록한다.
```

## Query Rewriting

추가 파일:

```text
include/advanced_query_transformer.hpp
src/advanced_query_transformer.cpp
```

`buildQueryRewritePrompt(query)`는 semantic document retrieval용으로 질의를 재작성하라고 지시한다. prompt는 원 질의를 `[Query]` 아래에 넣고 `[Rewritten Query]` 섹션만 출력하도록 요구한다.

`sanitizeQueryRewrite(generated_text, fallback_query)`는 생성 결과의 첫 non-empty line을 사용한다. 앞뒤 공백을 제거하고, 한 줄 전체가 큰따옴표로 감싸진 경우 따옴표를 제거한다. 최대 길이는 256자로 제한한다. 결과가 비어 있으면 원 질의로 fallback한다.

Pipeline mode `query_rewrite`에서는 final answer 생성 전에 Generator 호출이 한 번 추가된다. 검색에는 rewritten query를 쓰고, 최종 prompt에는 original query를 쓴다. 실패나 빈 출력은 전체 run 실패로 전파하지 않고 original query retrieval로 대체한다.

## HyDE

추가 파일:

```text
include/hyde_transformer.hpp
src/hyde_transformer.cpp
```

`buildHydePrompt(query)`는 답을 직접 만들라는 지시가 아니라 retrieval evidence 검색에 유용한 짧은 hypothetical document를 만들라고 지시한다. 출력 섹션은 `[Hypothetical Document]`다.

`sanitizeHydeDocument(generated_text, fallback_query)`는 전체 생성 결과를 trim해서 사용한다. 최대 길이는 768자다. 빈 결과나 trim 후 빈 문자열은 original query로 fallback한다.

Pipeline mode `hyde`에서는 검색 query가 hypothetical document로 바뀐다. final prompt에는 original query를 사용한다. HyDE 생성 실패를 실험 전체 실패로 보지 않도록 fallback 정책을 둔 것이 핵심이다.

## Rerank

추가 파일:

```text
include/reranker.hpp
src/reranker.cpp
include/bge_reranker.hpp
src/bge_reranker.cpp
configs/retrieval_scifact_bge_rerank.yaml
scripts/setup_bge_reranker.sh
```

`src/config.cpp`는 retrieval config에 다음 필드를 추가로 읽는다.

```yaml
rerank_candidate_k: 9

reranker:
  backend: "lexical" | "bge"
  model_path: "models/bge-reranker-base-onnx/model.onnx"
  sentencepiece_model_path: "models/bge-reranker-base-onnx/sentencepiece.bpe.model"
  max_length: 512
```

`rerank_candidate_k`가 없으면 `top_k * 3`을 기본값으로 쓴다. `reranker.backend` 기본값은 `lexical`이다. backend는 `lexical` 또는 `bge`만 허용한다. `backend=bge`인데 `model_path` 또는 `sentencepiece_model_path`가 비어 있으면 config load 단계에서 실패한다.

Native lexical reranker는 query term set과 context term set의 overlap을 다음 형태로 점수화한다.

```text
score = overlap_count / sqrt(query_terms_count * context_terms_count)
```

정렬 기준은 `rerank_score` 내림차순, 동점이면 dense `score` 내림차순, 그래도 동점이면 기존 candidate 순서다. 최종 반환 수는 `top_k`다. 반환된 context에는 `has_rerank_score=true`, `rerank_score=<computed>`가 기록된다.

BGE reranker는 `onnx-community/bge-reranker-base-ONNX`의 ONNX 모델과 `BAAI/bge-reranker-base`의 SentencePiece model을 기준으로 구현했다. `scripts/setup_bge_reranker.sh`는 기본적으로 `onnx/model_q4.onnx`, `sentencepiece.bpe.model`, `config.json`을 `models/bge-reranker-base-onnx/` 아래에 받는다.

BGE tokenization은 XLM-RoBERTa 계열 pair input을 따른다.

```text
<s> query </s></s> passage </s>
```

코드의 special token id는 다음과 같다.

```text
BOS: 0
EOS: 2
UNK: 3
```

SentencePiece piece id는 XLM-RoBERTa vocab offset에 맞춰 `piece_id + 1`로 변환하고, SentencePiece unk id는 `3`으로 보정한다. `max_length`는 pair 전체 길이 제한이며, 초과 시 passage token을 먼저 뒤에서 줄이고 passage가 비면 query token을 줄인다. ONNX session input name은 모델에서 동적으로 읽고, `input_ids`, `attention_mask`, `token_type_ids`를 지원한다. output tensor의 첫 번째 float logit을 rerank score로 사용한다.

`apps/query_main.cpp`는 실제 run이 `advanced=rerank`이고 `reranker.backend=bge`일 때만 `BgeReranker`를 생성한다. 즉 rerank가 아닌 run에서는 BGE 모델을 로드하지 않는다. SentencePiece C++ header/library가 없는 빌드에서 `backend=bge`를 실행하면 명시적 runtime error가 발생한다. lexical backend는 SentencePiece 없이 동작한다.

BGE reranker는 외부 모델을 하나 더 사용하는 검색 단계 cross-encoder다. 생성 모델 의존 기법으로 분류하지 않았고, 실험에서는 별도 latency/memory 비용을 측정해야 한다.

## Sentence Window Retrieval v2

추가 파일:

```text
include/sentence_window.hpp
src/sentence_window.cpp
```

변경 파일:

```text
include/text_chunker.hpp
src/text_chunker.cpp
apps/ingest_main.cpp
apps/query_main.cpp
include/types.hpp
```

이 구현에서 ingest는 corpus를 chunk store, FAISS index, sentence store로 변환하는 사전 색인 단계다. BGE rerank에는 새 ingest가 필요 없다. Sentence Window Retrieval에는 sentence metadata가 필요하므로 새 ingest 결과가 있으면 정확한 sentence window가 동작하고, legacy chunk store만 있으면 원 context를 그대로 쓰는 fallback으로 동작한다.

`Chunk`에는 다음 필드를 추가했다.

```text
doc_chunk_index
sentence_begin
sentence_end
```

`DocumentSentence` 구조체는 다음 필드를 가진다.

```text
doc_id
sentence_index
text
```

`TextChunker::chunkDocumentsWithSentences()`는 기존 chunk size와 overlap cadence를 유지하면서 문장 metadata를 계산한다. 문장 분리는 whitespace token을 순회하면서 `.`, `!`, `?`로 끝나는 token을 문장 끝으로 본다. 마지막 token까지 sentence terminator가 없으면 남은 token들을 하나의 문장으로 저장한다.

`apps/ingest_main.cpp`는 기존 chunk store JSONL에 `doc_chunk_index`, `sentence_begin`, `sentence_end`를 저장한다. 추가로 다음 경로에 sentence store JSONL을 저장한다.

```text
<chunk_store_path>.sentences.jsonl
```

sentence store row 형식:

```json
{"doc_id":"...", "sentence_index":0, "text":"..."}
```

`apps/query_main.cpp`는 query 실행 시 `<chunk_store_path>.sentences.jsonl`을 읽으려고 시도한다. 파일이 없으면 빈 sentence store를 사용한다. 기존 chunk store row에 sentence field가 없어도 optional field로 처리한다.

`selectRepresentativeSentence(query, hit, sentences)`는 hit chunk의 `[sentence_begin, sentence_end)` 범위 안에서 query와 lexical overlap이 가장 높은 문장을 고른다. 범위가 없거나 후보 문장이 없으면 `-1`을 반환한다.

`expandSentenceWindows(query, contexts, sentences, window_size)`는 각 retrieved hit마다 대표 문장을 고르고 같은 `doc_id` 안에서 `selected - window_size`부터 `selected + window_size`까지 문장을 붙여 context text를 교체한다. dense score, rerank score, rank 해석, doc_id는 원 hit 기준을 유지한다. 구현에서 pipeline window size는 현재 `1`로 고정했다.

이 설계는 sentence 단위 embedding/index를 만들지 않는다. 이유는 실험 통제 때문이다. 검색 단위와 chunking 조건을 바꾸면 sentence window 효과와 retrieval granularity 효과가 섞인다. 따라서 dense retrieval은 기존 chunk embedding을 그대로 사용하고, post-retrieval context text만 sentence window로 교체한다.

## Contextual Compression

추가 파일:

```text
include/contextual_compressor.hpp
src/contextual_compressor.cpp
```

`buildContextualCompressionPrompt(query, contexts)`는 query와 검색된 contexts를 넣고, 원문에 없는 사실을 추가하지 말고 query에 필요한 context statement만 추출하라고 지시한다. 출력은 답변이 아니라 `[Compressed Context]`다.

`buildCompressedContexts(generated_text, fallback_contexts)`는 생성 결과를 trim하고 최대 1600자로 제한한다. 결과가 비면 fallback contexts를 그대로 반환한다. 결과가 유효하면 하나의 synthetic context를 만든다.

```text
chunk_id = -1
doc_id = "compressed"
text = compressed output
```

fallback이 아닌 경우 최종 prompt에는 이 compressed context 하나가 들어간다. original retrieved metadata는 압축 context 내부에서 보존하지 않는다. batch/single output의 retrieved_contexts는 최종 prompt에 들어간 context를 기준으로 기록된다.

## 출력 계약

Single-run success output은 다음 필드를 추가로 가진다.

```text
advanced
retrieval_query
request.advanced
run_config.advanced
run_config.retrieval.rerank_candidate_k
run_config.retrieval.reranker.backend
run_config.retrieval.reranker.model_path
run_config.retrieval.reranker.sentencepiece_model_path
run_config.retrieval.reranker.max_length
trace.latency_ms.query_transform
trace.latency_ms.rerank
trace.latency_ms.context_transform
trace.retrieval_query
retrieved_contexts[].doc_chunk_index
retrieved_contexts[].sentence_begin
retrieved_contexts[].sentence_end
retrieved_contexts[].rerank_score
```

Batch prediction JSONL row는 다음 필드를 추가로 가진다.

```text
advanced
retrieval_query
retrieved_rerank_scores
retrieved_sentence_begins
retrieved_sentence_ends
trace.latency_ms.query_transform
trace.latency_ms.rerank
trace.latency_ms.context_transform
trace.retrieval_query
```

RAGAS에 직접 쓰는 네 필드 `user_input`, `response`, `retrieved_contexts`, `reference`는 유지했다.

## CMake와 의존성

`CMakeLists.txt`는 `rag_core`에 다음 source를 추가했다.

```text
src/advanced_mode.cpp
src/advanced_query_transformer.cpp
src/hyde_transformer.cpp
src/contextual_compressor.cpp
src/bge_reranker.cpp
src/reranker.cpp
src/sentence_window.cpp
```

`BUILD_RAG_TESTS` 옵션과 `rag_contract_tests` test executable을 추가했다. 테스트 executable은 외부 FAISS/llama runtime 없이 advanced contract helper들을 직접 검증하도록 구성했다.

SentencePiece 의존성은 optional compile flag로 처리한다.

```text
RAG_HAS_SENTENCEPIECE=1: BGE reranker 사용 가능
RAG_HAS_SENTENCEPIECE=0: backend=bge 실행 시 명시적 error
```

`scripts/setup_third_party.sh`는 SentencePiece `v0.2.0`을 `third_party/sentencepiece-install`에 설치하도록 확장했다. `.gitignore`에는 BGE SentencePiece model과 SentencePiece source/install directory를 추가했다.

## 자체 테스트

통합 후 직접 실행한 테스트:

```bash
g++ -std=c++17 \
  -Iinclude \
  -Ithird_party/llama.cpp/vendor \
  tests/advanced_contract_test.cpp \
  src/advanced_mode.cpp \
  src/advanced_query_transformer.cpp \
  src/hyde_transformer.cpp \
  src/contextual_compressor.cpp \
  src/sentence_window.cpp \
  src/text_chunker.cpp \
  src/query_request_io.cpp \
  src/reranker.cpp \
  -o /tmp/rag_all_contract_tests && \
  /tmp/rag_all_contract_tests
```

결과:

```text
pass
```

Pipeline syntax check:

```bash
g++ -std=c++17 \
  -DRAG_HAS_SENTENCEPIECE=0 \
  -Iinclude \
  -Ithird_party/faiss \
  -Ithird_party/llama.cpp/vendor \
  -Ithird_party/onnxruntime-linux-x64-1.24.3/include \
  -fsyntax-only src/rag_pipeline.cpp
```

결과:

```text
pass
```

CLI syntax check:

```bash
g++ -std=c++17 \
  -DRAG_HAS_SENTENCEPIECE=0 \
  -Iinclude \
  -Ithird_party/faiss \
  -Ithird_party/llama.cpp/vendor \
  -Ithird_party/onnxruntime-linux-x64-1.24.3/include \
  -fsyntax-only apps/query_main.cpp
```

결과:

```text
pass
```

BEIR export syntax check:

```bash
g++ -std=c++17 \
  -DRAG_HAS_SENTENCEPIECE=0 \
  -Iinclude \
  -Ithird_party/faiss \
  -Ithird_party/llama.cpp/vendor \
  -Ithird_party/onnxruntime-linux-x64-1.24.3/include \
  -fsyntax-only src/beir_export.cpp
```

결과:

```text
pass
```

Whitespace check:

```bash
git diff --check
```

결과:

```text
pass
```

전체 CMake configure:

```bash
cmake -S . -B build -DBUILD_RAG_TESTS=ON
```

결과:

```text
fail: faissConfig.cmake 또는 faiss-config.cmake를 찾지 못함
```

이 실패는 Advanced RAG 코드의 컴파일 오류가 아니라 현재 로컬 환경에서 FAISS CMake package path가 설정되지 않은 문제다. 해결하려면 FAISS development package를 설치하거나 `FAISS_ROOT`, `faiss_DIR`, `CMAKE_PREFIX_PATH` 중 하나를 FAISS config가 있는 경로로 지정해야 한다.

## 실험 방법론에 반영할 구현 가정

단일 run에는 advanced mode 하나만 적용한다. 조합 실험은 이번 구현 범위가 아니다.

`none`은 Naive RAG baseline이다. 추가 stage latency는 0에 가깝고, retrieval query는 original query와 같다.

`query_rewrite`와 `hyde`는 생성 모델에 의존한다. 둘 다 검색 query만 바꾸고 최종 답변 prompt의 query는 바꾸지 않는다. 생성 결과가 비어 있거나 sanitize 후 사용할 수 없으면 original query로 검색한다.

`rerank`는 검색 단계 최적화다. 1차 dense retrieval에서 `rerank_candidate_k`개를 가져온 뒤 reranker score로 `top_k`개를 고른다. BGE backend는 외부 ONNX cross-encoder와 SentencePiece tokenizer를 추가로 사용한다. BGE를 쓰면 모델 로딩, pair tokenization, candidate별 ONNX inference 비용이 latency/memory 측정에 포함되어야 한다.

`sentence_window`는 LLM을 호출하지 않는다. dense retrieval unit은 기존 chunk다. sentence metadata가 있을 때만 context text를 대표 문장 주변 window로 교체한다. metadata가 없으면 legacy chunk text로 fallback한다. 따라서 새 ingest는 sentence_window 품질 실험에는 필요하지만 BGE rerank 실험에는 필요 없다.

`contextual_compression`은 검색 후 prompt 전 context transform이다. Generator를 한 번 추가 호출해서 contexts를 하나의 compressed context로 만든다. 압축 실패 시 원 contexts를 그대로 사용한다. 이 mode는 추가 생성 호출 비용과 prompt token 감소 효과를 함께 측정해야 한다.

모든 mode에서 최종 품질 평가는 동일한 RAGAS 입력 필드 `user_input`, `response`, `retrieved_contexts`, `reference`를 사용한다. 효율 평가는 `trace.latency_ms`, `memory_mb`, `output_tokens_per_sec`, token count를 사용한다.
