# RAG Query I/O Contract

`rag_query`를 실험 자동화와 RAGAS 평가 파이프라인에서 사용하기 위한 입출력 규약이다.

## 범위

이 문서는 다음 실행 형태를 다룬다.

| 실행 형태 | 입력 | 출력 |
|---|---|---|
| 단일 수동 질의 | `--question` | `stdout` JSON object |
| 단일 구조화 질의 | `--request_json` | `stdout` JSON object |
| BEIR 배치 export | `--dataset`, `--split`, `--questions`, `--qrels`, `--corpus`, `--output_jsonl` | prediction JSONL, manifest JSON, `stdout` summary |

생성, 검색, 프롬프트 설정은 `--model_config`와 `--retrieval_config`로 지정한다.

## 출력 채널

| 채널 | 내용 |
|---|---|
| `stdout` | 기계 파싱용 JSON object 1개 |
| `stderr` | 진행 상태, 경고, 오류 로그 |
| prediction JSONL | 배치 샘플별 RAG 응답과 RAGAS 입력 필드 |
| manifest JSON | 배치 실행 단위 입력 경로와 실행 설정 |

실행 관측값은 단일 실행과 배치 실행 모두 `trace`에 기록한다.

## 공통 식별 필드

다음 필드는 단일 output, prediction row, manifest에서 같은 의미로 사용한다.

| 필드 | 타입 | 의미 |
|---|---|---|
| `mode` | string | 출력 종류와 실행 경로를 구분하는 식별자 |
| `dataset` | string \| null | 실험 데이터셋 이름. 예: `scifact`, `nq` |
| `split` | string \| null | 데이터셋 분할 이름. 예: `train`, `dev`, `test` |
| `sample_id` | string \| null | `dataset`/`split` 안에서 샘플을 식별하는 ID |
| `advanced` | string | Advanced RAG mode. 기본값은 `none` |

`mode` 값:

| 값 | 사용 위치 | 의미 |
|---|---|---|
| `naive_rag` | 단일 output | 현재 RAG pipeline 실행 결과 |
| `batch_export_jsonl` | batch summary, manifest | 배치 prediction JSONL export 결과 |

## Trace

단일 output과 prediction row는 같은 `trace` 구조를 사용한다.

| 필드 | 타입 | 의미 |
|---|---|---|
| `latency_ms.total` | integer | 검색 시작부터 생성 종료까지의 wall-clock 시간(ms) |
| `latency_ms.query_transform` | integer | advanced query transform 시간(ms). 미사용 시 0 |
| `latency_ms.retrieval` | integer | query embedding과 vector search 시간(ms) |
| `latency_ms.rerank` | integer | rerank 시간(ms). 미사용 시 0 |
| `latency_ms.context_transform` | integer | retrieved context 변환/압축/window 확장 시간(ms). 미사용 시 0 |
| `latency_ms.prompt_build` | integer | retrieved context를 prompt body로 구성한 시간(ms) |
| `latency_ms.generation` | integer | LLM 생성 시간(ms) |
| `output_tokens_per_sec` | number | `output_tokens / generation elapsed seconds` |
| `memory_mb` | number \| null | RAG 처리 완료 직후 `rag_query` 프로세스의 resident set size(RSS) MB. 측정 불가 시 `null` |
| `prompt_tokens_before_truncation` | integer | context window 제한 적용 전 prompt token 수 |
| `prompt_tokens` | integer | 실제 모델 입력에 사용된 prompt token 수 |
| `output_tokens` | integer | 생성된 token 수 |
| `prompt_truncated` | boolean | context window 제한으로 prompt가 잘렸는지 여부 |
| `retrieval_query` | string | 검색에 실제 사용한 query |

`memory_mb`는 `pipeline.run()`이 반환된 직후 `/proc/self/status`의 `VmRSS`를 읽은 값이다.
따라서 요청 처리 중 최대 RSS(peak RSS), `rag_http_server` 프로세스의 RSS, 장치 전체 메모리 사용량을 의미하지 않는다.
보고서와 결과표에서는 `end-of-request RSS (MB)`로 표기한다. Peak memory가 필요하면 `VmHWM` 또는
`getrusage(RUSAGE_SELF).ru_maxrss`를 별도 지표로 구현해야 한다.

## 단일 실행

### 수동 질의

```bash
rag_query --question "..."
```

질문 문자열 하나를 직접 실행한다.

### 구조화 질의

```bash
rag_query --request_json request.json
```

JSON 파일에서 질문과 샘플 식별 정보를 읽어 실행한다.

### Request JSON

| 필드 | 필수 여부 | 타입 | 의미 |
|---|---|---|---|
| `question` | 필수 | string | 질의문 |
| `run_id` | 선택 | string | 실행 추적 ID |
| `dataset` | 선택 | string | 데이터셋 이름 |
| `split` | 선택 | string | 데이터셋 분할 이름 |
| `sample_id` | 선택 | string | 샘플 ID |
| `advanced` | 선택 | string | Advanced RAG mode. 기본값은 `none` |

예시:

```json
{
  "dataset": "scifact",
  "split": "test",
  "sample_id": "1",
  "advanced": "none",
  "question": "0-dimensional biomaterials lack inductive properties."
}
```

Request JSON은 질문과 샘플 식별 정보만 포함한다. 생성 길이, sampling, 검색 top-k, 프롬프트 경로는 config 파일에서 지정한다.
지원하는 `advanced` 값은 `none`, `query_rewrite`, `hyde`, `rerank`, `sentence_window`, `contextual_compression`이다.

### Output JSON

성공 시 exit code는 `0`, 실패 시 exit code는 `1`이다.

최상위 필드:

| 필드 | 타입 | 의미 |
|---|---|---|
| `status` | string | `ok` 또는 `error` |
| `error` | object \| null | 실패 메시지 |
| `mode` | string | 출력 종류와 실행 경로를 구분하는 식별자 |
| `advanced` | string | Advanced RAG mode |
| `request` | object | 입력 질의와 샘플 식별 정보 |
| `answer` | string | 모델 응답 |
| `retrieval_query` | string \| null | 검색에 실제 사용한 query. 실패 시 `null` |
| `retrieved_contexts` | array | 검색된 context 목록 |
| `run_config` | object \| null | 실제 실행 설정 |
| `trace` | object | latency, throughput, memory, token count, truncation 등 실행 관측값 |

`request`:

| 필드 | 타입 | 의미 |
|---|---|---|
| `run_id` | string \| null | 실행 추적 ID |
| `dataset` | string \| null | 데이터셋 이름 |
| `split` | string \| null | 데이터셋 분할 이름 |
| `sample_id` | string \| null | 샘플 ID |
| `advanced` | string | Advanced RAG mode |
| `question` | string \| null | 질의문 |

`retrieved_contexts[]`:

| 필드 | 타입 | 의미 |
|---|---|---|
| `rank` | integer | 1부터 시작하는 검색 순위 |
| `doc_id` | string | 원문 문서 ID |
| `chunk_id` | integer | chunk store 내 chunk ID |
| `doc_chunk_index` | integer | 문서 내부 chunk 순서. legacy chunk store에서는 생략될 수 있음 |
| `sentence_begin` | integer | context text가 대응하는 문서 내 시작 sentence index. legacy chunk store에서는 생략될 수 있음 |
| `sentence_end` | integer | context text가 대응하는 문서 내 exclusive end sentence index. legacy chunk store에서는 생략될 수 있음 |
| `score` | number | retriever 원점수. 검색 순서는 `rank`를 기준으로 해석한다. |
| `rerank_score` | number | rerank 적용 시 reranker 점수 |
| `text` | string | context text |

`run_config`:

| 필드 | 타입 | 의미 |
|---|---|---|
| `model_config_path` | string \| null | model config 경로 |
| `retrieval_config_path` | string \| null | retrieval config 경로 |
| `advanced` | string | Advanced RAG mode |
| `generation` | object | 생성 설정 |
| `retrieval` | object | 검색 설정 |
| `prompt` | object | 프롬프트 설정 |

`generation`:
- `max_new_tokens`
- `temperature`
- `top_p`
- `top_k`
- `seed`
- `repetition_penalty`
- `repetition_penalty_last_n`
- `stop_strings`

`retrieval`:
- `top_k`
- `rerank_candidate_k`
- `chunk_size`
- `chunk_overlap`
- `data_dir`
- `index_path`
- `chunk_store_path`
- `reranker.backend`
- `reranker.model_path`
- `reranker.sentencepiece_model_path`
- `reranker.max_length`

`prompt`:
- `system_prompt_path`
- `use_model_chat_template`
- `prompt_builder`
- `context_section`
- `context_item_label`
- `query_section`
- `output_section`

`trace` 구조는 Trace 섹션을 따른다.

단일 실행의 `retrieved_contexts`는 디버깅과 검토를 위해 context별 metadata를 포함하는 object 배열이다.

성공 예시:

```json
{
  "status": "ok",
  "error": null,
  "mode": "naive_rag",
  "advanced": "none",
  "request": {
    "run_id": null,
    "dataset": "scifact",
    "split": "test",
    "sample_id": "1",
    "advanced": "none",
    "question": "0-dimensional biomaterials lack inductive properties."
  },
  "answer": "The retrieved evidence does not support the claim.",
  "retrieval_query": "0-dimensional biomaterials lack inductive properties.",
  "retrieved_contexts": [
    {
      "rank": 1,
      "doc_id": "31715818",
      "chunk_id": 42,
      "score": 0.8123,
      "text": "..."
    }
  ],
  "run_config": {
    "model_config_path": "configs/model.yaml",
    "retrieval_config_path": "configs/retrieval_scifact.yaml",
    "advanced": "none",
    "generation": {
      "max_new_tokens": 128,
      "temperature": 0.3,
      "top_p": 0.85,
      "top_k": 40,
      "seed": 42,
      "repetition_penalty": 1.1,
      "repetition_penalty_last_n": 64,
      "stop_strings": []
    },
    "retrieval": {
      "top_k": 3,
      "rerank_candidate_k": 9,
      "chunk_size": 256,
      "chunk_overlap": 32,
      "data_dir": "data/beir/scifact_corpus",
      "index_path": "data/indexes/scifact/faiss.index",
      "chunk_store_path": "data/indexes/scifact/chunks.jsonl",
      "reranker": {
        "backend": "lexical",
        "model_path": null,
        "sentencepiece_model_path": null,
        "max_length": 512
      }
    },
    "prompt": {
      "system_prompt_path": "configs/prompt_system_scifact.txt",
      "use_model_chat_template": false,
      "prompt_builder": "configurable_sectioned",
      "context_section": "Evidence",
      "context_item_label": "Evidence",
      "query_section": "Claim",
      "output_section": "Output"
    }
  },
  "trace": {
    "latency_ms": {
      "total": 1320,
      "query_transform": 0,
      "retrieval": 24,
      "rerank": 0,
      "context_transform": 0,
      "prompt_build": 1,
      "generation": 1295
    },
    "output_tokens_per_sec": 12.4,
    "memory_mb": 512.5,
    "prompt_tokens_before_truncation": 680,
    "prompt_tokens": 680,
    "output_tokens": 16,
    "prompt_truncated": false,
    "retrieval_query": "0-dimensional biomaterials lack inductive properties."
  }
}
```

실패 예시:

```json
{
  "status": "error",
  "error": {
    "message": "Request field question must not be blank."
  },
  "mode": "naive_rag",
  "advanced": "none",
  "retrieval_query": null,
  "request": {
    "run_id": null,
    "dataset": null,
    "split": null,
    "sample_id": null,
    "advanced": "none",
    "question": ""
  },
  "answer": "",
  "retrieved_contexts": [],
  "run_config": null,
  "trace": {
    "latency_ms": {
      "total": 0,
      "query_transform": 0,
      "retrieval": 0,
      "rerank": 0,
      "context_transform": 0,
      "prompt_build": 0,
      "generation": 0
    },
    "output_tokens_per_sec": 0.0,
    "memory_mb": null,
    "prompt_tokens_before_truncation": 0,
    "prompt_tokens": 0,
    "output_tokens": 0,
    "prompt_truncated": false,
    "retrieval_query": ""
  }
}
```

## BEIR 배치 Export

```bash
rag_query \
  --dataset scifact \
  --split test \
  --questions queries.jsonl \
  --qrels test.tsv \
  --corpus corpus.jsonl \
  --output_jsonl predictions.jsonl \
  [--limit N]
```

단일 실행 옵션(`--question`, `--request_json`)과 배치 출력 옵션(`--output_jsonl`)은 함께 쓰지 않는다.

### 입력 옵션

| 옵션 | 입력 형식 | 의미 |
|---|---|---|
| `--dataset` | string | prediction row와 manifest에 기록할 데이터셋 이름 |
| `--split` | string | prediction row와 manifest에 기록할 데이터셋 split |
| `--questions` | BEIR `queries.jsonl` | query ID와 질문 text를 읽는다. `_id` 또는 `id`, `text` 또는 `query` 필드를 사용한다. |
| `--qrels` | TSV | 평가 대상 query 순서와 positive 문서 ID를 읽는다. `score > 0` row만 사용한다. |
| `--corpus` | BEIR `corpus.jsonl` | qrels 문서 ID에 해당하는 reference text를 만든다. `_id` 또는 `id`, `title`, `text`/`contents`/`body` 필드를 사용한다. |
| `--output_jsonl` | path | prediction JSONL 출력 경로 |
| `--limit` | integer | qrels 순서 기준 최대 처리 샘플 수. 생략하거나 `0` 이하이면 제한 없음 |

qrels TSV 컬럼:

| 컬럼 | 의미 |
|---|---|
| `query-id` | `queries.jsonl`의 query ID |
| `corpus-id` | `corpus.jsonl`의 문서 ID |
| `score` | relevance score |

### 산출물

| 산출물 | 경로 | 의미 |
|---|---|---|
| prediction JSONL | `--output_jsonl` | 샘플별 RAG 응답과 RAGAS 입력 필드 |
| manifest JSON | `<output_jsonl>.manifest.json` | 실행 단위 입력 경로와 설정 |

`stdout` summary 예시:

```json
{
  "mode": "batch_export_jsonl",
  "status": "ok",
  "error": null,
  "dataset": "scifact",
  "split": "test",
  "output_jsonl": "results/runs/scifact_naive_rag/predictions.jsonl",
  "manifest_json": "results/runs/scifact_naive_rag/predictions.jsonl.manifest.json",
  "rows_written": 5
}
```

### Prediction Row

| 필드 | 타입 | 의미 |
|---|---|---|
| `sample_id` | string | 데이터셋 내 샘플 ID |
| `dataset` | string | 데이터셋 이름 |
| `split` | string | 데이터셋 분할 이름 |
| `advanced` | string | Advanced RAG mode |
| `user_input` | string | RAGAS `user_input` |
| `retrieval_query` | string | 검색에 실제 사용한 query |
| `response` | string | RAGAS `response` |
| `retrieved_contexts` | array[string] | RAGAS `retrieved_contexts` |
| `reference` | string | RAGAS `reference` |
| `retrieved_ranks` | array[integer] | 검색 순위 |
| `qrel_doc_ids` | array[string] | positive qrels 문서 ID |
| `retrieved_doc_ids` | array[string] | 검색된 문서 ID |
| `retrieved_scores` | array[number] | retriever 원점수 |
| `retrieved_rerank_scores` | array[number \| null] | rerank 적용 시 reranker 점수. 미적용 시 `null` |
| `retrieved_sentence_begins` | array[integer] | 각 context의 문서 내 시작 sentence index. metadata가 없으면 `-1` |
| `retrieved_sentence_ends` | array[integer] | 각 context의 문서 내 exclusive end sentence index. metadata가 없으면 `-1` |
| `trace` | object | 샘플별 실행 trace |

RAGAS 입력으로 직접 쓰는 필드는 `user_input`, `response`, `retrieved_contexts`, `reference`이다.
배치 row의 `retrieved_contexts`는 RAGAS 입력 형식에 맞춘 string 배열이다. 검색 metadata는 `retrieved_ranks`, `retrieved_doc_ids`, `retrieved_scores`, `trace`에 분리해 기록한다.
`retrieved_contexts`, `retrieved_ranks`, `retrieved_doc_ids`, `retrieved_scores`는 같은 길이의 배열이며, 같은 index가 같은 검색 결과를 의미한다.
`retrieved_rerank_scores`도 같은 길이이며, rerank 미적용 row에서는 각 값이 `null`이다.
`retrieved_sentence_begins`와 `retrieved_sentence_ends`도 같은 길이이며, sentence metadata가 없는 legacy row에서는 `-1`이다.

prediction JSONL row 예시:

```json
{
  "sample_id": "1",
  "dataset": "scifact",
  "split": "test",
  "advanced": "none",
  "user_input": "0-dimensional biomaterials lack inductive properties.",
  "retrieval_query": "0-dimensional biomaterials lack inductive properties.",
  "response": "The retrieved evidence does not support the claim.",
  "retrieved_contexts": [
    "..."
  ],
  "reference": "Biomaterials and inductive properties\n...",
  "retrieved_ranks": [
    1
  ],
  "qrel_doc_ids": [
    "31715818"
  ],
  "retrieved_doc_ids": [
    "31715818"
  ],
  "retrieved_scores": [
    0.8123
  ],
  "retrieved_rerank_scores": [
    null
  ],
  "retrieved_sentence_begins": [
    0
  ],
  "retrieved_sentence_ends": [
    2
  ],
  "trace": {
    "latency_ms": {
      "total": 1320,
      "query_transform": 0,
      "retrieval": 24,
      "rerank": 0,
      "context_transform": 0,
      "prompt_build": 1,
      "generation": 1295
    },
    "output_tokens_per_sec": 12.4,
    "memory_mb": 512.5,
    "prompt_tokens_before_truncation": 680,
    "prompt_tokens": 680,
    "output_tokens": 16,
    "prompt_truncated": false,
    "retrieval_query": "0-dimensional biomaterials lack inductive properties."
  }
}
```

### Manifest JSON

| 필드 | 타입 | 의미 |
|---|---|---|
| `mode` | string | `batch_export_jsonl` |
| `status` | string | `ok` 또는 `error` |
| `dataset` | string | 데이터셋 이름 |
| `split` | string | 데이터셋 분할 이름 |
| `output_jsonl` | string | prediction JSONL 경로 |
| `rows_written` | integer | 생성 row 수 |
| `inputs` | object | questions/qrels/corpus 경로 |
| `reference_policy` | string | reference 생성 정책 ID |
| `run_config` | object | 실제 실행 설정 |

`inputs`:
- `questions`
- `qrels`
- `corpus`

`run_config`는 단일 output과 같은 구조를 사용한다.

### BEIR Reference 정책

- qrels의 positive 문서만 사용한다.
- 각 문서는 `title + "\n" + text`로 구성한다.
- 여러 문서는 `"\n\n"`으로 연결한다.
- LLM으로 reference를 생성하지 않는다.

## Ingest 산출물

`rag_ingest`는 `retrieval_config.data_dir` 아래의 `.txt`, `.md`, `.jsonl`을 읽고 chunk store JSONL과 FAISS index를 생성한다.

JSONL 문서 ID 정책:
- `_id`가 있으면 `_id`를 canonical `doc_id`로 사용한다.
- `_id`가 없고 `id`가 있으면 `id`를 사용한다.
- 둘 다 없으면 `<relative-path>#<line-no>`를 사용한다.
- 숫자 ID는 문자열로 변환한다.

chunk store row:

```json
{
  "chunk_id": 0,
  "doc_id": "31715818",
  "text": "..."
}
```

## 자동화 처리

- `stdout`만 읽어 `json.loads()` 한다.
- 성공 판정: `returncode == 0` and `obj["status"] == "ok"`
- 실패 판정: `returncode != 0` or `obj["status"] == "error"`
- 오류 메시지: `obj["error"]["message"]`
