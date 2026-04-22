# RAG Query I/O Contract

`rag_query`를 자동화(sweep / RAGAS 연동)에서 안정적으로 쓰기 위한 입출력 규약이다.
코드 기준 파일:
- `apps/query_main.cpp`
- `src/scifact_export.cpp`

## 기본 원칙

- Naive RAG는 raw 결과만 출력한다.
- RAGAS 점수(faithfulness, answer relevancy, context precision/recall)는 여기서 계산/출력하지 않는다.
- `stdout`은 기계 파싱용 JSON만 출력한다.
- `stderr`는 운영 로그(trace/warn/error)만 출력한다.
- 성능 수치는 JSON 본문(`metrics`/`trace`)을 기준으로 본다.

## 실행 모드

1. 단일 질의
- `--request_json <path>` 권장
- `--question "..."` 간편 입력(수동 테스트용)

2. 배치 export (RAGAS 입력 JSONL 생성)
- `--questions --qrels --corpus --output_jsonl [--limit]`

제약:
- 단일 질의 옵션(`--request_json`/`--question`)과 배치 옵션(`--output_jsonl`)은 같이 사용할 수 없다.

## 단일 질의 입력 (`--request_json`)

스키마: `docs/schemas/rag_query_request.schema.json`

### 필드 규칙

| 필드 | 필수 | 타입 | 비고 |
|---|---|---|---|
| `run_id` | 선택 | string | 비어 있거나 없으면 자동 생성 |
| `dataset` | 필수 | string | 빈 문자열 불가 |
| `question_id` | 필수 | string | 빈 문자열 불가 |
| `question` | 필수 | string | 빈 문자열 불가 |
| `prompt_template_id` | 필수 | string | 빈 문자열 불가 |
| `retrieval.top_k` | 필수 | integer | `>= 1` |
| `generation.temperature` | 필수 | number | `>= 0` |
| `generation.top_p` | 필수 | number | `0 < top_p <= 1` |
| `generation.max_tokens` | 필수 | integer | `>= 1` |

`run_id` 자동 생성 규칙:
- 형식: `auto_<dataset>_<question_id>_k<top_k>_t<temperature*1000>_p<top_p*1000>_m<max_tokens>`
- sweep 자동화에서 별도 관리 ID가 있으면 입력에서 직접 지정하는 것을 권장

입력 예시:

```json
{
  "dataset": "scifact",
  "question_id": "q0001",
  "question": "0-dimensional biomaterials show inductive properties.",
  "prompt_template_id": "qa_only_context_v1",
  "retrieval": {
    "top_k": 3
  },
  "generation": {
    "temperature": 0.1,
    "top_p": 0.9,
    "max_tokens": 64
  }
}
```

## 단일 질의 출력 (`stdout` JSON 1개)

스키마: `docs/schemas/rag_query_response.schema.json`

### 공통

- `stdout`은 JSON object 1개만 출력
- 성공: `status="ok"`, exit code `0`
- 실패: `status="error"`, exit code `1`

### 필드 규칙

| 필드 | 항상 존재 | 타입 | 비고 |
|---|---|---|---|
| `run_id` | Y | string \| null | 입력값 또는 자동 생성값 |
| `mode` | Y | string | 현재 `naive_rag` |
| `dataset` | Y | string \| null | |
| `question_id` | Y | string \| null | |
| `question` | Y | string \| null | |
| `answer` | Y | string | 실패 시 `""` |
| `retrieved_contexts` | Y | array | 실패 시 `[]` |
| `prompt_template_id` | Y | string \| null | |
| `generation` | Y | object | `temperature/top_p/max_tokens` |
| `metrics` | Y | object | `latency_ms/throughput_tokens_per_sec/memory_mb` |
| `status` | Y | string | `ok` or `error` |
| `error` | Y | object \| null | 실패 시 `{"message": ...}` |

`retrieved_contexts[]` 항목:
- `rank` (int)
- `doc_id` (string)
- `chunk_id` (int)
- `score` (number)
- `text` (string)

`metrics.latency_ms` 항목:
- `total`
- `retrieval`
- `prompt_build`
- `generation`

## 배치 export (`--output_jsonl`)

- 역할: RAGAS 평가용 JSONL 생성
- 파일: `--output_jsonl` 경로
- 배치 실행 자체도 `stdout`에 JSON 1개를 출력

배치 실행 결과(stdout) 예시:

```json
{
  "mode": "batch_export_jsonl",
  "status": "ok",
  "error": null,
  "output_jsonl": "results/runs/scifact_naive_rag/predictions.jsonl"
}
```

JSONL row 스키마:
- `docs/schemas/scifact_predictions_row.schema.json`

row 핵심 필드:
- `sample_id`, `user_input`, `response`, `retrieved_contexts`
- `reference`, `qrel_doc_ids`
- `retrieved_doc_ids`, `retrieved_scores`
- `generation_config`, `retrieval_config`, `trace`

reference 생성 정책(SciFact):
- `qrels(test)`의 positive 문서만 사용
- 각 문서는 `title + "\n" + text`
- 여러 문서는 `"\n\n"`으로 연결
- LLM으로 reference를 생성하지 않음

## stderr 정책

- 허용: 경로/진행 상태/경고/예외 메시지
- 벤치마크 수치는 stderr에 싣지 않고 JSON(`metrics`/`trace`)만 사용

## Python 자동화 권장 처리

- `stdout`만 읽어 `json.loads()`
- 성공 판정: `returncode == 0` and `obj["status"] == "ok"`
- 실패 판정: `returncode != 0` or `obj["status"] == "error"`
- 오류 메시지: `obj["error"]["message"]`
