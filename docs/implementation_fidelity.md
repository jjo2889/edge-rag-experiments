# 기법 구현 충실도 감사

## 이 문서를 왜 읽어야 하는가

이 실험은 **"advanced RAG 기법 다섯 개가 온디바이스 SLM에서 효과적인가"**에 답하려 한다.
그리고 지금까지 나온 결과는 대체로 "효과 없음"이다.

음성 결과를 논문으로 주장하려면 반드시 먼저 증명해야 하는 것이 있다.

> **우리가 그 기법을 원 논문대로 구현했다.**

이게 안 되면 "효과가 없다"는 주장은 "너희가 잘못 구현했다"로 즉시 반박된다.
이 문서는 다섯 기법 각각에 대해 **원 논문의 정의**와 **우리 구현**을 나란히 놓고,
다른 점과 고치는 방법을 적는다.

읽는 순서는 각 절마다 동일하다.

1. 원 논문이 무엇을 하라고 했는가 (출처와 인용문)
2. 우리 코드가 실제로 무엇을 하는가 (파일·줄 번호)
3. 무엇이 다른가, 왜 그게 문제인가
4. 어떻게 고치는가 (구체적 변경 지점)

## 한눈에 보기

| 기법 | 원문 대비 | 심각도 | 현 상태로 음성 결과 주장 |
|---|---|---|---|
| HyDE | **수정 완료** (2026-08-03) | — | 가능 |
| Sentence Window | 문장 단위 색인이 아예 없음 | **치명적** | 불가 |
| Contextual Compression | **수정 완료** (2026-08-03) | — | 가능 |
| Query Rewriting | zero-shot(원문 few-shot), 단일 질의(원문 복수) | **중간** | 조건부 |
| Re-ranking (BGE) | **수정 완료** (2026-08-03) | — | 가능 |

---

# 1. HyDE

## 1.1 원 논문이 말하는 것

Gao et al., *Precise Zero-Shot Dense Retrieval without Relevance Labels*
(arXiv:2212.10496). 공식 구현: `github.com/texttron/hyde`.

**핵심 알고리즘** (`src/hyde/hyde.py`):

```python
for c in [query] + hypothesis_documents:
    c_emb = self.encoder.encode(c)
    all_emb_c.append(np.array(c_emb))
avg_emb_c = np.mean(all_emb_c, axis=0)
hyde_vector = avg_emb_c.reshape((1, len(avg_emb_c)))
```

세 가지가 핵심이다.

- **가상 문서를 여러 개 생성한다.** 논문 기본값 N=8.
- **원본 질의도 함께 인코딩한다.** `[query] + hypothesis_documents`
- **N+1개 임베딩의 평균을 검색 벡터로 쓴다.** 원본 질의가 앵커 역할을 하고,
  여러 생성의 평균이 개별 생성의 오류를 희석한다.

**SciFact 전용 프롬프트** (`src/hyde/promptor.py`):

```
Please write a scientific paper passage to support/refute the claim.
Claim: {}
Passage:
```

"scientific paper passage"라는 **문서 장르 지정**이 핵심이다. 임베딩 공간에서
논문 초록이 모여 있는 영역을 겨냥하기 위한 것이다.

## 1.2 우리 코드가 하는 것

`src/hyde_transformer.cpp`:

```cpp
std::string buildHydePrompt(const std::string& query) {
  oss << "[Task]\n";
  oss << "Write a short hypothetical document that would be useful for retrieving evidence for the query.\n";
  oss << "Preserve the query meaning and important entities.\n";
  oss << "Do not answer with uncertainty markers or explanations.\n";
  oss << "Return only the hypothetical document.\n\n";
  oss << "[Query]\n" << query << "\n\n";
  oss << "[Hypothetical Document]\n";
}
```

`src/rag_pipeline.cpp`:

```cpp
const HydeTransformResult hyde = sanitizeHydeDocument(hyde_trace.output_text, query);
retrieval_query = hyde.document;          // ← 원본 질의를 '대체'
...
contexts = retriever_->retrieve(retrieval_query, retrieve_k);
```

## 1.3 무엇이 다른가

**(1) 가상 문서가 1개다.** 원문은 8개를 만들어 평균한다. 1개만 쓰면 그 생성이
빗나갔을 때 복구 수단이 없다. 논문이 평균을 쓰는 이유가 정확히 이 분산을 줄이기
위해서다. 270M 모델의 생성 품질을 생각하면 분산은 더 크다.

**(2) 원본 질의를 버린다.** 이게 가장 큰 차이다. 원문은 `[query] + docs`를 평균해
원본 질의가 검색 벡터를 붙잡아 준다. 우리는 가상 문서로 **교체**하므로, 생성이
엉뚱하면 검색이 통째로 엉뚱한 곳을 본다. 알고리즘이 다르다고 봐야 한다.

**(3) 태스크 프롬프트가 없다.** 원문은 SciFact에 "scientific paper passage"를
명시한다. 우리 문구는 "hypothetical document"라 장르 신호가 없다.

## 1.4 수정 결과 (2026-08-03, `feature/hyde-embedding-average`)

원문과 일치하도록 구현했다. 대조 항목별 확인 결과는 이렇다.

| 원문 사양 | 확인 |
|---|---|
| `[query] + hypothesis_documents`를 각각 인코딩 | `meanEmbedding()`이 질의와 문서를 개별 `embed()` |
| 가중치 없는 산술 평균 | 합을 `documents.size() + 1`로 나눔 |
| 평균 벡터로 검색 | `Retriever::retrieveByVector()` 신설 |
| n=8 | `hyde.num_documents: 8` |
| temperature 0.7 | `hyde.temperature: 0.7` |
| max_tokens 512 | `hyde.max_new_tokens: 512` |
| SciFact 태스크 프롬프트 | `configs/prompt_hyde_scifact.txt`에 원문 그대로 |

**seed 처리는 원문에 없는 우리 쪽 결정이다.** 원문은 OpenAI API의 `n=8`로 한 번에
8개를 뽑으므로 seed 개념이 없다. 우리는 재현성을 위해 seed를 고정하는데, 그러면
8회 생성이 전부 동일해져 평균이 N=1로 붕괴한다. 그래서 draw마다
`config.seed + draw`를 쓴다. **모델 설정의 seed가 여전히 run을 지배하면서 한
질의 안의 draw만 달라진다.** 실측으로 8개 문서가 모두 다르고, 같은 요청을 두 번
보내면 문서 해시와 검색 결과가 동일하다.

**임베딩 정규화는 원문대로 두었다.** 우리 `embed()`는 L2 정규화하고 인덱스는
`IndexFlatIP`라 내적이 코사인이다. 정규화된 벡터 N+1개의 평균은 노름이 1보다
작아지지만, 내적 검색에서 질의 벡터의 스칼라 배는 순위를 바꾸지 않으므로 결과는
같다. 원문도 재정규화하지 않는다.

**다만 보고 시 주의가 필요하다.** HyDE의 `score` 값은 평균 벡터의 노름이 1보다
작은 만큼 다른 모드보다 작게 나온다. **순위는 정상이지만 점수 절댓값을 모드 간
비교하면 안 된다.**

**비용 실측.** gemma-3-270m 기준 쿼리당 57초(생성 8회, 보조 출력 1,826토큰).
300문항이면 약 4.75시간이다. `none`이 쿼리당 4초 수준이니 **14배**다. 이것이
원문대로 구현한 HyDE의 온디바이스 비용이며, 논문에 그대로 보고할 값이다.

## 1.5 원래 계획했던 수정 방법 (참고)

**변경 지점 A — 임베딩 평균 경로 (필수).**

현재 `Retriever::retrieve(text, k)`는 문자열만 받아 내부에서 임베딩한다.
벡터를 직접 받는 오버로드가 필요하다.

- `include/retriever.hpp`에 `std::vector<RetrievedChunk> retrieveByVector(const std::vector<float>& vec, int k) const;` 추가
- `include/embedding_model.hpp`의 인코딩 함수를 파이프라인에서 쓸 수 있게 노출
- `src/rag_pipeline.cpp`의 HyDE 분기를 아래로 교체

```
1. 가상 문서 N개 생성 (N은 설정값)
2. 원본 질의 + 가상 문서 N개를 각각 임베딩  → N+1개 벡터
3. 요소별 평균 → 검색 벡터 1개
4. retrieveByVector(평균벡터, k)
```

**변경 지점 B — N을 설정 가능하게.**

`configs/retrieval_*.yaml`에 `hyde_num_documents` 추가(기본 8). Pi 비용을 감안해
N을 줄여야 한다면 **줄인 값을 논문에 명시**해야 한다. 무단으로 1을 쓰는 것과
"비용 때문에 N=4로 낮췄다"고 밝히는 것은 완전히 다르다.

**변경 지점 C — 태스크별 프롬프트.**

`buildHydePrompt`가 데이터셋을 인자로 받아 분기하거나, 프롬프트 파일을
`configs/prompt_hyde_scifact.txt` 형태로 외부화한다. SciFact 문구는 원문 그대로
쓴다.

**주의.** N=8이면 쿼리당 생성이 8회다. Qwen3-0.6B 실측으로 생성 1회가 약 11초이니
쿼리당 88초, 300문항이면 7시간이다. **비용이 커서 N을 낮춘다면 그 자체가
"온디바이스에서는 HyDE의 원형을 감당할 수 없다"는 결과이며, 논문에 그렇게 써야
한다.** 몰래 N=1로 두고 "효과 없음"이라고 쓰는 것과는 정반대의 서술이다.

---

# 2. Sentence Window Retrieval

## 2.1 원 논문이 말하는 것

LlamaIndex `SentenceWindowNodeParser` + `MetadataReplacementPostProcessor`.

동작은 두 단계로 나뉜다.

**색인 시점** — 문서를 **문장 하나당 노드 하나**로 자른다. 각 노드는 자기 문장만
임베딩하고, 주변 문장 윈도우는 **메타데이터**에 담아둔다.

**검색 시점** — 문장 단위로 검색한 뒤, LLM에 넘기기 직전에
`MetadataReplacementPostProcessor`가 각 문장을 그 주변 윈도우로 **치환**한다.

기본 `window_size`는 3(양옆 각각 3문장, 총 7문장).

**이 기법의 핵심은 "검색은 문장 단위로 정밀하게, 컨텍스트는 넓게"다.** 정밀한
검색이 절반이고 넓은 컨텍스트가 나머지 절반이다.

## 2.2 우리 코드가 하는 것

`configs/retrieval_scifact_bge_rerank.yaml`:

```yaml
chunk_size: 256
chunk_overlap: 32
sentence_window_size: 1
```

`src/sentence_window.cpp`:

```cpp
for (const auto& context : contexts) {
    const int selected = selectRepresentativeSentence(query, context, sentences);
    const int begin = std::max(0, selected - safe_window_size);
    const int end = selected + safe_window_size + 1;
    ...
    out.chunk.text = joinSentences(window);
}
```

**256토큰 청크로 색인하고 검색한 뒤**, 그 청크 안에서 대표 문장 하나를 고르고
그 주변 ±1문장으로 확장한다.

## 2.3 무엇이 다른가

**(1) 문장 단위 색인이 없다.** 이게 결정적이다. 원문은 임베딩 대상이 문장 하나라
검색 자체가 정밀해진다. 우리는 256토큰 청크를 임베딩하므로 **검색 정밀도 향상이
전혀 없다.** 기법의 절반이 구현되지 않은 것이다.

우리 버전은 "검색된 청크를 문장 윈도우로 다시 자르는 것"에 가깝다. 오히려
컨텍스트가 **줄어드는** 경우도 있다 — 256토큰 청크가 ±1문장(총 3문장)으로 좁아질
수 있다. 원문은 넓히는 기법인데 우리는 좁힐 수 있다.

**(2) 윈도우 크기가 1이다.** 원문 기본값은 3이다. 총 3문장 대 총 7문장이다.

**(3) 대표 문장 선택이라는 단계가 원문에 없다.** 원문은 검색된 노드가 이미
문장이므로 고를 필요가 없다. 우리는 청크에서 골라야 하고, 그 선택 로직
(`selectRepresentativeSentence`)이 잘못 고르면 엉뚱한 윈도우가 나온다. **원문에
없는 실패 지점을 우리가 추가한 것이다.**

## 2.4 어떻게 고치는가

**변경 지점 A — 문장 단위 색인 (필수).**

`sentence_window` 모드 전용 인덱스를 따로 만든다. 같은 인덱스를 다른 모드와
공유할 수 없다. 청크 단위와 문장 단위는 색인 자체가 다르기 때문이다.

- `configs/retrieval_scifact_sentence_window.yaml`을 새로 만들고
  `index_path`/`chunk_store_path`를 별도 경로로 지정
- `src/text_chunker.cpp`에 문장 단위 분할 경로 추가 (이미
  `chunkDocumentsWithSentences`가 문장 메타데이터를 만들고 있으니 이를 활용)
- `rag_ingest`로 문장 인덱스를 별도 생성

**변경 지점 B — 윈도우 크기.**

`sentence_window_size: 1` → `3`. 원문 기본값에 맞춘다.

**변경 지점 C — 대표 문장 선택 제거.**

문장 단위 색인이 되면 `selectRepresentativeSentence`가 불필요해진다. 검색 결과가
이미 문장이므로 그 문장의 인덱스를 그대로 윈도우 중심으로 쓴다.

**주의.** 인덱스가 모드마다 달라지면 **BEIR 검색 지표를 다른 모드와 직접 비교할
수 없다.** 문장 단위 검색은 doc_id 단위 재현율이 달라진다. 논문에는 이 모드의
검색 지표를 별도 표로 내거나, 문장→문서 매핑 후 비교한다는 점을 명시해야 한다.

---

# 3. Contextual Compression

## 3.1 원 논문이 말하는 것

LangChain `LLMChainExtractor`. 프롬프트 원문:

```
Given the following question and context, extract any part of the context *AS IS*
that is relevant to answer the question. If none of the context is relevant return
NO_OUTPUT. Remember, *DO NOT* edit the extracted parts of the context.

> Question: {question}
> Context:
>>>
{context}
>>>
Extracted relevant parts:
```

**호출 단위가 문서 하나다.** 검색된 N개 문서를 순회하면서 각각에 대해 위 프롬프트를
한 번씩 실행한다. 결과는 N개의 (압축되었거나 비어 있는) 문서다.

세 가지 제약이 프롬프트에 박혀 있다.

- `*AS IS*` — 원문 그대로 발췌하라
- `*DO NOT* edit` — 편집 금지
- `NO_OUTPUT` — 관련 없으면 명시적으로 빈 결과

## 3.2 우리 코드가 하는 것

`src/contextual_compressor.cpp`:

```cpp
oss << "[Task]\n";
oss << "Extract only the context statements that help answer the query.\n";
oss << "Do not add facts that are not present in the contexts.\n";
oss << "Return a concise compressed context, not an answer.\n\n";
oss << "[Query]\n" << query << "\n\n";
oss << "[Contexts]\n";
for (size_t i = 0; i < contexts.size(); ++i) {
  oss << "Context " << (i + 1) << ":\n" << contexts[i].chunk.text << "\n\n";
}
oss << "[Compressed Context]\n";
```

**N개 문서를 한 프롬프트에 모두 넣고 한 번 호출**해서, 결과를 doc_id가
`"compressed"`인 **합성 청크 하나**로 만든다.

## 3.3 무엇이 다른가

**(1) 호출 단위가 다르다.** 원문은 문서마다 한 번(N회 호출), 우리는 전체를 한 번에
(1회 호출). 우리 방식은 모델에게 **다문서 종합**을 요구한다. 원문은 단일 문서
발췌만 요구한다. 작은 모델에게는 난이도 차이가 크다.

**(2) 발췌가 아니라 요약을 요구한다.** `*AS IS*`와 `*DO NOT* edit`이 없고,
대신 "concise compressed context"를 요구한다. 요약은 새 문장을 만들 여지를
주므로 **환각이 들어올 수 있다.** 원문은 그걸 막으려고 발췌를 강제한다.

**(3) `NO_OUTPUT` 규약이 없다.** 관련 문서가 하나도 없을 때 모델이 무엇을 해야
하는지 정의되지 않았다. 원문은 명시적 신호를 요구한다.

**(4) 문서별 provenance가 사라진다.** 합성 청크 하나가 되므로 어느 문서에서 온
내용인지 추적이 끊긴다. 코드가 `pre_transform_contexts`로 검색 출처를 따로
보존하고 있어 BEIR 지표는 살아 있지만, **생성기가 본 컨텍스트와 검색 결과의
대응 관계는 잃는다.**

## 3.4 수정 결과 (2026-08-03)

문서별 루프로 바꾸고 프롬프트를 LangChain 원문에 맞췄다(`*AS IS*`,
`*DO NOT* edit`, `NO_OUTPUT`). doc_id가 보존되므로 `pre_transform_contexts`
우회가 필요 없어졌다.

**NO_OUTPUT은 대소문자를 무시해 인식한다.** 원문 구현은 정확히 일치를 보지만,
Pi 실측에서 gemma-3-270m이 `No_OUTPUT`이라고 답했고 정확 일치로는 그 토큰 자체가
발췌 근거로 프롬프트에 들어갔다. **작은 모델이 규약을 정확히 지키지 못하는 것에
대한 방어이며, 원문과 다른 지점이므로 논문에 명시한다.**

호출이 1회에서 top_k회(현재 3)로 늘어난다. 원문대로 구현했을 때의 실제 비용이다.

## 3.5 원래 계획했던 수정 방법 (참고)

**변경 지점 A — 문서별 루프 (필수).**

`src/rag_pipeline.cpp`의 `ContextualCompression` 분기를 바꾼다.

```
현재: buildContextualCompressionPrompt(query, contexts) 1회 호출 → 합성 청크 1개
수정: for each context in contexts:
          buildContextualCompressionPrompt(query, context) 1회 호출
          결과가 NO_OUTPUT이면 그 문서를 버린다
          아니면 그 문서의 doc_id를 유지한 채 텍스트만 교체
```

이러면 `pre_transform_contexts` 우회가 필요 없어진다. doc_id가 보존되므로
BEIR 지표가 자연스럽게 계산된다.

**변경 지점 B — 프롬프트를 원문에 맞춘다.**

`buildContextualCompressionPrompt`를 단일 문서용으로 바꾸고 `*AS IS*`,
`*DO NOT* edit`, `NO_OUTPUT`을 포함한다.

**변경 지점 C — NO_OUTPUT 처리.**

`buildCompressedContexts`가 `NO_OUTPUT`을 인식해 해당 문서를 결과에서 제외해야
한다. 모든 문서가 NO_OUTPUT이면 컨텍스트가 빈 상태로 생성에 들어간다. 이건 정상
동작이며, `technique_effect`는 `applied`로 기록되어야 한다.

**주의.** 호출이 1회에서 N회(=top_k, 현재 3)로 늘어난다. 레이턴시가 약 3배가 된다.
이것도 **원문대로 구현했을 때의 실제 비용**이므로 논문에 그대로 보고할 값이다.

---

# 4. Query Rewriting

## 4.1 원 논문이 말하는 것

Ma et al., *Query Rewriting for Retrieval-Augmented Large Language Models*
(EMNLP 2023, arXiv:2305.14283). Table 1의 프롬프트 원문:

```
Prompts for LLM as a frozen rewriter

Open-domain QA:
Think step by step to answer this question, and provide search engine queries for
knowledge that you need. Split the queries with ';' and end the queries with '**'.
{demonstration} Question: {x} Answer:

Multiple choice QA:
Provide a better search query for web search engine to answer the given question,
end the queries with '**'. {demonstration} Question: {x} Answer:
```

세 가지가 눈에 띈다.

- **few-shot이다.** `{demonstration}` 자리에 예시가 들어간다. zero-shot이 아니다.
- **질의를 여러 개 낼 수 있다.** "Split the queries with ';'" — 하나로 제한하지 않는다.
- **웹 검색 엔진 질의를 만드는 태스크다.** 논문 실험은 Bing API 기반 open-domain QA다.

## 4.2 우리 코드가 하는 것

`src/advanced_query_transformer.cpp`:

```cpp
oss << "[Task]\n";
oss << "Rewrite the query for semantic document retrieval.\n";
oss << "Preserve the original meaning and entities.\n";
oss << "Return only one rewritten query, with no explanation.\n\n";
oss << "[Query]\n" << query << "\n\n";
oss << "[Rewritten Query]\n";
```

```cpp
std::string rewritten = firstNonEmptyLine(generated_text);   // 첫 줄만
```

## 4.3 무엇이 다른가

**(1) zero-shot이다.** 원문은 few-shot 예시를 준다. 다만 우리 앞선 실험에서
**few-shot 예시를 주면 270M 모델이 예시를 글자 그대로 복사**하는 것이 확인됐다
(`generation_quality_2026-08-02.md` 참조). 그래서 zero-shot으로 둔 데는 이유가
있지만, **그 이유를 논문에 적어야 한다.** 지금은 아무 설명 없이 다르다.

**(2) 질의 하나만 쓴다.** 원문은 `;`로 구분된 복수 질의를 허용한다. 우리는
`firstNonEmptyLine`으로 첫 줄만 취하고, `;`를 분리하지도 않는다. 복수 질의는
검색 결과를 합집합으로 모으는 것이 자연스러운데 그 경로가 없다.

**(3) 태스크가 다르다.** 원문은 웹 검색 질의 생성이고 우리는 밀집 검색 질의
재작성이다. "Provide a better search query for web search engine"과
"Rewrite the query for semantic document retrieval"은 겨냥이 다르다. 이건
데이터셋이 다르니 불가피한 측면이 있으나 명시해야 한다.

## 4.4 어떻게 고치는가

**변경 지점 A — 복수 질의 지원.**

`sanitizeQueryRewrite`가 `;` 분리를 지원하고, 파이프라인이 각 질의로 검색해
결과를 합집합으로 모은 뒤 중복을 제거한다. 원문과 같은 구조가 된다.

**변경 지점 B — few-shot 여부를 실험 변수로.**

zero-shot과 few-shot 양쪽을 돌려 비교한다. 예시 복사가 관측되면 그 자체가
**"few-shot이 SLM에서는 역효과"**라는 결과이며, 원문과 다르게 간 근거가 된다.

**변경 지점 C — 문서화.**

원문과 다른 두 지점(zero-shot, 단일 질의)을 논문 실험 설정 절에 명시한다.

**참고.** 이 기법은 원문 자체가 **웹 검색 + open-domain QA** 전제라, SciFact 같은
완결된 과학 클레임에는 적용점이 적을 수 있다. 실측에서도 재작성문이 문법 교정과
대문자화 수준에 그쳤고 MRR 순효과는 −0.024였다. 이건 구현 결함이 아니라
**태스크 부적합**일 가능성이 크며, 그렇게 서술하는 편이 정확하다.

---

# 5. Re-ranking (BGE cross-encoder)

## 5.1 표준이 말하는 것

BAAI `bge-reranker-base`는 크로스인코더다. `(query, passage)` 쌍을 하나의
시퀀스로 인코딩해 관련도 로짓 하나를 낸다. 백본은 XLM-RoBERTa다.

표준 사용법은 후보 k개를 받아 각각 점수를 매기고 상위 n개를 남기는 것이다.

## 5.2 우리 코드가 하는 것

`src/rag_pipeline.cpp`:

```cpp
const int retrieve_k = advanced == AdvancedMode::Rerank ? rerank_candidate_k_ : top_k_;
contexts = retriever_->retrieve(retrieval_query, retrieve_k);   // 9개
...
contexts = bge_reranker_->rerank(query, contexts, top_k_);      // → 3개
```

후보 9개를 검색해 3개로 줄인다. 구조는 표준과 같다. 원본 질의(`query`)로
재정렬하는 것도 맞다.

## 5.3 무엇이 다른가

**구조적 문제는 없다.** 다만 한 곳이 위험하다.

`src/bge_reranker.cpp`:

```cpp
for (const auto& name : input_names_) {
  if (name == "input_ids") {
    input_tensors.push_back(createTensor(input_ids));
  } else if (name == "attention_mask") {
    input_tensors.push_back(createTensor(attention_mask));
  } else if (name == "token_type_ids") {
    input_tensors.push_back(createTensor(token_type_ids));
  } else {
    input_tensors.push_back(createTensor(token_type_ids));   // ← 알 수 없는 입력에 token_type_ids
  }
}
```

**모르는 입력 이름이 오면 조용히 `token_type_ids`를 넣는다.** ONNX 모델을 다른
것으로 바꿨을 때 잘못된 텐서가 들어가도 예외 없이 실행되고, 점수만 조용히
틀려진다. 재정렬 결과가 사실상 무작위가 되어도 아무 로그가 남지 않는다.

또한 XLM-RoBERTa는 `token_type_ids`를 쓰지 않는다. 현재 모델에서는 무해하지만
폴백 자체가 위험한 설계다.

## 5.4 수정 결과 (2026-08-03)

알 수 없는 ONNX 입력 이름은 예외를 던진다. 조용히 `token_type_ids`를 넣어
점수만 틀려지는 경로를 없앴다.

## 5.5 원래 계획했던 수정 방법 (참고)

**변경 지점 — 알 수 없는 입력은 예외.**

```cpp
} else {
  throw std::runtime_error("Unexpected reranker input: " + name);
}
```

조용히 틀리는 것보다 시작 시점에 죽는 편이 낫다. 특히 이 기법은 **생성기와
무관해서 양성 대조로 쓸 후보**이므로, 여기가 조용히 망가지면 대조 자체가 무너진다.

---

# 6. 작업 순서 제안

의존 관계와 비용을 함께 고려한 순서다.

**1단계 — Re-ranking 폴백 수정 (30분).**
가장 싸고, 이걸 양성 대조로 쓰려면 먼저 신뢰할 수 있어야 한다.

**2단계 — Contextual Compression 원문화 (반나절).**
문서별 루프 + 프롬프트 교체 + NO_OUTPUT. 새 인덱스가 필요 없어 가장 단순하다.

**3단계 — HyDE 임베딩 평균 (1~2일).**
`Retriever`에 벡터 검색 인터페이스를 추가해야 해서 가장 침습적이다. N은
설정값으로 두고, 비용 때문에 낮춘다면 논문에 명시한다.

**4단계 — Sentence Window 문장 색인 (1~2일).**
별도 인덱스를 만들어야 하고, 그 결과 BEIR 지표 비교 방식도 정해야 한다.

**5단계 — Query Rewriting 복수 질의 (반나절).**
효과가 가장 불확실하다. 태스크 부적합일 가능성이 커서 우선순위가 낮다.

**모든 수정 후 재측정이 필요하다.** 지금까지 쌓인 결과(no-op률, MRR 변화 등)는
수정 전 구현에 대한 것이므로, 논문 수치로 쓸 수 없다. 다만 **수정 전후 비교
자체가 "구현 충실도가 결과를 얼마나 바꾸는가"라는 부수적 기여**가 될 수 있다.

## 참고 문헌

- HyDE: Gao et al., arXiv:2212.10496 · 구현 `github.com/texttron/hyde`
- Query Rewriting: Ma et al., EMNLP 2023, arXiv:2305.14283
- Contextual Compression: LangChain `LLMChainExtractor`
- Sentence Window: LlamaIndex `SentenceWindowNodeParser` +
  `MetadataReplacementPostProcessor`
- Re-ranking: BAAI `bge-reranker-base`
