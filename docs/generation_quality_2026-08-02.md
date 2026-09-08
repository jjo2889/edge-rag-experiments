# 생성 품질 감사 — 2026-08-02

첫 번째 본 실험을 14시간 만에 중단했다. 12개 모드/데이터셋 조합 중 둘이 완주한
시점이다. 왜 중단했고, 무엇을 고쳤고, 무엇이 미해결로 남았는지 기록한다.

## 그 run이 실제로 만들어낸 것

`none.json`과 `query_rewrite.json`(각 SciFact test 클레임 300건)의 `response`
필드를 직접 읽어 결함을 센 결과다.

| 결함 | none | query_rewrite |
|---|---|---|
| 클레임을 그대로 복사, 판정 없음 | 52 (17%) | 55 (18%) |
| 시스템 프롬프트를 답변으로 출력 | 30 (10%) | 29 (10%) |
| 구조 마커 누수 (`[Conclusion]`, `Evidence 4:`) | 84 (28%) | 85 (28%) |
| 빈 응답 | 0 | 0 |

두 모드의 점수는 구분되지 않는다.

| 지표 | none | query_rewrite |
|---|---|---|
| faithfulness | 0.350 | 0.330 |
| answer_relevancy | 0.563 | 0.565 |
| context_precision | 0.592 | 0.605 |
| context_recall | 0.629 | 0.625 |
| BEIR nDCG@10 | 0.597 | 0.599 |

검색은 정상이다. 생성기가 무엇을 검색해 왔든 같은 퇴화 출력을 내는 한, 논문이
필요로 하는 advanced RAG 모드 간 분리는 나타날 수 없다.

## 수정함: chat template 없이 raw completion으로 넣고 있었다

`use_model_chat_template`이 꺼져 있어서, instruction tuned 모델인
gemma-3-270m-it이 프롬프트를 raw completion으로 받았다. 지시를 따르는 대신
프롬프트를 이어썼다. 동일한 SciFact 클레임 30건 전후 비교다.

| | 수정 전 | 수정 후 |
|---|---|---|
| 시스템 프롬프트를 그대로 출력 | 6 | 1 |
| 클레임 그대로 복사 | 6 | 1 |
| 구조 마커 누수 | 6 | 0 |
| 128토큰 상한까지 폭주 | 3 | 0 |
| 빈 응답 | 0 | 0 |
| 출력 토큰 중앙값 | 29.5 | 21.5 |

llama.cpp의 Gemma template이 system role을 첫 user 턴에 알아서 병합해주므로
코드 수정은 필요 없었다. 이 플래그를 켜면 즉시 EOG로 빈 응답이 나온다던 기존
config 주석의 경고는 재현되지 않았다.

## 수정함: stop string이 답변 첫머리를 보지 못했다

모든 stop string이 선행 개행을 요구한다. 같은 단어가 문장 중간에 나왔을 때 잘리지
않게 하려는 장치인데, 그 때문에 답변이 구조 마커로 *시작*하면 `Evidence 4:`가
아무것도 매칭하지 못하고 `max_new_tokens`까지 폭주했다. `trimAtStopString`이
위치 0도 매칭하도록 하되, `I don't know`로 시작하는 정당한 답변이 살아남도록
섹션·항목 마커에만 적용한다.

## 미해결: 모델이 claim verification을 수행하지 못한다

프롬프트 엔지니어링으로는 구제되지 않는다. chat template을 켠 상태에서 SciFact
test split으로 네 가지 변형을 측정했다.

| 프롬프트 | 단일 라벨 출력 | 비고 |
|---|---|---|
| 현행 짧은 프롬프트 | 0/12 | 클레임을 재진술함 |
| 아카이브된 verbose 라벨 프롬프트 | 3/12 | 누수는 사라졌으나 여전히 재진술 위주 |
| 라벨 + few-shot 예시 | 8/12 | **예시의 이유 문장을 글자 그대로 복사** |
| 한 단어, 예시 없음 | 19/20 | 형식은 해결, 아래 참조 |

few-shot 변형은 읽어보기 전까지 승자처럼 보인다. 예시 문장 "The trial reported a
significant reduction in the vitamin group"이 무관한 클레임 8건에 그대로 돌아왔다.
프롬프트 안의 구체적 텍스트는 무조건 복사된다고 보면 된다.

한 단어 변형은 형식을 해결하면서 진짜 문제를 드러낸다. 대조 둘이다.

- **라벨 순서.** `SUPPORTS, REFUTES, NOT_ENOUGH_INFO` 순으로 제시하면 12/12가
  SUPPORTS. `REFUTES, NOT_ENOUGH_INFO, SUPPORTS`로 뒤집으면 같은 12건이 REFUTES.
  먼저 본 라벨을 그대로 뱉는다.
- **부정문 쌍.** 클레임과 그 부정문("0-dimensional biomaterials show / lack
  inductive properties")에 시험한 세 쌍 모두 동일 라벨.

정확도는 5/20으로, 이 표본의 gold 분포에서 항상 SUPPORTS만 찍었을 때와 같다.
상수 분류기이며 근거를 전혀 읽지 않는다.

270M 파라미터의 역량 한계이지 프롬프트 버그가 아니다. 해결은 연구적 판단이다 —
더 큰 생성기를 쓰거나, 생성 주장이 모델의 근거 추론에 의존하지 않도록 논문
프레이밍을 바꾸거나.

## Pi에서 측정한 생성기 후보들 (2026-08-03)

파라미터 총량은 오해를 부른다. 중요한 건 그중 얼마가 룩업 테이블이 아니라
트랜스포머인가다.

| | gemma-3-270m-it | SmolLM2-360M | Qwen3-0.6B | Qwen3-1.7B |
|---|---|---|---|---|
| GGUF 파라미터 | 268.10M | 361.82M | 751.63M | 1.72B |
| 어휘 | 262,144 | 49,152 | 151,936 | 151,936 |
| n_embd / n_layer | 640 / 18 | 960 / 32 | 1024 / 28 | 2048 / 28 |
| 임베딩 비중 | **62.6%** | 13.0% | 41.4% | — |
| **트랜스포머 본체** | **100.3M** | **314.6M** | **440.4M** | — |
| 시험한 양자화 | Q8_0 | F16 | F16 | Q4_K_M |
| llama-bench pp512 | 616 t/s | 128 t/s | 91.9 t/s | 52.4 t/s |
| llama-bench tg128 | 38.5 t/s | 16.0 t/s | 9.09 t/s | 10.1 t/s |
| 서빙 model_load p50 | 1,480 ms | 650 ms | 952 ms | 3,123 ms |
| 서빙 generation p50 | 3,777 ms | 6,926 ms | 10,915 ms | 18,188 ms |

gemma-3-270m은 3분의 2가 262k 항목짜리 임베딩 테이블이고, 실제로 계산을 하는
파라미터는 약 100M뿐이다. SmolLM2-360M은 어휘가 49k라 명목상 비슷한 크기에서
3.1배 큰 본체를 갖는다.

대조 실험은 모델 간 비교가 되도록 같은 Yes/No 프롬프트로 맞췄다. 라벨 기반
프롬프트는 SmolLM2에 쓸 수 없다 — 지시와 무관하게 라벨을 하나도 내지 않고
"Yes"만 답한다.

| 대조 | gemma-3-270m | SmolLM2-360M | Qwen3-0.6B |
|---|---|---|---|
| 라벨 순서 뒤집기 (12건) | **12/12 뒤집힘** | 12Y→10Y (거의 불변) | 11Y→9N (**뒤집힘**) |
| 예측 분포 (20건) | 전부 SUPPORTS | 전부 Yes | 19 Yes / 1 No |

gemma-3-270m만이 모든 대조를 노골적으로 실패한다. 두 sub-1B 후보 중에서는
Qwen3-0.6B가 선택지 순서에 *더* 민감하다. 어느 쪽도 그대로 쓸 수 없다.

Qwen3-1.7B(Q4_K_M)도 구제책이 못 된다. 부정문 쌍 0/3이다. 정확도 11/20은 gold
표본에 NOT_ENOUGH_INFO가 들어 있는 비율만큼 그 라벨을 찍어서 나온 것이지 근거를
읽어서 나온 값이 아니다.

**레이턴시는 병목이 아니다.** Qwen3-0.6B의 생성 중앙값 10.9초 기준으로 12개
모드/데이터셋 run 전체가 Pi 추론 약 12~14시간이며, 원래 계획과 같은 자릿수다.
병목은 역량이다.

### 부정문 쌍 대조에 대한 정정

이 문서의 이전 판은 2/3 결과를 근거 민감도로 읽었다. 그렇지 않다. 이유가 둘이다.

첫째, 그 대조에는 교란이 있다. **검색 질의가 곧 클레임**이므로, 부정형으로 바꾸면
검색되는 청크도 함께 바뀐다. 답이 달라진 것이 부정을 이해해서인지 다른 근거를
받아서인지 구분할 수 없다.

둘째, SciFact 안에 이미 부정문 쌍이 있고 Qwen3-0.6B는 그것을 틀린다. gold 라벨이
달린 클레임 12건(REFUTES 6건, SUPPORTS 6건)에 물었더니 **12건 전부** SUPPORTS(또는
Yes/No 프롬프트에서 "Yes")로 답했다.

    ALDH1 expression is associated with poorer prognosis in breast cancer.  [SUPPORTS] -> Yes
    ALDH1 expression is associated with better breast cancer outcomes.      [REFUTES]  -> Yes
    APOE4 ... tau phosphorylation causing GABA neuron degeneration.         [SUPPORTS] -> Yes
    APOE4 ... tau phosphorylation, delaying GABA neuron degeneration.       [REFUTES]  -> Yes

명시적 부정어를 넣어도 마찬가지다 — "ALDH1 expression is **not** associated with
poorer prognosis in breast cancer"도 Yes다. Qwen3-0.6B는 사실상 모든 클레임에
긍정으로 답하며, 20건 표본에서 나온 "No" 1건은 잡음이다.

**thinking 태그가 답변에 새어 나온다.** `/no_think`는 추론 블록을 비울 뿐 마커를
지우지 않는다. 모든 응답이 문자 그대로 `<think>\n\n</think>\n\nSUPPORTS`이며,
출력 7토큰 중 5토큰이 마크업이다. Qwen3 계열을 생성기로 쓰려면 RAGAS에 도달하기
전에 이것을 걷어내야 한다.

**레이턴시는 생성이 아니라 프롬프트에 묶여 있다.** 6~7토큰짜리 응답에 5~17초가
걸렸다. 프롬프트 처리가 91.9 t/s이므로 검색 컨텍스트 1,000토큰만으로 약 11초다.
레이턴시는 답변 길이가 아니라 컨텍스트 크기를 따라간다.

### 정확도 수치에 대한 단서

SciFact test 300건 중 `queries.jsonl` 메타데이터에 라벨이 있는 것은 188건이고,
나머지 112건을 NOT_ENOUGH_INFO로 채점했다. 이 문서의 이전 판은 그 매핑을 "아마
틀렸다"고 적었는데, 오히려 맞을 가능성이 높다 — SciFact의 NOINFO 클레임이 바로
근거 필드가 비어 있는 것들이며, 이는 문헌에도 언급된 데이터셋의 구조적 성질이다
(arXiv 2606.12854). 112건 모두 `qrels/test.tsv`에는 인용 문서를 갖고 있으므로,
검색에서 빠진 것이 아니라 라벨이 NOINFO인 것이다. BEIR 변환본이 아니라 Wadden
원본 배포본과 대조해 확정하는 작업은 여전히 해둘 가치가 있다.

진짜 단서는 다른 데 있다. 그 지름길이 존재하기 때문에, 파인튜닝된 모델은 추론이
아니라 **근거가 비었음을 탐지하는 것만으로** 높은 NEI F1을 얻을 수 있다. 우리
RAG 셋업에서는 top-k 검색이 항상 근거를 공급하므로 그 지름길이 없고, 따라서
NOINFO는 공개된 수치가 시사하는 것보다 여기서 더 어렵다.

어느 쪽이든 라벨 순서 뒤집기와 부정문 쌍 대조는 gold 라벨이 필요 없으므로 여전히
신뢰할 수 있는 근거다.

## 정리 작업

평가 저장소의 `reports/full-run/`을 `reports/archive-prechattemplate/`로 옮겼다.
`run_full_experiment.sh`가 `--skip-existing`을 넘기는데 이것이 `<mode>.json`의
존재 여부만 확인하므로, 수정 전 결과 파일 둘을 그대로 두면 다음 run이 그것들을
유효한 결과로 받아들였을 것이다.
