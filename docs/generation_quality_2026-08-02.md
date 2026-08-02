# Generation quality audit — 2026-08-02

The first full paper run was stopped 14 hours in, after two of the twelve
mode/dataset combinations had finished. This records why, what was fixed, and
what remains unresolved.

## What the run actually produced

Read directly from `none.json` and `query_rewrite.json` (300 SciFact test
claims each), counting defects in the `response` field:

| defect | none | query_rewrite |
|---|---|---|
| claim copied back verbatim, no verdict | 52 (17%) | 55 (18%) |
| system prompt emitted as the answer | 30 (10%) | 29 (10%) |
| section markers leaked (`[Conclusion]`, `Evidence 4:`) | 84 (28%) | 85 (28%) |
| empty replies | 0 | 0 |

Scores for the two modes are indistinguishable:

| metric | none | query_rewrite |
|---|---|---|
| faithfulness | 0.350 | 0.330 |
| answer_relevancy | 0.563 | 0.565 |
| context_precision | 0.592 | 0.605 |
| context_recall | 0.629 | 0.625 |
| BEIR nDCG@10 | 0.597 | 0.599 |

Retrieval is healthy. The separation the paper needs between advanced RAG
modes cannot appear while the generator emits the same degenerate output
regardless of what it retrieves.

## Fixed: raw completion instead of the chat template

`use_model_chat_template` was off, so gemma-3-270m-it — an instruction tuned
model — received the prompt as a raw completion and continued it rather than
following it. Over 30 SciFact claims, before and after enabling it:

| | before | after |
|---|---|---|
| system prompt echoed | 6 | 1 |
| claim copied verbatim | 6 | 1 |
| section markers leaked | 6 | 0 |
| ran to the 128-token cap | 3 | 0 |
| empty replies | 0 | 0 |
| median output tokens | 29.5 | 21.5 |

llama.cpp's Gemma template folds the system role into the first user turn on
its own, so no code change was needed for that. The config comment warning
that this flag caused immediate-EOG empty replies did not reproduce.

## Fixed: stop strings blind to the start of a reply

Every stop string carries a leading newline so the same word mid-sentence does
not trigger a cut. A reply that *opened* with `Evidence 4:` therefore matched
nothing and ran to `max_new_tokens`. `trimAtStopString` now also matches at
position 0, restricted to section and item markers so a reply that legitimately
opens with `I don't know` still survives.

## Not fixed: the model cannot do claim verification

Prompt engineering was tried and does not rescue this. Four variants were
measured on the SciFact test split with the chat template on:

| prompt | single label emitted | notes |
|---|---|---|
| current short prompt | 0/12 | restates the claim |
| archived verbose label prompt | 3/12 | leakage gone, still mostly restates |
| label + few-shot example | 8/12 | **copies the example's reason verbatim** |
| one word, no example | 19/20 | format solved, see below |

The few-shot variant looks like the winner until you read it: the example
sentence "The trial reported a significant reduction in the vitamin group"
came back unchanged for 8 unrelated claims. Any concrete text in the prompt
gets copied.

The one-word variant fixes the format and exposes the real problem. Two
controls:

- **Label order.** Listing `SUPPORTS, REFUTES, NOT_ENOUGH_INFO` produced
  SUPPORTS for 12/12 claims. Reordering to `REFUTES, NOT_ENOUGH_INFO,
  SUPPORTS` produced REFUTES for the same 12/12. The model emits whichever
  label it saw first.
- **Negation pairs.** Claims and their negations ("0-dimensional biomaterials
  show / lack inductive properties") received identical labels in all three
  pairs tested.

Accuracy was 5/20, matching what always-predicting-SUPPORTS scores against
this sample's gold distribution. The model is a constant classifier; it is not
reading the evidence at all.

This is a capability wall at 270M parameters, not a prompt bug. Resolving it is
a research decision — a larger generator, or a paper framing whose generation
claims do not depend on the model reasoning over retrieved evidence.

## Candidate generators measured on the Pi (2026-08-03)

The parameter count on the box is misleading. What matters is how much of it is
transformer rather than lookup table:

| | gemma-3-270m-it | SmolLM2-360M | Qwen3-0.6B | Qwen3-1.7B |
|---|---|---|---|---|
| GGUF params | 268.10M | 361.82M | 751.63M | 1.72B |
| vocab | 262,144 | 49,152 | 151,936 | 151,936 |
| n_embd / n_layer | 640 / 18 | 960 / 32 | 1024 / 28 | 2048 / 28 |
| embedding share | **62.6%** | 13.0% | 41.4% | — |
| **transformer body** | **100.3M** | **314.6M** | **440.4M** | — |
| quantization tested | Q8_0 | F16 | F16 | Q4_K_M |
| llama-bench pp512 | 616 t/s | 128 t/s | 91.9 t/s | 52.4 t/s |
| llama-bench tg128 | 38.5 t/s | 16.0 t/s | 9.09 t/s | 10.1 t/s |
| serving model_load p50 | 1,480 ms | 650 ms | 952 ms | 3,123 ms |
| serving generation p50 | 3,777 ms | 6,926 ms | 10,915 ms | 18,188 ms |

Two thirds of gemma-3-270m is its 262k-entry embedding table; only ~100M
parameters do any computation. SmolLM2-360M carries a 49k vocab and so fields a
3.1x larger transformer at a nominally similar size.

Controls, all on the same Yes/No prompt so the models are comparable (the
label-based prompt is not usable for SmolLM2 — it emits no label at all,
answering "Yes" regardless of instructions):

| control | gemma-3-270m | SmolLM2-360M | Qwen3-0.6B |
|---|---|---|---|
| label order flipped (12 claims) | **12/12 flip** | 12Y→10Y (stable) | 11Y→9N (**flips**) |
| negation pairs distinguished | **0/3** | 2/3 | 2/3 |
| prediction distribution (20) | all SUPPORTS | all Yes | 19 Yes / 1 No |

gemma-3-270m is the only one that fails every control outright. Both sub-1B
candidates distinguish 2 of 3 negation pairs — real evidence sensitivity that
gemma has none of — but both still answer Yes to essentially everything, and
Qwen3-0.6B is the *more* option-order dependent of the two. Neither is usable
as-is.

Qwen3-1.7B (Q4_K_M) does not rescue it either: 0/3 on negation pairs. Its 11/20
accuracy comes from predicting NOT_ENOUGH_INFO at roughly the rate the gold
sample contains it, not from reading evidence.

Latency is not the blocker. At Qwen3-0.6B's 10.9 s median generation, all twelve
mode/dataset runs come to roughly 12-14 hours of Pi inference, comparable to the
original plan. Capability is the blocker.

Caveat on accuracy figures: only 188 of the 300 SciFact test queries (63%) carry
a label in `queries.jsonl` metadata. The rest were scored as NOT_ENOUGH_INFO,
which is probably wrong. The order-flip and negation controls need no gold
labels and are the trustworthy evidence here.

## Housekeeping

`reports/full-run/` in the evaluator repo was moved to
`reports/archive-prechattemplate/`. `run_full_experiment.sh` passes
`--skip-existing`, which only tests whether `<mode>.json` exists, so leaving
the two pre-fix result files in place would have caused the next run to accept
them as valid.
