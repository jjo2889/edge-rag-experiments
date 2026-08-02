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

## Housekeeping

`reports/full-run/` in the evaluator repo was moved to
`reports/archive-prechattemplate/`. `run_full_experiment.sh` passes
`--skip-existing`, which only tests whether `<mode>.json` exists, so leaving
the two pre-fix result files in place would have caused the next run to accept
them as valid.
