# Measurement plan

## The question this experiment answers

**Are Re-ranking, Sentence Window Retrieval, Contextual Compression, HyDE, and
Query Rewriting effective when the generator is a small language model running
on-device?**

Not "can an SLM reason over evidence" — that is a precondition, not the claim.
The claim is about the five techniques. Everything below exists to make that
claim measurable.

## Why the first attempt could not answer it

The 2026-08-01 run produced `none` and `query_rewrite` on SciFact before it was
stopped. Neither mode separated from the other on any metric. Three independent
reasons, none of which is "the technique does not work":

1. **The technique did not run.** In `query_rewrite`, the rewritten query was
   byte-identical to the original in **252 of 300 claims (84%)**. Those rows
   performed exactly the same retrieval as `none`. `transform_used_fallback`
   counted 0, because producing an unchanged query is not treated as a failure —
   so the run reported 300 successful rewrites while doing 48.

2. **The generator ignores its context.** Every candidate tested so far answers
   the same regardless of evidence (see `generation_quality_2026-08-02.md`).
   Three of the five techniques *are implemented by calling that generator*, so
   its weakness is not confined to the final answer.

3. **No baselines.** Without a floor and a ceiling there is no denominator, so a
   0.5965 -> 0.599 delta cannot be called small, large, or absent.

## Layer 0 — hygiene: did the technique execute?

Report before any quality metric. A technique that silently no-ops makes every
downstream number meaningless.

| technique | executed when | current status |
|---|---|---|
| Query Rewriting | rewritten query != original | **16%** (48/300) |
| HyDE | generated document is not a restatement of the query | unmeasured |
| Contextual Compression | compressed context != input context | unmeasured |
| Re-ranking | final top-k order != retrieval order | unmeasured |
| Sentence Window | expanded chunk != original chunk | unmeasured |

`sanitizeQueryRewrite` and its siblings must count an unchanged output as a
no-op, distinct from both success and fallback.

## Layer 1 — baselines: how wide is the measurable band?

All five techniques compete inside the gap between these two. If the gap is
zero, no technique can show an effect and the experiment cannot produce a
signal.

| condition | context given to the generator | reads as |
|---|---|---|
| **closed-book** | none | floor — what the model knows without retrieval |
| **retrieved** | live retrieval (current pipeline) | the system as built |
| **oracle** | gold evidence only | ceiling — perfect retrieval |

Effect sizes get reported as a fraction of `oracle - closed-book`, not as a raw
delta.

Both are cheap to build. `buildPrompt(query, contexts)` is the only place
context enters, and every record already carries `reference_contexts`
(300/300 on SciFact). closed-book passes an empty vector; oracle passes the
gold chunks.

Running closed-book first would have exposed the generator problem in an hour
instead of fourteen.

## Layer 2 — evaluate each technique where it actually operates

The five do not act at the same layer, so scoring them with one metric on one
pipeline conflates them.

**Retrieval-layer techniques** — they change *which* documents come back.
Measured under live retrieval with BEIR metrics, which need no generator.

- Query Rewriting
- HyDE

**Context-layer techniques** — they change *what the generator sees* from a
given candidate pool. Measured under controlled context, so retrieval variance
does not mask the effect:

| condition | pool composition |
|---|---|
| A | gold chunks only |
| B | gold + k distractors, position and count controlled |
| C | distractors only, no gold |
| D | no context (shared floor) |

- Re-ranking — does it lift gold above distractors in condition B
- Sentence Window
- Contextual Compression

Condition B is where the interesting question lives: how much context noise an
SLM tolerates, and which technique recovers it. That is a sharper question for
an on-device paper than a leaderboard of five modes.

`rerank_candidate_k` is already 9 and `top_k` 3, so the existing configuration
supplies the pool size.

## Layer 3 — metrics

**Build these. They do not exist yet and nothing else answers the question.**

- SciFact: parse the label out of `response`, score macro-F1 against gold.
  300 test claims: 188 carry SUPPORT/CONTRADICT metadata, the other 112 are
  NOINFO by the empty-evidence convention.
- NQ: Exact Match and token F1 against gold short answers. **The answers are
  absent from the BEIR conversion** — recovered 352/500 (70%) by matching
  question text against `nq_open`; the rest need the full train split.

**Keep.** BEIR precision/recall/nDCG/MAP/MRR. Exact, free, reproducible from
qrels.

**Drop from the main run.** RAGAS `context_precision` and `context_recall`
duplicate BEIR while being slower, paid, and judge-dependent:

| | BEIR | RAGAS |
|---|---|---|
| none | ndcg@3 0.598 | context_precision 0.592 |
| none | recall@3 0.661 | context_recall 0.629 |
| query_rewrite | ndcg@3 0.600 | context_precision 0.605 |
| query_rewrite | recall@3 0.665 | context_recall 0.625 |

**Demote.** RAGAS `faithfulness` and `answer_relevancy` measure the generator,
so they are uninformative until the generator responds to context at all. Worth
reporting later on a subsample, not in the main sweep.

RAGAS dominates cost: `none` took 4.5 h and `query_rewrite` 5.0 h for 300 rows,
almost entirely judging. Without it the full twelve-run sweep fits in under ten
hours instead of eighty.

## Open decisions

- **Generator.** Four candidates fail the evidence-sensitivity controls
  (gemma-3-270m, SmolLM2-360M, Qwen3-0.6B, Qwen3-1.7B). Layer 1 turns candidate
  selection into a measurement: a usable model is one where oracle separates
  from closed-book. Until one exists, the two context-layer techniques and both
  end-task metrics stay blocked. The retrieval layer does not.
- **NQ corpus.** 20,623 of 2,681,468 BEIR passages (0.77%), so retrieval is
  easier than the benchmark it cites. Either scale the distractor pool up,
  or state the subsample and drop cross-paper comparison.
- **Scope.** If the generator cannot be fixed within budget, the retrieval-layer
  result stands on its own and the context-layer techniques become stated
  limitations rather than silent gaps.

## Order of work

1. Layer 0 hygiene counters — cheapest, and it already found an 84% no-op.
2. Layer 1 closed-book and oracle paths — small change, decides whether any
   quality experiment is worth running.
3. Layer 3 metrics — SciFact macro-F1 first, NQ answer recovery second.
4. Layer 2 sweeps, once 1-3 say there is a band to measure inside.
