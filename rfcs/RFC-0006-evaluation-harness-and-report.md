# RFC-0006 — Evaluation harness, statistics and report

**Status** Implemented · **Scope** MVP · **Depends on** RFC-0003, RFC-0004

## Summary

Two studies (judge-only, planning), one results store, one report command. Pre-registered hypotheses and analysis; all numbers regenerable offline from cached responses.

## Pre-registration

`docs/PREREG.md` is written and hashed before the planning study runs. It lists H1–H4 (constraint compliance, success preservation, paraphrase generalisation, cost/latency vs LLM), the primary metrics, the exact statistical tests and the ablation grid. Changes after the freeze are logged, not silently applied.

## Judge-only study

| Input | Source |
|---|---|
| 5,000 imagined rollouts | CEM candidates recorded during unconstrained LeWM planning (RFC-0008 traces) |
| 5,000 executed rollouts | Episodes from unconstrained and oracle-constrained planning |
| Labels | Oracle checkers on ground truth (executed) and oracle-on-probes (imagined; flagged) |
| Judges | Jev, Keyword, LLM-small (full), LLM-reasoning (500-item subset) |
| Inputs per judge | Identical state and keys |

Outputs per judge: precision/recall/F1 at 0.5, AUROC, ECE, reliability plot, per-constraint accuracy; **paraphrase table**: accuracy per constraint on canonical text vs the 5 paraphrases; **negatives**: false-positive rate on near-miss sentences; consistency: std-dev over 3 repeats on 200 items; latency and cost per 1,000 judgments.

## Planning study

Grid: conditions {LeWM, oracle-on-probes, keyword, Jev, LLM-small} × constraint sets {edges, edges+upright, implicit} × seeds {0..4} × 100 episodes.

Per episode: success, violation flag, violation steps, planning time per step (p50/p95), Jev calls, tokens, USD, abstentions.

Extra runs (Jev only): λ sweep {0.25, 0.5, 1, 2, 4}; K ∈ {4, 16, 64}; `final_only`; vocab `pusht@2`; `steps=H/2`; hard-rejection variant.

Shift robustness: `edges+upright` on Jev vs LeWM under two stable-worldmodel factors of variation (colour, lighting), 50 episodes × 3 seeds.

## Statistics

- Rates with 95 % bootstrap CIs (10,000 resamples, seed fixed).
- Condition comparisons: paired per episode; Wilcoxon signed-rank; Holm across the three constraint sets; report risk differences with CIs.
- Effect sizes always shown next to p-values.
- Pareto curve: success vs violation over λ with CI bands.

## Results store

Parquet tables: `judge_only.parquet`, `planning.parquet`, `ablations.parquet`, `shift.parquet`, each row one judgment or episode with all identifiers (condition, constraint set, seed, bank version, model version, cache hit).

## Report

`lejudge report` produces:

| Figure/table | Content |
|---|---|
| Fig 1 | Architecture (static asset) |
| Fig 2 | Success vs violation per condition with CIs, three constraint sets |
| Fig 3 | Pareto curve over λ |
| Fig 4 | Paraphrase heatmap: judge × constraint × paraphrase accuracy |
| Fig 5 | Latency and cost per 1,000 judgments, log scale |
| Fig 6 | Reliability diagrams |
| Fig 7 | Error decomposition: probe error, judge error, world-model error |
| Table 1 | Judge-only metrics |
| Table 2 | Planning metrics |
| Table 3 | Ablations |

All figures are regenerated from the Parquet tables; no figure reads an API.

## Acceptance

- `make paper` runs offline and byte-reproduces figure data.
- Pre-registration hash matches at report time.
