---
name: lejudge-experiments
description: Run, cache, analyse and report LeJudge experiments — judge-only and planning studies, ablations, bootstrap CIs, paired tests, Pareto curves, paraphrase heatmaps, and the `lejudge report` figures. Use this skill for anything under lejudge/eval/, artifacts/results/, paper/figures/, or when asked for numbers, tables, plots, statistics, p-values, confidence intervals, or "is the result significant" — even for a quick sanity check.
---

# Running and reporting LeJudge experiments

Every number in the paper comes from a Parquet row produced by a seeded, cached, traced run. Follow this order: pre-registration → run → results table → figure → sentence.

## Before running anything

1. `docs/PREREG.md` must exist and be hashed for the planning study. Do not change hypotheses or primary metrics after the hash.
2. `LEJUDGE_MODE=live` only for new samples; everything else `offline`.
3. Check the library, vocab and bank versions in `configs/` match what PREREG names.

## Commands

```
lejudge judge-study --env pusht --judges jev,keyword,llm-small --n 5000 --repeats 3 --out artifacts/results/judge_only.parquet
lejudge plan --env pusht --set edges --cond jev --episodes 100 --seeds 0-4 --lam 1.0 --out artifacts/results/planning.parquet
lejudge ablate --grid configs/ablations.yaml
lejudge report --offline --out paper/figures
```

## Statistics (do exactly this)

- Rates: 95 % bootstrap CI, 10,000 resamples, `numpy.random.default_rng(0)`.
- Condition A vs B: pair by (constraint set, seed, episode index); Wilcoxon signed-rank on per-episode outcomes; report risk difference with bootstrap CI; Holm correction across the three constraint sets.
- Judge accuracy: precision/recall/F1 at 0.5, AUROC, ECE with 10 equal-mass bins; reliability diagram.
- Consistency: per-question std-dev across repeats; report mean and max.
- Never report p without an effect size and CI. Never drop a seed.

## Figures (from `lejudge report`)

| Fig | Data | Notes |
|---|---|---|
| Success vs violation | `planning.parquet` | one point per condition per set, CI whiskers both axes |
| Pareto over λ | `planning.parquet` (jev, λ sweep) | line with CI band; mark λ = 1 |
| Paraphrase heatmap | `judge_only.parquet` | rows constraints, cols canonical + 5 paraphrases, one panel per judge |
| Latency/cost | `judge_only.parquet` | per 1,000 judgments; log scale |
| Reliability | `judge_only.parquet` | per judge |
| Error decomposition | probes meta + `judge_only` + `planning` | probe error, judge error on ground-truth words, world-model error (imagined vs executed) |

Style: matplotlib, colour-blind-safe palette, vector PDF, one message per figure, units in axis labels, caption states n and CI method.

## Writing results

- Lead with the number and its CI: "LeJudge reduced episode violations from 41 % [36, 46] to 9 % [6, 13] (oracle 5 %)".
- Say where the keyword baseline matches Jev; that honesty is a result.
- Every sentence with a number cites the table or figure.

## Common mistakes

- Comparing Jev's Noul threshold to a Choice probability.
- Mixing bank versions in one table.
- Running the LLM baseline with a longer prompt than Jev's state.
- Reporting imagined-state violations as if they were executed.
