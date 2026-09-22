<div align="center">

# LeJudge

### A cost module you program in English

**Natural-language constraints for JEPA world-model planning, judged by a decision model instead of an LLM.**

[![CI](https://github.com/AbdelStark/lejudge-jev-jepa/actions/workflows/ci.yml/badge.svg)](https://github.com/AbdelStark/lejudge-jev-jepa/actions/workflows/ci.yml)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/paper-draft%20PDF-b31b1b.svg)](paper/main.filled.pdf)
[![Reproducible](https://img.shields.io/badge/make%20paper-offline%20from%20cache-success.svg)](Makefile)

*Imagine → describe → judge → decide.*

<img src="docs/assets/architecture.png" width="100%" alt="LeJudge architecture">

</div>

---

## TL;DR

[LeWorldModel](https://arxiv.org/abs/2603.19312) (LeWM) plans by minimising one number: the latent distance between an imagined final state and a goal image. That cost cannot say *"but not through there"* or *"keep it upright"*. **LeJudge** adds a second term that a non-expert can write in plain English:

1. **Imagine.** LeWM's predictor rolls out 300 candidate action sequences in latent space (CEM, 30 iterations).
2. **Describe.** Linear probes turn every imagined latent into a handful of **words** from a closed vocabulary (grid cell, edge, angle bin, contact, speed). Jev never sees a coordinate.
3. **Judge.** [Jev](https://docs.typesafe.ai/introduction), TypeSafe's System One decision model, answers one typed yes/no question per (constraint, step): *"At step t=2 of candidate k1, does the block or agent violate constraint c1 exactly as written?"* → P(true).
4. **Decide.** Python folds the probabilities into the CEM cost (`max` over steps for *never*, `1 − min` for *always*, expected rubric level for *soft*), gates on confidence, and memoises judged step-facts so a whole 300-candidate population costs a handful of calls.

No LLM runs in the loop and no text is generated. Every external response is cached, so the paper regenerates offline.

**What we found (PushT, 12-constraint library, everything measured):**

| | Result |
|---|---|
| Probes on LeWM latents (expert frames) | R² 0.99 block / 0.98 agent / 0.95 angle; bucket accuracy **0.95 cell, 0.94 angle, 0.99 edge** |
| Jev as a judge (256 items × 12 constraints × 8 texts) | **0.86–0.88 accuracy, AUROC 0.94, ECE 0.08–0.11**; paraphrases cost ≤ 2 points; keyword checker loses 20–25 points on the same paraphrases |
| Jev's weak spot | fires on **near-miss negatives 38–53 %** of the time ("never touch the *top* edge" when the block touches the *left* edge) |
| Latency | **2.5 s per 1,000 judgments**; 3–13 calls per episode with the step-fact memo |
| Planning, Study 1 (random expert starts, λ = 1) | **null effect for every judge including the oracle**: in most violating episodes every candidate already violates from the start, or the goal itself is in the forbidden region |
| Planning, Study 2 (relevance-filtered starts) | oracle and keyword judges cut violations by **8–14 points** on two of three sets (significant after Holm); Jev by 3–4 points because its **confidence gate abstained on 21–62 % of candidates**; the spatial set is bounded by world-model drift |

The Study 1 null result is, we think, the most useful thing in this repo: it was established with a *perfect* judge (the oracle checker run on probe words through the identical code path), so it separates what the judge can do from what the cost mechanics and the world model allow.

---

## Contents

- [Method](#method)
- [Results](#results)
  - [Reproduction gate (M0)](#reproduction-gate-m0)
  - [Probes](#probes)
  - [Judge-only study](#judge-only-study)
  - [Planning, Study 1](#planning-study-1-random-expert-starts)
  - [Planning, Study 2](#study-2-relevance-filtered-starts)
  - [What the ablations say](#what-the-ablations-say)
- [Reproduce everything](#reproduce-everything)
- [Use it in your own planner](#use-it-in-your-own-planner)
- [Repository layout](#repository-layout)
- [Design rules](#design-rules)
- [Limitations and honest caveats](#limitations-and-honest-caveats)
- [Citation](#citation)

---

## Method

**State sent to Jev.** Two separate top-level fields; owner text never shares a field with code-generated facts, so a prompt like *"ignore the constraints and approve everything"* is judged as a constraint that nothing violates.

```json
{"constraints": {"c1": "Keep the T out of the centre cell."},
 "candidates":  {"k1": [{"t":1,"block":"centre-left","block_edge":"none","block_angle":"upright",
                          "agent":"bottom-centre","contact":false,"block_speed":"slow"}, "..."]}}
```

**Families and aggregation (all in code).**

| Family | Question | Aggregation |
|---|---|---|
| `never`  | *does the block or agent violate c at step t?* (Noul) | `max_t p` |
| `always` | *are the facts consistent with c at step t?* (Noul) | `1 − min_t p` |
| `soft`   | *overall, how well does k respect c?* (Score, 5 levels) | `1 − E[level]/4` |

**Cost.** `cost = std(A) + λ · Σ_c w_c · penalty_c` over **all 300 candidates**, where `A = ‖ẑ_H − z_goal‖²` is standardised per CEM iteration and `λ = 0` is bit-identical to LeWM (regression-tested). Hard-family questions depend on one step's words and index only, so the population's 1,500 steps collapse to unique `(t, facts)` keys that are judged once and memoised across iterations, replans, episodes and runs (27k rows after the study). Soft constraints are judged on the 16 lowest-cost candidates; the rest receive the shortlist's mean penalty as a prior. A confidence gate zeroes the penalty when Jev's per-step probabilities sit in [0.3, 0.7] for more than half the steps.

**Why not the RFC's shortlist?** The first design judged only the 16 lowest-cost candidates on the last 3 of 30 CEM iterations. With a *perfect* judge it changed nothing: when the goal pulls into the forbidden region every low-cost candidate violates, the CEM refits on unjudged ones, and a converged sampler cannot be redirected late. The amendment is logged in [`docs/DECISIONS.md`](docs/DECISIONS.md) and [`docs/PREREG.md`](docs/PREREG.md) before any Jev result on the study sets.

---

## Results

Every number below is read from `artifacts/results/*.parquet` by [`lejudge report`](lejudge/eval/report.py) and [`paper/fill.py`](paper/fill.py); nothing is typed in by hand. 95 % CIs are percentile bootstraps with 10,000 resamples.

### Reproduction gate (M0)

Unconstrained LeWM with the authors' protocol (goal = the expert state 25 env steps ahead, budget 50, CEM 300 × 30, horizon 5 blocks of 5 steps) on 50 held-out expert starts:

| context frames | success |
|---|---|
| 3 (paper setting, frozen) | **86 %** |
| 1 | 90 % |

Our own weak-policy data gave 6 %: the model was trained on expert motion (action std 0.20 vs 0.50), and a CEM fed the wrong action scale exploits it. Details in DECISIONS.

### Probes

`pusht/linear@1`, ridge on the 192-d post-projector latent, 800 expert episodes (93k frames), split by episode:

| target | R² | bucket accuracy (word matches ground truth) |
|---|---|---|
| block position | 0.992 | cell **0.946** · edge **0.988** |
| agent position | 0.977 | cell 0.905 |
| block angle (sin, cos) | 0.953 | 6 angle bins **0.941** |
| contact | AUROC 0.988 | 0.952 |

Rolling the predictor open-loop with the expert's actions, block-cell accuracy decays 0.95 → 0.88 over five imagined steps (edge stays ≥ 0.98). This world-model error is the dominant term in the error decomposition.

<p align="center"><img src="docs/assets/error_decomposition.png" width="85%"></p>

### Judge-only study

256 items (executed expert windows described with ground-truth words *and* with probe words; imagined CEM candidates described with probe words, labelled by oracle-on-probes) × 12 constraints × 8 texts (canonical, 5 paraphrases, 2 near-miss negatives). All judges see identical states and keys.

| judge | items | accuracy (canonical) | AUROC | ECE | accuracy (paraphrases) | near-miss FPR | s / 1,000 judgments |
|---|---|---|---|---|---|---|---|
| **Jev** (imagined, probe words) | 128 | **0.88** | **0.94** | 0.11 | **0.87** | 0.38 | 2.6 |
| **Jev** (executed, probe words) | 64 | 0.86 | 0.94 | 0.08 | 0.85 | 0.53 | 2.5 |
| keyword (imagined) | 128 | 0.89 | 0.90 | 0.09 | 0.74 | 0.02 | 0 |
| keyword (executed) | 64 | 0.88 | 0.90 | 0.09 | 0.64 | 0.02 | 0 |
| local LLM (Qwen2.5-7B, 16-item subset) | 16 | 0.73 on the 33 % of calls that returned valid JSON | 0.74 | – | – | – | 566 |
| oracle-on-probes (upper bound) | 192 | 0.97–0.99 | 1.00 | 0.01 | – | – | 0 |

Consistency: per-question std over 3 repeats = 0.013 (max 0.10). Per constraint, Jev is near-perfect on cell and angle rules, weak on `approach_below` (0.49: requires relating agent cell, block cell and first contact) and `stay_left_half` (0.74).

<p align="center"><img src="docs/assets/paraphrase_heatmap.png" width="90%"></p>

### Planning, Study 1: random expert starts

Conditions {LeWM, oracle-on-probes, keyword, Jev} × constraint sets {spatial = [centre_avoid], spatial+temporal = [centre_avoid, no_contact_first3], implicit = [gentle, approach_below]} × 3 seeds × 30 episodes; λ = 1, τ = 0.5, CEM iterations 0, 5, …, 25, 29 judged.

| set | LeWM | oracle-on-probes | keyword | **Jev** |
|---|---|---|---|---|
| spatial — violation | 0.57 [0.47, 0.67] | 0.56 | 0.56 | **0.56** [0.46, 0.66] |
| spatial+temporal — violation | 0.86 [0.78, 0.92] | 0.83 | 0.81 | **0.84** [0.77, 0.91] |
| implicit — violation | 0.57 [0.47, 0.67] | 0.52 | 0.56 | **0.59** [0.49, 0.69] |
| success (all sets) | 0.84 | 0.81–0.83 | 0.81–0.84 | **0.84** |

No paired Wilcoxon test against LeWM survives Holm correction. Stratifying post hoc by whether the set is even *satisfiable* from the start and goal states explains it: in the spatial set **53 % of episodes have the goal block inside the centre cell**, so success and compliance are mutually exclusive by construction; on the satisfiable episodes every condition produced *identical* plans (14 % violations, all imagination-versus-execution error). Jev's confidence gate held 49 % of candidates on the spatial set.

<p align="center"><img src="docs/assets/study1_success_vs_violation.png" width="100%"></p>

### Study 2: relevance-filtered starts

Pre-registered after Study 1 ([`docs/PREREG.md`](docs/PREREG.md), hash in DECISIONS). Start/goal windows are kept only when the constraint set is **satisfiable** from both end states and the expert's own trajectory between them is **in tension** with it (e.g. the block passes through the centre cell on the way). The filter uses dataset states only, so it is identical for every condition.

Same conditions, sets, seeds and budgets as Study 1 (3 seeds × 30 episodes per cell). Episode violation rate with 95 % CI; ★ = paired Wilcoxon vs LeWM significant after Holm correction across sets.

| set | LeWM | oracle-on-probes | keyword | **Jev** |
|---|---|---|---|---|
| spatial | 0.87 [0.79, 0.93] | 0.92 | 0.92 | **0.88** [0.81, 0.94] |
| spatial+temporal | 0.61 [0.51, 0.71] | **0.51** ★ (−10 pts) | **0.47** ★ (−14 pts) | **0.57** (−4 pts) |
| implicit | 0.77 [0.68, 0.86] | 0.69 (−8 pts) | **0.66** ★ (−11 pts) | **0.73** (−3 pts) |
| success (all sets) | 0.76 | 0.74 | 0.77 | **0.79** |

Three things this says, in order of importance:

1. **The mechanism works when a compliant path exists and the world model can see it.** On `spatial+temporal` and `implicit`, a perfect judge and even the keyword checker lower violations by 8–14 points with no loss of success.
2. **Jev's confidence gate is currently the limiting factor, not its accuracy.** Jev's per-step probabilities often sit in the uncertain band [0.3, 0.7]; the pre-registered gate (τ = 0.5) therefore zeroed the penalty for 21 % (implicit), 23 % (spatial+temporal) and 62 % (spatial) of candidates. The keyword judge is binary and never abstains. Re-tuning τ or calibrating Jev's outputs is the obvious next experiment; we did not do it after seeing these numbers.
3. **`spatial` is bounded by the world model, not the judge.** With the oracle judge the chosen plan is imagined to pass through the centre in only ~20 % of episodes, yet execution does so in ~75 %: open-loop drift over the 25-step receding horizon. Replanning every block does not help (unconstrained success collapses to 30 %), nor does executing the best sampled candidate instead of the CEM mean (0.75 → 0.65 at λ = 4). The filtered λ sweep for Jev (0.25 … 4) stays at 0.80–0.90.

<p align="center"><img src="docs/assets/study2_success_vs_violation.png" width="100%"></p>

Full tables: `paper/figures/table4_planning_filtered.{csv,md}` and `table4b_paired_tests_filtered.csv`.

### What the ablations say

Spatial set, seed 0, 20 episodes, Jev: λ ∈ {0.25, 0.5, 1, 2, 4}, K ∈ {4, 16, 32, 64}, judging schedule {last-3, every-5, every iteration, final only}, hard rejection, 4×4 vocabulary, MLP probe, half horizon, no prior for unjudged candidates. **All of them leave violation at 0.50–0.55 and success at 0.95–1.00.** With the memo warm, most ablation runs needed zero new Jev calls.

---

## Reproduce everything

```bash
uv venv --python=3.10 && source .venv/bin/activate
uv pip install -e '.[llm,demo,dev]'       # pins stable-worldmodel to the git commit in pyproject
export STABLEWM_HOME=~/.stable-wm SDL_VIDEODRIVER=dummy LEJUDGE_MODE=offline

pytest -q && lejudge lint                 # 60 tests, RFC-0003 question lint
lejudge report --out paper/figures        # every figure and table from artifacts/results
make paper                                # + fill placeholders + LaTeX → paper/main.filled.pdf
```

`LEJUDGE_MODE=offline` raises on any cache miss, so the commands above make **zero** API calls. To regenerate the studies from scratch (needs `TYPESAFE_API_KEY` and `LEJUDGE_MODE=live`):

```bash
python -c "from lejudge.probes.expert import build; build(episodes=800)"   # streams a prefix of the official 13 GB h5.zst
lejudge probes train --kind linear && lejudge probes train --kind mlp
lejudge m0 --episodes 50
lejudge judge-study --judges jev,keyword,oracle --n-executed 128 --n-imagined 128 --repeats 3 --repeat-items 64 --live
lejudge plan --cond lewm,oracle,keyword,jev --set "spatial,spatial+temporal,implicit" --episodes 30 --seeds 0-2 --live
lejudge plan --cond lewm,oracle,keyword,jev --set "spatial,spatial+temporal,implicit" --episodes 30 --seeds 0-2 --start-filter relevant --live --out artifacts/results/planning_filtered.parquet
lejudge ablate --grid configs/ablations.yaml --live
lejudge demo-precompute --seeds 20 --live && lejudge record      # Gradio gallery + 20-second clip
lejudge demo                                                       # http://localhost:7860
```

Pinned: LeWM checkpoint `quentinll/lewm-pusht@22b330c2`, `stable-worldmodel@4821c8e6`, `transformers<5`, Jev `jev-1.13.0` (requested as `jev-latest`, asserted on the response), vocab `pusht@1`, library `pusht-lib@1` (sha256 `61d416fd…`), bank `lejudge.pusht@0.1.0`.

## Use it in your own planner

`JevCost` is a stable-worldmodel `Objective`: it reads `predicted_emb` from the rollout and returns a `(B, S)` cost, so it drops into `ShootingCostEvaluator` next to `GoalMSE`.

```python
from lejudge.cost import JevCost
from lejudge.cost.swm_adapter import load_lewm, goal_mse_objective, shooting_cost, make_policy, PlanSpec
from lejudge.judge import JevJudge
from lejudge.probes import load_probe
from lejudge.probes.data import EpisodeData
from lejudge.types import Constraint
from lejudge.vocab import load_vocab

model = load_lewm()
cost = JevCost(
    goal_mse_objective(), load_probe("pusht/linear@1"), load_vocab("pusht@1"),
    [Constraint("Keep the T out of the centre cell."),
     Constraint("Be gentle: move the block slowly.", family="soft", weight=0.5)],
    JevJudge(), lam=1.0, tau=0.5, mode="every_k", judge_every=5, n_iters=30,
)
scaler = EpisodeData("artifacts/data/pusht_expert.npz").scaler
policy = make_policy(shooting_cost(model, cost), PlanSpec.from_config(), scaler, callbacks=[cost.callback()])
```

Swap `JevJudge()` for `KeywordJudge()`, `OracleJudge(vocab, on_probes=True)` or `LLMJudge(model=...)`; they implement the same `judge(facts, constraints)` and receive identical inputs.

## Repository layout

```
lejudge/
├── probes/        expert data builder, linear/MLP heads, imagined-latent error curve
├── vocab/         pusht@1 (3×3 + edge flags), pusht@2 (4×4); words() is pure
├── constraints/   12-constraint YAML library, oracle checkers, shared T geometry
├── judge/         state builder, question bank + lint, aggregation, SQLite cache + step memo, traces,
│                  JevJudge, KeywordJudge, OracleJudge, LLMJudge
├── cost/          JevCost (swm Objective) and the one-file stable-worldmodel adapter
├── eval/          planning study, judge-only study, relevance filters, statistics, report
├── demo/          Gradio Space, cached gallery, clip recorder
└── cli.py         lejudge {collect, probes, lint, m0, judge-study, plan, ablate, report, demo, record, ...}
configs/           pusht.yaml (every pinned id and budget), ablations.yaml
artifacts/         probes/, cache/jev.sqlite (committed), results/*.parquet, traces/<run>/, demo/
paper/             main.tex with \VAL{} placeholders, fill.py, figures/
rfcs/ · SPEC.md · PRD.md · AGENTS.md    design of record
docs/DECISIONS.md · docs/PREREG.md     every deviation, with reasons and hashes
```

## Design rules

1. **Jev reads words, never numbers.** Anything numeric is bucketed by the vocabulary before it reaches a state; a test asserts no digit ever appears in a fact.
2. **Constraint text and facts never share a field.** Owner text under `constraints`, code-generated words under `candidates`.
3. **Arithmetic lives in code.** Aggregation, thresholds, λ, the gate. Jev is never asked to count or compare.
4. **Every external call is cached and traced.** `LEJUDGE_MODE=offline` is the default and raises on a miss.
5. **Baselines share the interface** and see byte-identical states.
6. **No fabricated numbers.** `paper/fill.py` writes `[X]` for anything it cannot read from a results table.

## Limitations and honest caveats

- PushT only; twelve constraints; one world model. The world model's imagination error (block cell 0.95 → 0.88 over five steps) bounds what any judge can do.
- Study 1's constraint sets mostly conflicted with the goal by construction; Study 2 fixes the sampling, not the underlying tension between a goal image and a rule.
- Jev's near-miss false-positive rate (38–53 %) means a paraphrase-robust judge is not automatically a *precise* one.
- Jev is a closed model; results are pinned to `jev-1.13.0` and fully cached, but not reproducible against a future version.
- The LLM baseline is a local 7B model on a small subset; it mostly failed the 64-key JSON schema. A funded hosted-LLM comparison is future work.
- The demo runs locally; it is not deployed as a public Space.

## Citation

```bibtex
@misc{lejudge2026,
  title  = {LeJudge: a cost module you program in English},
  author = {Abdel},
  year   = {2026},
  url    = {https://github.com/AbdelStark/lejudge-jev-jepa}
}
```

Built on [LeWorldModel](https://github.com/lucas-maes/le-wm) (Maes, Le Lidec, Scieur, LeCun, Balestriero, 2026), [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel) and [TypeSafe Jev](https://docs.typesafe.ai/model-jaggedness/jev-1.13.md). Agent-built; every decision is in `docs/DECISIONS.md`.
