# SPEC — LeJudge MVP technical specification

**Status** Draft · **Date** 2026-09-22 · **Depends on** `stable-worldmodel` (planning + envs), LeWM checkpoint `quentinll/lewm-pusht`, `typesafe-sdk >= 0.5.7`, Python 3.10

## 1. Architecture

```
                 ┌──────────────── planning step (MPC) ────────────────┐
o_1 ──► enc ──► z_1                                                    │
                 │   CEM iteration i (×3)                              │
                 │   sample N=300 action seqs ──► pred rollouts ẑ_1..H │
                 │   cost A = ‖ẑ_H − z_g‖²  ──► rank ──► top-K elites  │
                 │   probes(ẑ_t) ──► symbols ──► words  (K × H facts)  │
                 │   Jev: one system_one call, Nouls per (k, c, t)     │
                 │   aggregate(family) ──► penalty_k                    │
                 │   cost_k = A_k + λ Σ_c w_c penalty_k,c  (gate by τ) │
                 │   refit CEM distribution on lowest-cost elites      │
                 └──────────────────────────────────────────────────────┘
execute first K_r actions ──► replan
```

Modules:

```
lejudge/
├── probes/        train/load probes; latent→symbols
├── vocab/         symbols→words; versioned vocabularies
├── constraints/   YAML library; oracle checkers; families; paraphrases
├── judge/         state builder; question bank builder; Jev client; cache; baselines (oracle/keyword/llm)
├── cost/          JevCost (swm-compatible); aggregation; confidence gate
├── eval/          judge-only study; planning study; metrics; stats; report
├── demo/          Gradio app
└── cli.py
```

## 2. Interfaces

### 2.1 Probes

```python
class Probe(Protocol):
    def __call__(self, z: Tensor[..., 192]) -> SymbolicState  # batched
@dataclass
class SymbolicState:  # PushT
    agent_xy: Tensor[..., 2]; block_xy: Tensor[..., 2]; block_angle: Tensor[...]; contact: Tensor[...]  # logits
```

Trained heads: linear (default) and 2-layer MLP (reported). Inputs are LeWM latents *after* the projector (the 192-d `z` used by the predictor). Targets from the dataset's ground-truth state. Saved as `probes/pusht/linear@1.pt` with metadata (checkpoint hash, R², bucket accuracy).

### 2.2 Vocabulary

```python
def words(state: SymbolicState, vocab: Vocab) -> list[StepFacts]
@dataclass
class StepFacts:
    t: int; block: str; block_edge: str; block_angle: str; agent: str; contact: bool; block_speed: str
```

Vocab `pusht@1`: 3×3 grid names for `block` and `agent` (`top-left … bottom-right`), `block_edge ∈ {none, left edge, right edge, top edge, bottom edge}` from a margin threshold, `block_angle ∈ {upright, tilted left, tilted right, on its side left, on its side right, upside down}` from 6 angle bins, `block_speed ∈ {still, slow, fast}` from Δposition between steps. Numeric thresholds live in `vocab/pusht@1.yaml`, never in prose.

### 2.3 Constraint

```yaml
- id: edges_never
  family: never          # never | always | soft | temporal_before
  text: "Never let the T touch the left or right edge."
  paraphrases: ["Keep the T off the sides.", "The block must not reach either side wall.", ...]   # 5
  negatives: ["Never let the T touch the top edge.", "Keep the agent off the sides."]              # near-miss, must NOT fire
  oracle: block_edge_lr    # name of a checker in constraints/oracles.py over ground-truth state
  weight: 1.0
```

Families define aggregation:

| Family | Per-step question | Aggregation (code) |
|---|---|---|
| `never` | "At step t, does candidate k violate constraint c as written?" | `max_t p` |
| `always` | "At step t, does candidate k satisfy constraint c as written?" | `1 − min_t p` |
| `soft` | one question per candidate: "Overall, how well does candidate k respect constraint c?" as Score `[not at all, poorly, partly, mostly, fully]` | `1 − E[score]/4` |
| `temporal_before` | two Nouls per step: "has event A happened by step t?", "has event B happened by step t?" | code compares first indices |

### 2.4 Judge call

```python
class JevJudge:
    def judge(self, facts: dict[str, list[StepFacts]], constraints: list[Constraint]) -> JudgeResult
```

State (one call):

```json
{"constraints": {"c1": "<text>", "c2": "<text>"},
 "candidates": {"k1": [{"t":1,"block":"centre-left","block_edge":"none","block_angle":"upright","agent":"bottom-centre","contact":false,"block_speed":"slow"}, ...], "k2": [...]}}
```

Questions: `f"{k}_{c}_{t}"` → `Noul(instructions=f"At step t={t} of candidate {k}, does the block violate constraint {c} as written in constraints.{c}?")`, criteria from the family template. All in one `client.system_one(model="jev-1.13", state=..., questions=...)`.

`JudgeResult`: `p[k][c][t]`, `confidence` where available, `latency_ms`, `input_tokens`, `response_model`, `cache_hit`.

Baselines implement the same interface: `OracleJudge` (ground-truth state of executed rollouts; for imagined rollouts uses probes + checker and is labelled "oracle-on-probes"), `KeywordJudge` (regex/dictionary over words), `LLMJudge(model)` (same JSON, asks for a probability per key).

### 2.5 JevCost

```python
class JevCost(nn.Module):  # swm cost-model interface
    def forward(self, z_rollout: Tensor[N, H, 192], z_goal: Tensor[192], actions) -> Tensor[N]
```

1. `A = ‖z_rollout[:, -1] − z_goal‖²`.
2. `idx = topk(−A, K)`; facts for `idx`.
3. `res = judge.judge(...)`; `pen[k] = Σ_c w_c · agg_c(res.p[k][c])`.
4. Gate: for candidate k, if `mean confidence < τ` (or fraction of low-confidence questions > 0.5) then `pen[k] = 0` and `held[k] = True`.
5. `cost = A; cost[idx] += λ · pen`. Non-elite candidates keep `A` only (they will not be selected anyway).

Configurable: `K` (default 16), `λ` (1.0; A is standardised per iteration so λ is unitless), `τ` (0.5), `steps` (all H or first H'), `mode ∈ {per_iter, final_only}`.

### 2.6 Planning config (frozen for the study)

`CEMSolver(num_samples=300, num_iters=3, elite_frac=0.1)`, `PlanConfig(horizon=10, receding_horizon=5)`, PushT episode length and success criterion as in stable-worldmodel's PushT-v1. If LeWM's reproduced numbers require different values, freeze those and record in `docs/DECISIONS.md`.

## 3. Data flows and formats

| Artefact | Format | Location |
|---|---|---|
| Probe training set | Lance/HDF5 from `galilai-group/lewm-pusht` with ground-truth state columns | `$STABLEWM_HOME` |
| Probe weights | `.pt` + `meta.json` | `artifacts/probes/` |
| Constraint library | YAML | `lejudge/constraints/pusht.yaml` |
| Jev cache | SQLite keyed by `sha256(model, bank_version, canonical_state_json)` → response JSON | `artifacts/cache/jev.sqlite` |
| Planning traces | JSONL per episode: step, candidates' facts, judge results, chosen action, executed state, oracle flags | `artifacts/traces/<run>/` |
| Results | Parquet tables per study | `artifacts/results/` |
| Figures | PDF/PNG via `lejudge report` | `paper/figures/` |

Canonical state JSON: sorted keys, fixed float formatting, candidate ids renamed `k1..kK` in cost order so cache hits survive reordering.

## 4. Budgets

| Item | Default | Notes |
|---|---|---|
| Jev calls per planning step | 3 (one per CEM iteration) | `final_only` mode → 1 |
| Questions per call | K × C × H = 16 × 2 × 10 = 320 | cap 512; above cap, split calls |
| State tokens per call | ~2,500 | measured, logged |
| Added latency per step | ≤ 0.4 s target | 3 × ~111 ms + overhead |
| Episodes in planning study | 100 × 5 seeds × 5 conditions × 3 constraint sets | cached responses make reruns free |

## 5. Evaluation harness

### 5.1 Judge-only study
Inputs: 5,000 imagined rollouts (from CEM candidates during unconstrained planning) and 5,000 executed rollouts, both described with probes and with ground truth. Labels: oracle checkers on ground truth. Judges: Jev, keyword, LLM-small (and a reasoning LLM on a 500-item subset for cost reasons). Outputs: precision/recall/F1 at 0.5, AUROC, ECE, reliability plot, per-constraint and per-paraphrase accuracy, agreement across paraphrases, near-miss negative false-positive rate, latency and cost per 1,000 judgments, consistency (std-dev over 3 repeats on 200 items).

### 5.2 Planning study
Conditions × constraint sets × seeds. Outputs: success rate, episode violation rate, violation steps, planning time p50/p95, Jev calls/tokens/USD per episode, abstention rate and its violation rate, λ sweep Pareto curve (Jev only), ablations (K, per_iter vs final_only, vocab 3×3 vs 4×4, steps H vs H/2).

### 5.3 Statistics
Bootstrap 95 % CIs (10,000 resamples) for all rates; paired Wilcoxon across conditions on per-episode outcomes; Holm correction across constraint sets; effect sizes (risk difference with CI).

## 6. Error handling

| Failure | Behaviour |
|---|---|
| Jev timeout / 5xx | Retry once (200 ms); then `pen = 0`, `held = True` for that iteration; logged |
| Cache miss offline (`--offline`) | Raise; the paper build must be fully cached |
| Probe confidence low (contact logit near 0) | `contact` omitted from facts for that step |
| Questions > cap | Split into multiple calls, same state |

## 7. Security and hygiene

- Constraint text is owner input; worded facts are generated by code; they never share a field.
- Free text from any external source (e.g. demo users) is length-capped (200 chars) and placed only under `constraints`.
- API keys via environment or the Space's secrets; never in traces.

## 8. Post-MVP hooks

- `Constraint.family = temporal_before` implemented but not in the MVP library.
- `goals/` package stub for language goals (RFC-0005).
- Environment adapters: `probes/`, `vocab/` and `constraints/` are keyed by env name so TwoRoom/Cube/Reacher add files, not code paths.
- Augmented Lagrangian solver variant: penalties exposed as constraint functions.

## 9. Definition of done (MVP)

- [x] LeWM PushT success reproduced with the authors' protocol (86 % with history 3, 90 % with history 1, 50 episodes; the paper text gives no single number to compare against — see DECISIONS)
- [x] Probes: R² and bucket accuracy reported in `artifacts/probes/pusht/linear@1/meta.json` (block cell 0.95, angle 0.94, edge 0.99)
- [x] Judge-only study complete: Jev, keyword, oracle on 256 items × 12 constraints × 8 texts with 3 repeats on 64 items; local-LLM baseline on a 16-item subset (8 executed windows, canonical texts); tables in `artifacts/results/judge_only.parquet`
- [x] Planning study complete (4 conditions × 3 sets × 3 seeds × 30 episodes) with paired tests; λ sweep and ablations in `ablations.parquet`
- [x] `make paper` regenerates every figure and number offline from the cache on a clean clone (no dataset, no traces); CI rebuilds `paper/values.json` and fails on any difference
- [~] Space runs locally (`lejudge demo`, cached gallery of 20 seeds × 6 presets); 20-second clip at `artifacts/demo/clip.mp4`; not deployed to Hugging Face in this build
- [x] Preprint draft `paper/main.tex` filled by `paper/fill.py` from `artifacts/results/`; unavailable numbers stay `[X]`
