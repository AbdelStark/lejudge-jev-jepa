# RFC-0003 — Judge: state, question bank, aggregation and baselines

**Status** Draft · **Scope** MVP · **Depends on** RFC-0001, RFC-0002

## Summary

One Jev call per CEM iteration judges K elite candidates against C constraints over H steps. This RFC fixes the state layout, the question templates, the aggregation per family, the confidence gate inputs, and the baseline judges that share the interface.

## State layout

```json
{"constraints": {"c1": "Never let the T touch the left or right edge.", "c2": "Keep the T roughly upright."},
 "candidates": {
   "k1": [{"t":1,"block":"centre-left","block_edge":"none","block_angle":"upright","agent":"bottom-centre","contact":false,"block_speed":"slow"},
          {"t":2,"block":"left","block_edge":"left edge","block_angle":"tilted left","agent":"centre-left","contact":true,"block_speed":"fast"}],
   "k2": ["..."]}}
```

Rules:
- `constraints` and `candidates` are separate top-level fields. Facts are code-generated words from a closed vocabulary; no free text ever appears under `candidates`.
- Candidate ids are `k1..kK` in cost order; steps are `t=1..H`.
- Fields with low probe confidence are omitted for that step, never guessed.

## Question templates

| Family | Key | Type | Instruction | Criteria |
|---|---|---|---|---|
| never | `{k}_{c}_{t}` | Noul | "At step t={t} of candidate {k}, does the block or agent violate constraint {c} exactly as written in constraints.{c}?" | "Read constraints.{c} literally. Use only the facts in candidates.{k} at step t={t}. Answer true only if those facts show the forbidden condition at this step." |
| always | `{k}_{c}_{t}` | Noul | "At step t={t} of candidate {k}, are the facts consistent with constraint {c} as written?" | "Answer true when the facts satisfy the requirement; answer false when they contradict it. If the facts do not mention what the constraint needs, answer true." |
| soft | `{k}_{c}` | Score `[not at all, poorly, partly, mostly, fully]` | "Overall across its steps, how well does candidate {k} respect constraint {c} as written?" | Level descriptions reference `block_speed` and `contact` words |
| temporal_before | `{k}_{c}_A_{t}`, `{k}_{c}_B_{t}` | Noul ×2 | "By step t={t} of candidate {k}, has event A of constraint {c} happened?" / B | code compares first true indices |

Question count per call = K·C·H for never/always; capped at 512, split across calls otherwise.

## Aggregation (code)

```python
def penalty(family, p):            # p: per-step probabilities for one (k, c)
    if family == "never":   return max(p)
    if family == "always":  return 1 - min(p)
    if family == "soft":    return 1 - expectation(p_levels) / 4
    if family == "temporal_before": return 0.0 if first_true(pA) < first_true(pB) else 1.0
```

Optional smoothing for `never`: `softmax-weighted max` with temperature 0.1 to reduce sensitivity to a single noisy step; ablated.

## Confidence gate inputs

Noul has no confidence field; Score and Choice do. For `never`/`always` we derive a per-candidate uncertainty proxy: fraction of step probabilities in the band [0.3, 0.7]. For `soft`, use the Score confidence directly. The gate (RFC-0004) holds a candidate when the proxy > 0.5 or Score confidence < τ.

## Baseline judges (same interface)

| Judge | Implementation |
|---|---|
| `OracleJudge` | Oracle checker on ground-truth executed states; `oracle-on-probes` on imagined states |
| `KeywordJudge` | Per constraint, a dictionary of trigger words over `StepFacts` fields written from the *canonical* text only (e.g. `block_edge ∈ {left edge, right edge}`); applied verbatim to paraphrases — the point is to show it does not transfer |
| `LLMJudge(model)` | Same JSON state; prompt lists the same keys and asks for a JSON object mapping key → probability; strip fences; parse failures → NaN, counted |

## Lint (automated, runs in CI on the constraint library and templates)

- No arithmetic, counting, or numeric comparison in any instruction.
- No negated Nouls (true must mean the condition is present).
- Criteria mention the state fields they rely on.
- Instruction ≤ 40 words; criteria ≤ 60 words.
- Each template renders with all vocabulary words without KeyError.

## Acceptance tests

- Golden test: a hand-built state with a known violation at step 2 yields `p[k1][c1][2] > 0.7` and `p[k1][c1][1] < 0.3` on `jev-1.13` (live test, cached).
- Consistency: 3 repeats on 200 states, per-question std-dev < 0.05.
- Baselines produce identical key sets to `JevJudge`.

## Alternatives

- One question per candidate summarising all steps: fewer questions, but hides which step violated and invites counting; kept as an ablation (`per_candidate` mode).
- Choice over candidates ("which is safest"): relative, not absolute; unsuitable for a penalty. Reserved for the language-goal retrieval (RFC-0005).
