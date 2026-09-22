# RFC-0004 — JevCost and planner integration

**Status** Draft · **Scope** MVP · **Depends on** RFC-0003

## Summary

A cost module compatible with `stable-worldmodel`'s solver interface that adds a language-constraint penalty to LeWM's goal-distance cost, judging only the elite shortlist each CEM iteration.

## Interface

`stable-worldmodel` solvers call a cost model on batched rollouts. `JevCost` wraps `AutoCostModel("pusht/lewm")`:

```python
class JevCost(nn.Module):
    def __init__(self, world_model, probes, vocab, constraints, judge, lam=1.0, K=16, tau=0.5,
                 mode="per_iter", steps=None, standardize=True): ...
    def rollout(self, z0, actions): return self.world_model.rollout(z0, actions)      # delegate
    def forward(self, z_rollout, z_goal, actions=None) -> Tensor[N]:                  # per-candidate cost
```

Verify the exact method names against `stable_worldmodel.policy.AutoCostModel` and `CEMSolver` at implementation time; the adapter layer lives in `cost/swm_adapter.py` so API drift touches one file.

## Algorithm per call

1. `A = ‖z_rollout[:, -1] − z_goal‖²`; if `standardize`, `A ← (A − mean) / std` over the batch so λ is unitless.
2. `idx = argsort(A)[:K]`.
3. `facts = words(probes(z_rollout[idx]))` for steps `1..H` (or `steps`).
4. `res = judge.judge(facts, constraints)` — one call (RFC-0003).
5. `pen[k] = Σ_c w_c · penalty(family_c, res.p[k][c])`.
6. Gate: `held[k] = uncertainty_proxy(res, k) > 0.5`; `pen[held] = 0`.
7. `cost = A.clone(); cost[idx] += lam · pen`.
8. Log `JudgeTrace` (RFC-0008).

`mode="final_only"`: skip steps 2–7 on all but the last CEM iteration.

## Where it sits in MPC

`WorldModelPolicy(solver=CEMSolver(model=JevCost(...), num_samples=300), config=PlanConfig(horizon=10, receding_horizon=5))`. Nothing else in the policy changes; replanning re-judges from the new observation.

## Behavioural guarantees

- With `lam=0` the policy is bit-identical to unconstrained LeWM (regression test).
- With `judge=OracleJudge` on probes it equals `oracle-on-probes` (upper bound in the same code path).
- Judge failure never blocks planning: penalty 0 and a `held` flag.

## Hyperparameters frozen for the study

`K=16`, `lam ∈ {0.25, 0.5, 1, 2, 4}` (sweep on Jev; 1.0 for cross-condition tables), `tau=0.5`, `mode=per_iter`, `steps=H`.

## Acceptance tests

- Regression: `lam=0` reproduces unconstrained success within seed noise (paired test not significant).
- Sanity: on the `edges` set, oracle-on-probes reduces violation rate by ≥ 50 % relative; Jev is within 10 points of it.
- Latency: added time per planning step ≤ 0.4 s p50 on the reference GPU.
- Unit: cost shape, standardisation, gate behaviour, `final_only` call count.

## Alternatives

- Hard rejection (`cost = +inf` when `p > 0.7`): brittle under probe noise; kept as an ablation.
- Augmented Lagrangian solver with penalties as constraints: post-MVP; the penalty function is exposed so this is a solver swap.
- Judging all N candidates: 300 × 2 × 10 = 6,000 questions per iteration; unnecessary because A already ranks; ablate K ∈ {4, 16, 64}.
