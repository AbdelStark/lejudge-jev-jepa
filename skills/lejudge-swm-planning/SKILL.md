---
name: lejudge-swm-planning
description: Work with stable-worldmodel and LeWorldModel checkpoints for LeJudge — loading the PushT LeWM checkpoint, running CEM/MPC planning, writing a cost model that plugs into WorldModelPolicy, collecting rollouts and traces. Use this skill whenever a task touches lejudge/cost/, planning configs, checkpoints, AutoCostModel, CEMSolver, WorldModelPolicy, PlanConfig, swm.World, PushT, or asks to reproduce LeWM numbers — even if it only says "run the planner".
---

# stable-worldmodel + LeWM for LeJudge

`stable-worldmodel` (swm) provides envs, data, solvers and MPC evaluation; LeWM is the world model. LeJudge adds a cost model. This skill tells you how to load, plan, extend and verify without guessing APIs.

## First: verify the installed API

The library is under active development. Before writing the adapter, run:

```
python -c "import stable_worldmodel as swm, inspect; print(swm.__version__); print(inspect.signature(swm.solver.CEMSolver)); print(inspect.signature(swm.policy.WorldModelPolicy)); print(inspect.signature(swm.policy.PlanConfig)); help(swm.policy.AutoCostModel)"
```

Record the observed signatures in `docs/DECISIONS.md` under "swm API snapshot". `references/swm-readme-snapshot.md` is the README as read on 2026-09-22; treat it as a hint, not truth.

## Loading LeWM (PushT)

```
hf download quentinll/lewm-pusht --local-dir $STABLEWM_HOME/hf_pusht
python scripts/convert_checkpoint.py --env pusht     # ports the le-wm README conversion; writes $STABLEWM_HOME/pusht/lewm_object.ckpt
```
```python
import stable_worldmodel as swm
cost = swm.policy.AutoCostModel("pusht/lewm")   # eval mode; .state_dict() available
```

The `_object.ckpt` is what `eval.py` and `AutoCostModel` expect; the HF repo ships `weights.pt` + `config.json`.

## Reproducing the paper's planning numbers (M0)

```python
world = swm.World("swm/PushT-v1", num_envs=8)
solver = swm.solver.CEMSolver(model=cost, num_samples=300)
policy = swm.policy.WorldModelPolicy(solver=solver, config=swm.policy.PlanConfig(horizon=10, receding_horizon=5))
world.set_policy(policy); results = world.evaluate(episodes=100, seed=0)
```

Match the budget to LeWM's `config/eval/pusht.yaml` from the le-wm repo if it differs; freeze the final values in `configs/pusht.yaml` and DECISIONS. Success within 5 points of the paper's PushT number is the gate for everything else.

## Writing a cost model that swm's solver accepts

- Inspect how `CEMSolver` calls the model (rollout method name, cost/objective method name, tensor shapes `[N, H, d]`).
- `JevCost` wraps the LeWM object: delegate rollout; override the per-candidate cost (RFC-0004).
- Keep the adapter in `lejudge/cost/swm_adapter.py`; nothing else imports swm internals.
- `lam=0` must be bit-identical to the wrapped model (regression test).

## Collecting rollouts and ground truth

- During planning, record CEM candidates' latents and, for executed steps, the env's ground-truth state (`world` exposes `info`/state; confirm field names for PushT: agent position, block position, block angle).
- Use RFC-0008's trace writer; never ad-hoc pickles.
- For probe training use the released dataset (`galilai-group/lewm-pusht`) which carries state columns; check with `swm inspect pusht_expert_train`.

## Factors of variation (shift study)

`swm fovs PushT-v1` lists visual/physical factors. Use colour and lighting for the robustness run (RFC-0006).

## Gotchas

- `$STABLEWM_HOME` default differs between le-wm docs (`~/.stable-wm/`) and swm README (`~/.stable_worldmodel/`); set it explicitly.
- GPU CEM is not bitwise deterministic across hardware; seeds fix the sampler, not CUDA kernels.
- One env instance per process when recording traces to keep episode ids simple.
