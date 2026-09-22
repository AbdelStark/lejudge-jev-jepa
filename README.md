# LeJudge — a cost module you program in English

**LeJudge** adds a natural-language cost module to JEPA world-model planning. LeWorldModel (LeWM) imagines latent futures; linear probes turn each imagined state into a few worded facts; Jev (TypeSafe's System One model) answers typed yes/no questions about those facts; code folds the calibrated probabilities into the CEM cost. The planner obeys constraints written in plain English, inside a sub-second MPC loop, with no LLM and no generated text.

> Imagine → describe → judge → decide.

## What is in this repo

| Path | What it is |
|---|---|
| `lejudge/` | The package: `probes/` (latent → symbols), `vocab/` (symbols → words), `constraints/` (library + oracle checkers), `judge/` (state builder, question bank, Jev + baselines, cache, traces), `cost/` (`JevCost`, swm adapter), `eval/` (planning study, judge-only study, stats, report), `demo/` (Gradio Space + clip), `cli.py` |
| `configs/pusht.yaml` | Frozen ids and budgets: checkpoint revision, swm commit, vocab/library/bank versions, planner config |
| `artifacts/` | `probes/` (weights + `meta.json`), `cache/jev.sqlite` (every Jev/LLM response), `traces/<run>/` (one JSONL per episode), `results/*.parquet`, `data/*.meta.json` |
| `paper/` | `main.tex` with `\VAL{}` placeholders, `fill.py`, `figures/` from `lejudge report` |
| `rfcs/`, `SPEC.md`, `PRD.md`, `AGENTS.md` | Design of record |
| `docs/DECISIONS.md`, `docs/PREREG.md` | Every choice not fixed by an RFC; the pre-registration and its hash |

## Quickstart

```bash
uv venv --python=3.10 && source .venv/bin/activate
uv pip install -e '.[llm,demo,dev]'          # pins stable-worldmodel to the git commit in pyproject
export STABLEWM_HOME=~/.stable-wm SDL_VIDEODRIVER=dummy
pytest -q && lejudge lint                    # offline; no API needed
lejudge report --out paper/figures           # regenerates every figure/table from artifacts/results
make paper                                   # report + fill placeholders + LaTeX (offline)
```

Live runs need `TYPESAFE_API_KEY` and `LEJUDGE_MODE=live`; everything else runs from the cache (`LEJUDGE_MODE=offline` raises on a miss).

```bash
python -c "from lejudge.probes.expert import build; build(episodes=800)"   # expert data → artifacts/data/pusht_expert.npz
lejudge probes train --kind linear && lejudge probes train --kind mlp
lejudge m0 --episodes 50                                                    # reproduction gate
lejudge judge-study --judges jev,keyword,oracle --n-executed 128 --n-imagined 128 --repeats 3 --repeat-items 64 --live
lejudge plan --cond lewm,oracle,keyword,jev --set "spatial,spatial+temporal,implicit" --episodes 30 --seeds 0-2 --live
lejudge ablate --grid configs/ablations.yaml --live
lejudge demo                                                                # Gradio Space (presets cached; live edits need the key)
lejudge record                                                              # 20-second clip
```

## Library use

```python
from lejudge.cost import JevCost
from lejudge.cost.swm_adapter import load_lewm, goal_mse_objective, shooting_cost, make_policy, PlanSpec
from lejudge.judge import JevJudge
from lejudge.probes import load_probe
from lejudge.vocab import load_vocab
from lejudge.types import Constraint

model = load_lewm()
cost = JevCost(goal_mse_objective(), load_probe("pusht/linear@1"), load_vocab("pusht@1"),
               [Constraint("Keep the T out of the centre cell."), Constraint("Be gentle: move the block slowly.", family="soft", weight=0.5)],
               JevJudge(), lam=1.0, K=16, tau=0.5, mode="last_n", n_iters=30, judge_last_n=3)
policy = make_policy(shooting_cost(model, cost), PlanSpec.from_config(), scaler, callbacks=[cost.callback()])
```

## Ground rules (see `AGENTS.md`)

Jev reads words, never numbers. Constraint text and facts never share a state field. Arithmetic lives in code. Every external call is cached and traced. Baselines share the judge interface. No fabricated numbers: every number in `paper/` traces to a row in `artifacts/results/`.

## Sources

LeWM paper https://arxiv.org/html/2603.19312v3 · repo https://github.com/lucas-maes/le-wm · checkpoints https://huggingface.co/collections/quentinll/lewm · stable-worldmodel https://github.com/galilai-group/stable-worldmodel · TypeSafe docs https://docs.typesafe.ai/introduction · Jev jaggedness https://docs.typesafe.ai/model-jaggedness/jev-1.13.md
