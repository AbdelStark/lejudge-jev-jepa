# PRD — LeJudge MVP

**Status** Draft · **Date** 2026-09-22 · **Owner** Abdel · **Build model** fully agent-built; no human time estimates anywhere in this repo

## 1. Problem

LeWorldModel plans by minimising one number: the latent distance between the imagined final state and a goal image. That cost cannot say "but not through there", "keep it upright" or "be gentle". Adding those today means writing a checker per constraint in code, or bolting an LLM onto a loop that must finish in under a second. LeCun's JEPA architecture draws a configurable cost module on the diagram; nobody has shipped one a non-expert can configure.

## 2. Product

LeJudge is a cost module for latent-MPC planners that accepts constraints in natural language and evaluates them inside the planning loop using a decision model. Delivered as:

1. A Python package `lejudge` that plugs into `stable-worldmodel`.
2. A reproducible study with an oracle, baselines and cached responses.
3. A Gradio Space where a sentence changes a plan.
4. A preprint.

## 3. Users and jobs

| User | Job | Done when |
|---|---|---|
| World-model researcher | Add constraints to a LeWM/DINO-WM planner without reward engineering | `policy = WorldModelPolicy(solver=CEMSolver(model=JevCost(lewm, constraints=[...])))` works |
| Robotics engineer | Encode operating rules ("never above table height") as sentences and get a violation rate they can trust | Violation and success numbers with CIs; abstention when unsure |
| TypeSafe / System One community | A rigorous, reproducible use of Jev in a control loop | Cached-response reproducibility; entry in awesome-typesafe |
| Public / social | Watch a tiny world model obey an English sentence | 20-second clip, live Space |

## 4. Goals (measurable)

| Goal | Target |
|---|---|
| G1 Constraint compliance | LeJudge violation rate within 10 points of the oracle checker on PushT across the 12-constraint library |
| G2 Task success preserved | Success within 10 points of unconstrained LeWM |
| G3 Latency | ≤ 0.4 s added per replanning step at the default budget (3 Jev calls) |
| G4 Generalisation | Judge accuracy on held-out paraphrases within 5 points of in-distribution; keyword checker drops ≥ 20 points on the same paraphrases |
| G5 Calibration | ECE ≤ 0.10 on judge-only labels; confidence gate reduces false-penalty rate |
| G6 Reproducibility | `make paper` regenerates every figure from cached responses with zero API calls |
| G7 Demo | Space loads in < 30 s and returns a constrained plan in < 15 s at demo budget |

## 5. Non-goals (MVP)

- Environments other than PushT.
- Language-specified goals (RFC-0005 is a stretch design, not MVP scope).
- Training or fine-tuning LeWM or probes beyond linear/MLP heads.
- Reasoning-LLM judges in the planning loop (judge-only study only).
- Any claim about Jev's adversarial robustness beyond a small appendix test.

## 6. User experience

### 6.1 Library

```python
import stable_worldmodel as swm
from lejudge import JevCost, Constraint

lewm = swm.policy.AutoCostModel("pusht/lewm")
cost = JevCost(
    world_model=lewm,
    probes="pusht/linear@1",
    constraints=[Constraint("Never let the T touch the left or right edge."),
                 Constraint("Keep the T roughly upright.", weight=0.5)],
    lam=1.0, confidence_floor=0.5, elites_per_iter=16,
)
policy = swm.policy.WorldModelPolicy(solver=swm.solver.CEMSolver(model=cost, num_samples=300),
                                     config=swm.policy.PlanConfig(horizon=10, receding_horizon=5))
```

### 6.2 CLI

```
lejudge probes train --env pusht --checkpoint pusht/lewm
lejudge judge-study --env pusht --conditions oracle,keyword,jev,llm-small --n 5000
lejudge plan --env pusht --constraint-set edges --condition jev --episodes 100 --seeds 5
lejudge report --out paper/figures
lejudge demo
```

### 6.3 Space

Type a constraint. See the stock plan and the LeJudge plan side by side, with the imagined frames Jev judged and per-step probability bars. Edit the sentence; the plan changes. A tab shows the exact JSON sent and the call's latency.

## 7. Requirements

**Functional**
- F1 Probes from LeWM latents to symbolic PushT state with reported R² and bucket accuracy.
- F2 Deterministic symbols→words vocabulary; raw numbers never reach Jev.
- F3 Constraint library: 12 constraints × (text, family, oracle checker, 5 paraphrases, 2 near-miss negatives).
- F4 One Jev call per CEM iteration over the elite shortlist; per-step Nouls; aggregation in code by family.
- F5 `JevCost` compatible with `WorldModelPolicy` and `CEMSolver`; confidence gate; λ and weights configurable.
- F6 Baselines: oracle, keyword, LLM-small, all on identical inputs and seeds.
- F7 Evaluation harness producing success, violation, latency, cost, judge accuracy, calibration, paraphrase robustness with bootstrap CIs.
- F8 Response cache keyed by (model version, state hash, question bank version).
- F9 Gradio Space with the side-by-side view.

**Non-functional**
- N1 Single-GPU; no LeWM training.
- N2 Every Jev call logs `response.model`, latency, tokens.
- N3 Constraint text and worded facts are never in the same state field.
- N4 All randomness seeded; all traces saved.

## 8. Success metrics for the launch

| Metric | Target |
|---|---|
| Preprint + Space + repo live the same day | yes |
| LeWM authors or TypeSafe repost or comment | at least one |
| `stable-worldmodel` discussion/PR opened for `JevCost` | yes |
| Space sessions in week one | 2,000 |
| GitHub stars in month one | 300 |

## 9. Risks

| Risk | Mitigation |
|---|---|
| Probes degrade on imagined latents over the horizon | Measure per-step probe error; judge only the first H' steps if needed; report as world-model error |
| Jev misreads spatial vocabulary | Vocabulary designed per the jaggedness guide; judge-only study gates the planning study |
| Keyword matches Jev on hard rules | Expected and stated; the claim rests on paraphrases and soft/implicit constraints |
| Closed model | Cached responses; pinned version; open-model judge as a secondary condition |
| Name | Ask the LeWM authors; fallback name WordCost |

## 10. Open decisions

Tracked in `docs/DECISIONS.md`. Initial: grid granularity (3×3 + edge flags vs 4×4); Score vs Noul for soft constraints; whether the demo runs CEM live or replays cached plans for speed.
