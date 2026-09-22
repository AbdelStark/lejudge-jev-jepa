# Pre-registration — LeJudge planning study

Written before the planning study runs. Hash recorded in `docs/DECISIONS.md`; changes after the
freeze are logged there, never silently applied.

## Hypotheses
- H1 LeJudge (Jev) episode violation rate is within 10 points of oracle-on-probes on all three constraint sets.
- H2 LeJudge success is within 10 points of unconstrained LeWM.
- H3 Jev's judge accuracy on the five held-out paraphrases is within 5 points of its accuracy on the canonical text; the keyword checker drops by at least 20 points on the same paraphrases.
- H4 The small LLM judge matches Jev's violation rate within 5 points but its per-judgment latency is at least 10× Jev's.

## Primary metrics
Episode violation rate (oracle checker on executed ground-truth states); success rate (PushT `eval_state` criterion within the budget); planning time p50 per replanning step; judge calls and input tokens per episode.

## Design
Conditions {LeWM, oracle-on-probes, keyword, Jev} × constraint sets {spatial = [centre_avoid], spatial+temporal = [centre_avoid, no_contact_first3], implicit = [gentle, approach_below]} × seeds {0, 1, 2} × 30 episodes per cell. LLM-small (local Qwen2.5-7B-Instruct via Ollama) runs on the `spatial` set, seed 0, 10 episodes, and is reported as a subset. Episodes start from (state, goal) pairs drawn from held-out expert PushT episodes with the le-wm protocol: goal = the state 25 env steps later, budget 50 env steps, episode ends at success. Planner: CEM 300 samples × 30 iterations, top-30 elites, horizon 5 blocks of 5 env steps, receding horizon 5 blocks, 3 context frames. LeJudge: K = 16, λ = 1.0, τ = 0.5, judge the last 3 CEM iterations, all horizon steps. Conditions share seeds, start states and CEM sampler seeds. Violations are checked by the oracle on executed ground-truth states at the planner's cadence (one state per 5-step action block, plus the final state).

The RFC-0002 sets `edges` and `edges+upright` were replaced before any study run: on the M0 episodes unconstrained LeWM violated `edges_never` in 0 of 50 episodes (the expert start/goal windows almost never bring the block to a side wall) and `upright_always` in 92 % (the block is rarely upright at the start and cannot be righted within the budget). The study sets were chosen from constraints whose M0 base violation rate lies between 30 % and 80 % so that task and constraint conflict: centre_avoid 0.54, no_contact_first3 0.76, gentle 0.30, approach_below 0.50 (history-3 M0 run, planner cadence).

## Analysis
Rates with 95 % bootstrap CIs (10,000 resamples, `numpy.random.default_rng(0)`). Condition vs LeWM paired by (constraint set, seed, episode): Wilcoxon signed-rank; Holm across the three sets; risk differences with bootstrap CIs. Judge-only: precision/recall/F1 at 0.5, AUROC, ECE with 10 equal-mass bins, reliability diagrams, per-constraint and per-paraphrase accuracy, near-miss negative false-positive rate on items whose canonical condition is present, consistency as per-question std-dev over 3 repeats.

## Ablations (exploratory, labelled as such)
On the `spatial` set, seed 0, 20 episodes each: λ ∈ {0.25, 0.5, 1, 2, 4}; K ∈ {4, 16, 64}; `final_only`; vocab `pusht@2`; `steps = 2`; hard rejection; MLP probe.

## Exclusions
None. Episodes with judge API failures are kept (penalty 0, `held` flag) and counted.

## Versions
LeWM checkpoint `quentinll/lewm-pusht` revision `22b330c28c27ead4bfd1888615af1340e3fe9052`; stable-worldmodel commit `4821c8e6a3f0f83b7e6a80da3a757e026ea9026b`; probes `pusht/linear@1`; vocab `pusht@1`; constraint library `pusht-lib@1`; bank `lejudge.pusht@0.1.0`; Jev requested as `jev-latest`, pinned to responses whose model starts with `jev-1.13`.

## Amendments after the hash (logged in DECISIONS)
- 2026-09-22: JevCost judges every candidate (step-level de-duplication and memo for hard families; K-shortlist with mean prior for soft families) on CEM iterations 0, 5, 10, 15, 20, 25 and 29 (`every_k`), replacing the K = 16 shortlist judged on the last 3 iterations. Reason: the oracle judge showed the original mechanism could not change plans. Seeds, episodes, sets, λ, τ and the analysis are unchanged.
