# Pre-registration — LeJudge planning study (template)

Hash this file (`sha256sum docs/PREREG.md`) and record the hash in DECISIONS before running the planning study.

## Hypotheses
- H1 LeJudge episode violation rate within 10 points of oracle-on-probes on all three constraint sets.
- H2 LeJudge success within 10 points of unconstrained LeWM.
- H3 Judge accuracy on paraphrases within 5 points of canonical for Jev; keyword drops ≥ 20 points.
- H4 LLM-small matches Jev's violation rate within 5 points but planning time ≥ 10× and cost ≥ 20×.

## Primary metrics
Episode violation rate (oracle on executed states); success rate (env criterion); planning time p50 per step; USD per episode.

## Design
Conditions {LeWM, oracle-on-probes, keyword, jev, llm-small} × sets {edges, edges+upright, implicit} × seeds {0..4} × 100 episodes. λ = 1.0; K = 16; τ = 0.5; H = 10; receding 5; CEM 300 × 3.

## Analysis
Bootstrap CIs (10,000); paired Wilcoxon; Holm across sets; risk differences with CIs.

## Ablations (exploratory, labelled as such)
λ sweep; K ∈ {4,16,64}; final_only; vocab pusht@2; steps H/2; hard rejection.

## Exclusions
None. Episodes with judge API failures are kept (penalty 0, held flag) and counted.

## Versions
LeWM checkpoint revision: …  probes: pusht/linear@1  vocab: pusht@1  bank: lejudge.pusht@0.1.0  jev: jev-1.13  swm: …
