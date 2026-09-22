# Glossary

- **LeWM / LeWorldModel** — 15M-parameter JEPA world model trained end to end from pixels (Maes, Le Lidec, Scieur, LeCun, Balestriero, 2026).
- **JEPA** — Joint-Embedding Predictive Architecture: predict future embeddings, not pixels.
- **CEM** — Cross-Entropy Method; sampling-based trajectory optimiser used for planning.
- **MPC / receding horizon** — plan H steps, execute K, replan.
- **Cost A** — LeWM's terminal latent goal distance `‖ẑ_H − z_g‖²`.
- **Penalty** — LeJudge's constraint term added to cost A.
- **Probe** — linear/MLP head from latent to symbolic state.
- **Vocabulary** — deterministic mapping from symbols to words.
- **Facts / StepFacts** — worded per-step description of an imagined state.
- **Constraint family** — never · always · soft · temporal_before; decides question template and aggregation.
- **Oracle** — checker on ground-truth simulator state; `oracle-on-probes` when applied to probe outputs.
- **Jev** — TypeSafe's System One model; typed Choice/Score/Noul answers with probabilities.
- **Noul** — yes/no question returning P(true).
- **Confidence gate** — hold (penalty 0) when the judge is unsure.
- **Bank** — versioned set of question templates.
- **Trace** — JSONL record of one CEM iteration.
