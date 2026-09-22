# LeJudge — a cost module you program in English

**LeJudge** adds a natural-language cost module to JEPA world-model planning. LeWorldModel (LeWM) imagines latent futures; LeWM's own linear probes turn each imagined state into a few worded facts; Jev (TypeSafe's System One model) answers typed yes/no questions about those facts; code folds the calibrated probabilities into the CEM cost. The planner obeys constraints written in plain English, inside a sub-second MPC loop, with no LLM and no generated text.

> Imagine → describe → judge → decide.

## This package

| Path | What it is |
|---|---|
| `PRD.md` | Product requirements for the MVP |
| `SPEC.md` | Technical specification: architecture, interfaces, data formats, budgets |
| `rfcs/` | Eight RFCs that together implement the MVP; each is independently reviewable |
| `AGENTS.md` | Operating manual for the coding agents that build this repo |
| `skills/` | Custom skills the agents load: writing Jev questions, working with stable-worldmodel, running experiments |
| `docs/` | Glossary and decision log template |

## MVP in one paragraph

PushT only. Twelve natural-language constraints across three families (spatial exclusion, orientation, soft/implicit), each with an oracle checker and five paraphrases. Linear probes from LeWM latents to agent position, block position, block angle and contact. A `JevCost` that plugs into `stable-worldmodel`'s `WorldModelPolicy`. Five conditions: unconstrained LeWM, oracle, keyword checker, LeJudge (Jev), and one small LLM judge. A judge-only study and a planning study with bootstrap CIs. A Gradio Space showing the plan change when the sentence changes. Everything cached so the paper's plots regenerate without API spend.

## Non-goals for the MVP

Other environments, language goals by retrieval, LeWM training, real robots, reasoning-LLM judges. Each has an RFC stub or a line in `SPEC.md` § Post-MVP so the design does not paint us into a corner.

## Sources

LeWM paper https://arxiv.org/html/2603.19312v3 · repo https://github.com/lucas-maes/le-wm · checkpoints https://huggingface.co/collections/quentinll/lewm · stable-worldmodel https://github.com/galilai-group/stable-worldmodel · TypeSafe docs https://docs.typesafe.ai/introduction · Jev jaggedness https://docs.typesafe.ai/model-jaggedness/jev-1.13.md

Prepared 2026-09-22.
