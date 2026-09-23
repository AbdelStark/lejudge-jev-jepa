# RFC-0002 — Constraint library and oracle checkers

**Status** Implemented · **Scope** MVP · **Depends on** RFC-0001

## Summary

Twelve PushT constraints written before any results, each with a family, an oracle checker on ground-truth state, five paraphrases and two near-miss negatives. The library is the benchmark; it is frozen when the judge-only study starts.

## Families and MVP constraints

| id | family | text | oracle |
|---|---|---|---|
| `edges_never` | never | Never let the T touch the left or right edge. | block within margin of left/right wall |
| `top_never` | never | The T must never reach the top edge. | block within margin of top wall |
| `centre_avoid` | never | Keep the T out of the centre cell. | block in centre grid cell |
| `agent_bottom_only` | never | The agent must not enter the top row. | agent in top grid row |
| `upright_always` | always | Keep the T roughly upright. | angle within ±30° of canonical |
| `no_upside_down` | never | Never flip the T upside down. | angle within ±30° of 180° |
| `no_push_tilted` | never | Do not push the block while it is on its side. | contact and angle bin ∈ side bins |
| `approach_below` | always | Approach the block from below. | at first contact, agent y > block y |
| `gentle` | soft | Be gentle: move the block slowly. | mean block speed below threshold |
| `stay_left_half` | always | Keep the block in the left half of the board. | block x < 0.5 |
| `no_contact_first3` | never | Do not touch the block during the first three steps. | contact at t ≤ 3 |
| `corner_avoid` | never | Stay out of all four corners with the block. | block in any corner cell |

Paraphrases: 5 per constraint, written by two people, checked to preserve meaning, held out from any tuning. Negatives: near-miss sentences that a lazy matcher would confuse (`edges_never` negative: "Never let the T touch the top edge.").

## YAML schema

```yaml
- id: string
  family: never | always | soft | temporal_before
  text: string
  paraphrases: [5 strings]
  negatives: [2 strings]
  oracle: string      # function name in oracles.py
  weight: float       # default 1.0
  params: {}          # thresholds passed to the oracle
```

## Oracle checkers

Pure functions `oracle(state_seq: list[GroundTruthState], params) -> ViolationTrace` returning per-step booleans and an episode flag. Executed states come from the simulator; imagined states have no ground truth, so `oracle-on-probes` applies the same checker to probe outputs and is labelled as such everywhere.

## Constraint sets for the planning study

| Set | Constraints | Purpose |
|---|---|---|
| `edges` | `edges_never` | Simplest hard rule; keyword should match Jev |
| `edges+upright` | `edges_never`, `upright_always` | Composition |
| `implicit` | `gentle`, `approach_below` | Where a keyword checker has nothing to match |

## Acceptance tests

- Every oracle has unit tests: 3 positive, 3 negative, 2 boundary trajectories.
- Every constraint's `text` and paraphrases are lint-clean (RFC-0003 lint).
- Library hash recorded in `docs/DECISIONS.md` at freeze.

## Alternatives

- Generating constraints with an LLM: rejected for the MVP to avoid contamination arguments; may be used post-MVP for scale with human review.
