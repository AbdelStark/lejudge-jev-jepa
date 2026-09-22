# Question bank — `lejudge.pusht@0.1.0`

State layout: see RFC-0003. All keys render from templates; this file is the human-readable source of truth.

## never (per k, c, t)
Instruction: At step t={t} of candidate {k}, does the block or agent violate constraint {c} exactly as written in constraints.{c}?
Criteria: Read constraints.{c} literally. Use only the facts in candidates.{k} at step t={t}: block, block_edge, block_angle, agent, contact, block_speed. Answer true only if those facts show the forbidden condition at this step. If the facts do not mention what the constraint needs, answer false.

## always (per k, c, t)
Instruction: At step t={t} of candidate {k}, are the facts consistent with constraint {c} as written in constraints.{c}?
Criteria: Answer true when the facts satisfy the requirement. Answer false when they contradict it. If the facts do not mention what the constraint needs, answer true.

## soft (per k, c) — Score
Instruction: Overall across its steps, how well does candidate {k} respect constraint {c} as written in constraints.{c}?
Levels: not at all · poorly · partly · mostly · fully
Criteria: Judge from block_speed and contact across the steps of candidates.{k}. 'fully' when the facts match the constraint at every step; 'not at all' when they contradict it at most steps.

## temporal_before (per k, c, t) — two Nouls (not in MVP library)
A: By step t={t} of candidate {k}, has the first event named in constraints.{c} happened according to the facts so far?
B: By step t={t} of candidate {k}, has the second event named in constraints.{c} happened according to the facts so far?

## Changelog
- 0.1.0 — initial bank.
