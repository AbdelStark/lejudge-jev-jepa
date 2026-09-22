---
name: lejudge-jev-questions
description: Write, lint and validate Jev (TypeSafe System One) questions, criteria and state layouts for LeJudge. Use this skill whenever you touch anything under lejudge/judge/, the constraint library, question templates, aggregation code, or when a task mentions Jev, Noul, Choice, Score, system_one, jaggedness, state JSON, or "how should I ask the model" — even if the user does not say "Jev" explicitly.
---

# Writing Jev questions for LeJudge

Jev is a decision model, not a chat model. You send a JSON `state` and typed `questions` (Noul → P(true); Choice → option + probabilities + confidence; Score → level + probabilities + confidence). Jev reads literally, cannot count or do arithmetic, degrades with irrelevant state, and can be steered by adversarial text. Every question in this repo follows the rules below. Run `lejudge lint` before committing.

## Rules

1. **One judgment per question.** If you need "A and B", ask two Nouls and combine in code.
2. **Literal instructions.** Name the exact state fields the question depends on (`candidates.k1` at `t=2`, `constraints.c1`). Put boundary cases in `criteria`.
3. **True means present.** A Noul's true must mean the condition is present. Never "is it NOT touching".
4. **No numbers, counts, dates, or comparisons.** Words only. Aggregation (`max` over steps, first-index, expectation) is Python.
5. **Separate fields for trust domains.** Owner text under `constraints`; code-generated facts under `candidates`. Never inline constraint text into a question or a fact.
6. **Small state.** Only the fields the question needs. Cap questions per call at 512; split otherwise.
7. **Never rely on structural invariants.** A Noul and a yes/no Choice on the "same" question are not comparable; do not carry thresholds across primitives. `P(noul)` and `1 − P(not noul)` are not interchangeable.
8. **Criteria extend instructions.** Same polarity, same vocabulary; if you find yourself explaining what you "really meant", that explanation belongs in `criteria`.
9. **Version the bank.** Any wording change bumps the bank version; results are never mixed across versions.

## Templates (copy, then fill)

```python
Noul(
  instructions=f"At step t={t} of candidate {k}, does the block or agent violate constraint {c} exactly as written in constraints.{c}?",
  criteria=f"Read constraints.{c} literally. Use only the facts in candidates.{k} at step t={t}: block, block_edge, block_angle, agent, contact, block_speed. Answer true only if those facts show the forbidden condition at this step. If the facts do not mention what the constraint needs, answer false.",
)
```

```python
Score(
  instructions=f"Overall across its steps, how well does candidate {k} respect constraint {c} as written in constraints.{c}?",
  levels=["not at all", "poorly", "partly", "mostly", "fully"],
  criteria="'fully': every step's block_speed is still or slow and contact is gentle. 'not at all': most steps show fast block_speed. Use only the listed facts.",
)
```

## Workflow

1. Draft the question in `references/bank.md` under the right family.
2. Run `lejudge lint` (checks rules 1–8 mechanically where possible).
3. Add a golden case to `tests/judge/golden/` with a hand-built state and expected direction (`> 0.7` or `< 0.3`).
4. Run the live golden test once (`LEJUDGE_MODE=live`), which populates the cache; CI then runs offline.
5. Check consistency: 3 repeats, std-dev < 0.05 (`lejudge consistency --bank <version>`).

## When Jev is the wrong tool

- Extracting a value from text → regex or a generative model, then Jev picks among candidates.
- Anything a parser can compute → compute it.
- Generating an explanation → templates in code.

## References

- `references/jaggedness-summary.md` — condensed failure modes from TypeSafe's Jev 1.13 page with the fix for each.
- `references/bank.md` — the current question bank with versions.
- TypeSafe docs: https://docs.typesafe.ai/primitives · https://docs.typesafe.ai/model-jaggedness/jev-1.13.md · https://docs.typesafe.ai/patterns
