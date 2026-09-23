"""Judge interface shared by Jev and the baselines."""

from __future__ import annotations

import math
from typing import Any, Protocol

from lejudge.judge.bank import QuestionSpec, questions_for
from lejudge.judge.state import BuiltState, build_state
from lejudge.types import Constraint, GroundTruthState, JudgeResult, StepFacts


class Judge(Protocol):
    """Every judge receives identical inputs and returns probabilities keyed by the caller's ids.

    A judge may set ``local = True`` when it is free and deterministic (oracle, keyword); the cost
    module then judges every candidate's full sequence directly instead of memoising step facts.
    """

    name: str

    def judge(
        self,
        facts: dict[str, list[StepFacts]],
        constraints: list[Constraint],
        *,
        states: dict[str, list[GroundTruthState]] | None = None,
    ) -> JudgeResult: ...


def plan_questions(built: BuiltState, constraints: list[Constraint]) -> list[QuestionSpec]:
    """All question specs for one state, in a deterministic order."""
    specs: list[QuestionSpec] = []
    for k in built.state["candidates"]:
        for c in constraints:
            specs.extend(questions_for(c.family, k, built.c_map[c.id], built.steps[k]))
    return specs


def empty_result(
    facts: dict[str, list[StepFacts]], constraints: list[Constraint], name: str, failed: bool = True
) -> JudgeResult:
    p: dict[str, dict[str, list[float]]] = {}
    conf: dict[str, dict[str, float | None]] = {}
    for k, seq in facts.items():
        p[k] = {}
        conf[k] = {}
        for c in constraints:
            n = (
                5
                if c.family == "soft"
                else (2 * len(seq) if c.family == "temporal_before" else len(seq))
            )
            p[k][c.id] = [math.nan] * n
            conf[k][c.id] = None
    return JudgeResult(
        p=p,
        confidence=conf,
        latency_ms=0.0,
        input_tokens=0,
        output_tokens=0,
        response_model="",
        cache_hit=False,
        n_calls=0,
        judge_name=name,
        failed=failed,
    )


def assemble(
    built: BuiltState,
    constraints: list[Constraint],
    specs: list[QuestionSpec],
    answers: dict[str, dict[str, Any]],
    *,
    latency_ms: float,
    input_tokens: int,
    output_tokens: int,
    response_model: str,
    cache_hit: bool,
    n_calls: int,
    name: str,
) -> JudgeResult:
    """Turn ``{key: answer_dict}`` into a ``JudgeResult`` keyed by caller ids.

    Answer dicts are ``{"type": "noul", "noul": p}`` or ``{"type": "score", "probabilities":
    {"0": p0, ...}, "confidence": c}``. Missing keys become NaN and mark the result failed.
    """
    inv_k, inv_c = built.inv_k(), built.inv_c()
    fam = {c.id: c.family for c in constraints}
    p: dict[str, dict[str, list[float]]] = {
        inv_k[k]: {c.id: [] for c in constraints} for k in built.state["candidates"]
    }
    conf: dict[str, dict[str, float | None]] = {
        inv_k[k]: {c.id: None for c in constraints} for k in built.state["candidates"]
    }
    missing = 0
    # group specs by (k, c) preserving order (steps ascending; A before B for temporal)
    for spec in specs:
        kk, cc = inv_k[spec.k], inv_c[spec.c]
        a = answers.get(spec.key)
        if a is None:
            missing += 1
            if spec.kind == "score":
                p[kk][cc] = [math.nan] * 5
            else:
                p[kk][cc].append(math.nan)
            continue
        if spec.kind == "score":
            probs = a.get("probabilities", {})
            vec = [float(probs.get(str(i), probs.get(i, math.nan))) for i in range(5)]
            p[kk][cc] = vec
            conf[kk][cc] = float(a["confidence"]) if a.get("confidence") is not None else None
        else:
            p[kk][cc].append(float(a.get("noul", math.nan)))
    # temporal: reorder to [A_1..A_H, B_1..B_H]
    for c in constraints:
        if fam[c.id] == "temporal_before":
            for kk in p:
                seq = p[kk][c.id]
                p[kk][c.id] = seq[0::2] + seq[1::2]
    return JudgeResult(
        p=p,
        confidence=conf,
        latency_ms=latency_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        response_model=response_model,
        cache_hit=cache_hit,
        n_calls=n_calls,
        judge_name=name,
        keys=[s.key for s in specs],
        failed=missing > 0,
    )


__all__ = ["Judge", "assemble", "build_state", "empty_result", "plan_questions"]
