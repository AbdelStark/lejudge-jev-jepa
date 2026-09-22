"""KeywordJudge: regex dictionary over the *canonical* constraint texts.

Rules map a regex on the constraint sentence to a predicate over ``StepFacts``. The
dictionary is written from the canonical texts only and applied verbatim to paraphrases;
when no rule matches a sentence the judge fires nothing (p = 0). That failure to transfer
is the point of the baseline.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from lejudge.judge.base import build_state, plan_questions
from lejudge.types import Constraint, GroundTruthState, JudgeResult, StepFacts

Pred = Callable[[StepFacts, int], bool]  # (facts, t) -> violates?


@dataclass(frozen=True)
class Rule:
    pattern: str
    violates: Pred
    family_hint: str = "never"


def _edge(*names: str) -> Pred:
    s = set(names)
    return lambda f, t: f.block_edge in s


def _cells(field: str, *names: str) -> Pred:
    s = set(names)
    return lambda f, t: getattr(f, field) in s


def _angle(*names: str) -> Pred:
    s = set(names)
    return lambda f, t: f.block_angle in s


RULES: list[Rule] = [
    Rule(r"\bT\b.*\btouch\b.*\bleft or right edge\b", _edge("left edge", "right edge")),
    Rule(r"\bT\b.*\bnever reach the top edge\b", _edge("top edge")),
    Rule(r"\bT\b.*\bout of the centre cell\b", _cells("block", "centre")),
    Rule(r"\bagent\b.*\bnot enter the top row\b", _cells("agent", "top-left", "top-centre", "top-right")),
    Rule(r"\bT\b.*\broughly upright\b", lambda f, t: f.block_angle != "upright", "always"),
    Rule(r"\bflip the T upside down\b", _angle("upside down")),
    Rule(r"\bpush the block while it is on its side\b", lambda f, t: bool(f.contact) and f.block_angle in {"on its side left", "on its side right"}),
    Rule(r"\bApproach the block from below\b", lambda f, t: bool(f.contact) and f.agent in {"top-left", "top-centre", "top-right"}, "always"),
    Rule(r"\bgentle\b.*\bslowly\b", lambda f, t: f.block_speed == "fast", "soft"),
    Rule(r"\bblock in the left half\b", _cells("block", "top-centre", "centre", "bottom-centre", "top-right", "centre-right", "bottom-right"), "always"),
    Rule(r"\btouch the block during the first three steps\b", lambda f, t: bool(f.contact) and t <= 3),
    Rule(r"\bout of all four corners with the block\b", _cells("block", "top-left", "top-right", "bottom-left", "bottom-right")),
]


def match_rule(text: str) -> Rule | None:
    for r in RULES:
        if re.search(r.pattern, text):
            return r
    return None


class KeywordJudge:
    name = "keyword"

    def judge(
        self,
        facts: dict[str, list[StepFacts]],
        constraints: list[Constraint],
        *,
        states: dict[str, list[GroundTruthState]] | None = None,
    ) -> JudgeResult:
        built = build_state(facts, constraints)
        specs = plan_questions(built, constraints)
        p: dict[str, dict[str, list[float]]] = {}
        conf: dict[str, dict[str, float | None]] = {}
        for k, seq in facts.items():
            p[k] = {}
            conf[k] = {}
            for c in constraints:
                rule = match_rule(c.text)
                flags = [bool(rule.violates(f, f.t)) if rule else False for f in seq]
                if c.family == "never":
                    p[k][c.id] = [1.0 if v else 0.0 for v in flags]
                elif c.family == "always":
                    p[k][c.id] = [0.0 if v else 1.0 for v in flags]
                elif c.family == "soft":
                    frac_ok = 1.0 - (sum(flags) / len(flags) if flags else 0.0)
                    level = int(round(frac_ok * 4))
                    vec = [0.0] * 5
                    vec[level] = 1.0
                    p[k][c.id] = vec
                    conf[k][c.id] = 1.0 if rule else 0.0
                    continue
                else:  # temporal_before: no keyword implementation
                    p[k][c.id] = [0.0] * (2 * len(seq))
                conf[k][c.id] = None
        return JudgeResult(p=p, confidence=conf, latency_ms=0.0, input_tokens=0, output_tokens=0, response_model="keyword@1", cache_hit=True, n_calls=0, judge_name=self.name, keys=[s.key for s in specs])
