"""Question bank ``lejudge.pusht@0.1.0`` (RFC-0003) and its lint.

Templates render to ``typesafe_sdk`` questions. Any wording change bumps ``BANK_VERSION``;
results are never mixed across versions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

BANK_VERSION = "lejudge.pusht@0.1.0"
MAX_QUESTIONS_PER_CALL = 512
SOFT_LEVELS = ("not at all", "poorly", "partly", "mostly", "fully")
FACT_FIELDS = ("block", "block_edge", "block_angle", "agent", "contact", "block_speed")


@dataclass(frozen=True)
class QuestionSpec:
    key: str
    kind: str  # "noul" | "score"
    instructions: str
    criteria: Any  # dict(true=..., false=...) for noul; list of level descriptions for score
    k: str
    c: str
    t: int | None
    family: str


def never_question(k: str, c: str, t: int) -> QuestionSpec:
    return QuestionSpec(
        key=f"{k}_{c}_{t}",
        kind="noul",
        instructions=(
            f"At step t={t} of candidate {k}, does the block or agent violate constraint {c} "
            f"exactly as written in constraints.{c}?"
        ),
        criteria={
            "true": (
                f"Read constraints.{c} literally. Use only the facts in candidates.{k} at step t={t}: "
                "block, block_edge, block_angle, agent, contact, block_speed. The facts at this step "
                "show the forbidden condition."
            ),
            "false": (
                f"The facts in candidates.{k} at step t={t} do not show the forbidden condition, "
                f"or do not mention what constraints.{c} needs."
            ),
        },
        k=k,
        c=c,
        t=t,
        family="never",
    )


def always_question(k: str, c: str, t: int) -> QuestionSpec:
    return QuestionSpec(
        key=f"{k}_{c}_{t}",
        kind="noul",
        instructions=(
            f"At step t={t} of candidate {k}, are the facts consistent with constraint {c} "
            f"as written in constraints.{c}?"
        ),
        criteria={
            "true": (
                f"The facts in candidates.{k} at step t={t} (block, block_edge, block_angle, agent, "
                f"contact, block_speed) satisfy the requirement in constraints.{c}, or do not mention "
                "what the requirement needs."
            ),
            "false": (
                f"The facts in candidates.{k} at step t={t} contradict the requirement in constraints.{c}."
            ),
        },
        k=k,
        c=c,
        t=t,
        family="always",
    )


def soft_question(k: str, c: str) -> QuestionSpec:
    return QuestionSpec(
        key=f"{k}_{c}",
        kind="score",
        instructions=(
            f"Overall across its steps, how well does candidate {k} respect constraint {c} "
            f"as written in constraints.{c}?"
        ),
        criteria=[
            f"not at all: the facts of candidates.{k} (block_speed, contact, block, block_angle, agent) contradict constraints.{c} at most steps.",
            f"poorly: the facts of candidates.{k} contradict constraints.{c} at many steps.",
            f"partly: the facts of candidates.{k} match constraints.{c} at about half of the steps.",
            f"mostly: the facts of candidates.{k} match constraints.{c} at nearly every step.",
            f"fully: the facts of candidates.{k} match constraints.{c} at every step.",
        ],
        k=k,
        c=c,
        t=None,
        family="soft",
    )


def temporal_questions(k: str, c: str, t: int) -> list[QuestionSpec]:
    return [
        QuestionSpec(
            key=f"{k}_{c}_A_{t}",
            kind="noul",
            instructions=f"By step t={t} of candidate {k}, has the first event named in constraints.{c} happened according to the facts so far?",
            criteria={
                "true": f"Some step of candidates.{k} up to t={t} (block, block_edge, block_angle, agent, contact, block_speed) shows the first event named in constraints.{c}.",
                "false": f"No step of candidates.{k} up to t={t} shows the first event named in constraints.{c} in its facts.",
            },
            k=k,
            c=c,
            t=t,
            family="temporal_before",
        ),
        QuestionSpec(
            key=f"{k}_{c}_B_{t}",
            kind="noul",
            instructions=f"By step t={t} of candidate {k}, has the second event named in constraints.{c} happened according to the facts so far?",
            criteria={
                "true": f"Some step of candidates.{k} up to t={t} (block, block_edge, block_angle, agent, contact, block_speed) shows the second event named in constraints.{c}.",
                "false": f"No step of candidates.{k} up to t={t} shows the second event named in constraints.{c} in its facts.",
            },
            k=k,
            c=c,
            t=t,
            family="temporal_before",
        ),
    ]


def questions_for(family: str, k: str, c: str, steps: list[int]) -> list[QuestionSpec]:
    if family == "never":
        return [never_question(k, c, t) for t in steps]
    if family == "always":
        return [always_question(k, c, t) for t in steps]
    if family == "soft":
        return [soft_question(k, c)]
    if family == "temporal_before":
        return [q for t in steps for q in temporal_questions(k, c, t)]
    raise ValueError(family)


def to_sdk(spec: QuestionSpec) -> Any:
    from typesafe_sdk import Noul, Score

    if spec.kind == "noul":
        return Noul(instructions=spec.instructions, criteria=dict(spec.criteria))
    return Score(instructions=spec.instructions, criteria=list(spec.criteria))


def spec_json(spec: QuestionSpec) -> dict[str, Any]:
    return {"key": spec.key, "kind": spec.kind, "instructions": spec.instructions, "criteria": spec.criteria}


# ---------------------------------------------------------------------------- lint
_ARITH = re.compile(r"(\bcount\b|\bhow many\b|\bsum\b|\bmore than\b|\bless than\b|\bgreater\b|\bfewer\b|\btimes\b|[+*/<>=]|\d+\s*%)", re.I)
_NEGATED = re.compile(r"\b(does not|doesn't|isn't|is not|never|no longer|not)\b", re.I)


def _word_count(s: str) -> int:
    return len(re.findall(r"\S+", s))


def lint_question(spec: QuestionSpec, allow_numbers: bool = False) -> list[str]:
    problems: list[str] = []
    instr = spec.instructions
    crit_text = " ".join(spec.criteria.values()) if isinstance(spec.criteria, dict) else " ".join(spec.criteria)
    # rule 4: no arithmetic, counting, numeric comparison (t=<n> and k<n>/c<n> are keys, not numbers)
    stripped = re.sub(r"\b(t=\d+|k\d+|c\d+)\b", "", instr + " " + crit_text)
    if _ARITH.search(stripped):
        problems.append(f"{spec.key}: arithmetic/counting/comparison wording")
    if not allow_numbers and re.search(r"\d", stripped):
        problems.append(f"{spec.key}: bare number in question text")
    # rule 3: no negated nouls (true must mean present)
    if spec.kind == "noul" and _NEGATED.search(re.sub(r"\bnot\b(?= mention)", "", instr)):
        problems.append(f"{spec.key}: negated Noul instruction")
    # criteria mention state fields
    if not any(f in crit_text for f in FACT_FIELDS):
        problems.append(f"{spec.key}: criteria do not name any state field")
    if _word_count(instr) > 40:
        problems.append(f"{spec.key}: instruction longer than 40 words")
    if _word_count(crit_text) > 60 * (5 if spec.kind == "score" else 2):
        problems.append(f"{spec.key}: criteria too long")
    return problems


def lint_constraint_text(text: str) -> list[str]:
    problems: list[str] = []
    if len(text) > 200:
        problems.append("text longer than 200 characters")
    if re.search(r"\d", text) and not re.search(r"\b(one|two|three)\b", text):
        problems.append(f"digit in constraint text: {text!r}")
    return problems


def lint_bank(vocab_words: dict[str, tuple[str, ...]]) -> list[str]:
    """Render every template with every vocabulary word; collect lint problems."""
    problems: list[str] = []
    for fam in ("never", "always", "soft", "temporal_before"):
        for q in questions_for(fam, "k1", "c1", [1, 2]):
            problems.extend(lint_question(q))
            try:
                q.instructions.format()
            except (KeyError, IndexError, ValueError) as e:  # pragma: no cover
                problems.append(f"{q.key}: template does not render: {e}")
    for field, ws in vocab_words.items():
        for w in ws:
            if re.search(r"\d", w):
                problems.append(f"vocab word with digit in {field}: {w!r}")
    return problems
