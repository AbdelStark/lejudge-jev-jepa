"""Canonical state builder (RFC-0003 / RFC-0008).

``constraints`` and ``candidates`` are separate top-level fields. Candidate ids are renamed
``k1..kK`` in the order given (cost order), constraint ids ``c1..cC``. The JSON is
serialised with sorted keys and no floats, so identical content hashes identically.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from lejudge.types import Constraint, StepFacts


@dataclass(frozen=True)
class BuiltState:
    state: dict[str, Any]
    k_map: dict[str, str]  # caller key -> k1..
    c_map: dict[str, str]  # constraint id -> c1..
    steps: dict[str, list[int]]  # k -> list of t present

    @property
    def canonical_json(self) -> str:
        return canonical_json(self.state)

    def inv_k(self) -> dict[str, str]:
        return {v: k for k, v in self.k_map.items()}

    def inv_c(self) -> dict[str, str]:
        return {v: k for k, v in self.c_map.items()}


def canonical_json(obj: Any) -> str:
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def build_state(facts: dict[str, list[StepFacts]], constraints: list[Constraint]) -> BuiltState:
    if not facts:
        raise ValueError("no candidates")
    if not constraints:
        raise ValueError("no constraints")
    k_map = {k: f"k{i}" for i, k in enumerate(facts.keys(), 1)}
    c_map = {c.id: f"c{i}" for i, c in enumerate(constraints, 1)}
    if len(set(c_map)) != len(constraints):
        raise ValueError("duplicate constraint ids in one call")
    cands: dict[str, list[dict[str, Any]]] = {}
    steps: dict[str, list[int]] = {}
    for k, seq in facts.items():
        rows = [f.to_json() for f in seq]
        for r in rows:
            _assert_words_only(r)
        cands[k_map[k]] = rows
        steps[k_map[k]] = [f.t for f in seq]
    state = {
        "constraints": {c_map[c.id]: c.text for c in constraints},
        "candidates": cands,
    }
    return BuiltState(state=state, k_map=k_map, c_map=c_map, steps=steps)


def _assert_words_only(row: dict[str, Any]) -> None:
    for key, val in row.items():
        if key == "t":
            if not isinstance(val, int):
                raise TypeError("t must be an int")
            continue
        if key == "contact":
            if not isinstance(val, bool):
                raise TypeError("contact must be a bool")
            continue
        if not isinstance(val, str):
            raise TypeError(f"fact field {key!r} must be a word, got {type(val).__name__}")
        if any(ch.isdigit() for ch in val):
            raise ValueError(f"fact field {key!r} contains a digit: {val!r}")
