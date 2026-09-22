"""Load, validate and hash the constraint library."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from lejudge.constraints.oracles import ViolationTrace, get_oracle
from lejudge.types import Constraint, GroundTruthState
from lejudge.vocab.pusht import Vocab

TEXT_VARIANTS = ("canonical", "p1", "p2", "p3", "p4", "p5", "n1", "n2")


@dataclass(frozen=True)
class Library:
    version: str
    constraints: tuple[Constraint, ...]
    sets: dict[str, tuple[str, ...]]
    sha256: str

    def get(self, cid: str) -> Constraint:
        for c in self.constraints:
            if c.id == cid:
                return c
        raise KeyError(cid)

    def set(self, name: str) -> list[Constraint]:
        return [self.get(cid) for cid in self.sets[name]]

    def ids(self) -> list[str]:
        return [c.id for c in self.constraints]

    def variants(self, cid: str) -> dict[str, Constraint]:
        """Canonical text plus paraphrases ``p1..p5`` and negatives ``n1..n2``."""
        c = self.get(cid)
        out = {"canonical": c}
        for i, p in enumerate(c.paraphrases, 1):
            out[f"p{i}"] = c.with_text(p, suffix=f"@p{i}")
        for i, n in enumerate(c.negatives, 1):
            out[f"n{i}"] = c.with_text(n, suffix=f"@n{i}")
        return out


def _pkg_path(env: str) -> Path:
    return Path(str(resources.files("lejudge.constraints").joinpath(f"{env}.yaml")))


def load_library(env: str = "pusht") -> Library:
    path = Path(env) if env.endswith(".yaml") else _pkg_path(env)
    text = path.read_text()
    raw: dict[str, Any] = yaml.safe_load(text)
    constraints = []
    for item in raw["constraints"]:
        c = Constraint(
            text=str(item["text"]),
            family=str(item["family"]),
            id=str(item["id"]),
            weight=float(item.get("weight", 1.0)),
            oracle=str(item["oracle"]),
            params=dict(item.get("params", {}) or {}),
            paraphrases=tuple(str(p) for p in item.get("paraphrases", [])),
            negatives=tuple(str(n) for n in item.get("negatives", [])),
        )
        if len(c.paraphrases) != 5:
            raise ValueError(f"{c.id}: expected 5 paraphrases, got {len(c.paraphrases)}")
        if len(c.negatives) != 2:
            raise ValueError(f"{c.id}: expected 2 negatives, got {len(c.negatives)}")
        get_oracle(c.oracle or "")  # raises if unknown
        constraints.append(c)
    ids = [c.id for c in constraints]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate constraint ids")
    sets = {str(k): tuple(str(x) for x in v) for k, v in raw.get("sets", {}).items()}
    for name, members in sets.items():
        for m in members:
            if m not in ids:
                raise ValueError(f"set {name!r} references unknown constraint {m!r}")
    return Library(
        version=str(raw["version"]),
        constraints=tuple(constraints),
        sets=sets,
        sha256=hashlib.sha256(text.encode()).hexdigest(),
    )


def check(constraint: Constraint, seq: list[GroundTruthState], vocab: Vocab) -> ViolationTrace:
    """Run a constraint's oracle on a state sequence (index 0 = current state)."""
    if not constraint.oracle:
        raise ValueError(f"constraint {constraint.id} has no oracle")
    return get_oracle(constraint.oracle)(seq, constraint.params, vocab)
