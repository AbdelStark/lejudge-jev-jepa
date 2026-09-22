"""Core dataclasses shared across the package.

Rule 1 of AGENTS.md: Jev reads words, never numbers. ``StepFacts`` therefore holds only
closed-vocabulary strings and one boolean; anything numeric lives in ``SymbolicState`` or
``GroundTruthState`` and is bucketed by ``lejudge.vocab`` before it reaches a state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

ARENA_PX = 512.0
"""PushT window size in pixels; every coordinate is normalised by this."""
BLOCK_SCALE = 30.0 / ARENA_PX
"""Default PushT block scale (``variation_space.block.scale`` = 30 px), normalised."""
AGENT_RADIUS = 0.375 * 40.0 / ARENA_PX
"""Default PushT agent radius (``add_circle``: 0.375 × scale 40 px), normalised."""
CONTACT_TOLERANCE = 0.006
"""Agent disc within this distance (≈3 px) of the T polygon counts as contact (DECISIONS 2026-09-22)."""


@dataclass(frozen=True)
class GroundTruthState:
    """One PushT state in normalised coordinates ``[0, 1]^2`` (origin top-left, y down).

    ``block_angle`` is the pymunk body angle in radians wrapped to ``[0, 2π)``.
    ``contact`` is True when the simulator reported agent/block contact points.
    """

    agent_xy: tuple[float, float]
    block_xy: tuple[float, float]
    block_angle: float
    contact: bool

    @classmethod
    def from_env(cls, state: np.ndarray, contact: bool | int | float | None = None, arena: float = ARENA_PX) -> GroundTruthState:
        """Build from the env's 7-d ``state`` (ax, ay, bx, by, angle, vx, vy).

        ``contact`` defaults to the geometric test (agent disc within ``CONTACT_TOLERANCE`` of the
        T polygon at this frame), which is the definition used everywhere in the repo; pass a
        value to override (e.g. the simulator's per-step contact counter).
        """
        s = np.asarray(state, dtype=np.float64).reshape(-1)
        axy = (float(s[0]) / arena, float(s[1]) / arena)
        bxy = (float(s[2]) / arena, float(s[3]) / arena)
        ang = float(s[4]) % (2 * math.pi)
        if contact is None:
            c = bool(geometric_contact(np.array(axy), np.array(bxy), np.array(ang)))
        else:
            c = bool(contact > 0) if not isinstance(contact, bool) else contact
        return cls(agent_xy=axy, block_xy=bxy, block_angle=ang, contact=c)

    def to_json(self) -> dict[str, Any]:
        return {
            "agent_xy": [round(self.agent_xy[0], 5), round(self.agent_xy[1], 5)],
            "block_xy": [round(self.block_xy[0], 5), round(self.block_xy[1], 5)],
            "block_angle": round(self.block_angle, 5),
            "contact": self.contact,
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> GroundTruthState:
        return cls(
            agent_xy=(float(d["agent_xy"][0]), float(d["agent_xy"][1])),
            block_xy=(float(d["block_xy"][0]), float(d["block_xy"][1])),
            block_angle=float(d["block_angle"]),
            contact=bool(d["contact"]),
        )


@dataclass
class SymbolicState:
    """Batched probe outputs. Leading dims are arbitrary; last dims as documented.

    ``agent_xy``/``block_xy``: ``[..., 2]`` normalised; ``block_angle``: ``[...]`` radians;
    ``contact_logit``: ``[...]`` (positive means contact).
    """

    agent_xy: np.ndarray
    block_xy: np.ndarray
    block_angle: np.ndarray
    contact_logit: np.ndarray

    def __post_init__(self) -> None:
        self.agent_xy = np.asarray(self.agent_xy, dtype=np.float64)
        self.block_xy = np.asarray(self.block_xy, dtype=np.float64)
        self.block_angle = np.asarray(self.block_angle, dtype=np.float64) % (2 * math.pi)
        self.contact_logit = np.asarray(self.contact_logit, dtype=np.float64)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.block_angle.shape)

    def __getitem__(self, idx: Any) -> SymbolicState:
        return SymbolicState(self.agent_xy[idx], self.block_xy[idx], self.block_angle[idx], self.contact_logit[idx])

    def to_ground_truth(self, contact_band: float = 0.0) -> list[GroundTruthState]:
        """Flatten a 1-d symbolic sequence into ``GroundTruthState`` objects (oracle-on-probes)."""
        assert self.block_angle.ndim == 1, "to_ground_truth expects a 1-d sequence"
        out = []
        for i in range(self.block_angle.shape[0]):
            out.append(
                GroundTruthState(
                    agent_xy=(float(self.agent_xy[i, 0]), float(self.agent_xy[i, 1])),
                    block_xy=(float(self.block_xy[i, 0]), float(self.block_xy[i, 1])),
                    block_angle=float(self.block_angle[i]),
                    contact=bool(self.contact_logit[i] > 0),
                )
            )
        return out

    @classmethod
    def from_ground_truth(cls, seq: list[GroundTruthState]) -> SymbolicState:
        return cls(
            agent_xy=np.array([s.agent_xy for s in seq]),
            block_xy=np.array([s.block_xy for s in seq]),
            block_angle=np.array([s.block_angle for s in seq]),
            contact_logit=np.array([10.0 if s.contact else -10.0 for s in seq]),
        )


@dataclass(frozen=True)
class StepFacts:
    """Worded description of one imagined (or executed) step. Words only."""

    t: int
    block: str
    block_edge: str
    block_angle: str
    agent: str
    contact: bool | None
    block_speed: str

    def to_json(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "t": self.t,
            "block": self.block,
            "block_edge": self.block_edge,
            "block_angle": self.block_angle,
            "agent": self.agent,
            "block_speed": self.block_speed,
        }
        if self.contact is not None:
            d["contact"] = self.contact
        return d

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> StepFacts:
        return cls(
            t=int(d["t"]),
            block=str(d["block"]),
            block_edge=str(d["block_edge"]),
            block_angle=str(d["block_angle"]),
            agent=str(d["agent"]),
            contact=(None if "contact" not in d else bool(d["contact"])),
            block_speed=str(d["block_speed"]),
        )


def geometric_contact(agent_xy: np.ndarray, block_xy: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """Vectorised geometric contact flag in normalised units (see ``CONTACT_TOLERANCE``)."""
    from lejudge.constraints.geometry import contact_from_geometry

    return contact_from_geometry(agent_xy, block_xy, angle, BLOCK_SCALE, AGENT_RADIUS, CONTACT_TOLERANCE)


def contact_flags_from_states(states: np.ndarray, arena: float = ARENA_PX) -> np.ndarray:
    """Geometric contact for an ``[N, 7]`` env state array."""
    s = np.asarray(states, dtype=np.float64)
    return geometric_contact(s[:, 0:2] / arena, s[:, 2:4] / arena, s[:, 4])


FAMILIES = ("never", "always", "soft", "temporal_before")


@dataclass(frozen=True)
class Constraint:
    """A natural-language constraint. ``text`` is owner input and never mixed with facts."""

    text: str
    family: str = "never"
    id: str = ""
    weight: float = 1.0
    oracle: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    paraphrases: tuple[str, ...] = ()
    negatives: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise ValueError(f"unknown family {self.family!r}; expected one of {FAMILIES}")
        if not self.id:
            object.__setattr__(self, "id", _slug(self.text))
        if len(self.text) > 200:
            raise ValueError("constraint text is capped at 200 characters")

    def with_text(self, text: str, suffix: str = "") -> Constraint:
        """Same constraint (family, oracle, weight) with different wording, e.g. a paraphrase."""
        return Constraint(
            text=text,
            family=self.family,
            id=self.id + suffix,
            weight=self.weight,
            oracle=self.oracle,
            params=dict(self.params),
        )


def _slug(text: str) -> str:
    import re

    s = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return s[:40] or "constraint"


@dataclass
class JudgeResult:
    """Output of any judge. Keyed by the caller's candidate keys and constraint ids.

    ``p[k][c]`` is a list of per-step probabilities for ``never``/``always`` families,
    or the 5-level probability vector for ``soft``. ``confidence[k][c]`` is the Score
    confidence for ``soft`` and ``None`` otherwise. ``held`` is filled by the cost
    module's gate, not by the judge.
    """

    p: dict[str, dict[str, list[float]]]
    confidence: dict[str, dict[str, float | None]]
    latency_ms: float
    input_tokens: int
    output_tokens: int
    response_model: str
    cache_hit: bool
    n_calls: int = 1
    judge_name: str = ""
    keys: list[str] = field(default_factory=list)
    failed: bool = False

    def to_json(self) -> dict[str, Any]:
        return {
            "p": self.p,
            "confidence": self.confidence,
            "latency_ms": round(self.latency_ms, 3),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "response_model": self.response_model,
            "cache_hit": self.cache_hit,
            "n_calls": self.n_calls,
            "judge_name": self.judge_name,
            "failed": self.failed,
        }
