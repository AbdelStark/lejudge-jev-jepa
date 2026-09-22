"""Oracle checkers on ground-truth (or probe-derived) PushT states.

Every checker is a pure function ``(seq, params, vocab) -> ViolationTrace`` where ``seq`` is
the list of states for steps ``t = 0..H`` (index 0 = current state, as in ``words()``) and
the per-step flags are reported for ``t = 1..H``. Checkers detect *violations*; for
``always`` constraints the violation is the negation of the requirement.

Executed states come from the simulator; imagined states come from probes, in which case
the result is labelled ``oracle-on-probes`` by the caller.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from lejudge.types import GroundTruthState, SymbolicState
from lejudge.vocab.pusht import Vocab


@dataclass(frozen=True)
class ViolationTrace:
    steps: tuple[bool, ...]  # per step t=1..H
    episode: bool

    @property
    def first_step(self) -> int | None:
        for i, v in enumerate(self.steps):
            if v:
                return i + 1
        return None


OracleFn = Callable[[list[GroundTruthState], dict[str, Any], Vocab], ViolationTrace]
_REGISTRY: dict[str, OracleFn] = {}


def register(name: str) -> Callable[[OracleFn], OracleFn]:
    def deco(fn: OracleFn) -> OracleFn:
        _REGISTRY[name] = fn
        return fn

    return deco


def get_oracle(name: str) -> OracleFn:
    if name not in _REGISTRY:
        raise KeyError(f"unknown oracle {name!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def oracle_names() -> list[str]:
    return sorted(_REGISTRY)


def _sym(seq: list[GroundTruthState]) -> SymbolicState:
    return SymbolicState.from_ground_truth(seq)


def _trace(flags: np.ndarray) -> ViolationTrace:
    steps = tuple(bool(x) for x in np.asarray(flags)[1:])
    return ViolationTrace(steps=steps, episode=any(steps))


@register("block_edge")
def block_edge(seq: list[GroundTruthState], params: dict[str, Any], vocab: Vocab) -> ViolationTrace:
    """Violation when the T touches one of ``params['walls']`` (left/right/top/bottom)."""
    s = _sym(seq)
    edge = vocab.edge_name(s.block_xy, s.block_angle)
    walls = {f"{w} edge" for w in params.get("walls", ["left", "right"])}
    return _trace(np.array([e in walls for e in edge]))


@register("block_in_cells")
def block_in_cells(seq: list[GroundTruthState], params: dict[str, Any], vocab: Vocab) -> ViolationTrace:
    """Violation when the block centroid is in any of ``params['cells']``.

    ``centroid_x_min``, when given, overrides the cell list with a half-plane test so the
    checker is exact for half-board constraints regardless of grid granularity.
    """
    s = _sym(seq)
    centroid = vocab.block_centroid(s.block_xy, s.block_angle)
    if "centroid_x_min" in params:
        return _trace(centroid[:, 0] >= float(params["centroid_x_min"]))
    cells = set(params.get("cells", []))
    names = vocab.cell_name(centroid)
    return _trace(np.array([n in cells for n in names]))


@register("agent_in_cells")
def agent_in_cells(seq: list[GroundTruthState], params: dict[str, Any], vocab: Vocab) -> ViolationTrace:
    s = _sym(seq)
    cells = set(params.get("cells", []))
    names = vocab.cell_name(s.agent_xy)
    return _trace(np.array([n in cells for n in names]))


@register("block_angle_in")
def block_angle_in(seq: list[GroundTruthState], params: dict[str, Any], vocab: Vocab) -> ViolationTrace:
    s = _sym(seq)
    bins = set(params.get("bins", []))
    names = vocab.angle_name(s.block_angle)
    return _trace(np.array([n in bins for n in names]))


@register("block_angle_not_in")
def block_angle_not_in(seq: list[GroundTruthState], params: dict[str, Any], vocab: Vocab) -> ViolationTrace:
    s = _sym(seq)
    bins = set(params.get("bins", []))
    names = vocab.angle_name(s.block_angle)
    return _trace(np.array([n not in bins for n in names]))


@register("contact_while_angle_in")
def contact_while_angle_in(seq: list[GroundTruthState], params: dict[str, Any], vocab: Vocab) -> ViolationTrace:
    s = _sym(seq)
    bins = set(params.get("bins", []))
    names = vocab.angle_name(s.block_angle)
    contact = np.array([st.contact for st in seq])
    return _trace(np.array([c and (n in bins) for c, n in zip(contact, names)]))


@register("first_contact_not_below")
def first_contact_not_below(seq: list[GroundTruthState], params: dict[str, Any], vocab: Vocab) -> ViolationTrace:
    """Violation at the first contact step if the agent is not below the block centroid.

    Screen y grows downward, so "below" means ``agent_y > block_centroid_y``. Steps after
    the first contact inherit no flag; the episode flag is the first-contact flag. If the
    rollout starts already in contact (index 0), the first imagined step is judged.
    """
    s = _sym(seq)
    centroid = vocab.block_centroid(s.block_xy, s.block_angle)
    flags = np.zeros(len(seq), dtype=bool)
    for i in range(1, len(seq)):
        if seq[i].contact and not seq[i - 1].contact or (i == 1 and seq[0].contact and seq[1].contact):
            flags[i] = not (seq[i].agent_xy[1] > centroid[i, 1])
            break
    return _trace(flags)


@register("block_fast")
def block_fast(seq: list[GroundTruthState], params: dict[str, Any], vocab: Vocab) -> ViolationTrace:
    """Per-step violation when the block moves ``fast``; episode violation when the
    fraction of fast steps exceeds ``max_fraction_fast``."""
    s = _sym(seq)
    centroid = vocab.block_centroid(s.block_xy, s.block_angle)
    disp = np.linalg.norm(np.diff(centroid, axis=0), axis=-1)
    fast = vocab.speed_name(disp) == "fast"
    steps = tuple(bool(x) for x in fast)
    frac = float(np.mean(fast)) if len(fast) else 0.0
    return ViolationTrace(steps=steps, episode=frac > float(params.get("max_fraction_fast", 0.34)))


@register("contact_in_steps")
def contact_in_steps(seq: list[GroundTruthState], params: dict[str, Any], vocab: Vocab) -> ViolationTrace:
    """Violation when there is contact at any of the listed steps (1-indexed, t=1 is the
    first imagined step)."""
    steps_set = {int(t) for t in params.get("steps", [1, 2, 3])}
    flags = np.zeros(len(seq), dtype=bool)
    for i in range(1, len(seq)):
        flags[i] = (i in steps_set) and seq[i].contact
    return _trace(flags)
