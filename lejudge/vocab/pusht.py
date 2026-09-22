"""PushT vocabulary: turn a ``SymbolicState`` sequence into ``StepFacts``.

``words()`` is pure and deterministic. All thresholds come from ``pusht@<n>.yaml``;
no numeric value appears in any output field.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from lejudge.constraints.geometry import (
    angle_bin,
    grid_cell,
    tee_centroid,
    tee_vertices,
    wall_distances,
)
from lejudge.types import StepFacts, SymbolicState

EDGE_WORDS = ("none", "left edge", "right edge", "top edge", "bottom edge")
SPEED_WORDS = ("still", "slow", "fast")


@dataclass(frozen=True)
class Vocab:
    version: str
    grid: int
    grid_names: tuple[tuple[str, ...], ...]
    wall_inset: float
    edge_margin: float
    block_scale: float
    angle_bins_deg: dict[str, tuple[float, float]]
    speed_thresholds: dict[str, float]
    contact_logit_band: float

    @property
    def angle_words(self) -> tuple[str, ...]:
        return tuple(self.angle_bins_deg.keys())

    @property
    def cell_words(self) -> tuple[str, ...]:
        return tuple(w for row in self.grid_names for w in row)

    def all_words(self) -> dict[str, tuple[str, ...]]:
        return {
            "block": self.cell_words,
            "agent": self.cell_words,
            "block_edge": EDGE_WORDS,
            "block_angle": self.angle_words,
            "block_speed": SPEED_WORDS,
        }

    # ---- primitive word functions (also used by the oracles) -------------------------
    def cell_name(self, xy: np.ndarray) -> np.ndarray:
        row, col = grid_cell(xy, self.grid)
        names = np.array(self.grid_names, dtype=object)
        return names[row, col]

    def block_centroid(self, block_xy: np.ndarray, angle: np.ndarray) -> np.ndarray:
        return tee_centroid(block_xy, angle, self.block_scale)

    def edge_name(self, block_xy: np.ndarray, angle: np.ndarray) -> np.ndarray:
        verts = tee_vertices(block_xy, angle, self.block_scale)
        d = wall_distances(verts, self.wall_inset)
        stacked = np.stack([d["left"], d["right"], d["top"], d["bottom"]], -1)
        nearest = np.argmin(stacked, -1)
        touching = np.min(stacked, -1) <= self.edge_margin
        names = np.array(["left edge", "right edge", "top edge", "bottom edge"], dtype=object)
        out = np.where(touching, names[nearest], "none")
        return out.astype(object)

    def angle_name(self, angle: np.ndarray) -> np.ndarray:
        return angle_bin(angle, self.angle_bins_deg)

    def speed_name(self, displacement: np.ndarray) -> np.ndarray:
        d = np.asarray(displacement, dtype=np.float64)
        out = np.where(d < self.speed_thresholds["still"], "still", np.where(d < self.speed_thresholds["slow"], "slow", "fast"))
        return out.astype(object)


def _pkg_path(name: str) -> Path:
    return Path(str(resources.files("lejudge.vocab").joinpath(f"{name}.yaml")))


def load_vocab(name: str = "pusht@1") -> Vocab:
    path = Path(name) if name.endswith(".yaml") else _pkg_path(name)
    raw: dict[str, Any] = yaml.safe_load(path.read_text())
    return Vocab(
        version=str(raw["version"]),
        grid=int(raw["grid"]),
        grid_names=tuple(tuple(str(w) for w in row) for row in raw["grid_names"]),
        wall_inset=float(raw["wall_inset"]),
        edge_margin=float(raw["edge_margin"]),
        block_scale=float(raw["block_scale"]),
        angle_bins_deg={str(k): (float(v[0]), float(v[1])) for k, v in raw["angle_bins_deg"].items()},
        speed_thresholds={str(k): float(v) for k, v in raw["speed_thresholds"].items()},
        contact_logit_band=float(raw["contact_logit_band"]),
    )


def words(states: SymbolicState, vocab: Vocab, start_t: int = 1) -> list[StepFacts]:
    """Describe a 1-d sequence of ``H+1`` symbolic states as ``H`` step facts.

    Index 0 is the current (observed) state and only serves to compute the speed of step
    ``t=1``; facts are emitted for indices ``1..H`` with ``t = start_t .. start_t+H-1``.
    """
    assert states.block_angle.ndim == 1, "words() expects a 1-d sequence (H+1 states)"
    n = states.block_angle.shape[0]
    if n < 2:
        raise ValueError("need at least two states (current + one imagined step)")
    centroid = vocab.block_centroid(states.block_xy, states.block_angle)
    block_cells = vocab.cell_name(centroid)
    agent_cells = vocab.cell_name(states.agent_xy)
    edges = vocab.edge_name(states.block_xy, states.block_angle)
    angles = vocab.angle_name(states.block_angle)
    disp = np.linalg.norm(np.diff(centroid, axis=0), axis=-1)
    speeds = vocab.speed_name(disp)
    out: list[StepFacts] = []
    for i in range(1, n):
        logit = float(states.contact_logit[i])
        contact: bool | None = None if abs(logit) < vocab.contact_logit_band else bool(logit > 0)
        out.append(
            StepFacts(
                t=start_t + i - 1,
                block=str(block_cells[i]),
                block_edge=str(edges[i]),
                block_angle=str(angles[i]),
                agent=str(agent_cells[i]),
                contact=contact,
                block_speed=str(speeds[i - 1]),
            )
        )
    return out
