"""PushT block geometry shared by the vocabulary and the oracle checkers.

Both sides call the same functions so a word ("left edge") and its checker agree exactly.
"""

from __future__ import annotations

import math

import numpy as np

# Offsets from PushT.add_tee (scale = 1): bar 4x1 on top, stem 1x3 below, body origin at
# the bar's top-centre. Centre of gravity of the two rectangles is (0, 1.5).
_TEE_VERTICES_UNIT = np.array(
    [
        # bar
        [-2.0, 1.0],
        [2.0, 1.0],
        [2.0, 0.0],
        [-2.0, 0.0],
        # stem
        [-0.5, 1.0],
        [-0.5, 4.0],
        [0.5, 4.0],
        [0.5, 1.0],
    ]
)
_TEE_COG_UNIT = np.array([0.0, 1.5])


def tee_vertices(block_xy: np.ndarray, angle: np.ndarray, scale: float) -> np.ndarray:
    """World-frame vertices ``[..., 8, 2]`` of the T given body origin, angle and scale."""
    block_xy = np.asarray(block_xy, dtype=np.float64)
    angle = np.asarray(angle, dtype=np.float64)
    c, s = np.cos(angle), np.sin(angle)
    rot = np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)  # [..., 2, 2]
    local = _TEE_VERTICES_UNIT * scale  # [8, 2]
    world = np.einsum("...ij,vj->...vi", rot, local) + block_xy[..., None, :]
    return world


def tee_centroid(block_xy: np.ndarray, angle: np.ndarray, scale: float) -> np.ndarray:
    """World-frame centre of gravity ``[..., 2]`` of the T."""
    block_xy = np.asarray(block_xy, dtype=np.float64)
    angle = np.asarray(angle, dtype=np.float64)
    c, s = np.cos(angle), np.sin(angle)
    local = _TEE_COG_UNIT * scale
    dx = c * local[0] - s * local[1]
    dy = s * local[0] + c * local[1]
    return block_xy + np.stack([dx, dy], -1)


def wall_distances(vertices: np.ndarray, wall_inset: float) -> dict[str, np.ndarray]:
    """Min distance from any vertex to each wall: keys left, right, top, bottom."""
    x = vertices[..., 0]
    y = vertices[..., 1]
    lo, hi = wall_inset, 1.0 - wall_inset
    return {
        "left": np.min(x - lo, axis=-1),
        "right": np.min(hi - x, axis=-1),
        "top": np.min(y - lo, axis=-1),
        "bottom": np.min(hi - y, axis=-1),
    }


def grid_cell(xy: np.ndarray, grid: int) -> tuple[np.ndarray, np.ndarray]:
    """Row and column indices of a point on a ``grid x grid`` partition of the unit square."""
    xy = np.asarray(xy, dtype=np.float64)
    col = np.clip(np.floor(xy[..., 0] * grid), 0, grid - 1).astype(int)
    row = np.clip(np.floor(xy[..., 1] * grid), 0, grid - 1).astype(int)
    return row, col


def signed_angle_deg(angle: np.ndarray) -> np.ndarray:
    """Wrap radians to degrees in ``[0, 360)`` measured from the canonical orientation."""
    return (np.degrees(np.asarray(angle, dtype=np.float64)) % 360.0 + 360.0) % 360.0


def angle_bin(angle: np.ndarray, bins_deg: dict[str, tuple[float, float]]) -> np.ndarray:
    """Name of the bin containing each angle; bins are ``[lo, hi)`` in degrees, wrapping at 360."""
    deg = signed_angle_deg(angle)
    out = np.empty(deg.shape, dtype=object)
    out[...] = ""
    for name, (lo, hi) in bins_deg.items():
        lo_w, hi_w = lo % 360.0, hi % 360.0
        if lo_w < hi_w:
            mask = (deg >= lo_w) & (deg < hi_w)
        else:  # wraps around 0
            mask = (deg >= lo_w) | (deg < hi_w)
        out[mask & (out == "")] = name
    assert np.all(out != ""), "angle bins must cover the full circle"
    return out


def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def point_to_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance from points ``p [..., 2]`` to segments ``a→b`` (same leading dims)."""
    ab = b - a
    t = np.clip(np.sum((p - a) * ab, -1) / np.maximum(np.sum(ab * ab, -1), 1e-12), 0.0, 1.0)
    proj = a + t[..., None] * ab
    return np.linalg.norm(p - proj, axis=-1)


def _point_in_convex_quad(p: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """``quad [..., 4, 2]`` in consistent winding; returns bool ``[...]``."""
    sign = None
    inside = np.ones(p.shape[:-1], dtype=bool)
    for i in range(4):
        a, b = quad[..., i, :], quad[..., (i + 1) % 4, :]
        cross = (b[..., 0] - a[..., 0]) * (p[..., 1] - a[..., 1]) - (b[..., 1] - a[..., 1]) * (
            p[..., 0] - a[..., 0]
        )
        s = cross >= 0
        if sign is None:
            sign = s
        inside &= s == sign
    return inside


def agent_block_distance(
    agent_xy: np.ndarray, block_xy: np.ndarray, angle: np.ndarray, scale: float
) -> np.ndarray:
    """Signed distance from the agent centre to the T (negative inside). ``[...]``."""
    verts = tee_vertices(block_xy, angle, scale)  # [..., 8, 2]
    p = np.asarray(agent_xy, dtype=np.float64)
    d = np.full(p.shape[:-1], np.inf)
    for quad in (verts[..., 0:4, :], verts[..., 4:8, :]):
        for i in range(4):
            d = np.minimum(
                d, point_to_segment_distance(p, quad[..., i, :], quad[..., (i + 1) % 4, :])
            )
        inside = _point_in_convex_quad(p, quad)
        d = np.where(inside, -np.abs(d), d)
    return d


def contact_from_geometry(
    agent_xy: np.ndarray,
    block_xy: np.ndarray,
    angle: np.ndarray,
    scale: float,
    agent_radius: float,
    tolerance: float,
) -> np.ndarray:
    """True when the agent disc is within ``tolerance`` of the T polygon (all in normalised units)."""
    return agent_block_distance(agent_xy, block_xy, angle, scale) <= agent_radius + tolerance
