import math

import numpy as np
import pytest

from lejudge.constraints.geometry import tee_centroid, tee_vertices, wall_distances
from lejudge.types import GroundTruthState, SymbolicState
from lejudge.vocab import words


def _seq(*states):
    return SymbolicState.from_ground_truth(list(states))


def gs(ax, ay, bx, by, ang=0.0, contact=False):
    return GroundTruthState((ax, ay), (bx, by), ang, contact)


def test_words_is_pure_and_deterministic(vocab):
    s = _seq(gs(0.5, 0.9, 0.5, 0.5), gs(0.5, 0.8, 0.45, 0.5, 0.1, True), gs(0.3, 0.5, 0.2, 0.5))
    a = words(s, vocab)
    b = words(s, vocab)
    assert a == b
    assert [f.t for f in a] == [1, 2]


def test_no_numbers_in_facts(vocab):
    rng = np.random.default_rng(0)
    for _ in range(50):
        st = SymbolicState(
            rng.random((6, 2)),
            rng.random((6, 2)) * 0.8 + 0.1,
            rng.random(6) * 2 * math.pi,
            rng.normal(size=6) * 3,
        )
        for f in words(st, vocab):
            for k, v in f.to_json().items():
                if k == "t":
                    assert isinstance(v, int)
                elif k == "contact":
                    assert isinstance(v, bool)
                else:
                    assert isinstance(v, str) and not any(ch.isdigit() for ch in v)


def test_all_words_are_in_vocab(vocab):
    rng = np.random.default_rng(1)
    allowed = vocab.all_words()
    st = SymbolicState(
        rng.random((200, 2)),
        rng.random((200, 2)),
        rng.random(200) * 2 * math.pi,
        rng.normal(size=200) * 3,
    )
    for f in words(st, vocab):
        assert f.block in allowed["block"]
        assert f.agent in allowed["agent"]
        assert f.block_edge in allowed["block_edge"]
        assert f.block_angle in allowed["block_angle"]
        assert f.block_speed in allowed["block_speed"]


def test_grid_cells(vocab):
    assert vocab.cell_name(np.array([0.1, 0.1])) == "top-left"
    assert vocab.cell_name(np.array([0.5, 0.5])) == "centre"
    assert vocab.cell_name(np.array([0.9, 0.9])) == "bottom-right"
    assert vocab.cell_name(np.array([0.5, 0.9])) == "bottom-centre"


def test_angle_bins(vocab):
    deg = {
        0: "upright",
        29: "upright",
        31: "tilted right",
        100: "on its side right",
        180: "upside down",
        200: "upside down",
        240: "on its side left",
        300: "tilted left",
        359: "upright",
        -20: "upright",
    }
    for d, name in deg.items():
        assert vocab.angle_name(np.array([math.radians(d)]))[0] == name, d


def test_edge_words(vocab):
    # T with the bar along the top wall: body origin at wall_inset + margin/2, angle 0
    y = vocab.wall_inset + vocab.edge_margin / 2
    assert vocab.edge_name(np.array([0.5, y]), np.array([0.0]))[0] == "top edge"
    # far from every wall
    assert vocab.edge_name(np.array([0.5, 0.5]), np.array([0.0]))[0] == "none"
    # bar end touching the left wall: bar half-length is 2*scale
    x = vocab.wall_inset + 2 * vocab.block_scale + vocab.edge_margin / 2
    assert vocab.edge_name(np.array([x, 0.5]), np.array([0.0]))[0] == "left edge"


def test_speed_words(vocab):
    assert vocab.speed_name(np.array([0.0]))[0] == "still"
    assert vocab.speed_name(np.array([0.01]))[0] == "slow"
    assert vocab.speed_name(np.array([0.2]))[0] == "fast"


def test_contact_omitted_when_unsure(vocab):
    st = SymbolicState(
        np.zeros((2, 2)) + 0.5, np.zeros((2, 2)) + 0.5, np.zeros(2), np.array([0.0, 0.1])
    )
    f = words(st, vocab)
    assert f[0].contact is None
    assert "contact" not in f[0].to_json()


def test_geometry_shapes():
    v = tee_vertices(np.zeros((3, 2)) + 0.5, np.zeros(3), 0.05)
    assert v.shape == (3, 8, 2)
    c = tee_centroid(np.array([0.5, 0.5]), np.array(0.0), 0.05)
    assert c.shape == (2,) and c[1] > 0.5
    d = wall_distances(v, 0.01)
    assert set(d) == {"left", "right", "top", "bottom"}


def test_words_requires_two_states(vocab):
    with pytest.raises(ValueError):
        words(_seq(gs(0.5, 0.5, 0.5, 0.5)), vocab)
