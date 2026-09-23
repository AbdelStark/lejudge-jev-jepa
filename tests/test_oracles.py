"""Every oracle: 3 positive, 3 negative, 2 boundary trajectories (RFC-0002)."""

import math

import pytest

from lejudge.constraints import check, load_library
from lejudge.types import GroundTruthState

LIB = load_library("pusht")


def gs(ax=0.5, ay=0.9, bx=0.5, by=0.5, ang=0.0, contact=False):
    return GroundTruthState((ax, ay), (bx, by), ang, contact)


def rad(d):
    return math.radians(d)


# Each case: (constraint id, sequence (index 0 = current), expected episode flag)
CASES = {
    "edges_never": {
        "pos": [
            [gs(), gs(bx=0.10), gs(bx=0.10)],
            [gs(), gs(), gs(bx=0.88)],
            [gs(), gs(bx=0.05, ang=rad(90))],
        ],
        "neg": [[gs(), gs(), gs()], [gs(), gs(by=0.05)], [gs(), gs(bx=0.3, ang=rad(45))]],
        "boundary": [
            ([gs(), gs(bx=0.0098 + 2 * 0.05859375 + 0.019)], True),
            ([gs(), gs(bx=0.0098 + 2 * 0.05859375 + 0.021)], False),
        ],
    },
    "top_never": {
        "pos": [[gs(), gs(by=0.02)], [gs(), gs(), gs(by=0.01)], [gs(), gs(by=0.02, bx=0.3)]],
        "neg": [[gs(), gs()], [gs(), gs(bx=0.05)], [gs(), gs(by=0.95, ang=rad(180))]],
        "boundary": [([gs(), gs(by=0.0098 + 0.019)], True), ([gs(), gs(by=0.0098 + 0.021)], False)],
    },
    "centre_avoid": {
        "pos": [
            [gs(bx=0.9, by=0.9), gs(bx=0.5, by=0.4)],
            [gs(), gs(bx=0.4, by=0.4)],
            [gs(), gs(bx=0.6, by=0.5, ang=rad(180))],
        ],
        "neg": [
            [gs(bx=0.9), gs(bx=0.1, by=0.1)],
            [gs(), gs(bx=0.5, by=0.8)],
            [gs(), gs(bx=0.9, by=0.5)],
        ],
        "boundary": [
            ([gs(), gs(bx=0.5, by=1 / 3 - 0.0879 + 0.001)], True),
            ([gs(), gs(bx=0.5, by=1 / 3 - 0.0879 - 0.001)], False),
        ],
    },
    "agent_bottom_only": {
        "pos": [[gs(), gs(ay=0.1)], [gs(), gs(), gs(ax=0.9, ay=0.3)], [gs(), gs(ax=0.1, ay=0.2)]],
        "neg": [[gs(ay=0.1), gs(ay=0.5)], [gs(), gs(ay=0.9)], [gs(), gs(ay=0.4)]],
        "boundary": [([gs(), gs(ay=1 / 3 - 0.001)], True), ([gs(), gs(ay=1 / 3 + 0.001)], False)],
    },
    "upright_always": {
        "pos": [[gs(), gs(ang=rad(45))], [gs(), gs(ang=rad(180))], [gs(), gs(ang=rad(-60))]],
        "neg": [
            [gs(), gs(ang=rad(10))],
            [gs(), gs(ang=rad(-25))],
            [gs(ang=rad(90)), gs(ang=rad(0))],
        ],
        "boundary": [([gs(), gs(ang=rad(30.5))], True), ([gs(), gs(ang=rad(29.5))], False)],
    },
    "no_upside_down": {
        "pos": [[gs(), gs(ang=rad(180))], [gs(), gs(ang=rad(160))], [gs(), gs(ang=rad(-155))]],
        "neg": [
            [gs(), gs(ang=rad(0))],
            [gs(), gs(ang=rad(100))],
            [gs(ang=rad(180)), gs(ang=rad(30))],
        ],
        "boundary": [([gs(), gs(ang=rad(150.5))], True), ([gs(), gs(ang=rad(149.5))], False)],
    },
    "no_push_tilted": {
        "pos": [
            [gs(), gs(ang=rad(100), contact=True)],
            [gs(), gs(ang=rad(-100), contact=True)],
            [gs(), gs(), gs(ang=rad(120), contact=True)],
        ],
        "neg": [
            [gs(), gs(ang=rad(100))],
            [gs(), gs(ang=rad(0), contact=True)],
            [gs(ang=rad(100), contact=True), gs(ang=rad(180), contact=True)],
        ],
        "boundary": [
            ([gs(), gs(ang=rad(90.5), contact=True)], True),
            ([gs(), gs(ang=rad(89.5), contact=True)], False),
        ],
    },
    "approach_below": {
        "pos": [
            [gs(ay=0.2), gs(ay=0.3, contact=True)],
            [gs(), gs(ay=0.5, contact=True)],
            [gs(ay=0.1), gs(ay=0.2), gs(ay=0.3, contact=True)],
        ],
        "neg": [
            [gs(ay=0.9), gs(ay=0.8, contact=True)],
            [gs(), gs(), gs()],
            [gs(ay=0.9), gs(ay=0.7, contact=True), gs(ay=0.2, contact=True)],
        ],
        "boundary": [
            ([gs(), gs(ay=0.5 + 0.0879 - 0.001, contact=True)], True),
            ([gs(), gs(ay=0.5 + 0.0879 + 0.001, contact=True)], False),
        ],
    },
    "gentle": {
        "pos": [
            [gs(bx=0.1), gs(bx=0.3), gs(bx=0.5)],
            [gs(bx=0.1), gs(bx=0.2), gs(bx=0.5), gs(bx=0.5)],
            [gs(by=0.1), gs(by=0.4)],
        ],
        "neg": [
            [gs(), gs(bx=0.51), gs(bx=0.52)],
            [gs(), gs(), gs()],
            [gs(bx=0.1), gs(bx=0.12), gs(bx=0.14), gs(bx=0.16)],
        ],
        "boundary": [
            ([gs(bx=0.1), gs(bx=0.1 + 0.031)], True),
            ([gs(bx=0.1), gs(bx=0.1 + 0.029)], False),
        ],
    },
    "stay_left_half": {
        "pos": [
            [gs(bx=0.2), gs(bx=0.7)],
            [gs(bx=0.2), gs(bx=0.3), gs(bx=0.9)],
            [gs(bx=0.2), gs(bx=0.55)],
        ],
        "neg": [[gs(bx=0.2), gs(bx=0.3)], [gs(bx=0.7), gs(bx=0.4)], [gs(bx=0.1), gs(bx=0.45)]],
        "boundary": [([gs(), gs(bx=0.501)], True), ([gs(), gs(bx=0.499)], False)],
    },
    "no_contact_first3": {
        "pos": [
            [gs(), gs(contact=True)],
            [gs(), gs(), gs(), gs(contact=True)],
            [gs(), gs(), gs(contact=True), gs(), gs()],
        ],
        "neg": [
            [gs(), gs(), gs(), gs(), gs(contact=True)],
            [gs(contact=True), gs(), gs(), gs()],
            [gs(), gs(), gs()],
        ],
        "boundary": [
            ([gs(), gs(), gs(), gs(contact=True)], True),
            ([gs(), gs(), gs(), gs(), gs(contact=True)], False),
        ],
    },
    "corner_avoid": {
        "pos": [
            [gs(), gs(bx=0.1, by=0.05)],
            [gs(), gs(bx=0.9, by=0.9)],
            [gs(), gs(), gs(bx=0.1, by=0.85)],
        ],
        "neg": [
            [gs(), gs(bx=0.5, by=0.1)],
            [gs(), gs(bx=0.1, by=0.5)],
            [gs(bx=0.1, by=0.05), gs(bx=0.5, by=0.5)],
        ],
        "boundary": [
            ([gs(), gs(bx=1 / 3 - 0.001, by=0.1)], True),
            ([gs(), gs(bx=1 / 3 + 0.001, by=0.1)], False),
        ],
    },
}


def test_every_constraint_has_cases():
    assert set(CASES) == set(LIB.ids())


@pytest.mark.parametrize("cid", sorted(CASES))
def test_oracle_cases(cid, vocab):
    c = LIB.get(cid)
    cases = CASES[cid]
    assert len(cases["pos"]) == 3 and len(cases["neg"]) == 3 and len(cases["boundary"]) == 2
    for seq in cases["pos"]:
        assert check(c, seq, vocab).episode is True, f"{cid} positive"
    for seq in cases["neg"]:
        assert check(c, seq, vocab).episode is False, f"{cid} negative"
    for seq, expected in cases["boundary"]:
        assert check(c, seq, vocab).episode is expected, f"{cid} boundary"


def test_library_shape():
    assert len(LIB.constraints) == 12
    assert LIB.sets["edges"] == ("edges_never",)
    assert len(LIB.sha256) == 64
    for c in LIB.constraints:
        assert len(c.paraphrases) == 5 and len(c.negatives) == 2
        v = LIB.variants(c.id)
        assert set(v) == {"canonical", "p1", "p2", "p3", "p4", "p5", "n1", "n2"}
