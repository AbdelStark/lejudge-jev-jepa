from lejudge.eval.filters import in_tension, relevant, satisfiable
from lejudge.types import GroundTruthState


def gs(ax=0.5, ay=0.9, bx=0.2, by=0.2, ang=0.0, contact=False):
    return GroundTruthState((ax, ay), (bx, by), ang, contact)


def test_spatial_filter(vocab, library):
    start, goal = gs(bx=0.2, by=0.2), gs(bx=0.8, by=0.8)
    assert satisfiable("spatial", start, goal, vocab, library)
    assert not satisfiable("spatial", gs(bx=0.5, by=0.45), goal, vocab, library)
    through_centre = [start] + [gs(bx=0.5, by=0.45)] * 10 + [goal]
    around = [start] + [gs(bx=0.2, by=0.8)] * 10 + [goal]
    assert in_tension("spatial", through_centre, vocab, library)
    assert not in_tension("spatial", around, vocab, library)
    assert relevant("spatial", through_centre, vocab, library) and not relevant("spatial", around, vocab, library)


def test_temporal_filter(vocab, library):
    start = gs(contact=False)
    early = [start] + [gs(contact=True)] * 5 + [gs()] * 20
    late = [start] + [gs()] * 20 + [gs(contact=True)] * 5
    assert in_tension("spatial+temporal", early, vocab, library)
    assert not in_tension("spatial+temporal", late, vocab, library)
    assert not satisfiable("spatial+temporal", gs(contact=True), gs(), vocab, library)
