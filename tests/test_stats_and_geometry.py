import numpy as np
import pandas as pd
import pytest

from lejudge.constraints.geometry import agent_block_distance, contact_from_geometry
from lejudge.eval.stats import auroc, bootstrap_ci, ece, holm, prf, risk_difference, wilcoxon_paired
from lejudge.types import (
    AGENT_RADIUS,
    BLOCK_SCALE,
    CONTACT_TOLERANCE,
    GroundTruthState,
    contact_flags_from_states,
)


def test_agent_block_distance_geometry():
    # agent right on the bar's top-centre (body origin) is inside → negative
    d = agent_block_distance(np.array([0.5, 0.5 + 0.01]), np.array([0.5, 0.5]), np.array(0.0), BLOCK_SCALE)
    assert d < 0
    # agent far away
    d = agent_block_distance(np.array([0.1, 0.1]), np.array([0.5, 0.5]), np.array(0.0), BLOCK_SCALE)
    assert d > 0.3
    # touching: exactly agent_radius above the bar top edge
    y = 0.5 - AGENT_RADIUS - 0.001
    assert contact_from_geometry(np.array([0.5, y]), np.array([0.5, 0.5]), np.array(0.0), BLOCK_SCALE, AGENT_RADIUS, CONTACT_TOLERANCE)
    y = 0.5 - AGENT_RADIUS - CONTACT_TOLERANCE - 0.002
    assert not contact_from_geometry(np.array([0.5, y]), np.array([0.5, 0.5]), np.array(0.0), BLOCK_SCALE, AGENT_RADIUS, CONTACT_TOLERANCE)


def test_from_env_geometric_contact():
    s = np.array([256.0, 256.0 - 15 - 1, 256.0, 256.0, 0.0, 0.0, 0.0])
    assert GroundTruthState.from_env(s).contact is True
    s2 = s.copy()
    s2[1] = 100.0
    assert GroundTruthState.from_env(s2).contact is False
    assert GroundTruthState.from_env(s2, contact=3).contact is True
    flags = contact_flags_from_states(np.stack([s, s2]))
    assert flags.tolist() == [True, False]


def test_bootstrap_ci_and_rd():
    x = np.array([1, 0, 1, 1, 0, 1, 1, 1, 0, 1], dtype=float)
    m, lo, hi = bootstrap_ci(x, n=2000)
    assert lo <= m <= hi and abs(m - 0.7) < 1e-9
    rd, lo, hi = risk_difference(x, 1 - x, n=2000)
    assert abs(rd - 0.4) < 1e-9 and lo <= rd <= hi


def test_wilcoxon_and_holm():
    a = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 1], dtype=float)
    b = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0], dtype=float)
    assert wilcoxon_paired(a, b) < 0.05
    assert wilcoxon_paired(a, a) == 1.0
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj["a"] == pytest.approx(0.03) and adj["c"] == pytest.approx(0.06) and adj["b"] == pytest.approx(0.06)


def test_classification_metrics():
    y = np.array([1, 1, 0, 0, 1, 0])
    s = np.array([0.9, 0.6, 0.4, 0.2, 0.3, 0.8])
    m = prf(y, s)
    assert m["tp"] == 2 and m["fp"] == 1 and m["fn"] == 1
    assert 0 < auroc(y, s) < 1
    e, table = ece(y, s, bins=3)
    assert 0 <= e <= 1 and len(table) == 3
    assert np.isnan(auroc(np.zeros(3), s[:3]))


def test_planning_table_and_tests():
    from lejudge.eval.report import paired_tests, planning_table

    rows = []
    rng = np.random.default_rng(0)
    for cond, pv in (("lewm", 0.6), ("jev", 0.1)):
        for ep in range(30):
            rows.append({"condition": cond, "constraint_set": "edges", "seed": 0, "episode": ep, "success": bool(rng.random() < 0.5), "violation": bool(rng.random() < pv), "violation_steps": 1, "plan_time_p50_s": 1.0, "plan_time_p95_s": 1.2, "judge_calls": 6, "tokens_in": 100, "abstention_rate": 0.0, "lam": 1.0, "tag": ""})
    df = pd.DataFrame(rows)
    t = planning_table(df)
    assert set(t.condition) == {"lewm", "jev"}
    pt = paired_tests(df, ref="lewm", metric="violation")
    assert len(pt) == 1 and pt.iloc[0].risk_difference < 0 and pt.iloc[0].p_holm <= 1.0
