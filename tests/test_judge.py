import json
import math

import numpy as np
import pytest

from lejudge.judge import (
    BANK_VERSION,
    CacheMiss,
    JevJudge,
    KeywordJudge,
    LLMJudge,
    OracleJudge,
    TraceWriter,
    build_state,
    lint_bank,
    lint_constraint_text,
    penalty,
    plan_questions,
    uncertainty_proxy,
)
from lejudge.judge.cache import Cache, CachedCaller, cache_key
from lejudge.judge.llm import parse_llm_json
from lejudge.types import Constraint, GroundTruthState, StepFacts, SymbolicState
from lejudge.vocab import words


def gs(ax=0.5, ay=0.9, bx=0.5, by=0.5, ang=0.0, contact=False):
    return GroundTruthState((ax, ay), (bx, by), ang, contact)


@pytest.fixture
def sample(vocab, library):
    seq = [gs(), gs(bx=0.45, contact=True), gs(bx=0.06, ax=0.3, ay=0.5, contact=True)]
    facts = {
        "cand_a": words(SymbolicState.from_ground_truth(seq), vocab),
        "cand_b": words(SymbolicState.from_ground_truth([gs(), gs(), gs()]), vocab),
    }
    states = {"cand_a": seq, "cand_b": [gs(), gs(), gs()]}
    cs = library.set("edges+upright") + [library.get("gentle")]
    return facts, states, cs


def test_state_layout(sample):
    facts, _, cs = sample
    b = build_state(facts, cs)
    assert set(b.state) == {"constraints", "candidates"}
    assert list(b.state["candidates"]) == ["k1", "k2"]
    assert list(b.state["constraints"]) == ["c1", "c2", "c3"]
    js = json.loads(b.canonical_json)
    for rows in js["candidates"].values():
        for r in rows:
            for k, v in r.items():
                assert k == "t" or k == "contact" or isinstance(v, str)
    # constraint text never appears under candidates
    assert "Never let" not in json.dumps(js["candidates"])


def test_canonical_json_stable_under_reordering(sample):
    facts, _, cs = sample
    a = build_state(facts, cs).canonical_json
    b = build_state(facts, cs).canonical_json
    assert a == b
    # candidate ids renamed in cost order → same content different caller keys hashes equal
    renamed = {"zzz": facts["cand_a"], "aaa": facts["cand_b"]}
    assert build_state(renamed, cs).canonical_json == a


def test_rejects_numbers_in_facts(sample):
    _, _, cs = sample
    bad = {"k": [StepFacts(1, "centre", "none", "upright", "x=0.5", False, "slow")]}
    with pytest.raises(ValueError):
        build_state(bad, cs)


def test_question_count_and_keys(sample):
    facts, _, cs = sample
    b = build_state(facts, cs)
    specs = plan_questions(b, cs)
    # 2 candidates × (2 steps × 2 never/always + 1 soft)
    assert len(specs) == 2 * (2 * 2 + 1)
    assert specs[0].key == "k1_c1_1"
    assert any(s.key == "k1_c3" and s.kind == "score" for s in specs)


def test_baselines_share_key_set(sample, vocab):
    facts, states, cs = sample
    kw = KeywordJudge().judge(facts, cs)
    oc = OracleJudge(vocab).judge(facts, cs, states=states)
    assert kw.keys == oc.keys
    assert set(kw.p) == set(oc.p) == {"cand_a", "cand_b"}
    assert set(kw.p["cand_a"]) == {c.id for c in cs}
    # canonical text: keyword equals oracle on this simple case
    assert kw.p["cand_a"]["edges_never"] == oc.p["cand_a"]["edges_never"] == [0.0, 1.0]
    # paraphrase: keyword does not transfer
    para = [cs[0].with_text(cs[0].paraphrases[0], "@p1")]
    assert KeywordJudge().judge(facts, para).p["cand_a"][para[0].id] == [0.0, 0.0]


def test_penalty_aggregation():
    assert penalty("never", [0.1, 0.9, 0.2]) == pytest.approx(0.9)
    assert penalty("always", [0.9, 0.4, 1.0]) == pytest.approx(0.6)
    assert penalty("soft", [0, 0, 0, 0, 1]) == pytest.approx(0.0)
    assert penalty("soft", [1, 0, 0, 0, 0]) == pytest.approx(1.0)
    assert penalty("soft", [0, 0, 1, 0, 0]) == pytest.approx(0.5)
    assert penalty("temporal_before", [0.9, 0.9, 0.1, 0.9]) == 0.0
    assert penalty("temporal_before", [0.1, 0.9, 0.9, 0.9]) == 1.0
    assert penalty("never", []) == 0.0
    assert penalty("never", [float("nan"), 0.3]) == pytest.approx(0.3)
    sm = penalty("never", [0.1, 0.9, 0.2], smooth_temperature=0.1)
    assert 0.8 < sm <= 0.9


def test_uncertainty_proxy():
    assert uncertainty_proxy("never", [0.5, 0.9, 0.1], None) == pytest.approx(1 / 3)
    assert uncertainty_proxy("soft", [0.2] * 5, 0.3) == pytest.approx(0.7)
    assert uncertainty_proxy("never", [], None) == 1.0


def test_lint_bank_clean(vocab, library):
    assert lint_bank(vocab.all_words()) == []
    for c in library.constraints:
        for t in (c.text, *c.paraphrases, *c.negatives):
            assert lint_constraint_text(t) == []


def test_cache_roundtrip_and_offline(tmp_path, monkeypatch):
    c = Cache(tmp_path / "c.sqlite")
    caller = CachedCaller(c)
    calls = []

    def fn():
        calls.append(1)
        return {"k": {"type": "noul", "noul": 0.5}}, "jev-1.13.0", 10, 2

    monkeypatch.setenv("LEJUDGE_MODE", "offline")
    with pytest.raises(CacheMiss):
        caller.call("jev-1.13", BANK_VERSION, "{}", "[]", fn)
    monkeypatch.setenv("LEJUDGE_MODE", "live")
    r1 = caller.call("jev-1.13", BANK_VERSION, "{}", "[]", fn)
    assert r1.cache_hit is False and len(calls) == 1
    monkeypatch.setenv("LEJUDGE_MODE", "offline")
    r2 = caller.call("jev-1.13", BANK_VERSION, "{}", "[]", fn)
    assert r2.cache_hit is True and r2.payload == r1.payload and len(calls) == 1
    monkeypatch.setenv("LEJUDGE_MODE", "refresh")
    r3 = caller.call("jev-1.13", BANK_VERSION, "{}", "[]", fn, uid="rep1")
    assert r3.cache_hit is False and len(calls) == 2
    assert cache_key("a", "b", "c", "d") != cache_key("a", "b", "c", "d", uid="x")
    assert c.count() == 2


def test_jev_judge_offline_raises_on_miss(sample, tmp_path, monkeypatch):
    facts, _, cs = sample
    monkeypatch.setenv("LEJUDGE_MODE", "offline")
    j = JevJudge(cache_path=str(tmp_path / "x.sqlite"))
    with pytest.raises(CacheMiss):
        j.judge(facts, cs)


class _FakeAnswer:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _FakeResp:
    def __init__(self, answers, model="jev-1.13.0"):
        self.answers = answers
        self.model = model
        self.usage = _FakeAnswer(input_tokens=100, output_tokens=5)


class _FakeClient:
    def __init__(self, model="jev-1.13.0"):
        self.calls = 0
        self.model = model

    def system_one(self, state, questions, model):
        self.calls += 1
        answers = {}
        for k, q in questions.items():
            if q.type == "score":
                answers[k] = _FakeAnswer(
                    type="score",
                    score=3.0,
                    confidence=0.8,
                    probabilities={0: 0.0, 1: 0.0, 2: 0.2, 3: 0.6, 4: 0.2},
                )
            else:
                answers[k] = _FakeAnswer(type="noul", noul=0.8 if k.endswith("_2") else 0.1)
        return _FakeResp(answers, self.model)


def test_jev_judge_live_with_fake_client(sample, tmp_path, monkeypatch):
    facts, _, cs = sample
    monkeypatch.setenv("LEJUDGE_MODE", "live")
    fake = _FakeClient()
    j = JevJudge(cache_path=str(tmp_path / "x.sqlite"), client=fake)
    r = j.judge(facts, cs)
    assert r.cache_hit is False and fake.calls == 1
    assert r.p["cand_a"]["edges_never"] == [0.1, 0.8]
    assert r.p["cand_a"]["gentle"] == pytest.approx([0.0, 0.0, 0.2, 0.6, 0.2])
    assert r.confidence["cand_a"]["gentle"] == 0.8
    assert r.response_model == "jev-1.13.0" and r.input_tokens == 100
    r2 = j.judge(facts, cs)
    assert r2.cache_hit is True and fake.calls == 1
    assert r2.p == r.p


def test_jev_judge_pin_violation_is_a_soft_failure(sample, tmp_path, monkeypatch):
    facts, _, cs = sample
    monkeypatch.setenv("LEJUDGE_MODE", "live")
    j = JevJudge(cache_path=str(tmp_path / "x.sqlite"), client=_FakeClient(model="jev-2.0.0"))
    r = j.judge(facts, cs)
    assert r.failed is True
    assert all(math.isnan(x) for x in r.p["cand_a"]["edges_never"])


def test_question_cap_splits_calls(vocab, library, tmp_path, monkeypatch):
    monkeypatch.setenv("LEJUDGE_MODE", "live")
    seq = [gs()] + [gs(bx=0.5 + 0.01 * i) for i in range(10)]
    facts = {f"k{i}": words(SymbolicState.from_ground_truth(seq), vocab) for i in range(16)}
    cs = library.set("edges+upright") + [library.get("centre_avoid"), library.get("top_never")]
    fake = _FakeClient()
    j = JevJudge(cache_path=str(tmp_path / "x.sqlite"), client=fake)
    r = j.judge(facts, cs)  # 16 × 4 × 10 = 640 questions > 512
    assert fake.calls == 2 and r.n_calls == 2
    assert len(r.keys) == 640


def test_parse_llm_json():
    class S:
        def __init__(self, key, kind):
            self.key, self.kind = key, kind

    specs = [S("k1_c1_1", "noul"), S("k1_c2", "score"), S("k1_c1_2", "noul")]
    txt = '```json\n{"k1_c1_1": 0.9, "k1_c2": {"fully": 0.5, "mostly": 0.5}, "k1_c1_2": "bad"}\n```'
    out = parse_llm_json(txt, specs)
    assert out["k1_c1_1"]["noul"] == 0.9
    assert out["k1_c2"]["probabilities"]["4"] == 0.5
    assert "k1_c1_2" not in out
    assert parse_llm_json("nonsense", specs) == {}


def test_llm_judge_offline_raises(sample, tmp_path, monkeypatch):
    facts, _, cs = sample
    monkeypatch.setenv("LEJUDGE_MODE", "offline")
    with pytest.raises(CacheMiss):
        LLMJudge(cache_path=str(tmp_path / "x.sqlite")).judge(facts, cs)


def test_trace_roundtrip(tmp_path):
    w = TraceWriter("run_x", root=tmp_path)
    w.open_episode(3)
    w.context = {"cond": "jev", "set": "edges", "seed": 0}
    w.write(
        {
            "step": 1,
            "iter": 2,
            "penalty": np.array([0.1, 0.2]),
            "judge": {"p": {"k1": {"c1": [0.1, 0.9]}}},
        }
    )
    w.close()
    recs = TraceWriter.read(tmp_path / "run_x" / "ep_3.jsonl")
    assert recs[0]["episode"] == 3 and recs[0]["cond"] == "jev" and recs[0]["penalty"] == [0.1, 0.2]
    assert recs[0]["judge"]["p"]["k1"]["c1"] == [0.1, 0.9]


def test_constraint_validation():
    with pytest.raises(ValueError):
        Constraint("x", family="sometimes")
    with pytest.raises(ValueError):
        Constraint("y" * 201)
    c = Constraint("Keep the T upright.")
    assert c.id and c.family == "never"
