import numpy as np
import torch

from lejudge.cost import JevCost
from lejudge.judge import KeywordJudge, OracleJudge
from lejudge.types import SymbolicState


class ToyBase(torch.nn.Module):
    def forward(self, info):
        z = info["predicted_emb"]
        return ((z[:, :, -1] - info["goal_emb"][:, None, -1]) ** 2).sum(-1)


class ToyProbe:
    """Latent = [ax, ay, bx, by, sin, cos, contact, 0...]: identity probe."""

    def symbolic(self, z):
        z = np.asarray(z)
        return SymbolicState(z[..., 0:2], z[..., 2:4], np.arctan2(z[..., 4], z[..., 5]), z[..., 6])


def _info(n=40, h=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    z = torch.zeros(1, n, h + 1, 192)
    z[..., 0:4] = torch.rand(1, n, h + 1, 4, generator=g) * 0.6 + 0.2
    z[..., 5] = 1.0
    z[..., 6] = -5.0
    # candidates 0..9 drive the block to the left wall at the last step
    z[0, :10, -1, 2] = 0.05
    # the goal image has the block at the left wall, so the wall candidates are the cheapest in A
    goal = torch.zeros(1, 1, 192)
    goal[..., 0:4] = torch.tensor([0.5, 0.5, 0.05, 0.5])
    goal[..., 5] = 1.0
    goal[..., 6] = -5.0
    z[0, :10, -1, 0:2] = 0.5
    z[0, :10, -1, 3] = 0.5
    return {"predicted_emb": z, "goal_emb": goal, "action_candidates": torch.zeros(1, n, h, 10)}


def test_lam_zero_is_identity(vocab, library):
    info = _info()
    base = ToyBase()
    cost = JevCost(
        base,
        ToyProbe(),
        vocab,
        library.set("edges"),
        OracleJudge(vocab, on_probes=True),
        lam=0.0,
        mode="per_iter",
    )
    out = cost(info)
    assert torch.equal(out, base(info))


def test_penalty_applies_to_elites_and_shape(vocab, library):
    info = _info()
    cost = JevCost(
        ToyBase(),
        ToyProbe(),
        vocab,
        library.set("edges"),
        OracleJudge(vocab, on_probes=True),
        lam=1.0,
        K=16,
        mode="per_iter",
    )
    out = cost(info)
    assert out.shape == (1, 40)
    tr = cost.history[-1]
    assert tr.judged and len(tr.elites) == 16 and tr.n_candidates == 40
    # the wall candidates are among the elites and get penalty 1
    pen = dict(zip(tr.elites, tr.penalty))
    assert all(pen[i] == 1.0 for i in range(10) if i in pen)
    assert sum(1 for i in range(10) if i in pen) >= 5
    # standardised cost: penalised candidates are no longer the cheapest
    assert int(out[0].argmin()) >= 10


def test_modes_control_call_count(vocab, library):
    for mode, n_iters, last_n, expected in (
        ("per_iter", 4, 0, 4),
        ("final_only", 4, 0, 1),
        ("last_n", 4, 2, 2),
        ("every_k", 7, 0, 3),
    ):
        cost = JevCost(
            ToyBase(),
            ToyProbe(),
            vocab,
            library.set("edges"),
            OracleJudge(vocab, on_probes=True),
            mode=mode,
            n_iters=n_iters,
            judge_last_n=last_n,
            judge_every=3,
        )
        cb = cost.callback()
        cb.reset()
        for i in range(n_iters):
            cost(_info(seed=i))
            cb(step=i)
        assert sum(t.judged for t in cost.history) == expected, mode


def test_gate_holds_uncertain_candidates(vocab, library):
    class Unsure:
        name = "unsure"

        def judge(self, facts, constraints, states=None):
            from lejudge.types import JudgeResult

            p = {k: {c.id: [0.5] * len(v) for c in constraints} for k, v in facts.items()}
            conf = {k: {c.id: None for c in constraints} for k in facts}
            return JudgeResult(p, conf, 1.0, 0, 0, "x", True)

    info = _info()
    cost = JevCost(
        ToyBase(),
        ToyProbe(),
        vocab,
        library.set("edges"),
        Unsure(),
        lam=1.0,
        mode="per_iter",
        tau=0.5,
    )
    cost(info)
    tr = cost.history[-1]
    assert all(tr.held) and all(p == 0.0 for p in tr.penalty)
    assert cost.stats()["held"] == 40


def test_keyword_judge_in_the_loop(vocab, library):
    info = _info()
    cost = JevCost(
        ToyBase(), ToyProbe(), vocab, library.set("edges"), KeywordJudge(), lam=2.0, mode="per_iter"
    )
    out = cost(info)
    assert torch.isfinite(out).all()


def test_hard_reject(vocab, library):
    info = _info()
    cost = JevCost(
        ToyBase(),
        ToyProbe(),
        vocab,
        library.set("edges"),
        OracleJudge(vocab, on_probes=True),
        mode="per_iter",
        hard_reject=True,
    )
    out = cost(info)
    assert torch.isinf(out).any()


class CountingJudge:
    """Fake remote judge: counts calls and questions; flags any step whose block is 'centre-left'."""

    name = "counting"
    model_id = "fake-model"

    def __init__(self):
        self.calls = 0
        self.questions = 0

    def judge(self, facts, constraints, states=None):
        from lejudge.types import JudgeResult

        self.calls += 1
        p, conf = {}, {}
        for k, seq in facts.items():
            p[k], conf[k] = {}, {}
            for c in constraints:
                self.questions += len(seq)
                p[k][c.id] = [1.0 if f.block_edge == "left edge" else 0.0 for f in seq]
                conf[k][c.id] = None
        return JudgeResult(p, conf, 1.0, 10, 1, "fake-model", False, n_calls=1)


def test_step_dedupe_path_judges_every_candidate_and_memoises(
    vocab, library, tmp_path, monkeypatch
):
    import lejudge.cost.jevcost as jc
    from lejudge.judge.cache import Cache

    monkeypatch.setattr(jc, "get_cache", lambda: Cache(tmp_path / "c.sqlite"))
    info = _info()
    judge = CountingJudge()
    cost = JevCost(
        ToyBase(), ToyProbe(), vocab, library.set("edges"), judge, lam=1.0, K=16, mode="per_iter"
    )
    out = cost(info)
    tr = cost.history[-1]
    assert tr.n_candidates == 40 and judge.calls == 1
    # far fewer questions than candidates × steps thanks to de-duplication of identical step facts
    assert judge.questions <= 40 * 3
    assert tr.unique_steps == tr.new_steps > 0
    # every wall candidate is penalised (by exactly lam) even if outside the K shortlist
    A = ToyBase()(info)
    A_std = (A - A.mean(dim=1, keepdim=True)) / A.std(dim=1, keepdim=True)
    assert torch.allclose(out[0, :10] - A_std[0, :10], torch.ones(10), atol=1e-5)
    assert torch.allclose(out[0, 10:], A_std[0, 10:], atol=1e-5)
    # second call on the same population: everything memoised, no new judge call
    cost(info)
    assert judge.calls == 1 and cost.history[-1].new_steps == 0
    # a fresh JevCost with a fresh in-process cache finds the steps in the persistent table
    cost2 = JevCost(
        ToyBase(), ToyProbe(), vocab, library.set("edges"), judge, lam=1.0, K=16, mode="per_iter"
    )
    cost2(info)
    assert judge.calls == 1 and cost2.stats()["step_db_hits"] > 0
