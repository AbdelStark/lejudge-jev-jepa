import torch

from lejudge.cost.swm_adapter import BestSampleCEMSolver


class FakeInner:
    def __init__(self):
        self.callbacks = []
        self.configured = None

    def configure(self, **kw):
        self.configured = kw

    @property
    def action_dim(self):
        return 10

    @property
    def n_envs(self):
        return 1

    @property
    def horizon(self):
        return 5

    def solve(self, info_dict, init_action=None):
        cands = torch.randn(1, 8, 5, 10)
        costs = torch.arange(8.0).flip(0).view(1, 8)  # candidate 7 is the cheapest
        for cb in self.callbacks:
            cb.reset()
            cb.start_batch()
            cb(step=0, candidates=cands, costs=costs)
            cb.end_solve()
        self._cands = cands
        return {"actions": cands[:, :4].mean(dim=1), "costs": [0.0]}


def test_best_sample_solver_returns_lowest_cost_candidate():
    inner = FakeInner()
    s = BestSampleCEMSolver(inner)
    s.configure(action_space=None, n_envs=1, config=None)
    out = s.solve({"x": torch.zeros(1)})
    assert torch.equal(out["actions"], inner._cands[:, 7])
    assert "mean_actions" in out and s.horizon == 5 and s.action_dim == 10
