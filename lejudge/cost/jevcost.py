"""JevCost: a stable-worldmodel ``Objective`` that adds a language-constraint penalty to
LeWM's goal-distance cost (RFC-0004, amended 2026-09-22 — see docs/DECISIONS.md).

Every candidate is penalised, not only a shortlist:

* ``never`` / ``always`` constraints ask one question per (constraint, step) that depends only
  on that step's words and index, so the population's ``S × H`` steps are de-duplicated into
  unique ``(t, facts)`` keys, judged once, and cached in-process across CEM iterations,
  replans and episodes. Candidates inherit their steps' probabilities.
* ``soft`` / ``temporal_before`` constraints need whole sequences: the ``K`` lowest-cost
  candidates are judged; the others receive the shortlist's mean penalty as a prior.
* Judges that are free and local (oracle-on-probes, keyword) judge every candidate's full
  sequence directly.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import nn

from lejudge.judge.aggregate import penalty as agg_penalty
from lejudge.judge.aggregate import uncertainty_proxy
from lejudge.judge.bank import BANK_VERSION
from lejudge.judge.base import Judge
from lejudge.judge.cache import TraceWriter, get_cache
from lejudge.types import Constraint, JudgeResult, StepFacts
from lejudge.vocab import Vocab, words

MODES = ("per_iter", "final_only", "last_n", "every_k")
HARD = ("never", "always")


@dataclass
class JudgeTrace:
    """What one ``forward`` did; appended to ``JevCost.history`` and to the trace file."""

    iter: int
    judged: bool
    elites: list[int] = field(default_factory=list)
    costA: list[float] = field(default_factory=list)
    penalty: list[float] = field(default_factory=list)
    held: list[bool] = field(default_factory=list)
    facts: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    judge: dict[str, Any] | None = None
    p: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    wall_ms: float = 0.0
    n_candidates: int = 0
    unique_steps: int = 0
    new_steps: int = 0
    penalised_fraction: float = 0.0


class JevCost(nn.Module):
    """``cost = standardise(A) + lam · Σ_c w_c · penalty_c`` with a confidence gate.

    Args:
        base: objective producing ``A`` of shape ``(B, S)`` (``GoalMSE``).
        probe, vocab: latent → symbols → words.
        constraints: the constraint set (canonical or paraphrased texts).
        judge: any ``Judge``.
        lam: penalty weight (``lam=0`` returns ``base`` unchanged — bit-identical baseline).
        K: shortlist size for sequence-level (soft/temporal) judging and for elite traces.
        tau: gate threshold on the uncertainty proxy.
        mode: ``per_iter`` | ``final_only`` | ``last_n`` | ``every_k``.
        n_iters, judge_last_n, judge_every: schedule parameters.
        steps: judge only the first ``steps`` imagined steps (``None`` = all).
        standardize: standardise ``A`` per batch row before adding the penalty.
        hard_reject: ablation — ``+inf`` instead of ``lam * pen`` when ``pen > 0.7``.
        unjudged: prior for candidates a sequence-level judge did not see: ``"mean"`` | ``"none"``.
        max_new_steps: cap on newly judged unique step-facts per iteration, lowest-cost candidates'
            steps first. Steps over the cap stay unjudged this iteration: a candidate's hard-family
            penalty uses its known steps only, and a candidate with no known step is held.
        smooth_temperature: ablation — softmax-weighted mean instead of ``max`` for ``never``.
        record_only: judge for the trace only; never change the cost (demo: stock plan bars).
    """

    def __init__(
        self,
        base: Any,
        probe: Any,
        vocab: Vocab,
        constraints: list[Constraint],
        judge: Judge,
        lam: float = 1.0,
        K: int = 16,
        tau: float = 0.5,
        mode: str = "every_k",
        n_iters: int = 30,
        judge_last_n: int = 3,
        judge_every: int = 5,
        steps: int | None = None,
        standardize: bool = True,
        hard_reject: bool = False,
        smooth_temperature: float | None = None,
        trace: TraceWriter | None = None,
        record_only: bool = False,
        unjudged: str = "mean",
        max_new_steps: int = 1024,
    ) -> None:
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if unjudged not in ("mean", "none"):
            raise ValueError("unjudged must be 'mean' or 'none'")
        self.base = base
        self.probe = probe
        self.vocab = vocab
        self.constraints = list(constraints)
        self.judge = judge
        self.lam = float(lam)
        self.K = int(K)
        self.tau = float(tau)
        self.mode = mode
        self.n_iters = int(n_iters)
        self.judge_last_n = int(judge_last_n)
        self.judge_every = max(1, int(judge_every))
        self.steps = steps
        self.standardize = standardize
        self.hard_reject = hard_reject
        self.smooth_temperature = smooth_temperature
        self.trace = trace
        self.record_only = record_only
        self.unjudged = unjudged
        self.max_new_steps = int(max_new_steps)
        self.local_judge = bool(getattr(judge, "local", False))
        self._step_cache: dict[tuple[str, str], float] = {}
        self._judge_id = str(
            getattr(judge, "model_id", None) or getattr(judge, "name", type(judge).__name__)
        )
        self._persist = not self.local_judge
        self._db = get_cache() if self._persist else None
        self._iter = 0
        self.history: list[JudgeTrace] = []
        self.reset_stats()

    # -- CEM iteration bookkeeping (see ``IterCallback``) --------------------------------
    def reset_iter(self) -> None:
        self._iter = 0

    def callback(self) -> IterCallback:
        return IterCallback(self)

    def _should_judge(self) -> bool:
        if not self.constraints or (self.lam == 0.0 and not self.record_only):
            return False
        if self.mode == "per_iter":
            return True
        if self.mode == "final_only":
            return self._iter >= self.n_iters - 1
        if self.mode == "every_k":
            return self._iter % self.judge_every == 0 or self._iter >= self.n_iters - 1
        return self._iter >= self.n_iters - self.judge_last_n

    # -- objective ------------------------------------------------------------------------
    def forward(self, info_dict: dict[str, Any]) -> torch.Tensor:
        self.calls += 1
        A = self.base(info_dict)  # (B, S)
        if not self._should_judge():
            self._record(JudgeTrace(iter=self._iter, judged=False))
            self._iter += 1
            return A
        t0 = time.perf_counter()
        cost = A.clone()
        if self.standardize and not self.record_only:
            mean = A.mean(dim=1, keepdim=True)
            std = A.std(dim=1, keepdim=True).clamp_min(1e-8)
            cost = (A - mean) / std
        z = info_dict["predicted_emb"]  # (B, S, H_ctx + horizon, D)
        B, S = A.shape
        h_ctx = (
            z.shape[2] - info_dict["action_candidates"].shape[2]
            if "action_candidates" in info_dict
            else 1
        )
        for b in range(B):
            zb = z[b][:, h_ctx - 1 :]
            if self.steps is not None:
                zb = zb[:, : self.steps + 1]
            sym = self.probe.symbolic(zb.detach().float().cpu())  # (S, H+1)
            facts_all = [words(sym[j], self.vocab) for j in range(S)]
            order = torch.argsort(cost[b])
            pen, held, meta = self._penalties(facts_all, sym, order.tolist())
            pen_t = torch.tensor(pen, dtype=cost.dtype, device=cost.device)
            if self.record_only:
                pass
            elif self.hard_reject:
                cost[b] = torch.where(pen_t > 0.7, torch.full_like(pen_t, float("inf")), cost[b])
            else:
                cost[b] += self.lam * pen_t
            k = min(self.K, S)
            idx = order[:k].tolist()
            self.held_total += int(sum(held))
            self.judged_total += S
            self._record(
                JudgeTrace(
                    iter=self._iter,
                    judged=True,
                    elites=[int(i) for i in idx],
                    costA=[float(A[b, i]) for i in idx],
                    penalty=[float(pen[i]) for i in idx],
                    held=[bool(held[i]) for i in idx],
                    facts={f"e{r}": [f.to_json() for f in facts_all[i]] for r, i in enumerate(idx)},
                    judge=meta.get("judge"),
                    p={f"e{r}": meta["p"][i] for r, i in enumerate(idx)},
                    wall_ms=(time.perf_counter() - t0) * 1000.0,
                    n_candidates=S,
                    unique_steps=int(meta.get("unique_steps", 0)),
                    new_steps=int(meta.get("new_steps", 0)),
                    penalised_fraction=float(np.mean(np.asarray(pen) > 0.5)),
                )
            )
        self._iter += 1
        return cost

    # -- penalty computation ---------------------------------------------------------------
    def _penalties(
        self, facts_all: list[list[StepFacts]], sym: Any, order: list[int]
    ) -> tuple[list[float], list[bool], dict[str, Any]]:
        S = len(facts_all)
        total = np.zeros(S)
        unsure = np.zeros(S)
        meta: dict[str, Any] = {"unique_steps": 0, "new_steps": 0, "judge": None}
        probs: dict[int, dict[str, list[float]]] = {j: {} for j in range(S)}
        meta["p"] = probs
        hard = [c for c in self.constraints if c.family in HARD]
        seqc = [c for c in self.constraints if c.family not in HARD]
        # ---- local judges (oracle / keyword): judge every candidate's full sequence -------
        if self.local_judge:
            facts = {f"e{j}": facts_all[j] for j in range(S)}
            states = {f"e{j}": sym[j].to_ground_truth() for j in range(S)}
            res = self.judge.judge(facts, self.constraints, states=states)
            self._account(res)
            meta["judge"] = res.to_json()
            for j in range(S):
                for c in self.constraints:
                    p = res.p[f"e{j}"][c.id]
                    probs[j][c.id] = [float(x) for x in p]
                    total[j] += c.weight * agg_penalty(c.family, p, self.smooth_temperature)
                    unsure[j] = max(
                        unsure[j], uncertainty_proxy(c.family, p, res.confidence[f"e{j}"][c.id])
                    )
            held = unsure > self.tau
            return list(np.where(held, 0.0, total)), [bool(h) for h in held], meta
        # ---- hard families: de-duplicated (t, facts) steps, cached across iterations ------
        if hard:
            keys: list[list[str]] = []
            uniq: dict[str, StepFacts] = {}
            for j in range(S):
                row = []
                for f in facts_all[j]:
                    kkey = json.dumps(f.to_json(), sort_keys=True)
                    row.append(kkey)
                    uniq.setdefault(kkey, f)
                keys.append(row)
            # first level: in-process dict; second level: persistent stepfacts table shared by every run
            if self._db is not None:
                lookup = {
                    (c.id, k): self._db.step_key(self._judge_id, BANK_VERSION, c.text, c.family, k)
                    for c in hard
                    for k in uniq
                    if (c.id, k) not in self._step_cache
                }
                if lookup:
                    found = self._db.get_steps(list(lookup.values()))
                    for (cid, k), sk in lookup.items():
                        if sk in found:
                            self._step_cache[(cid, k)] = found[sk]
                    self.step_db_hits += len(found)
            new = [k for k in uniq if any((c.id, k) not in self._step_cache for c in hard)]
            meta["unique_steps"] = len(uniq)
            # judge the most promising new steps first (those of the lowest-cost candidates)
            if len(new) > self.max_new_steps:
                prio: dict[str, int] = {}
                for rank, j in enumerate(order):
                    for kkey in keys[j]:
                        prio.setdefault(kkey, rank)
                new = sorted(new, key=lambda k: prio.get(k, 10**9))[: self.max_new_steps]
            meta["new_steps"] = len(new)
            if new:
                facts = {f"s{i}": [uniq[k]] for i, k in enumerate(new)}
                res = self.judge.judge(facts, hard)
                self._account(res)
                meta["judge"] = res.to_json()
                rows = []
                for i, k in enumerate(new):
                    for c in hard:
                        p = res.p[f"s{i}"][c.id]
                        val = float(p[0]) if p else float("nan")
                        self._step_cache[(c.id, k)] = val
                        if self._db is not None and not np.isnan(val) and not res.failed:
                            rows.append(
                                (
                                    self._db.step_key(
                                        self._judge_id, BANK_VERSION, c.text, c.family, k
                                    ),
                                    self._judge_id,
                                    BANK_VERSION,
                                    val,
                                )
                            )
                if rows:
                    self._db.put_steps(rows)
            for j in range(S):
                for c in hard:
                    p = [self._step_cache.get((c.id, k), float("nan")) for k in keys[j]]
                    probs[j][c.id] = [float(x) for x in p]
                    known = [x for x in p if not np.isnan(x)]
                    if not known:
                        unsure[j] = 1.0
                        continue
                    total[j] += c.weight * agg_penalty(c.family, known, self.smooth_temperature)
                    unsure[j] = max(unsure[j], uncertainty_proxy(c.family, known, None))
        # ---- sequence families: shortlist + prior ---------------------------------------
        if seqc:
            k = min(self.K, S)
            idx = order[:k]
            facts = {f"e{j}": facts_all[j] for j in idx}
            res = self.judge.judge(facts, seqc)
            self._account(res)
            if meta["judge"] is None:
                meta["judge"] = res.to_json()
            short = np.zeros(S)
            for j in idx:
                for c in seqc:
                    p = res.p[f"e{j}"][c.id]
                    probs[j][c.id] = [float(x) for x in p]
                    conf = res.confidence[f"e{j}"][c.id]
                    if all(np.isnan(p)):
                        unsure[j] = 1.0
                        continue
                    short[j] += c.weight * agg_penalty(c.family, p, self.smooth_temperature)
                    unsure[j] = max(unsure[j], uncertainty_proxy(c.family, p, conf))
            prior = (
                float(np.mean([short[j] for j in idx]))
                if (self.unjudged == "mean" and idx)
                else 0.0
            )
            for j in range(S):
                total[j] += short[j] if j in idx else prior
        held = unsure > self.tau
        return list(np.where(held, 0.0, total)), [bool(h) for h in held], meta

    def _account(self, res: JudgeResult) -> None:
        self.judge_calls += res.n_calls
        self.judge_wall_ms += res.latency_ms
        self.tokens_in += res.input_tokens
        self.tokens_out += res.output_tokens
        self.judge_failures += int(res.failed)

    def _record(self, tr: JudgeTrace) -> None:
        self.history.append(tr)
        if self.trace is not None and tr.judged:
            self.trace.write(
                {
                    "iter": tr.iter,
                    "elites": tr.elites,
                    "costA": tr.costA,
                    "penalty": tr.penalty,
                    "held": tr.held,
                    "facts": tr.facts,
                    "p": tr.p,
                    "judge": tr.judge,
                    "wall_ms": round(tr.wall_ms, 2),
                    "n_candidates": tr.n_candidates,
                    "unique_steps": tr.unique_steps,
                    "new_steps": tr.new_steps,
                    "penalised_fraction": round(tr.penalised_fraction, 4),
                }
            )

    def stats(self) -> dict[str, Any]:
        return {
            "forward_calls": self.calls,
            "judge_calls": self.judge_calls,
            "judge_wall_ms": round(self.judge_wall_ms, 1),
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "held": self.held_total,
            "judged": self.judged_total,
            "judge_failures": self.judge_failures,
            "step_cache_size": len(self._step_cache),
            "step_db_hits": self.step_db_hits,
        }

    def reset_stats(self) -> None:
        self.history = []
        self.calls = self.judge_calls = 0
        self.judge_wall_ms = 0.0
        self.tokens_in = self.tokens_out = 0
        self.held_total = self.judged_total = 0
        self.judge_failures = 0
        self.step_db_hits = 0

    def penalties(self, res: JudgeResult, keys: list[str]) -> tuple[list[float], list[bool]]:
        """Aggregate a sequence-level ``JudgeResult`` (kept for tests and the demo)."""
        pen: list[float] = []
        held: list[bool] = []
        for kk in keys:
            total = 0.0
            unsure = 0.0
            for c in self.constraints:
                p = res.p[kk][c.id]
                conf = res.confidence[kk][c.id]
                if all(np.isnan(p)):
                    unsure = 1.0
                    continue
                total += c.weight * agg_penalty(c.family, p, self.smooth_temperature)
                unsure = max(unsure, uncertainty_proxy(c.family, p, conf))
            h = unsure > self.tau
            pen.append(0.0 if h else total)
            held.append(h)
        return pen, held


class IterCallback:
    """CEM solver callback that resets the iteration counter at each solve."""

    output_key = "lejudge"

    def __init__(self, cost: JevCost) -> None:
        self.cost = cost
        self.history: list[Any] = []

    def reset(self) -> None:
        self.cost.reset_iter()

    def start_batch(self) -> None:
        self.cost.reset_iter()

    def __call__(self, **kwargs: Any) -> None:
        self.cost._iter = int(kwargs.get("step", self.cost._iter - 1)) + 1

    def end_solve(self) -> None:
        pass
