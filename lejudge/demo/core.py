"""Demo backend: plan the same start/goal with stock LeWM and with LeJudge, return frames,
per-step probabilities from the trace, the JSON state sent and the call metadata."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from lejudge.constraints import check, load_library
from lejudge.cost.jevcost import JevCost
from lejudge.cost.swm_adapter import (
    PlanSpec,
    goal_mse_objective,
    load_config,
    load_lewm,
    make_policy,
    make_world,
    shooting_cost,
)
from lejudge.judge import JevJudge, OracleJudge
from lejudge.judge.aggregate import penalty
from lejudge.probes.data import EpisodeData
from lejudge.probes.model import load_probe
from lejudge.types import Constraint, GroundTruthState
from lejudge.vocab import load_vocab

GALLERY_DIR = Path("artifacts/demo/gallery")
PRESETS = [
    "Never let the T touch the left or right edge.",
    "Keep the T roughly upright.",
    "Keep the T out of the centre cell.",
    "The agent must not enter the top row.",
    "Be gentle: move the block slowly.",
    "Stay out of all four corners with the block.",
]
MAX_LINES, MAX_CHARS = 3, 200


@dataclass
class PlanResult:
    frames: list[np.ndarray]
    states: list[GroundTruthState]
    success: bool
    probs: dict[
        str, list[float]
    ]  # constraint text -> per-step p of the chosen elite (last judged iter)
    penalties: dict[str, float]
    state_json: str
    call: dict[str, Any]
    facts: list[dict[str, Any]]
    oracle: dict[str, bool]
    plan_time_s: float
    held: bool = False


@dataclass
class DemoBackend:
    device: str | None = None
    data_path: str = "artifacts/data/pusht_expert.npz"
    probes: str = "pusht/linear@1"
    vocab_name: str = "pusht@1"
    budget: int = 50
    goal_offset: int = 25
    num_samples: int = 100
    n_steps: int = 10
    judge_last_n: int = 2
    K: int = 8
    _res: dict[str, Any] = field(default_factory=dict)

    def load(self) -> None:
        if self._res:
            return
        self._res["config"] = load_config()
        self._res["model"] = load_lewm(self.device)
        self._res["data"] = EpisodeData(self.data_path)
        self._res["probe"] = load_probe(self.probes)
        self._res["vocab"] = load_vocab(self.vocab_name)
        self._res["lib"] = load_library()
        self._res["world"] = make_world(1, 100)

    def episode_spec(self, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        from lejudge.eval.planning import sample_episode_specs

        es = sample_episode_specs(self._res["data"], seed + 1, 777, self.goal_offset)[seed]
        world = self._res["world"]
        world.reset(seed=0, options=[{"state": es.state, "goal_state": es.goal_state}])
        goal_img = world.infos["goal"][0, -1].copy()
        return es.state, es.goal_state, goal_img

    @staticmethod
    def parse_constraints(text: str) -> list[Constraint]:
        lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()][:MAX_LINES]
        out = []
        for i, ln in enumerate(lines):
            ln = ln[:MAX_CHARS]
            fam = (
                "soft"
                if any(w in ln.lower() for w in ("gentle", "slowly", "softly", "slow "))
                else (
                    "always"
                    if any(
                        w in ln.lower()
                        for w in ("keep the t", "stay ", "always", "must remain", "approach")
                    )
                    and "out of" not in ln.lower()
                    and "away" not in ln.lower()
                    and "off" not in ln.lower()
                    else "never"
                )
            )
            out.append(Constraint(text=ln, family=fam, id=f"user{i + 1}", weight=1.0))
        return out

    def plan(
        self,
        seed: int,
        constraints: list[Constraint],
        lam: float = 1.0,
        tau: float = 0.5,
        mode: str = "jev",
        record_only: bool = False,
    ) -> PlanResult:
        self.load()
        model, probe, vocab, world = (
            self._res["model"],
            self._res["probe"],
            self._res["vocab"],
            self._res["world"],
        )
        state, goal_state, _ = self.episode_spec(seed)
        spec = PlanSpec.from_config(
            self._res["config"],
            seed=1234 + seed,
            device=self.device,
            num_samples=self.num_samples,
            n_steps=self.n_steps,
        )
        judge = JevJudge() if mode == "jev" else OracleJudge(vocab, on_probes=True)
        objective = JevCost(
            goal_mse_objective(),
            probe,
            vocab,
            constraints,
            judge,
            lam=lam,
            K=self.K,
            tau=tau,
            mode="every_k",
            n_iters=spec.n_steps,
            judge_every=3,
            record_only=record_only,
        )
        policy = make_policy(
            shooting_cost(model, objective),
            spec,
            self._res["data"].scaler,
            callbacks=[objective.callback()],
        )
        world.set_policy(policy)
        world.reset(seed=0, options=[{"state": state, "goal_state": goal_state}])
        frames = [world.infos["pixels"][0, -1].copy()]
        states = [GroundTruthState.from_env(world.infos["state"][0, -1])]
        success = False
        t0 = time.time()
        for _ in range(self.budget):
            a = policy.get_action(world.infos)
            _, _, term, trunc, world.infos = world.envs.step(a)
            frames.append(world.infos["pixels"][0, -1].copy())
            states.append(GroundTruthState.from_env(world.infos["state"][0, -1]))
            if bool(term[0]):
                success = True
                break
        plan_time = time.time() - t0
        probs: dict[str, list[float]] = {}
        pens: dict[str, float] = {}
        state_json, call, facts, held = "{}", {}, [], False
        judged = [t for t in objective.history if t.judged and t.p]
        if judged:
            tr = judged[-1]
            # chosen elite = lowest A + penalty among elites of the last judged iteration
            tot = [a + (0.0 if record_only else lam * p) for a, p in zip(tr.costA, tr.penalty)]
            j = int(np.argmin(tot))
            key = f"e{j}"
            for c in constraints:
                p = tr.p.get(key, {}).get(c.id, [])
                probs[c.text] = [float(x) for x in p]
                pens[c.text] = penalty(c.family, [x for x in p if x == x]) if p else 0.0
            held = bool(tr.held[j])
            facts = tr.facts[key]
            state_json = json.dumps(
                {
                    "constraints": {f"c{i + 1}": c.text for i, c in enumerate(constraints)},
                    "candidates": {"k1": facts},
                },
                indent=1,
            )
            call = {
                k: (tr.judge or {}).get(k)
                for k in (
                    "latency_ms",
                    "input_tokens",
                    "output_tokens",
                    "response_model",
                    "cache_hit",
                    "n_calls",
                )
            }
            call["all_steps_memoised"] = tr.judge is None
            call["judged_iterations"] = len(judged)
            call["unique_steps_last_iter"] = tr.unique_steps
            call["new_steps_last_iter"] = tr.new_steps
        oracle = {}
        for c in constraints:
            lib_c = next((x for x in self._res["lib"].constraints if x.text == c.text), None)
            if lib_c is not None:
                oracle[c.text] = bool(check(lib_c, states, vocab).episode)
        return PlanResult(
            frames, states, success, probs, pens, state_json, call, facts, oracle, plan_time, held
        )

    def plan_pair(
        self, seed: int, text: str, lam: float = 1.0, tau: float = 0.5, mode: str = "jev"
    ) -> tuple[PlanResult, PlanResult]:
        cs = self.parse_constraints(text)
        stock = self.plan(seed, cs, lam=0.0, tau=tau, mode=mode, record_only=True)
        ours = self.plan(seed, cs, lam=lam, tau=tau, mode=mode)
        return stock, ours

    # ---- gallery ----------------------------------------------------------------------
    def gallery_path(self, seed: int, text: str) -> Path:
        import hashlib

        h = hashlib.sha1(text.strip().encode()).hexdigest()[:10]
        return GALLERY_DIR / f"s{seed}_{h}.npz"

    def precompute(
        self,
        seeds: range,
        presets: list[str],
        lam: float = 1.0,
        tau: float = 0.5,
        mode: str = "jev",
    ) -> None:
        GALLERY_DIR.mkdir(parents=True, exist_ok=True)
        for s in seeds:
            for p in presets:
                path = self.gallery_path(s, p)
                if path.exists():
                    continue
                stock, ours = self.plan_pair(s, p, lam, tau, mode)
                save_pair(path, stock, ours)


def save_pair(path: Path, stock: PlanResult, ours: PlanResult) -> None:
    np.savez_compressed(
        path,
        stock_frames=np.stack(stock.frames),
        ours_frames=np.stack(ours.frames),
        meta=json.dumps({"stock": _meta(stock), "ours": _meta(ours)}),
    )


def _meta(r: PlanResult) -> dict[str, Any]:
    return {
        "success": r.success,
        "probs": r.probs,
        "penalties": r.penalties,
        "state_json": r.state_json,
        "call": r.call,
        "facts": r.facts,
        "oracle": r.oracle,
        "plan_time_s": r.plan_time_s,
        "held": r.held,
        "states": [s.to_json() for s in r.states],
    }


def load_pair(path: Path) -> tuple[PlanResult, PlanResult]:
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))

    def mk(frames: np.ndarray, m: dict[str, Any]) -> PlanResult:
        return PlanResult(
            list(frames),
            [GroundTruthState.from_json(s) for s in m["states"]],
            m["success"],
            m["probs"],
            m["penalties"],
            m["state_json"],
            m["call"],
            m["facts"],
            m["oracle"],
            m["plan_time_s"],
            m.get("held", False),
        )

    return mk(z["stock_frames"], meta["stock"]), mk(z["ours_frames"], meta["ours"])
