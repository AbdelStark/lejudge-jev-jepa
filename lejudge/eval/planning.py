"""Planning study runner (RFC-0006): conditions × constraint sets × seeds × episodes.

Episodes start from (state, goal) pairs sampled from the collected episodes with the
le-wm protocol (goal = the state ``goal_offset`` env steps later; budget ``eval_budget``).
Every CEM iteration that judges is traced (RFC-0008); every episode becomes one row.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from lejudge.constraints import Library, check, load_library
from lejudge.cost.jevcost import JevCost
from lejudge.cost.swm_adapter import (
    ActionScaler,
    PlanSpec,
    goal_mse_objective,
    load_config,
    load_lewm,
    make_policy,
    make_world,
    shooting_cost,
)
from lejudge.judge import JevJudge, KeywordJudge, LLMJudge, OracleJudge, TraceWriter
from lejudge.judge.bank import BANK_VERSION
from lejudge.probes.data import EpisodeData
from lejudge.probes.model import load_probe
from lejudge.types import Constraint, GroundTruthState
from lejudge.vocab import Vocab, load_vocab

CONDITIONS = ("lewm", "oracle", "keyword", "jev", "llm-small")
RESULTS_DIR = Path("artifacts/results")


@dataclass
class EpisodeSpec:
    episode: int
    source_episode: int
    start_step: int
    state: np.ndarray
    goal_state: np.ndarray


@dataclass
class RunConfig:
    condition: str = "jev"
    constraint_set: str = "edges"
    seed: int = 0
    episodes: int = 20
    lam: float = 1.0
    K: int = 16
    tau: float = 0.5
    mode: str = "every_k"
    judge_last_n: int = 3
    judge_every: int = 5
    steps: int | None = None
    vocab: str = "pusht@1"
    probes: str = "pusht/linear@1"
    hard_reject: bool = False
    unjudged: str = "mean"
    start_filter: str = "none"
    paraphrase: str = "canonical"  # which text variant the judge sees
    data_path: str = "artifacts/data/pusht_expert.npz"
    device: str | None = None
    tag: str = ""
    plan_overrides: dict[str, Any] = field(default_factory=dict)
    llm_model: str = "qwen2.5:7b-instruct"

    def to_json(self) -> dict[str, Any]:
        return {k: (v if not isinstance(v, dict) else dict(v)) for k, v in self.__dict__.items()}


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "nogit"


def run_id(cfg: RunConfig) -> str:
    h = hashlib.sha256(json.dumps(cfg.to_json(), sort_keys=True).encode()).hexdigest()[:10]
    return f"{git_commit()}_{cfg.condition}_{cfg.constraint_set}_s{cfg.seed}_{h}"


def sample_episode_specs(data: EpisodeData, n: int, seed: int, goal_offset: int = 25, min_start: int = 0, start_filter: str = "none", constraint_set: str | None = None, vocab: Vocab | None = None, lib: Library | None = None) -> list[EpisodeSpec]:
    """Deterministic (episode, start) pairs: one per evaluation episode, seeded.

    ``start_filter="relevant"`` (Study 2) keeps only windows for which ``constraint_set`` is
    satisfiable from the start and goal states and the expert trajectory is in tension with it
    (see ``lejudge.eval.filters``); windows are then drawn without replacement.
    """
    rng = np.random.default_rng(10_000 + seed)
    eps = data.split(seed=0)["test"] if len(data.episodes) >= 20 else data.episodes
    out: list[EpisodeSpec] = []
    if start_filter == "none":
        for i in range(n):
            e = int(rng.choice(eps))
            n_frames = data.length(e)
            t0 = int(rng.integers(min_start, n_frames - goal_offset - 1))
            ep = data.episode(e)
            out.append(EpisodeSpec(i, e, t0, ep["state"][t0].copy(), ep["state"][t0 + goal_offset].copy()))
        return out
    from lejudge.eval.filters import relevant

    assert constraint_set is not None
    vocab = vocab or load_vocab("pusht@1")
    lib = lib or load_library()
    pool: list[tuple[int, int]] = []
    for e in eps:
        ep = data.episode(int(e))
        gts = [GroundTruthState.from_env(ep["state"][i]) for i in range(len(ep["state"]))]
        for t0 in range(min_start, len(gts) - goal_offset - 1):
            if relevant(constraint_set, gts[t0 : t0 + goal_offset + 1], vocab, lib):
                pool.append((int(e), t0))
    if not pool:
        raise ValueError(f"no relevant windows for set {constraint_set!r}")
    replace = len(pool) < n
    picks = rng.choice(len(pool), size=n, replace=replace)
    for i, k in enumerate(picks):
        e, t0 = pool[int(k)]
        ep = data.episode(e)
        out.append(EpisodeSpec(i, e, t0, ep["state"][t0].copy(), ep["state"][t0 + goal_offset].copy()))
    return out


def build_judge(condition: str, vocab: Vocab, cfg: RunConfig) -> Any:
    if condition == "oracle":
        return OracleJudge(vocab, on_probes=True)
    if condition == "keyword":
        return KeywordJudge()
    if condition == "jev":
        return JevJudge()
    if condition == "llm-small":
        return LLMJudge(model=cfg.llm_model)
    raise ValueError(condition)


def constraints_for(lib: Library, cfg: RunConfig) -> list[Constraint]:
    cs = lib.set(cfg.constraint_set)
    if cfg.paraphrase == "canonical":
        return cs
    return [lib.variants(c.id)[cfg.paraphrase] for c in cs]


class Runner:
    """Holds the loaded model/probe/vocab so several runs share them."""

    def __init__(self, device: str | None = None, data_path: str = "artifacts/data/pusht_expert.npz"):
        self.config = load_config()
        self.device = device
        self.model = load_lewm(device)
        self.data = EpisodeData(data_path)
        self.scaler: ActionScaler = self.data.scaler
        self.lib = load_library()
        self._probes: dict[str, Any] = {}
        self._vocabs: dict[str, Vocab] = {}
        self.world = None
        self.max_episode_steps = int(self.config["eval"]["max_episode_steps"])

    def probe(self, name: str) -> Any:
        if name not in self._probes:
            self._probes[name] = load_probe(name)
        return self._probes[name]

    def vocab(self, name: str) -> Vocab:
        if name not in self._vocabs:
            self._vocabs[name] = load_vocab(name)
        return self._vocabs[name]

    def _world(self) -> Any:
        if self.world is None:
            self.world = make_world(1, self.max_episode_steps)
        return self.world

    def run(self, cfg: RunConfig, trace_root: Path | str = "artifacts/traces", progress: bool = True) -> pd.DataFrame:
        ecfg = self.config["eval"]
        goal_offset, budget = int(ecfg["goal_offset_steps"]), int(ecfg["eval_budget"])
        spec = PlanSpec.from_config(self.config, seed=1234 + cfg.seed, device=cfg.device or self.device, **cfg.plan_overrides)
        vocab = self.vocab(cfg.vocab)
        probe = self.probe(cfg.probes)
        constraints = constraints_for(self.lib, cfg)
        canonical = self.lib.set(cfg.constraint_set)
        rid = run_id(cfg)
        trace = TraceWriter(rid, trace_root)
        base = goal_mse_objective()
        if cfg.condition == "lewm":
            objective: Any = JevCost(base, probe, vocab, constraints, OracleJudge(vocab, on_probes=True), lam=0.0, n_iters=spec.n_steps, trace=trace)
        else:
            objective = JevCost(base, probe, vocab, constraints, build_judge(cfg.condition, vocab, cfg), lam=cfg.lam, K=cfg.K, tau=cfg.tau, mode=cfg.mode, n_iters=spec.n_steps, judge_last_n=cfg.judge_last_n, judge_every=cfg.judge_every, steps=cfg.steps, hard_reject=cfg.hard_reject, unjudged=cfg.unjudged, trace=trace)
        cost = shooting_cost(self.model, objective)
        policy = make_policy(cost, spec, self.scaler, callbacks=[objective.callback()])
        world = self._world()
        world.set_policy(policy)
        specs = sample_episode_specs(self.data, cfg.episodes, cfg.seed, goal_offset, start_filter=cfg.start_filter, constraint_set=cfg.constraint_set, vocab=self.vocab("pusht@1"), lib=self.lib)
        (Path(trace_root) / rid).mkdir(parents=True, exist_ok=True)
        (Path(trace_root) / rid / "config.json").write_text(json.dumps({"run": cfg.to_json(), "plan": spec.to_json(), "constraints": [c.text for c in constraints], "library_sha256": self.lib.sha256, "bank": BANK_VERSION, "checkpoint": self.config["checkpoint"], "device": str(next(self.model.parameters()).device), "torch": torch.__version__}, indent=2))
        rows = []
        t_run = time.time()
        for es in specs:
            row = self._episode(world, policy, objective, es, cfg, canonical, vocab, trace, budget, rid, spec)
            rows.append(row)
            if progress:
                print(f"[{cfg.condition}/{cfg.constraint_set}/s{cfg.seed}] ep {es.episode + 1}/{cfg.episodes} success={row['success']} viol={row['violation']} plan_p50={row['plan_time_p50_s']:.2f}s judge_calls={row['judge_calls']} ({time.time() - t_run:.0f}s)", flush=True)
        trace.close()
        return pd.DataFrame(rows)

    def _episode(self, world: Any, policy: Any, objective: JevCost, es: EpisodeSpec, cfg: RunConfig, canonical: list[Constraint], vocab: Vocab, trace: TraceWriter, budget: int, rid: str, spec: PlanSpec) -> dict[str, Any]:
        trace.open_episode(es.episode)
        trace.context = {"cond": cfg.condition, "set": cfg.constraint_set, "seed": cfg.seed, "paraphrase": cfg.paraphrase}
        objective.reset_stats()
        # fresh policy buffers for a new episode
        policy._action_buffer = [type(policy._action_buffer[0])(maxlen=policy.flatten_receding_horizon)]
        policy._next_init = None
        world.reset(seed=int(es.source_episode * 1000 + es.start_step), options=[{"state": es.state, "goal_state": es.goal_state}])
        executed: list[GroundTruthState] = [GroundTruthState.from_env(world.infos["state"][0, -1])]
        plan_times: list[float] = []
        success = False
        step_records = []
        for t in range(budget):
            t0 = time.perf_counter()
            action = policy.get_action(world.infos)
            dt = time.perf_counter() - t0
            if dt > 0.05:  # a replan happened (buffer refill); buffered steps are ~free
                plan_times.append(dt)
                trace.context["step"] = t
            _, _, term, trunc, world.infos = world.envs.step(action)
            gt = GroundTruthState.from_env(world.infos["state"][0, -1])
            executed.append(gt)
            step_records.append({"t": t, "state": gt.to_json(), "terminated": bool(term[0])})
            if bool(term[0]):
                success = True  # the reference protocol stops the episode at success
                break
            if bool(trunc[0]):
                break
        # oracle on executed states (ground truth), per canonical constraint, at the planning
        # cadence (one state per action block) so words and thresholds match what was judged
        cadence = executed_at_cadence(executed, spec.action_block)
        viol_flags = {}
        viol_steps = {}
        study_vocab = self.vocab("pusht@1")  # executed-state labels always use the study vocabulary
        for c in canonical:
            tr = check(c, cadence, study_vocab)
            viol_flags[c.id] = bool(tr.episode)
            viol_steps[c.id] = int(sum(tr.steps))
        trace.write({"iter": -1, "summary": True, "executed": [s.to_json() for s in executed], "oracle": viol_flags, "success": success})
        st = objective.stats()
        return {
            "run_id": rid,
            "condition": cfg.condition,
            "constraint_set": cfg.constraint_set,
            "paraphrase": cfg.paraphrase,
            "seed": cfg.seed,
            "episode": es.episode,
            "source_episode": es.source_episode,
            "start_step": es.start_step,
            "success": bool(success),
            "violation": bool(any(viol_flags.values())),
            "violation_steps": int(sum(viol_steps.values())),
            "violations_by_constraint": json.dumps(viol_flags),
            "steps": len(executed) - 1,
            "plan_time_p50_s": float(np.median(plan_times)) if plan_times else float("nan"),
            "plan_time_p95_s": float(np.percentile(plan_times, 95)) if plan_times else float("nan"),
            "replans": len(plan_times),
            "judge_calls": int(st["judge_calls"]),
            "judge_wall_ms": float(st["judge_wall_ms"]),
            "tokens_in": int(st["tokens_in"]),
            "tokens_out": int(st["tokens_out"]),
            "held": int(st["held"]),
            "judged": int(st["judged"]),
            "abstention_rate": float(st["held"] / st["judged"]) if st["judged"] else 0.0,
            "judge_failures": int(st.get("judge_failures", 0)),
            "lam": cfg.lam,
            "K": cfg.K,
            "tau": cfg.tau,
            "mode": cfg.mode,
            "judge_last_n": cfg.judge_last_n,
            "judge_every": cfg.judge_every,
            "steps_judged": cfg.steps if cfg.steps is not None else spec.horizon,
            "hard_reject": cfg.hard_reject,
            "unjudged": cfg.unjudged,
            "start_filter": cfg.start_filter,
            "vocab": cfg.vocab,
            "probes": cfg.probes,
            "library": self.lib.version,
            "bank": BANK_VERSION,
            "checkpoint_revision": self.config["checkpoint"]["revision"],
            "response_model": _last_model(objective),
            "tag": cfg.tag,
        }


def executed_at_cadence(executed: list[GroundTruthState], action_block: int) -> list[GroundTruthState]:
    """Subsample executed env states at the planner's cadence (indices 0, ab, 2ab, ...), always
    keeping the final state so a violation at the end of a short episode is not dropped."""
    idx = list(range(0, len(executed), action_block))
    if idx[-1] != len(executed) - 1:
        idx.append(len(executed) - 1)
    return [executed[i] for i in idx]


def _last_model(objective: JevCost) -> str:
    for tr in reversed(objective.history):
        if tr.judge and tr.judge.get("response_model"):
            return str(tr.judge["response_model"])
    return ""


def append_results(df: pd.DataFrame, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = pd.read_parquet(path)
        # never overwrite: drop previous rows of the same run_id (re-run) then append
        old = old[~old["run_id"].isin(df["run_id"].unique())]
        df = pd.concat([old, df], ignore_index=True)
    df.to_parquet(path, index=False)
    return path
