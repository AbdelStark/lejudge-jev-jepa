"""Collect PushT episodes with the repo's data-collection policy and encode them with LeWM.

Output: ``artifacts/data/<name>.npz`` with flat arrays ``emb [N,192]``, ``state [N,7]``,
``contact [N]``, ``action [N,2]`` (action taken *from* that frame; NaN on the last frame),
``episode_idx [N]``, ``step_idx [N]``, ``seed [N]``, plus ``<name>.meta.json``.
Pixels are not stored; the latents are what the probes and the planner consume.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lejudge.cost.swm_adapter import (
    ActionScaler,
    encode_frames,
    load_config,
    load_lewm,
    make_world,
    save_json,
)

DATA_DIR = Path("artifacts/data")


def collect(
    name: str = "pusht_weak_v1",
    episodes: int = 200,
    num_envs: int = 8,
    seed: int = 1000,
    max_episode_steps: int = 100,
    device: str | None = None,
    out_dir: Path | str = DATA_DIR,
    model: torch.nn.Module | None = None,
    progress: bool = True,
) -> Path:
    from stable_worldmodel.envs.pusht import WeakPolicy

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = model or load_lewm(device)
    world = make_world(num_envs, max_episode_steps)
    policy = WeakPolicy(dist_constraint=100, seed=seed)
    world.set_policy(policy)

    seeds = [seed + i for i in range(num_envs)]
    next_seed = seed + num_envs
    world.reset(seed=seeds)
    buffers: list[dict[str, list]] = [_new_buffer() for _ in range(num_envs)]
    env_seed = list(seeds)
    for i in range(num_envs):
        _push(buffers[i], world.infos, i)

    episodes_out: list[dict[str, np.ndarray]] = []
    t0 = time.time()
    while len(episodes_out) < episodes:
        actions = policy.get_action(world.infos)
        for i in range(num_envs):
            buffers[i]["action"].append(np.asarray(actions[i], dtype=np.float32).copy())
        _, _, term, trunc, world.infos = world.envs.step(actions)
        for i in range(num_envs):
            _push(buffers[i], world.infos, i)
        done = np.asarray(term) | np.asarray(trunc)
        if not done.any():
            continue
        reset_seeds: list[int | None] = [None] * num_envs
        for i in np.where(done)[0]:
            ep = _finalize(buffers[i], model, env_seed[i], len(episodes_out))
            episodes_out.append(ep)
            if progress and len(episodes_out) % 10 == 0:
                print(
                    f"[collect] {len(episodes_out)}/{episodes} episodes, {time.time() - t0:.0f}s",
                    flush=True,
                )
            buffers[i] = _new_buffer()
            reset_seeds[i] = next_seed
            env_seed[i] = next_seed
            next_seed += 1
            if len(episodes_out) >= episodes:
                break
        if len(episodes_out) >= episodes:
            break
        _, world.infos = world.envs.reset(seed=reset_seeds, mask=done)
        for i in np.where(done)[0]:
            _push(buffers[i], world.infos, i)
    world.close()

    flat = {k: np.concatenate([e[k] for e in episodes_out], 0) for k in episodes_out[0]}
    scaler = ActionScaler.fit(flat["action"])
    path = out_dir / f"{name}.npz"
    np.savez_compressed(path, **flat)
    cfg = load_config()
    meta: dict[str, Any] = {
        "name": name,
        "episodes": len(episodes_out),
        "frames": int(flat["emb"].shape[0]),
        "seed": seed,
        "num_envs": num_envs,
        "max_episode_steps": max_episode_steps,
        "collection_policy": "stable_worldmodel.envs.pusht.WeakPolicy(dist_constraint=100)",
        "checkpoint": cfg["checkpoint"],
        "stable_worldmodel_commit": cfg["stable_worldmodel_commit"],
        "action_scaler": scaler.to_json(),
        "elapsed_s": round(time.time() - t0, 1),
    }
    save_json(out_dir / f"{name}.meta.json", meta)
    return path


def _new_buffer() -> dict[str, list]:
    return {"pixels": [], "state": [], "contact": [], "action": []}


def _push(buf: dict[str, list], infos: dict[str, Any], i: int) -> None:
    buf["pixels"].append(np.asarray(infos["pixels"][i, -1]).copy())
    buf["state"].append(np.asarray(infos["state"][i, -1], dtype=np.float64).copy())
    buf["contact"].append(int(np.asarray(infos["n_contacts"][i]).reshape(-1)[-1] > 0))


def _finalize(
    buf: dict[str, list], model: torch.nn.Module, seed: int, episode_idx: int
) -> dict[str, np.ndarray]:
    pixels = np.stack(buf["pixels"], 0)
    n = len(pixels)
    emb = encode_frames(model, pixels)
    actions = np.stack(buf["action"], 0) if buf["action"] else np.zeros((0, 2), dtype=np.float32)
    act = np.full((n, 2), np.nan, dtype=np.float32)
    act[: len(actions)] = actions[:n]
    return {
        "emb": emb.astype(np.float32),
        "state": np.stack(buf["state"], 0),
        "contact": np.asarray(buf["contact"], dtype=np.int8),
        "action": act,
        "episode_idx": np.full(n, episode_idx, dtype=np.int32),
        "step_idx": np.arange(n, dtype=np.int32),
        "seed": np.full(n, seed, dtype=np.int64),
    }


class EpisodeData:
    """Convenience reader for a collected ``.npz``."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        z = np.load(self.path)
        self.emb = z["emb"]
        self.state = z["state"]
        from lejudge.types import contact_flags_from_states

        self.sim_contact = z["contact"] if "contact" in z else None
        self.contact = contact_flags_from_states(z["state"]).astype(
            np.int8
        )  # geometric, frame-level (repo definition)
        self.action = z["action"]
        self.episode_idx = z["episode_idx"]
        self.step_idx = z["step_idx"]
        self.seed = z["seed"]
        self.meta = (
            json.loads(self.path.with_suffix("").with_suffix(".meta.json").read_text())
            if self.path.with_suffix("").with_suffix(".meta.json").exists()
            else {}
        )
        self.episodes = np.unique(self.episode_idx)
        self._starts = {int(e): int(np.nonzero(self.episode_idx == e)[0][0]) for e in self.episodes}
        self._lens = {int(e): int((self.episode_idx == e).sum()) for e in self.episodes}

    @property
    def scaler(self) -> ActionScaler:
        if "action_scaler" in self.meta:
            return ActionScaler.from_json(self.meta["action_scaler"])
        return ActionScaler.fit(self.action)

    def episode(self, e: int) -> dict[str, np.ndarray]:
        s, n = self._starts[int(e)], self._lens[int(e)]
        sl = slice(s, s + n)
        return {
            "emb": self.emb[sl],
            "state": self.state[sl],
            "contact": self.contact[sl],
            "action": self.action[sl],
            "seed": int(self.seed[s]),
        }

    def length(self, e: int) -> int:
        return self._lens[int(e)]

    def split(
        self, fractions: tuple[float, float, float] = (0.8, 0.1, 0.1), seed: int = 0
    ) -> dict[str, np.ndarray]:
        if len(self.episodes) < 30:  # tiny fixtures (CI): everything is "test"
            return {"train": self.episodes[:0], "val": self.episodes[:0], "test": self.episodes}
        rng = np.random.default_rng(seed)
        eps = rng.permutation(self.episodes)
        n = len(eps)
        a = int(round(fractions[0] * n))
        b = a + int(round(fractions[1] * n))
        return {"train": np.sort(eps[:a]), "val": np.sort(eps[a:b]), "test": np.sort(eps[b:])}

    def mask(self, eps: np.ndarray) -> np.ndarray:
        return np.isin(self.episode_idx, eps)
