"""Build the expert PushT dataset (``artifacts/data/pusht_expert.npz``) from the official
``pusht_expert_train.h5`` (or a readable prefix of it): LeWM latents + states + actions.

The action scaler is fitted on every readable expert action so it matches the authors' eval
(``StandardScaler`` on the dataset's ``action`` column).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lejudge.cost.swm_adapter import ActionScaler, encode_frames, load_config, load_lewm, save_json

H5_DEFAULT = Path("artifacts/data/expert/pusht_expert_prefix.h5")


def _open(path: Path):
    import h5py
    import hdf5plugin  # noqa: F401 — registers the pixel compression filter

    try:
        return h5py.File(path, "r", libver="latest", swmr=True)
    except Exception:  # noqa: BLE001
        return h5py.File(path, "r")


def readable_range(f: Any, probe_rows: int = 2_000_000) -> int:
    """Number of leading frames with populated metadata columns (prefix files end in zeros)."""
    n = min(probe_rows, f["state"].shape[0])
    st = np.asarray(f["state"][:n])
    valid = ~np.all(st == 0, axis=1)
    if valid.all():
        return n
    return int(np.argmin(valid))


def renderer_check(f: Any, n_frames: int = 8, seed: int = 0) -> dict[str, float]:
    """Render dataset states with our env and compare to the stored pixels."""
    from lejudge.cost.swm_adapter import make_world

    rng = np.random.default_rng(seed)
    n_ok = readable_range(f, 200_000)
    idx = np.sort(rng.integers(0, n_ok, size=n_frames))
    states = np.asarray(f["state"][idx])
    world = make_world(1, 100)
    diffs = []
    for s in states:
        world.reset(seed=0, options=[{"state": s, "goal_state": s}])
        diffs.append(world.infos["pixels"][0, -1].astype(np.float64))
    world.close()
    pix = np.asarray(f["pixels"][idx]).astype(np.float64)
    mad = [float(np.abs(a - b).mean()) for a, b in zip(diffs, pix)]
    return {"mean_abs_diff": float(np.mean(mad)), "max_frame_mad": float(np.max(mad)), "frames": n_frames}


def build(
    h5_path: Path | str = H5_DEFAULT,
    name: str = "pusht_expert",
    episodes: int = 800,
    out_dir: Path | str = "artifacts/data",
    device: str | None = None,
    min_len: int = 40,
    model: torch.nn.Module | None = None,
    pixel_batch: int = 256,
    progress: bool = True,
) -> Path:
    t0 = time.time()
    out_dir = Path(out_dir)
    f = _open(Path(h5_path))
    n_ok = readable_range(f)
    ep_idx = np.asarray(f["episode_idx"][:n_ok])
    step_idx = np.asarray(f["step_idx"][:n_ok])
    state = np.asarray(f["state"][:n_ok]).astype(np.float64)
    action = np.asarray(f["action"][:n_ok]).astype(np.float32)
    scaler = ActionScaler.fit(action)
    # complete episodes only (the last one may be cut by the prefix)
    uniq, starts, counts = np.unique(ep_idx, return_index=True, return_counts=True)
    order = np.argsort(starts)
    uniq, starts, counts = uniq[order], starts[order], counts[order]
    keep = [(int(e), int(s), int(c)) for e, s, c in zip(uniq, starts, counts) if c >= min_len and s + c <= n_ok and step_idx[s] == 0]
    keep = keep[: episodes + 1]
    if keep and keep[-1][1] + keep[-1][2] >= n_ok:
        keep = keep[:-1]
    keep = keep[:episodes]
    model = model or load_lewm(device)
    parts: dict[str, list[np.ndarray]] = {k: [] for k in ("emb", "state", "action", "episode_idx", "step_idx", "seed")}
    rows = 0
    for i, (e, s, c) in enumerate(keep):
        pix = np.asarray(f["pixels"][s : s + c])
        emb = encode_frames(model, pix, batch=pixel_batch)
        act = action[s : s + c].copy()
        # the dataset stores the action taken from each frame; the last frame of an episode has none
        act[-1] = np.nan
        parts["emb"].append(emb.astype(np.float32))
        parts["state"].append(state[s : s + c])
        parts["action"].append(act)
        parts["episode_idx"].append(np.full(c, i, dtype=np.int32))
        parts["step_idx"].append(np.arange(c, dtype=np.int32))
        parts["seed"].append(np.full(c, e, dtype=np.int64))
        rows += c
        if progress and (i + 1) % 50 == 0:
            print(f"[expert] {i + 1}/{len(keep)} episodes, {rows} frames, {time.time() - t0:.0f}s", flush=True)
    flat = {k: np.concatenate(v, 0) for k, v in parts.items()}
    path = out_dir / f"{name}.npz"
    np.savez_compressed(path, **flat)
    cfg = load_config()
    meta: dict[str, Any] = {
        "name": name,
        "source": str(h5_path),
        "source_repo": "quentinll/lewm-pusht (pusht_expert_train.h5.zst)",
        "episodes": len(keep),
        "frames": int(rows),
        "readable_prefix_frames": int(n_ok),
        "action_scaler": scaler.to_json(),
        "action_scaler_frames": int(len(action)),
        "checkpoint": cfg["checkpoint"],
        "stable_worldmodel_commit": cfg["stable_worldmodel_commit"],
        "contact": "geometric (lejudge.types.CONTACT_TOLERANCE)",
        "elapsed_s": round(time.time() - t0, 1),
    }
    save_json(out_dir / f"{name}.meta.json", meta)
    return path
