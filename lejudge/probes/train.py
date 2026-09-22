"""Train and evaluate probes; measure bucket accuracy and the imagined-latent error curve."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lejudge.cost.swm_adapter import ActionScaler, load_config, load_lewm
from lejudge.probes.data import EpisodeData
from lejudge.probes.model import PROBE_ROOT, LinearProbe, MLPProbe, targets_from_state
from lejudge.types import GroundTruthState, SymbolicState
from lejudge.vocab import Vocab, load_vocab


def r2(y: np.ndarray, p: np.ndarray) -> float:
    ss_res = float(((y - p) ** 2).sum())
    ss_tot = float(((y - y.mean(0)) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y).astype(int)
    if y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, s))


def _gt_seq(state: np.ndarray, contact: np.ndarray) -> list[GroundTruthState]:
    return [GroundTruthState.from_env(state[i], contact[i]) for i in range(len(state))]


def bucket_accuracy(pred: SymbolicState, truth: SymbolicState, vocab: Vocab) -> dict[str, float]:
    """Fraction of frames whose word matches, per vocabulary field (speed needs pairs)."""
    out: dict[str, float] = {}
    pc = vocab.block_centroid(pred.block_xy, pred.block_angle)
    tc = vocab.block_centroid(truth.block_xy, truth.block_angle)
    out["block"] = float(np.mean(vocab.cell_name(pc) == vocab.cell_name(tc)))
    out["agent"] = float(np.mean(vocab.cell_name(pred.agent_xy) == vocab.cell_name(truth.agent_xy)))
    out["block_edge"] = float(np.mean(vocab.edge_name(pred.block_xy, pred.block_angle) == vocab.edge_name(truth.block_xy, truth.block_angle)))
    out["block_angle"] = float(np.mean(vocab.angle_name(pred.block_angle) == vocab.angle_name(truth.block_angle)))
    out["contact"] = float(np.mean((pred.contact_logit > 0) == (truth.contact_logit > 0)))
    return out


def evaluate_probe(probe: torch.nn.Module, data: EpisodeData, eps: np.ndarray, vocab: Vocab) -> dict[str, Any]:
    m = data.mask(eps)
    x = data.emb[m]
    y = targets_from_state(data.state[m], data.contact[m])
    pred = probe(torch.tensor(x)).cpu().numpy()
    metrics: dict[str, Any] = {
        "n": int(m.sum()),
        "r2": {"agent_xy": r2(y[:, 0:2], pred[:, 0:2]), "block_xy": r2(y[:, 2:4], pred[:, 2:4]), "angle_sincos": r2(y[:, 4:6], pred[:, 4:6])},
        "mse": {"agent_xy": float(np.mean((y[:, 0:2] - pred[:, 0:2]) ** 2)), "block_xy": float(np.mean((y[:, 2:4] - pred[:, 2:4]) ** 2)), "angle_sincos": float(np.mean((y[:, 4:6] - pred[:, 4:6]) ** 2))},
        "position_error_px": {"agent": float(np.mean(np.linalg.norm(y[:, 0:2] - pred[:, 0:2], axis=1)) * 512), "block": float(np.mean(np.linalg.norm(y[:, 2:4] - pred[:, 2:4], axis=1)) * 512)},
        "contact": {"auroc": auroc(y[:, 6], pred[:, 6]), "accuracy": float(np.mean((pred[:, 6] > 0) == (y[:, 6] > 0.5)))},
    }
    ang_true = np.arctan2(y[:, 4], y[:, 5])
    ang_pred = np.arctan2(pred[:, 4], pred[:, 5])
    d = np.abs((ang_pred - ang_true + math.pi) % (2 * math.pi) - math.pi)
    metrics["angle_error_deg"] = float(np.degrees(d.mean()))
    truth = SymbolicState.from_ground_truth(_gt_seq(data.state[m], data.contact[m]))
    metrics["bucket_accuracy"] = bucket_accuracy(probe.symbolic(torch.tensor(x)), truth, vocab)
    return metrics


@torch.inference_mode()
def imagined_curve(probe: torch.nn.Module, model: torch.nn.Module, data: EpisodeData, eps: np.ndarray, vocab: Vocab, scaler: ActionScaler, horizon: int = 5, action_block: int = 5, n_starts: int = 400, seed: int = 0) -> dict[str, Any]:
    """Roll the predictor open-loop from dataset frames with dataset actions and report probe
    error and bucket accuracy as a function of the imagined step h = 1..horizon."""
    rng = np.random.default_rng(seed)
    device = next(model.parameters()).device
    starts: list[tuple[int, int]] = []
    for e in eps:
        n = data.length(int(e))
        max_t0 = n - 1 - horizon * action_block
        if max_t0 <= 0:
            continue
        for t0 in rng.integers(0, max_t0 + 1, size=max(1, n_starts // max(1, len(eps)))):
            starts.append((int(e), int(t0)))
    starts = starts[:n_starts]
    embs, acts, truths = [], [], []
    for e, t0 in starts:
        ep = data.episode(e)
        embs.append(ep["emb"][t0])
        a = ep["action"][t0 : t0 + horizon * action_block]  # (H*ab, 2)
        acts.append(scaler.transform(a).reshape(horizon, action_block * 2))
        idx = [t0 + h * action_block for h in range(0, horizon + 1)]
        truths.append(_gt_seq(ep["state"][idx], ep["contact"][idx]))
    emb0 = torch.tensor(np.stack(embs), dtype=torch.float32, device=device)  # (N, 192)
    action_seq = torch.tensor(np.stack(acts), dtype=torch.float32, device=device).unsqueeze(0)  # (1, N, H, 10)
    info = {"emb": emb0[None, :, None, :], "pixels": torch.zeros(1, len(embs), 1, 1, device=device)}
    info = model.rollout(info, action_seq)
    pred = info["predicted_emb"][0].float().cpu().numpy()  # (N, H+1, 192)
    sym = probe.symbolic(torch.tensor(pred))  # leading dims (N, H+1)
    curve = []
    for h in range(0, horizon + 1):
        truth_h = SymbolicState.from_ground_truth([tr[h] for tr in truths])
        pred_h = sym[:, h]
        acc = bucket_accuracy(pred_h, truth_h, vocab)
        pos_err = float(np.mean(np.linalg.norm(pred_h.block_xy - truth_h.block_xy, axis=1)) * 512)
        agent_err = float(np.mean(np.linalg.norm(pred_h.agent_xy - truth_h.agent_xy, axis=1)) * 512)
        d = np.abs((pred_h.block_angle - truth_h.block_angle + math.pi) % (2 * math.pi) - math.pi)
        curve.append({"h": h, "bucket_accuracy": acc, "block_error_px": pos_err, "agent_error_px": agent_err, "angle_error_deg": float(np.degrees(d.mean()))})
    return {"n_starts": len(starts), "horizon": horizon, "action_block": action_block, "curve": curve}


def train(
    data_path: Path | str = "artifacts/data/pusht_expert.npz",
    kind: str = "linear",
    name: str | None = None,
    vocab_name: str = "pusht@1",
    root: Path | str = PROBE_ROOT,
    seed: int = 0,
    device: str | None = None,
    with_imagined: bool = True,
) -> Path:
    t0 = time.time()
    data = EpisodeData(data_path)
    vocab = load_vocab(vocab_name)
    split = data.split(seed=seed)
    xtr, xva = data.emb[data.mask(split["train"])], data.emb[data.mask(split["val"])]
    ytr = targets_from_state(data.state[data.mask(split["train"])], data.contact[data.mask(split["train"])])
    yva = targets_from_state(data.state[data.mask(split["val"])], data.contact[data.mask(split["val"])])
    probe: LinearProbe | MLPProbe = LinearProbe() if kind == "linear" else MLPProbe()
    hp = probe.fit(xtr, ytr, x_val=xva, y_val=yva) if kind == "linear" else probe.fit(xtr, ytr, xva, yva, seed=seed)
    probe.eval()
    metrics = {s: evaluate_probe(probe, data, split[s], vocab) for s in ("train", "val", "test")}
    cfg = load_config()
    meta: dict[str, Any] = {
        "name": name or f"pusht/{kind}@1",
        "kind": kind,
        "hparams": hp,
        "dataset": {"path": str(data_path), "episodes": int(len(data.episodes)), "frames": int(len(data.emb)), "meta": data.meta},
        "split": {k: [int(e) for e in v] for k, v in split.items()},
        "checkpoint": cfg["checkpoint"],
        "vocab": vocab.version,
        "metrics": metrics,
        "seed": seed,
    }
    if with_imagined:
        model = load_lewm(device)
        meta["imagined"] = imagined_curve(probe, model, data, split["test"], vocab, data.scaler)
    meta["elapsed_s"] = round(time.time() - t0, 1)
    out = probe.save(Path(root) / (name or f"pusht/{kind}@1"), meta)
    return out


def summary(meta_path: Path | str) -> str:
    m = json.loads(Path(meta_path).read_text())
    t = m["metrics"]["test"]
    lines = [f"probe {m['name']} ({m['kind']}) test n={t['n']}"]
    lines.append("R2: " + ", ".join(f"{k}={v:.3f}" for k, v in t["r2"].items()))
    lines.append("bucket acc: " + ", ".join(f"{k}={v:.3f}" for k, v in t["bucket_accuracy"].items()))
    lines.append(f"contact AUROC={t['contact']['auroc']:.3f} acc={t['contact']['accuracy']:.3f}; angle err={t['angle_error_deg']:.1f} deg; block err={t['position_error_px']['block']:.1f}px")
    if "imagined" in m:
        for row in m["imagined"]["curve"]:
            ba = row["bucket_accuracy"]
            lines.append(f"  h={row['h']}: block={ba['block']:.2f} edge={ba['block_edge']:.2f} angle={ba['block_angle']:.2f} agent={ba['agent']:.2f} contact={ba['contact']:.2f} | block err {row['block_error_px']:.1f}px")
    return "\n".join(lines)
