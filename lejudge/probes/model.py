"""Probe heads. Inputs: 192-d LeWM latents after the projector. Outputs: 7 numbers
(agent_xy, block_xy, sin θ, cos θ, contact logit) that the vocabulary immediately buckets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from lejudge.types import SymbolicState

PROBE_ROOT = Path("artifacts/probes")
TARGET_NAMES = ("agent_x", "agent_y", "block_x", "block_y", "sin", "cos", "contact")


def targets_from_state(state: np.ndarray, contact: np.ndarray, arena: float = 512.0) -> np.ndarray:
    """``[N, 7]`` regression/classification targets from the env state and contact flags."""
    s = np.asarray(state, dtype=np.float64)
    ang = s[:, 4]
    return np.stack(
        [
            s[:, 0] / arena,
            s[:, 1] / arena,
            s[:, 2] / arena,
            s[:, 3] / arena,
            np.sin(ang),
            np.cos(ang),
            np.asarray(contact, dtype=np.float64),
        ],
        1,
    )


class _ProbeBase(nn.Module):
    kind = "base"

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("x_mean", torch.zeros(192))
        self.register_buffer("x_std", torch.ones(192))

    def _norm(self, z: torch.Tensor) -> torch.Tensor:
        return (z - self.x_mean) / self.x_std

    def raw(self, z: torch.Tensor) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    @torch.inference_mode()
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        dev = self.x_mean.device
        return self.raw(torch.as_tensor(z, dtype=torch.float32, device=dev))

    @torch.inference_mode()
    def symbolic(self, z: torch.Tensor | np.ndarray) -> SymbolicState:
        """Batched: any leading dims, last dim 192 → SymbolicState with the same leading dims."""
        out = (
            self.forward(torch.as_tensor(np.asarray(z) if not torch.is_tensor(z) else z))
            .float()
            .cpu()
            .numpy()
        )
        angle = np.arctan2(out[..., 4], out[..., 5])
        return SymbolicState(
            agent_xy=np.clip(out[..., 0:2], 0, 1),
            block_xy=np.clip(out[..., 2:4], 0, 1),
            block_angle=angle,
            contact_logit=out[..., 6],
        )

    __call__ = forward  # type: ignore[assignment]

    def save(self, out_dir: Path | str, meta: dict[str, Any]) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "kind": self.kind,
                "state_dict": self.state_dict(),
                "hparams": getattr(self, "hparams", {}),
            },
            out_dir / "weights.pt",
        )
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
        return out_dir


class LinearProbe(_ProbeBase):
    kind = "linear"

    def __init__(self, in_dim: int = 192, out_dim: int = 7) -> None:
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.hparams: dict[str, Any] = {}

    def raw(self, z: torch.Tensor) -> torch.Tensor:
        return self.lin(self._norm(z))

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        ridge: float = 1.0,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        ridge_grid: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0),
    ) -> dict[str, Any]:
        """Closed-form ridge for the 6 regression targets; logistic regression for contact.
        ``ridge`` is chosen on the validation split when one is provided."""
        from sklearn.linear_model import LogisticRegression

        x = np.asarray(x, dtype=np.float64)
        mu, sd = x.mean(0), x.std(0) + 1e-6
        xn = (x - mu) / sd
        self.x_mean.copy_(torch.tensor(mu, dtype=torch.float32))
        self.x_std.copy_(torch.tensor(sd, dtype=torch.float32))
        best = (None, np.inf, ridge)
        grid = ridge_grid if x_val is not None else (ridge,)
        for lam in grid:
            w, b = _ridge(xn, y[:, :6], lam)
            if x_val is not None:
                pred = (
                    torch.tensor((x_val - mu) / sd, dtype=torch.float64) @ torch.tensor(w)
                    + torch.tensor(b)
                ).numpy()
                err = float(np.mean((pred - y_val[:, :6]) ** 2))
            else:
                err = 0.0
            if err < best[1]:
                best = ((w, b), err, lam)
        (w, b), _, lam = best
        clf = LogisticRegression(max_iter=2000, C=1.0)
        yc = y[:, 6].astype(int)
        if yc.min() == yc.max():
            wc, bc = np.zeros(x.shape[1]), np.array([-10.0 if yc.max() == 0 else 10.0])
        else:
            clf.fit(xn, yc)
            wc, bc = clf.coef_[0], clf.intercept_
        W = np.concatenate([w.T, wc[None]], 0)  # [7, 192]
        B = np.concatenate([b, bc], 0)
        with torch.no_grad():
            self.lin.weight.copy_(torch.tensor(W, dtype=torch.float32))
            self.lin.bias.copy_(torch.tensor(B, dtype=torch.float32))
        self.hparams = {"ridge": float(lam)}
        return self.hparams


def _ridge(x: np.ndarray, y: np.ndarray, lam: float) -> tuple[np.ndarray, np.ndarray]:
    # torch matmul avoids spurious Accelerate BLAS warnings on macOS for float64 arrays
    xt = torch.tensor(x, dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.float64)
    xm, ym = xt.mean(0), yt.mean(0)
    xc, yc = xt - xm, yt - ym
    a = xc.T @ xc + lam * torch.eye(x.shape[1], dtype=torch.float64)
    w = torch.linalg.solve(a, xc.T @ yc)
    b = ym - xm @ w
    return w.numpy(), b.numpy()


class MLPProbe(_ProbeBase):
    kind = "mlp"

    def __init__(
        self, in_dim: int = 192, hidden: int = 256, out_dim: int = 7, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )
        self.hparams: dict[str, Any] = {"hidden": hidden, "dropout": dropout}

    def raw(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(self._norm(z))

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        epochs: int = 60,
        lr: float = 1e-3,
        batch: int = 512,
        seed: int = 0,
    ) -> dict[str, Any]:
        torch.manual_seed(seed)
        x = np.asarray(x, dtype=np.float32)
        mu, sd = x.mean(0), x.std(0) + 1e-6
        self.x_mean.copy_(torch.tensor(mu))
        self.x_std.copy_(torch.tensor(sd))
        xt, yt = torch.tensor(x), torch.tensor(np.asarray(y, dtype=np.float32))
        opt = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-4)
        n = len(xt)
        best_state, best_val = None, np.inf
        for _ep in range(epochs):
            self.train()
            perm = torch.randperm(n)
            for i in range(0, n, batch):
                idx = perm[i : i + batch]
                out = self.net(self._norm(xt[idx]))
                loss = nn.functional.mse_loss(
                    out[:, :6], yt[idx, :6]
                ) + nn.functional.binary_cross_entropy_with_logits(out[:, 6], yt[idx, 6])
                opt.zero_grad()
                loss.backward()
                opt.step()
            self.eval()
            if x_val is not None:
                with torch.no_grad():
                    pv = self.net(self._norm(torch.tensor(np.asarray(x_val, dtype=np.float32))))
                    v = float(
                        nn.functional.mse_loss(
                            pv[:, :6], torch.tensor(np.asarray(y_val[:, :6], dtype=np.float32))
                        )
                    )
                if v < best_val:
                    best_val, best_state = v, {k: t.clone() for k, t in self.state_dict().items()}
        if best_state is not None:
            self.load_state_dict(best_state)
        self.eval()
        self.hparams.update({"epochs": epochs, "lr": lr, "val_mse": float(best_val)})
        return self.hparams


def load_probe(
    name: str = "pusht/linear@1", root: Path | str = PROBE_ROOT, device: str | torch.device = "cpu"
) -> _ProbeBase:
    d = Path(name) if Path(name).exists() else Path(root) / name
    payload = torch.load(d / "weights.pt", map_location="cpu", weights_only=False)
    probe: _ProbeBase = (
        LinearProbe()
        if payload["kind"] == "linear"
        else MLPProbe(
            **{k: v for k, v in payload.get("hparams", {}).items() if k in ("hidden", "dropout")}
        )
    )
    probe.load_state_dict(payload["state_dict"])
    probe.eval()
    probe.hparams = payload.get("hparams", {})
    probe.meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}  # type: ignore[attr-defined]
    return probe.to(device)
