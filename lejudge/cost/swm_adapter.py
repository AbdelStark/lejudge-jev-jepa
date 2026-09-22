"""Everything that touches stable-worldmodel internals lives here (RFC-0004).

API snapshot (stable-worldmodel @ 4821c8e): ``World``, ``WorldModelPolicy(solver, config,
process, transform)``, ``PlanConfig(horizon, receding_horizon, history_len, action_block,
warm_start)``, ``planning.CEMSolver(cost=..., num_samples, n_steps, topk, device, seed,
callbacks)``, ``planning.ShootingCostEvaluator(model, objective)``, objectives are callables
``info_dict -> (B, S)`` reading ``predicted_emb`` ``(B, S, H_ctx + horizon, 192)``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("STABLEWM_HOME", os.path.expanduser("~/.stable-wm"))

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "pusht.yaml"


def load_config(path: Path | str = CONFIG_PATH) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text())


def pick_device(pref: str | None = None) -> torch.device:
    if pref:
        return torch.device(pref)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_lewm(device: torch.device | str | None = None, repo: str | None = None) -> torch.nn.Module:
    """Load the pinned LeWM PushT checkpoint in eval mode."""
    import stable_worldmodel as swm

    cfg = load_config()
    repo = repo or cfg["checkpoint"]["repo"]
    model = swm.wm.utils.load_pretrained(repo)
    model.eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    return model.to(pick_device(device) if not isinstance(device, torch.device) else device)


def preprocess_pixels(pixels: np.ndarray | torch.Tensor) -> torch.Tensor:
    """uint8 ``[N, H, W, 3]`` → float ``[N, 3, 224, 224]`` with ImageNet normalisation.

    Matches the eval transform in le-wm/eval.py: ToImage, ToDtype(float32, scale=True),
    Normalize(ImageNet), Resize(224).
    """
    x = torch.as_tensor(np.asarray(pixels)) if not torch.is_tensor(pixels) else pixels
    if x.ndim == 3:
        x = x.unsqueeze(0)
    x = x.permute(0, 3, 1, 2).float() / 255.0
    if x.shape[-2:] != (224, 224):
        x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", antialias=True, align_corners=False)
    return (x - IMAGENET_MEAN) / IMAGENET_STD


def _tv_transform():
    """The exact torchvision v2 pipeline used by ``WorldModelPolicy.transform``."""
    from torchvision.transforms import v2 as transforms

    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=IMAGENET_MEAN.flatten().tolist(), std=IMAGENET_STD.flatten().tolist()),
            transforms.Resize(size=224),
        ]
    )


@torch.inference_mode()
def encode_frames(model: torch.nn.Module, pixels: np.ndarray, batch: int = 128) -> np.ndarray:
    """Encode uint8 frames ``[N, H, W, 3]`` into projector latents ``[N, 192]``."""
    device = next(model.parameters()).device
    out = []
    for i in range(0, len(pixels), batch):
        x = preprocess_pixels(pixels[i : i + batch]).to(device).unsqueeze(1)  # (b, 1, 3, H, W)
        info = model.encode({"pixels": x})
        out.append(info["emb"][:, 0].float().cpu().numpy())
    return np.concatenate(out, 0) if out else np.zeros((0, 192), dtype=np.float32)


@dataclass
class ActionScaler:
    """StandardScaler equivalent for env actions (the model was trained on standardised actions).

    ``clip`` bounds the *env-space* action on ``inverse_transform`` to the env's declared action
    space; PushT itself does not clip and a kinematic agent can otherwise leave the arena.
    """

    mean: np.ndarray
    std: np.ndarray
    clip: tuple[float, float] | None = None

    @classmethod
    def fit(cls, actions: np.ndarray) -> ActionScaler:
        a = np.asarray(actions, dtype=np.float64).reshape(-1, actions.shape[-1])
        a = a[~np.isnan(a).any(axis=1)]
        return cls(mean=a.mean(0), std=a.std(0) + 1e-8)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float64) - self.mean) / self.std

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        a = np.asarray(x, dtype=np.float64) * self.std + self.mean
        if self.clip is not None:
            a = np.clip(a, self.clip[0], self.clip[1])
        return a

    def to_json(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> ActionScaler:
        return cls(np.asarray(d["mean"], dtype=np.float64), np.asarray(d["std"], dtype=np.float64))


def make_world(num_envs: int, max_episode_steps: int = 100, image_shape: tuple[int, int] = (224, 224)) -> Any:
    import stable_worldmodel as swm

    return swm.World("swm/PushT-v1", num_envs=num_envs, image_shape=image_shape, max_episode_steps=max_episode_steps)


@dataclass
class PlanSpec:
    horizon: int = 5
    receding_horizon: int = 5
    action_block: int = 5
    history_len: int = 1
    warm_start: bool = True
    num_samples: int = 300
    n_steps: int = 30
    topk: int = 30
    var_scale: float = 1.0
    seed: int = 1234
    device: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None = None, **overrides: Any) -> PlanSpec:
        cfg = cfg or load_config()
        d = {**cfg["plan"], **cfg["solver"]}
        d.update(overrides)
        return cls(**d)

    def to_json(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if k != "extra"}


def make_policy(cost: Any, spec: PlanSpec, scaler: ActionScaler, callbacks: list[Any] | None = None) -> Any:
    """Build ``WorldModelPolicy(CEMSolver(cost))`` with the frozen transforms and scaler."""
    import stable_worldmodel as swm
    from stable_worldmodel.planning import CEMSolver

    device = str(pick_device(spec.device))
    solver = CEMSolver(
        cost=cost,
        num_samples=spec.num_samples,
        n_steps=spec.n_steps,
        topk=spec.topk,
        var_scale=spec.var_scale,
        device=device,
        seed=spec.seed,
        callbacks=callbacks or [],
    )
    config = swm.PlanConfig(
        horizon=spec.horizon,
        receding_horizon=spec.receding_horizon,
        action_block=spec.action_block,
        history_len=spec.history_len,
        warm_start=spec.warm_start,
    )
    tf = _tv_transform()
    return swm.policy.WorldModelPolicy(solver=solver, config=config, process={"action": scaler}, transform={"pixels": tf, "goal": tf})


def goal_mse_objective() -> Any:
    from stable_worldmodel.planning import GoalMSE

    return GoalMSE()


def shooting_cost(model: torch.nn.Module, objective: Any) -> Any:
    from stable_worldmodel.planning import ShootingCostEvaluator

    return ShootingCostEvaluator(model, objective)


def save_json(path: Path | str, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, default=_default))


def _default(o: Any) -> Any:
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if hasattr(o, "to_json"):
        return o.to_json()
    raise TypeError(type(o).__name__)
