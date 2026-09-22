"""Render the 20-second clip: stock plan → type constraint → LeJudge plan → edit → new plan → end card."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lejudge.demo.core import PRESETS, DemoBackend, load_pair

FPS = 10


def _text_panel(lines: list[str], size: tuple[int, int] = (224 * 2 + 16, 60)) -> np.ndarray:
    from PIL import Image, ImageDraw

    img = Image.new("RGB", size, (250, 250, 250))
    d = ImageDraw.Draw(img)
    y = 6
    for ln in lines:
        d.text((8, y), ln, fill=(20, 20, 20))
        y += 14
    return np.asarray(img)


def _compose(left: np.ndarray, right: np.ndarray, caption: list[str]) -> np.ndarray:
    gap = np.full((224, 16, 3), 255, dtype=np.uint8)
    row = np.concatenate([left, gap, right], 1)
    return np.concatenate([_text_panel(caption), row], 0)


def _hold(frames: list[np.ndarray], seconds: float) -> list[np.ndarray]:
    return [frames[-1]] * int(seconds * FPS)


def record_clip(out: str = "artifacts/demo/clip.mp4", seed: int = 0, backend: DemoBackend | None = None) -> str:
    import imageio

    backend = backend or DemoBackend()
    first, second = PRESETS[0], PRESETS[1]
    pairs = []
    for text in (first, second):
        path = backend.gallery_path(seed, text)
        pairs.append(load_pair(path) if path.exists() else backend.plan_pair(seed, text))
    frames: list[np.ndarray] = []
    (stock, ours) = pairs[0]
    n = max(len(stock.frames), len(ours.frames))
    for i in range(n):
        frames.append(_compose(stock.frames[min(i, len(stock.frames) - 1)], stock.frames[min(i, len(stock.frames) - 1)], ["LeWorldModel plans toward the goal image.", "No constraint.", ""]))
    frames += _hold(frames, 1.0)
    for i in range(n):
        frames.append(_compose(stock.frames[min(i, len(stock.frames) - 1)], ours.frames[min(i, len(ours.frames) - 1)], [f'Constraint: "{first}"', f"stock: violation={stock.oracle.get(first)}   LeJudge: violation={ours.oracle.get(first)}", "Jev judges words from probes; code adds the penalty."]))
    frames += _hold(frames, 1.0)
    (stock2, ours2) = pairs[1]
    n2 = max(len(stock2.frames), len(ours2.frames))
    for i in range(n2):
        frames.append(_compose(stock2.frames[min(i, len(stock2.frames) - 1)], ours2.frames[min(i, len(ours2.frames) - 1)], [f'Edit the sentence: "{second}"', f"stock: violation={stock2.oracle.get(second)}   LeJudge: violation={ours2.oracle.get(second)}", "Same planner, same goal; only the sentence changed."]))
    frames += _hold(frames, 1.0)
    end = _compose(np.full((224, 224, 3), 255, np.uint8), np.full((224, 224, 3), 255, np.uint8), ["LeJudge — a cost module you program in English", "LeWM + probes + Jev (TypeSafe System One)", "github: lejudge"])
    frames += [end] * (2 * FPS)
    target = 20 * FPS
    if len(frames) > target:
        idx = np.linspace(0, len(frames) - 1, target).astype(int)
        frames = [frames[i] for i in idx]
    elif len(frames) < target:  # short episodes: hold the end card so the clip is exactly 20 s
        frames += [end] * (target - len(frames))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(out, fps=FPS, codec="libx264", macro_block_size=1) as w:
        for f in frames:
            w.append_data(f)
    return f"{out} ({len(frames) / FPS:.1f}s, {len(frames)} frames)"
