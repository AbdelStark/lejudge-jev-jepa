"""Judge-only study (RFC-0006 §judge-only): imagined + executed rollouts, oracle labels,
all judges on identical inputs, canonical text + paraphrases + near-miss negatives."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from lejudge.constraints import TEXT_VARIANTS, Library, check, load_library
from lejudge.cost.swm_adapter import load_config, load_lewm
from lejudge.judge import JevJudge, KeywordJudge, LLMJudge, OracleJudge, penalty
from lejudge.judge.bank import BANK_VERSION
from lejudge.probes.data import EpisodeData
from lejudge.probes.model import load_probe
from lejudge.types import Constraint, GroundTruthState, StepFacts, SymbolicState
from lejudge.vocab import Vocab, load_vocab, words


@dataclass
class Item:
    item_id: str
    source: str  # "executed" | "imagined"
    description: str  # "gt-words" | "probe-words"
    facts: list[StepFacts]
    states: list[GroundTruthState]  # what the oracle sees (GT for executed; probes for imagined)
    labels: dict[str, bool]  # constraint id -> oracle episode flag
    label_steps: dict[str, list[bool]]


def _gt(state: np.ndarray, contact: np.ndarray) -> list[GroundTruthState]:
    return [GroundTruthState.from_env(state[i], contact[i]) for i in range(len(state))]


def executed_items(data: EpisodeData, probe: Any, vocab: Vocab, lib: Library, n: int, seed: int, horizon: int = 5, action_block: int = 5) -> list[Item]:
    rng = np.random.default_rng(seed)
    eps = data.split(seed=0)["test"]
    items: list[Item] = []
    i = 0
    while len(items) < 2 * n:
        e = int(rng.choice(eps))
        n_frames = data.length(e)
        t0 = int(rng.integers(0, n_frames - horizon * action_block))
        ep = data.episode(e)
        idx = [t0 + h * action_block for h in range(horizon + 1)]
        gt = _gt(ep["state"][idx], ep["contact"][idx])
        labels = {c.id: check(c, gt, vocab).episode for c in lib.constraints}
        label_steps = {c.id: list(check(c, gt, vocab).steps) for c in lib.constraints}
        gt_words = words(SymbolicState.from_ground_truth(gt), vocab)
        sym = probe.symbolic(torch.tensor(ep["emb"][idx]))
        probe_words = words(sym, vocab)
        base = f"ex_{e}_{t0}"
        items.append(Item(base + "_gt", "executed", "gt-words", gt_words, gt, labels, label_steps))
        items.append(Item(base + "_pr", "executed", "probe-words", probe_words, gt, labels, label_steps))
        i += 1
    return items


@torch.inference_mode()
def imagined_items(data: EpisodeData, model: Any, probe: Any, vocab: Vocab, lib: Library, n: int, seed: int, horizon: int = 5, action_block: int = 5, num_samples: int = 300, k: int = 16, goal_offset: int = 25) -> list[Item]:
    """CEM-style candidates: sample action sequences from the standard-normal prior, roll the
    predictor, keep the ``k`` cheapest by goal distance, describe with probes. Labels are
    oracle-on-probes (flagged by ``description='probe-words'`` and ``source='imagined'``)."""
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    device = next(model.parameters()).device
    eps = data.split(seed=0)["test"]
    items: list[Item] = []
    while len(items) < n:
        e = int(rng.choice(eps))
        n_frames = data.length(e)
        t0 = int(rng.integers(0, n_frames - goal_offset - 1))
        ep = data.episode(e)
        emb0 = torch.tensor(ep["emb"][t0], device=device).view(1, 1, 1, -1)
        goal = torch.tensor(ep["emb"][t0 + goal_offset], device=device).view(1, 1, -1)
        acts = torch.randn(1, num_samples, horizon, 2 * action_block, generator=gen).to(device)
        info = model.rollout({"emb": emb0.expand(1, num_samples, 1, -1), "pixels": torch.zeros(1, num_samples, 1, 1, device=device)}, acts)
        pred = info["predicted_emb"][0]  # (S, H+1, D)
        A = ((pred[:, -1] - goal[0]) ** 2).sum(-1)
        idx = torch.topk(A, k=k, largest=False).indices
        sym = probe.symbolic(pred[idx].float().cpu())
        for j in range(k):
            st = sym[j].to_ground_truth()
            labels = {c.id: check(c, st, vocab).episode for c in lib.constraints}
            label_steps = {c.id: list(check(c, st, vocab).steps) for c in lib.constraints}
            items.append(Item(f"im_{e}_{t0}_{j}", "imagined", "probe-words", words(sym[j], vocab), st, labels, label_steps))
            if len(items) >= n:
                break
    return items


def _score(family: str, p: list[float]) -> float:
    return penalty(family, p)


def run_judges(items: list[Item], judges: dict[str, Any], lib: Library, vocab: Vocab, variants: tuple[str, ...] = TEXT_VARIANTS, batch: int = 16, constraints_per_call: int = 4, repeats: int = 1, repeat_items: int = 0, progress: bool = True) -> pd.DataFrame:
    """Every judge sees identical states and keys. Returns one row per (item, judge, constraint, variant, repeat)."""
    rows: list[dict[str, Any]] = []
    cids = lib.ids()
    t_start = time.time()
    n_batches = math.ceil(len(items) / batch)
    for bi in range(n_batches):
        chunk = items[bi * batch : (bi + 1) * batch]
        facts = {it.item_id: it.facts for it in chunk}
        states = {it.item_id: it.states for it in chunk}
        for variant in variants:
            groups = [cids[i : i + constraints_per_call] for i in range(0, len(cids), constraints_per_call)]
            for group in groups:
                cs: list[Constraint] = [lib.variants(cid)[variant] for cid in group]
                for jname, judge in judges.items():
                    n_rep = repeats if (variant == "canonical" and repeat_items and bi * batch < repeat_items) else 1
                    for rep in range(n_rep):
                        if rep > 0 and hasattr(judge, "uid"):
                            judge.uid = f"rep{rep}"
                        res = judge.judge(facts, cs, states=states)
                        if hasattr(judge, "uid"):
                            judge.uid = ""
                        per_q = len(chunk) * sum(1 if c.family == "soft" else len(chunk[0].facts) for c in cs)
                        for it in chunk:
                            for c, cid in zip(cs, group):
                                p = res.p[it.item_id][c.id]
                                fam = c.family
                                rows.append(
                                    {
                                        "item_id": it.item_id,
                                        "source": it.source,
                                        "description": it.description,
                                        "judge": jname,
                                        "constraint": cid,
                                        "family": fam,
                                        "variant": variant,
                                        "text": c.text,
                                        "repeat": rep,
                                        "score": _score(fam, p),
                                        "p": json.dumps([None if (isinstance(x, float) and math.isnan(x)) else x for x in p]),
                                        "confidence": res.confidence[it.item_id][c.id],
                                        "label": bool(it.labels[cid]),
                                        "label_steps": json.dumps(it.label_steps[cid]),
                                        "failed": bool(res.failed),
                                        "cache_hit": bool(res.cache_hit),
                                        "latency_ms_per_q": (res.latency_ms / per_q) if per_q else 0.0,
                                        "tokens_in_per_q": (res.input_tokens / per_q) if per_q else 0.0,
                                        "tokens_out_per_q": (res.output_tokens / per_q) if per_q else 0.0,
                                        "response_model": res.response_model,
                                        "bank": BANK_VERSION,
                                        "library": lib.version,
                                        "vocab": vocab.version,
                                    }
                                )
        if progress:
            print(f"[judge-study] batch {bi + 1}/{n_batches} rows={len(rows)} ({time.time() - t_start:.0f}s)", flush=True)
    return pd.DataFrame(rows)


def build_items(n_executed: int, n_imagined: int, seed: int = 0, data_path: str = "artifacts/data/pusht_expert.npz", probes: str = "pusht/linear@1", vocab_name: str = "pusht@1", device: str | None = None) -> list[Item]:
    data = EpisodeData(data_path)
    probe = load_probe(probes)
    vocab = load_vocab(vocab_name)
    lib = load_library()
    items = executed_items(data, probe, vocab, lib, n_executed // 2, seed)
    if n_imagined:
        model = load_lewm(device)
        items += imagined_items(data, model, probe, vocab, lib, n_imagined, seed)
    return items


def make_judges(names: list[str], vocab: Vocab, llm_model: str = "qwen2.5:7b-instruct") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for n in names:
        if n == "jev":
            out[n] = JevJudge()
        elif n == "keyword":
            out[n] = KeywordJudge()
        elif n == "oracle":
            out[n] = OracleJudge(vocab, on_probes=True)
        elif n.startswith("llm"):
            out[n] = LLMJudge(model=llm_model)
        else:
            raise ValueError(n)
    return out


def metrics_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per judge × variant-group metrics: P/R/F1@0.5, AUROC, ECE, negatives FPR, latency, tokens."""
    from lejudge.eval.stats import auroc, ece, prf

    rows = []
    base = df[df.repeat == 0]
    for (judge, source, description), g in base.groupby(["judge", "source", "description"]):
        canon = g[g.variant == "canonical"]
        paras = g[g.variant.str.startswith("p")]
        negs = g[g.variant.str.startswith("n")]
        r: dict[str, Any] = {"judge": judge, "source": source, "description": description, "n_items": canon.item_id.nunique()}
        for name, sub in (("canonical", canon), ("paraphrase", paras)):
            m = prf(sub.label.to_numpy(), sub.score.to_numpy())
            r[f"{name}_precision"], r[f"{name}_recall"], r[f"{name}_f1"], r[f"{name}_accuracy"] = m["precision"], m["recall"], m["f1"], m["accuracy"]
            r[f"{name}_auroc"] = auroc(sub.label.to_numpy(), sub.score.to_numpy())
            r[f"{name}_ece"] = ece(sub.label.to_numpy(), sub.score.to_numpy())[0]
        trap = negs[negs.label]  # canonical condition present; negative sentence must not fire
        # the oracle judge ignores the wording (it runs the canonical checker), so its FPR is undefined
        r["negative_fpr"] = float("nan") if str(judge).startswith("oracle") else (float((trap.score > 0.5).mean()) if len(trap) else float("nan"))
        r["latency_ms_per_1000"] = float(g.latency_ms_per_q.mean() * 1000)
        r["tokens_in_per_1000"] = float(g.tokens_in_per_q.mean() * 1000)
        r["failed_rate"] = float(g.failed.mean())
        rows.append(r)
    return pd.DataFrame(rows)


def per_constraint_table(df: pd.DataFrame) -> pd.DataFrame:
    from lejudge.eval.stats import prf

    rows = []
    base = df[(df.repeat == 0)]
    for (judge, description, cid, variant), g in base.groupby(["judge", "description", "constraint", "variant"]):
        m = prf(g.label.to_numpy(), g.score.to_numpy())
        rows.append({"judge": judge, "description": description, "constraint": cid, "variant": variant, "accuracy": m["accuracy"], "f1": m["f1"], "recall": m["recall"], "precision": m["precision"], "n": m["n"], "positives": int(g.label.sum())})
    return pd.DataFrame(rows)


def consistency_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    reps = df[df.variant == "canonical"]
    for judge, g in reps.groupby("judge"):
        multi = g.groupby(["item_id", "constraint"]).filter(lambda x: x.repeat.nunique() > 1)
        if multi.empty:
            continue
        sd = multi.groupby(["item_id", "constraint"]).score.std(ddof=0)
        rows.append({"judge": judge, "n_pairs": int(len(sd)), "repeats": int(multi.repeat.nunique()), "score_std_mean": float(sd.mean()), "score_std_max": float(sd.max())})
    return pd.DataFrame(rows)


def save_config_snapshot(path: Path | str, extra: dict[str, Any]) -> None:
    cfg = load_config()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"config": cfg, **extra}, indent=2, sort_keys=True))
