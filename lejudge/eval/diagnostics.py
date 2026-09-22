"""Trace-level diagnostics for the paper: imagined vs executed violations and gate statistics."""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

from lejudge.judge import TraceWriter


def imagined_vs_executed(results: Path | str = "artifacts/results", traces: Path | str = "artifacts/traces", tables: tuple[str, ...] = ("planning_filtered.parquet", "planning_gate.parquet", "planning.parquet")) -> pd.DataFrame:
    """Per (table, condition, set): fraction of episodes whose chosen plan was *imagined* to violate
    (penalty of the lowest-cost elite at the last judged iteration of the first replan > 0.5)
    against the fraction that violated in *execution* (oracle on executed states)."""
    rows = []
    for tname in tables:
        p = Path(results) / tname
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        df = df[df.condition != "lewm"]
        for rid, g in df.groupby("run_id"):
            cond, cset = g.condition.iloc[0], g.constraint_set.iloc[0]
            tau = float(g.tau.iloc[0]) if "tau" in g else float("nan")
            run_dir = Path(traces) / rid
            if not run_dir.exists():
                continue
            imag, execd, both = [], [], []
            for f in sorted(glob.glob(str(run_dir / "ep_*.jsonl"))):
                recs = TraceWriter.read(f)
                summ = [r for r in recs if r.get("summary")]
                its = [r for r in recs if not r.get("summary") and r.get("penalty")]
                if not summ or not its:
                    continue
                first_step = min(r.get("step", 0) for r in its)
                last = [r for r in its if r.get("step", 0) == first_step][-1]
                pen0 = float(last["penalty"][0]) if last["penalty"] else 0.0
                iv = pen0 > 0.5
                ev = bool(any(summ[-1]["oracle"].values()))
                imag.append(iv)
                execd.append(ev)
                both.append((iv, ev))
            if not imag:
                continue
            n = len(imag)
            rows.append({
                "table": tname.replace(".parquet", ""),
                "condition": cond,
                "constraint_set": cset,
                "tau": tau,
                "seed": int(g.seed.iloc[0]),
                "n": n,
                "imagined_violation": float(np.mean(imag)),
                "executed_violation": float(np.mean(execd)),
                "executed_given_imagined_clean": float(np.mean([e for i, e in both if not i])) if any(not i for i, _ in both) else float("nan"),
                "executed_given_imagined_violating": float(np.mean([e for i, e in both if i])) if any(i for i, _ in both) else float("nan"),
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    agg = out.groupby(["table", "condition", "constraint_set", "tau"], dropna=False).agg(n=("n", "sum"), imagined_violation=("imagined_violation", "mean"), executed_violation=("executed_violation", "mean"), executed_given_imagined_clean=("executed_given_imagined_clean", "mean"), executed_given_imagined_violating=("executed_given_imagined_violating", "mean")).reset_index()
    return agg


def penalised_fraction_curve(results: Path | str = "artifacts/results", traces: Path | str = "artifacts/traces", table: str = "planning_filtered.parquet", condition: str = "oracle") -> pd.DataFrame:
    """Mean penalised fraction of the CEM population per judged iteration (first replan)."""
    p = Path(results) / table
    df = pd.read_parquet(p)
    df = df[df.condition == condition]
    rows = []
    for rid, g in df.groupby("run_id"):
        for f in sorted(glob.glob(str(Path(traces) / rid / "ep_*.jsonl"))):
            recs = [r for r in TraceWriter.read(f) if not r.get("summary") and "penalised_fraction" in r]
            if not recs:
                continue
            first_step = min(r.get("step", 0) for r in recs)
            for r in recs:
                if r.get("step", 0) == first_step:
                    rows.append({"constraint_set": g.constraint_set.iloc[0], "iter": r["iter"], "penalised_fraction": r["penalised_fraction"]})
    out = pd.DataFrame(rows)
    return out.groupby(["constraint_set", "iter"]).penalised_fraction.agg(["mean", "count"]).reset_index() if not out.empty else out


if __name__ == "__main__":  # pragma: no cover
    print(imagined_vs_executed().to_string())
