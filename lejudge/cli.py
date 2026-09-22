"""``lejudge`` command line."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import typer

app = typer.Typer(help="LeJudge: a natural-language cost module for LeWM planning.", no_args_is_help=True)
probes_app = typer.Typer(help="Probe training and inspection.")
app.add_typer(probes_app, name="probes")


def _mode(live: bool) -> None:
    os.environ["LEJUDGE_MODE"] = "live" if live else os.environ.get("LEJUDGE_MODE", "offline")


@app.command()
def collect(name: str = "pusht_weak_v1", episodes: int = 200, num_envs: int = 8, seed: int = 1000, device: str | None = None) -> None:
    """Collect PushT episodes with the weak collection policy and encode them with LeWM."""
    from lejudge.probes.data import collect as _collect

    path = _collect(name=name, episodes=episodes, num_envs=num_envs, seed=seed, device=device)
    typer.echo(f"wrote {path}")


@probes_app.command("train")
def probes_train(env: str = "pusht", kind: str = "linear", data: str = "artifacts/data/pusht_expert.npz", name: str | None = None, vocab: str = "pusht@1", seed: int = 0, no_imagined: bool = False) -> None:
    """Train a probe head and write weights + meta.json with R², bucket accuracy and the imagined curve."""
    from lejudge.probes.train import summary, train

    out = train(data_path=data, kind=kind, name=name, vocab_name=vocab, seed=seed, with_imagined=not no_imagined)
    typer.echo(summary(out / "meta.json"))


@probes_app.command("show")
def probes_show(name: str = "pusht/linear@1") -> None:
    from lejudge.probes.train import summary

    typer.echo(summary(Path("artifacts/probes") / name / "meta.json"))


@app.command()
def lint(env: str = "pusht") -> None:
    """Lint the question bank and the constraint library (RFC-0003 rules)."""
    from lejudge.constraints import load_library
    from lejudge.judge import lint_bank, lint_constraint_text
    from lejudge.vocab import load_vocab

    vocab = load_vocab(f"{env}@1")
    lib = load_library(env)
    problems = lint_bank(vocab.all_words())
    for c in lib.constraints:
        for t in (c.text, *c.paraphrases, *c.negatives):
            problems += [f"{c.id}: {p}" for p in lint_constraint_text(t)]
    if problems:
        for p in problems:
            typer.echo(f"LINT {p}")
        raise typer.Exit(code=1)
    typer.echo(f"lint clean: bank + {len(lib.constraints)} constraints × 8 texts (library {lib.version} sha256 {lib.sha256[:12]})")


@app.command()
def plan(
    env: str = "pusht",
    cond: str = "jev",
    set: str = "edges",
    episodes: int = 20,
    seeds: str = "0",
    lam: float = 1.0,
    k: int = 16,
    tau: float = 0.5,
    mode: str = "every_k",
    judge_last_n: int = 3,
    judge_every: int = 5,
    steps: int | None = None,
    vocab: str = "pusht@1",
    probes: str = "pusht/linear@1",
    paraphrase: str = "canonical",
    hard_reject: bool = False,
    unjudged: str = "mean",
    tag: str = "",
    out: str = "artifacts/results/planning.parquet",
    live: bool = False,
    device: str | None = None,
    data: str = "artifacts/data/pusht_expert.npz",
    history_len: int | None = None,
) -> None:
    """Run the planning study for one or more conditions / sets / seeds and append rows to a Parquet table."""
    _mode(live)
    from lejudge.eval.planning import RunConfig, Runner, append_results

    runner = Runner(device=device, data_path=data)
    seed_list = _parse_range(seeds)
    conds = cond.split(",")
    sets = set.split(",")
    overrides = {"history_len": history_len} if history_len is not None else {}
    for c in conds:
        for s in sets:
            for sd in seed_list:
                cfg = RunConfig(condition=c, constraint_set=s, seed=sd, episodes=episodes, lam=lam, K=k, tau=tau, mode=mode, judge_last_n=judge_last_n, judge_every=judge_every, steps=steps, vocab=vocab, probes=probes, hard_reject=hard_reject, unjudged=unjudged, paraphrase=paraphrase, tag=tag, device=device, data_path=data, plan_overrides=overrides)
                df = runner.run(cfg)
                append_results(df, out)
                typer.echo(f"{c}/{s}/seed{sd}: success={df.success.mean():.3f} violation={df.violation.mean():.3f} n={len(df)} → {out}")


@app.command("judge-study")
def judge_study(
    env: str = "pusht",
    judges: str = "jev,keyword",
    n_executed: int = 128,
    n_imagined: int = 128,
    repeats: int = 1,
    repeat_items: int = 0,
    seed: int = 0,
    out: str = "artifacts/results/judge_only.parquet",
    live: bool = False,
    probes: str = "pusht/linear@1",
    vocab: str = "pusht@1",
    data: str = "artifacts/data/pusht_expert.npz",
    llm_model: str = "qwen2.5:7b-instruct",
    variants: str = "canonical,p1,p2,p3,p4,p5,n1,n2",
    device: str | None = None,
) -> None:
    """Judge-only study: all judges on identical inputs; canonical, paraphrases and negatives."""
    _mode(live)
    from lejudge.constraints import load_library
    from lejudge.eval.judge_study import (
        build_items,
        make_judges,
        metrics_table,
        run_judges,
        save_config_snapshot,
    )
    from lejudge.vocab import load_vocab

    t0 = time.time()
    items = build_items(n_executed, n_imagined, seed=seed, data_path=data, probes=probes, vocab_name=vocab, device=device)
    vocab_obj = load_vocab(vocab)
    lib = load_library(env)
    js = make_judges(judges.split(","), vocab_obj, llm_model=llm_model)
    df = run_judges(items, js, lib, vocab_obj, variants=tuple(variants.split(",")), repeats=repeats, repeat_items=repeat_items)
    df["seed"] = seed
    df["probes"] = probes
    out_p = Path(out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    if out_p.exists():
        import pandas as pd

        old = pd.read_parquet(out_p)
        old = old[~old.judge.isin(df.judge.unique()) | (old.seed != seed) | (old.probes != probes)]
        df = pd.concat([old, df], ignore_index=True)
    df.to_parquet(out_p, index=False)
    save_config_snapshot(out_p.with_suffix(".config.json"), {"judges": judges, "n_executed": n_executed, "n_imagined": n_imagined, "seed": seed, "elapsed_s": time.time() - t0})
    typer.echo(metrics_table(df).to_string())
    typer.echo(f"rows={len(df)} → {out_p} ({time.time() - t0:.0f}s)")


@app.command()
def ablate(grid: str = "configs/ablations.yaml", live: bool = False, out: str = "artifacts/results/ablations.parquet", device: str | None = None) -> None:
    """Run the ablation grid (λ sweep, K, final_only, vocab, steps, hard rejection)."""
    _mode(live)
    import yaml

    from lejudge.eval.planning import RunConfig, Runner, append_results

    spec = yaml.safe_load(Path(grid).read_text())
    runner = Runner(device=device, data_path=spec.get("data", "artifacts/data/pusht_expert.npz"))
    base = spec.get("base", {})
    for run in spec["runs"]:
        params = {**base, **{k: v for k, v in run.items() if k != "plan_overrides"}}
        cfg = RunConfig(**params, plan_overrides=run.get("plan_overrides", base.get("plan_overrides", {})))
        df = runner.run(cfg)
        append_results(df, out)
        typer.echo(f"{cfg.tag or cfg.condition}: success={df.success.mean():.3f} violation={df.violation.mean():.3f}")


@app.command()
def report(out: str = "paper/figures", results: str = "artifacts/results", offline: bool = True, probes: str = "pusht/linear@1") -> None:
    """Regenerate every figure and table from the Parquet tables (never touches an API)."""
    os.environ["LEJUDGE_MODE"] = "offline"
    from lejudge.eval.report import build_report

    s = build_report(results, out, probes_meta_path=Path("artifacts/probes") / probes / "meta.json")
    typer.echo(f"figures: {s['figures']}\ntables: {list(s['tables'])}\n→ {out}")


@app.command()
def consistency(n_items: int = 64, repeats: int = 3, out: str = "artifacts/results/consistency.parquet", live: bool = False, seed: int = 0) -> None:
    """3 repeats on a subset of items (refresh mode with a uid) → per-question std-dev."""
    _mode(live)
    if live:
        os.environ["LEJUDGE_MODE"] = "live"
    from lejudge.constraints import load_library
    from lejudge.eval.judge_study import build_items, consistency_table, make_judges, run_judges
    from lejudge.vocab import load_vocab

    items = build_items(n_items, 0, seed=seed)
    vocab = load_vocab("pusht@1")
    lib = load_library()
    js = make_judges(["jev"], vocab)
    # repeats use LEJUDGE_MODE=refresh semantics via uid: rep>0 rows are fresh calls (or cached by uid)
    df = run_judges(items, js, lib, vocab, variants=("canonical",), repeats=repeats, repeat_items=len(items))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    typer.echo(consistency_table(df).to_string())


@app.command()
def m0(episodes: int = 50, seed: int = 0, out: str = "artifacts/results/m0.parquet", device: str | None = None, history_len: int | None = None, data: str = "artifacts/data/pusht_expert.npz", tag: str = "m0") -> None:
    """M0: unconstrained LeWM planning success (reproduction gate)."""
    os.environ["LEJUDGE_MODE"] = "offline"
    from lejudge.eval.planning import RunConfig, Runner, append_results

    runner = Runner(device=device, data_path=data)
    overrides = {"history_len": history_len} if history_len is not None else {}
    df = runner.run(RunConfig(condition="lewm", constraint_set="edges", seed=seed, episodes=episodes, tag=tag, device=device, data_path=data, plan_overrides=overrides))
    append_results(df, out)
    typer.echo(f"M0 success={df.success.mean():.3f} (n={len(df)}) plan p50={df.plan_time_p50_s.median():.2f}s")


@app.command()
def demo(share: bool = False, port: int = 7860, live: bool = False) -> None:
    """Launch the Gradio Space locally."""
    _mode(live)
    from lejudge.demo.app import launch

    launch(share=share, port=port)


@app.command()
def record(out: str = "artifacts/demo/clip.mp4", seed: int = 0, live: bool = False) -> None:
    """Render the 20-second demo clip from a deterministic script."""
    _mode(live)
    from lejudge.demo.record import record_clip

    typer.echo(record_clip(out, seed=seed))


@app.command()
def cache_stats() -> None:
    from lejudge.judge import get_cache

    c = get_cache()
    typer.echo(json.dumps({"path": str(c.path), "responses": c.count(), "jev": c.count("jev-1.13")}, indent=2))


def _parse_range(s: str) -> list[int]:
    out: list[int] = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


if __name__ == "__main__":
    app()


@app.command("merge-results")
def merge_results(pattern: str = "artifacts/results/planning_*.parquet", out: str = "artifacts/results/planning.parquet") -> None:
    """Merge per-process planning tables into the main table (rows keyed by run_id; re-runs replace)."""
    import glob

    import pandas as pd

    from lejudge.eval.planning import append_results

    for f in sorted(glob.glob(pattern)):
        df = pd.read_parquet(f)
        append_results(df, out)
        typer.echo(f"merged {f}: {len(df)} rows ({df.run_id.nunique()} runs)")
    typer.echo(f"→ {out}: {len(pd.read_parquet(out))} rows")


@app.command("demo-precompute")
def demo_precompute(seeds: int = 20, live: bool = False, judge: str = "jev") -> None:
    """Precompute the demo gallery: seeds × presets, stock and LeJudge plans (Jev responses cached)."""
    _mode(live)
    from lejudge.demo.core import PRESETS, DemoBackend

    b = DemoBackend()
    b.precompute(range(seeds), PRESETS, mode=judge)
    typer.echo(f"gallery: {seeds} seeds × {len(PRESETS)} presets → artifacts/demo/gallery")
