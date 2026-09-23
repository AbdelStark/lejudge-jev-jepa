"""``lejudge report``: figures and tables from the Parquet results, offline only (RFC-0006)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lejudge.eval.stats import bootstrap_ci, holm, risk_difference, wilcoxon_paired

RESULTS = Path("artifacts/results")
PALETTE = {
    "lewm": "#7f7f7f",
    "oracle": "#1b9e77",
    "keyword": "#d95f02",
    "jev": "#7570b3",
    "llm-small": "#e7298a",
}
LABELS = {
    "lewm": "LeWM (unconstrained)",
    "oracle": "oracle-on-probes",
    "keyword": "keyword",
    "jev": "LeJudge (Jev)",
    "llm-small": "LLM-small",
}


def _plt():
    import os

    import matplotlib

    # Byte-stable PDFs: matplotlib stamps CreationDate from SOURCE_DATE_EPOCH when it is set,
    # so re-running `make paper` on unchanged results leaves every figure file unchanged.
    os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150}
    )
    return plt


def planning_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cond, cset), g in df.groupby(["condition", "constraint_set"]):
        s, slo, shi = bootstrap_ci(g.success.to_numpy(dtype=float))
        v, vlo, vhi = bootstrap_ci(g.violation.to_numpy(dtype=float))
        rows.append(
            {
                "condition": cond,
                "constraint_set": cset,
                "n": len(g),
                "seeds": g.seed.nunique(),
                "success": s,
                "success_lo": slo,
                "success_hi": shi,
                "violation": v,
                "violation_lo": vlo,
                "violation_hi": vhi,
                "violation_steps": float(g.violation_steps.mean()),
                "plan_p50_s": float(np.nanmedian(g.plan_time_p50_s)),
                "plan_p95_s": float(np.nanpercentile(g.plan_time_p95_s.dropna(), 95))
                if g.plan_time_p95_s.notna().any()
                else float("nan"),
                "judge_calls_per_episode": float(g.judge_calls.mean()),
                "tokens_in_per_episode": float(g.tokens_in.mean()),
                "abstention_rate": float(g.abstention_rate.mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["constraint_set", "condition"]).reset_index(drop=True)


def paired_tests(df: pd.DataFrame, ref: str = "lewm", metric: str = "violation") -> pd.DataFrame:
    """Each condition vs ``ref``, paired by (constraint set, seed, episode); Holm across sets."""
    rows = []
    for cond in sorted(df.condition.unique()):
        if cond == ref:
            continue
        pvals: dict[str, float] = {}
        tmp = []
        for cset, g in df[df.condition.isin([cond, ref])].groupby("constraint_set"):
            a = g[g.condition == cond].set_index(["seed", "episode"])[metric].astype(float)
            b = g[g.condition == ref].set_index(["seed", "episode"])[metric].astype(float)
            common = a.index.intersection(b.index)
            if len(common) < 2:
                continue
            a, b = a.loc[common].to_numpy(), b.loc[common].to_numpy()
            rd, lo, hi = risk_difference(a, b)
            p = wilcoxon_paired(a, b)
            pvals[cset] = p
            tmp.append(
                {
                    "condition": cond,
                    "ref": ref,
                    "constraint_set": cset,
                    "metric": metric,
                    "n_pairs": len(common),
                    "risk_difference": rd,
                    "rd_lo": lo,
                    "rd_hi": hi,
                    "p": p,
                }
            )
        adj = holm(pvals) if pvals else {}
        for r in tmp:
            r["p_holm"] = adj.get(r["constraint_set"], float("nan"))
            rows.append(r)
    return pd.DataFrame(rows)


def fig_success_vs_violation(df: pd.DataFrame, out: Path) -> None:
    plt = _plt()
    t = planning_table(df)
    sets = list(dict.fromkeys(t.constraint_set))
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(sets), figsize=(3.2 * len(sets), 3.0), squeeze=False)
    for ax, cset in zip(axes[0], sets):
        g = t[t.constraint_set == cset]
        for _, r in g.iterrows():
            c = PALETTE.get(str(r.condition).split(":")[0], "k")
            marker = {"jev:tau0.75": "s", "jev:tau1.0": "^"}.get(str(r.condition), "o")
            ax.errorbar(
                r.violation,
                r.success,
                xerr=[[r.violation - r.violation_lo], [r.violation_hi - r.violation]],
                yerr=[[r.success - r.success_lo], [r.success_hi - r.success]],
                fmt=marker,
                color=c,
                capsize=2,
                label=LABELS.get(r.condition, r.condition),
            )
        ax.set_title(f"set: {cset} (n={int(g.n.max())}/cond)")
        ax.set_xlabel("episode violation rate (oracle on executed states)")
        ax.set_ylabel("success rate")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
    axes[0][0].legend(fontsize=7, loc="best")
    fig.suptitle(
        "Success vs violation per condition; 95% bootstrap CIs (10,000 resamples)", fontsize=9
    )
    fig.tight_layout()
    fig.savefig(out / "fig2_success_vs_violation.pdf")
    fig.savefig(out / "fig2_success_vs_violation.png")
    plt.close(fig)


def fig_pareto(df: pd.DataFrame, out: Path) -> None:
    plt = _plt()
    out.mkdir(parents=True, exist_ok=True)
    g = df[
        (df.condition == "jev") & (df.tag.astype(str).str.contains("lam_sweep") | (df.lam != 1.0))
    ]
    base = df[(df.condition == "jev") & (df.lam == 1.0)]
    g = pd.concat([g, base])
    if g.empty:
        return
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    for cset, gg in g.groupby("constraint_set"):
        rows = []
        for lam, h in gg.groupby("lam"):
            s, slo, shi = bootstrap_ci(h.success.to_numpy(dtype=float))
            v, vlo, vhi = bootstrap_ci(h.violation.to_numpy(dtype=float))
            rows.append((lam, s, slo, shi, v, vlo, vhi))
        rows.sort()
        lam, s, slo, shi, v, vlo, vhi = map(np.array, zip(*rows))
        ax.plot(v, s, "-o", label=cset)
        ax.fill_betweenx(s, vlo, vhi, alpha=0.15)
        for L, x, y in zip(lam, v, s):
            ax.annotate(f"λ={L:g}", (x, y), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("episode violation rate")
    ax.set_ylabel("success rate")
    ax.set_title("λ sweep (Jev); bands: 95% CI on violation", fontsize=8)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig3_pareto_lambda.pdf")
    fig.savefig(out / "fig3_pareto_lambda.png")
    plt.close(fig)


def fig_paraphrase_heatmap(jdf: pd.DataFrame, out: Path) -> None:
    plt = _plt()
    from lejudge.eval.judge_study import per_constraint_table

    t = per_constraint_table(
        jdf[jdf.description != "gt-words"] if (jdf.description == "probe-words").any() else jdf
    )
    judges = list(dict.fromkeys(t.judge))
    variants = ["canonical", "p1", "p2", "p3", "p4", "p5", "n1", "n2"]
    fig, axes = plt.subplots(1, len(judges), figsize=(2.6 * len(judges) + 1, 3.6), squeeze=False)
    for ax, j in zip(axes[0], judges):
        g = t[t.judge == j].pivot_table(index="constraint", columns="variant", values="accuracy")
        g = g.reindex(columns=[v for v in variants if v in g.columns])
        im = ax.imshow(g.to_numpy(dtype=float), vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(g.columns)))
        ax.set_xticklabels(g.columns, rotation=60, fontsize=6)
        ax.set_yticks(range(len(g.index)))
        ax.set_yticklabels(g.index if ax is axes[0][0] else [], fontsize=6)
        ax.set_title(j, fontsize=8)
    fig.colorbar(im, ax=axes[0].tolist(), label="accuracy @0.5", shrink=0.8)
    fig.suptitle(
        "Accuracy per constraint × text variant (canonical, paraphrases p1–p5, near-miss negatives n1–n2)",
        fontsize=8,
    )
    fig.savefig(out / "fig4_paraphrase_heatmap.pdf", bbox_inches="tight")
    fig.savefig(out / "fig4_paraphrase_heatmap.png", bbox_inches="tight")
    plt.close(fig)


def fig_latency_cost(jdf: pd.DataFrame, out: Path) -> None:
    plt = _plt()
    g = (
        jdf[jdf.repeat == 0]
        .groupby("judge")
        .agg(latency=("latency_ms_per_q", "mean"), tokens=("tokens_in_per_q", "mean"))
        .reset_index()
    )
    g["latency_per_1000_s"] = g.latency  # ms per question × 1000 questions = seconds
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    x = np.arange(len(g))
    ax.bar(
        x,
        np.maximum(g.latency_per_1000_s, 1e-3),
        color=[PALETTE.get(j.split(":")[0], "k") for j in g.judge],
    )
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(g.judge, rotation=20, fontsize=7)
    ax.set_ylabel("wall-clock seconds per 1,000 judgments (log)")
    ax.set_title("Judge latency (measured on live calls)", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig5_latency_cost.pdf")
    fig.savefig(out / "fig5_latency_cost.png")
    plt.close(fig)


def fig_reliability(jdf: pd.DataFrame, out: Path) -> None:
    plt = _plt()
    from lejudge.eval.stats import ece

    judges = [j for j in dict.fromkeys(jdf.judge) if j != "oracle"]
    fig, axes = plt.subplots(1, len(judges), figsize=(2.6 * len(judges), 2.8), squeeze=False)
    for ax, j in zip(axes[0], judges):
        g = jdf[(jdf.judge == j) & (jdf.repeat == 0) & (jdf.variant == "canonical")]
        e, table = ece(g.label.to_numpy(), g.score.to_numpy())
        if table:
            ax.plot(
                [r["confidence"] for r in table],
                [r["accuracy"] for r in table],
                "o-",
                color=PALETTE.get(j.split(":")[0], "k"),
            )
        ax.plot([0, 1], [0, 1], "--", color="#999", lw=0.8)
        ax.set_title(f"{j}  ECE={e:.3f}", fontsize=8)
        ax.set_xlabel("mean score in bin")
        ax.set_ylabel("fraction violated")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    fig.suptitle("Reliability diagrams (10 equal-mass bins, canonical texts)", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig6_reliability.pdf")
    fig.savefig(out / "fig6_reliability.png")
    plt.close(fig)


def fig_error_decomposition(
    jdf: pd.DataFrame | None, probes_meta: dict[str, Any] | None, out: Path
) -> None:
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.8))
    if probes_meta and "imagined" in probes_meta:
        curve = probes_meta["imagined"]["curve"]
        hs = [r["h"] for r in curve]
        for field in ("block", "block_edge", "block_angle", "agent", "contact"):
            axes[0].plot(hs, [r["bucket_accuracy"][field] for r in curve], "o-", label=field, ms=3)
        axes[0].set_xlabel("imagined step h (0 = encoded frame)")
        axes[0].set_ylabel("bucket accuracy vs ground truth")
        axes[0].set_title("Probe + world-model error over the horizon", fontsize=8)
        axes[0].legend(fontsize=6)
        axes[0].set_ylim(0, 1)
    if jdf is not None and not jdf.empty:
        from lejudge.eval.stats import prf

        rows = []
        for (j, desc), g in jdf[(jdf.repeat == 0) & (jdf.variant == "canonical")].groupby(
            ["judge", "description"]
        ):
            rows.append((f"{j}\n{desc}", prf(g.label.to_numpy(), g.score.to_numpy())["accuracy"]))
        if rows:
            names, vals = zip(*rows)
            axes[1].bar(range(len(vals)), vals, color="#7570b3")
            axes[1].set_xticks(range(len(vals)))
            axes[1].set_xticklabels(names, fontsize=6, rotation=30)
            axes[1].set_ylim(0, 1)
            axes[1].set_ylabel("judge accuracy vs oracle")
            axes[1].set_title("Judge error on ground-truth words vs probe words", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig7_error_decomposition.pdf")
    fig.savefig(out / "fig7_error_decomposition.png")
    plt.close(fig)


def _md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(
                floatfmt.format(v)
                if isinstance(v, (float, np.floating))
                and not (isinstance(v, float) and np.isnan(v))
                else str(v)
            )
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_report(
    results: Path | str = RESULTS,
    out: Path | str = "paper/figures",
    probes_meta_path: Path | str = "artifacts/probes/pusht/linear@1/meta.json",
) -> dict[str, Any]:
    results, out = Path(results), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"figures": [], "tables": {}}
    pdf = (
        pd.read_parquet(results / "planning.parquet")
        if (results / "planning.parquet").exists()
        else None
    )
    jdf = (
        pd.read_parquet(results / "judge_only.parquet")
        if (results / "judge_only.parquet").exists()
        else None
    )
    adf = (
        pd.read_parquet(results / "ablations.parquet")
        if (results / "ablations.parquet").exists()
        else None
    )
    probes_meta = (
        json.loads(Path(probes_meta_path).read_text()) if Path(probes_meta_path).exists() else None
    )
    if pdf is not None and not pdf.empty:
        t2 = planning_table(pdf)
        t2.to_csv(out / "table2_planning.csv", index=False)
        (out / "table2_planning.md").write_text(_md_table(t2))
        summary["tables"]["planning"] = t2.to_dict(orient="records")
        tests = paired_tests(pdf, ref="lewm", metric="violation")
        tests_s = paired_tests(pdf, ref="lewm", metric="success")
        tests_all = pd.concat([tests, tests_s])
        tests_all.to_csv(out / "table2b_paired_tests.csv", index=False)
        (out / "table2b_paired_tests.md").write_text(_md_table(tests_all, "{:.4f}"))
        summary["tables"]["paired_tests"] = tests_all.to_dict(orient="records")
        fig_success_vs_violation(pdf, out)
        summary["figures"].append("fig2_success_vs_violation")
        try:
            sat = annotate_satisfiable(pdf)
            t2s = stratified_table(sat)
            t2s.to_csv(out / "table2c_stratified.csv", index=False)
            (out / "table2c_stratified.md").write_text(_md_table(t2s))
            summary["tables"]["stratified"] = t2s.to_dict(orient="records")
            fig_success_vs_violation(sat[sat.satisfiable], out / "satisfiable")
        except FileNotFoundError:
            pass
        full = pd.concat([pdf, adf]) if adf is not None else pdf
        fig_pareto(full, out)
        summary["figures"].append("fig3_pareto_lambda")
    fdf = (
        pd.read_parquet(results / "planning_filtered.parquet")
        if (results / "planning_filtered.parquet").exists()
        else None
    )
    if fdf is not None and not fdf.empty:
        t4 = planning_table(fdf)
        t4.to_csv(out / "table4_planning_filtered.csv", index=False)
        (out / "table4_planning_filtered.md").write_text(_md_table(t4))
        summary["tables"]["planning_filtered"] = t4.to_dict(orient="records")
        ft = pd.concat(
            [
                paired_tests(fdf, ref="lewm", metric="violation"),
                paired_tests(fdf, ref="lewm", metric="success"),
            ]
        )
        ft.to_csv(out / "table4b_paired_tests_filtered.csv", index=False)
        (out / "table4b_paired_tests_filtered.md").write_text(_md_table(ft, "{:.4f}"))
        summary["tables"]["paired_tests_filtered"] = ft.to_dict(orient="records")
        fig_success_vs_violation(fdf, out / "filtered")
        summary["figures"].append("filtered/fig2_success_vs_violation")
        afd = (
            pd.read_parquet(results / "ablations_filtered.parquet")
            if (results / "ablations_filtered.parquet").exists()
            else None
        )
        if afd is not None and not afd.empty:
            fig_pareto(pd.concat([fdf, afd]), out / "filtered")
            summary["figures"].append("filtered/fig3_pareto_lambda")
    gdf = (
        pd.read_parquet(results / "planning_gate.parquet")
        if (results / "planning_gate.parquet").exists()
        else None
    )
    if gdf is not None and not gdf.empty and fdf is not None:
        sets = sorted(gdf.constraint_set.unique())
        base = fdf[
            fdf.constraint_set.isin(sets) & fdf.condition.isin(["lewm", "oracle", "keyword", "jev"])
        ].copy()
        base["condition"] = base.condition.where(base.condition != "jev", "jev:tau0.5")
        g = gdf.copy()
        g["condition"] = "jev:tau" + g.tau.astype(str)
        both = pd.concat([base, g], ignore_index=True)
        t5 = planning_table(both)
        t5.to_csv(out / "table5_gate.csv", index=False)
        (out / "table5_gate.md").write_text(_md_table(t5))
        summary["tables"]["gate"] = t5.to_dict(orient="records")
        tests5 = pd.concat(
            [
                paired_tests(both, ref="lewm", metric="violation"),
                paired_tests(both, ref="jev:tau0.5", metric="violation"),
                paired_tests(both, ref="lewm", metric="success"),
            ]
        )
        tests5.to_csv(out / "table5b_paired_tests_gate.csv", index=False)
        (out / "table5b_paired_tests_gate.md").write_text(_md_table(tests5, "{:.4f}"))
        summary["tables"]["paired_tests_gate"] = tests5.to_dict(orient="records")
        fig_success_vs_violation(both, out / "gate")
        summary["figures"].append("gate/fig2_success_vs_violation")
    if adf is not None and not adf.empty:
        t3 = planning_table(adf.assign(condition=adf.condition + ":" + adf.tag.astype(str)))
        t3.to_csv(out / "table3_ablations.csv", index=False)
        (out / "table3_ablations.md").write_text(_md_table(t3))
        summary["tables"]["ablations"] = t3.to_dict(orient="records")
    if jdf is not None and not jdf.empty:
        from lejudge.eval.judge_study import consistency_table, metrics_table, per_constraint_table

        t1 = metrics_table(jdf)
        t1.to_csv(out / "table1_judge_only.csv", index=False)
        (out / "table1_judge_only.md").write_text(_md_table(t1))
        summary["tables"]["judge_only"] = t1.to_dict(orient="records")
        per_constraint_table(jdf).to_csv(out / "table1b_per_constraint.csv", index=False)
        ct = consistency_table(jdf)
        ct.to_csv(out / "table1c_consistency.csv", index=False)
        summary["tables"]["consistency"] = ct.to_dict(orient="records")
        fig_paraphrase_heatmap(jdf, out)
        fig_latency_cost(jdf, out)
        fig_reliability(jdf, out)
        summary["figures"] += ["fig4_paraphrase_heatmap", "fig5_latency_cost", "fig6_reliability"]
    fig_error_decomposition(jdf, probes_meta, out)
    summary["figures"].append("fig7_error_decomposition")
    if probes_meta:
        summary["probes"] = {
            "name": probes_meta.get("name"),
            "test": probes_meta["metrics"]["test"],
            "imagined": probes_meta.get("imagined"),
        }
    try:
        from lejudge.eval.paper_figures import build_paper_figures

        summary["paper_figures"] = build_paper_figures(results, out / "paper", probes_meta_path)
    except Exception as e:  # noqa: BLE001 — paper figures must never break the core report
        summary["paper_figures_error"] = repr(e)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=_json_default))
    return summary


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o).__name__)


# ------------------------------------------------------------------ post-hoc stratification
EPISODE_STATES = RESULTS / "episode_states.parquet"
"""Start and goal ground-truth states of every planning episode, keyed by
(start_filter, constraint_set, seed, episode). Committed so the report and the paper build on a
clean clone without the 69 MB expert dataset, which is only needed to regenerate this table."""


def _episode_key(r: Any, has_filter: bool) -> tuple[str, str, int, int]:
    filt = str(r.start_filter) if has_filter else "none"
    return (filt, str(r.constraint_set) if filt != "none" else "", int(r.seed), int(r.episode))


def episode_states(
    df: pd.DataFrame, data_path: str = "artifacts/data/pusht_expert.npz"
) -> dict[tuple[str, str, int, int], tuple[Any, Any]]:
    """``{key: (start, goal)}`` as ``GroundTruthState`` for every episode row of ``df``.

    Read from ``EPISODE_STATES`` when it covers ``df``; otherwise re-derived from the expert
    dataset with the planner's own sampler and written back (raises ``FileNotFoundError`` when
    neither is available).
    """
    from lejudge.types import GroundTruthState

    has_filter = "start_filter" in df.columns
    need = {_episode_key(r, has_filter) for r in df.itertuples()}
    states: dict[tuple[str, str, int, int], tuple[Any, Any]] = {}
    if EPISODE_STATES.exists():
        for r in pd.read_parquet(EPISODE_STATES).itertuples():
            states[(r.start_filter, r.constraint_set, int(r.seed), int(r.episode))] = (
                GroundTruthState.from_json(json.loads(r.start_state)),
                GroundTruthState.from_json(json.loads(r.goal_state)),
            )
        if need <= states.keys():
            return states
    from lejudge.constraints import load_library
    from lejudge.eval.planning import sample_episode_specs
    from lejudge.probes.data import EpisodeData
    from lejudge.vocab import load_vocab

    if not Path(data_path).exists():
        raise FileNotFoundError(
            f"{EPISODE_STATES} does not cover these runs and {data_path} is absent"
        )
    data, vocab, lib = EpisodeData(data_path), load_vocab("pusht@1"), load_library()
    for filt, cset, seed in sorted({k[:3] for k in need - states.keys()}):
        rows = df[(df.seed == seed) & ((df.constraint_set == cset) if filt != "none" else True)]
        n = int(rows.episode.max()) + 1
        specs = sample_episode_specs(
            data,
            n,
            seed,
            25,
            start_filter=filt,
            constraint_set=cset or str(rows.constraint_set.iloc[0]),
            vocab=vocab,
            lib=lib,
        )
        for es in specs:
            states[(filt, cset, seed, es.episode)] = (
                GroundTruthState.from_env(es.state),
                GroundTruthState.from_env(es.goal_state),
            )
    table = pd.DataFrame(
        [
            {
                "start_filter": k[0],
                "constraint_set": k[1],
                "seed": k[2],
                "episode": k[3],
                "start_state": json.dumps(s0.to_json()),
                "goal_state": json.dumps(g.to_json()),
            }
            for k, (s0, g) in sorted(states.items())
        ]
    )
    EPISODE_STATES.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(EPISODE_STATES, index=False)
    return states


def annotate_satisfiable(
    df: pd.DataFrame, data_path: str = "artifacts/data/pusht_expert.npz"
) -> pd.DataFrame:
    """Add ``satisfiable``: whether the constraint set can be met while reaching the goal, judged
    from the start and goal states only (exploratory, not pre-registered; see DECISIONS).

    centre_avoid: neither the start nor the goal block centroid is in the centre cell.
    no_contact_first3 / approach_below: the agent is not already touching the block at the start.
    gentle: always satisfiable.
    """
    from lejudge.constraints import load_library
    from lejudge.vocab import load_vocab

    vocab = load_vocab("pusht@1")
    lib = load_library()
    states = episode_states(df, data_path)
    has_filter = "start_filter" in df.columns
    flags = []
    for r in df.itertuples():
        s0, g = states[_episode_key(r, has_filter)]
        ok = True
        for cid in lib.sets.get(str(r.constraint_set), ()):
            if cid == "centre_avoid":
                for st in (s0, g):
                    cell = vocab.cell_name(
                        vocab.block_centroid(np.array(st.block_xy), np.array(st.block_angle))
                    )
                    ok &= str(cell) != "centre"
            elif cid in ("no_contact_first3", "approach_below"):
                ok &= not s0.contact
        flags.append(bool(ok))
    out = df.copy()
    out["satisfiable"] = flags
    return out


def stratified_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cond, cset, sat), g in df.groupby(["condition", "constraint_set", "satisfiable"]):
        s, slo, shi = bootstrap_ci(g.success.to_numpy(dtype=float))
        v, vlo, vhi = bootstrap_ci(g.violation.to_numpy(dtype=float))
        rows.append(
            {
                "condition": cond,
                "constraint_set": cset,
                "satisfiable": sat,
                "n": len(g),
                "success": s,
                "success_lo": slo,
                "success_hi": shi,
                "violation": v,
                "violation_lo": vlo,
                "violation_hi": vhi,
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["constraint_set", "satisfiable", "condition"])
        .reset_index(drop=True)
    )
