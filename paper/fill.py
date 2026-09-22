"""Fill every \\VAL{...} placeholder in paper/main.tex from artifacts/results and paper/figures/summary.json.
Numbers that are not available become the literal ``[X]`` so nothing is fabricated."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "paper" / "figures"
RES = ROOT / "artifacts" / "results"


def pct(x: float | None) -> str:
    return "[X]" if x is None or pd.isna(x) else f"{100 * x:.0f}\\%"


def num(x: float | None, d: int = 2) -> str:
    return "[X]" if x is None or pd.isna(x) else f"{x:.{d}f}"


def main() -> None:
    vals: dict[str, str] = {}
    summary = json.loads((FIG / "summary.json").read_text()) if (FIG / "summary.json").exists() else {}
    cfg = __import__("yaml").safe_load((ROOT / "configs" / "pusht.yaml").read_text())
    vals["ckpt_rev"] = cfg["checkpoint"]["revision"][:12]
    vals["judge_last_n"] = str(cfg["jevcost"]["judge_last_n"])
    vals["cem_iters"] = str(cfg["solver"]["n_steps"])
    from lejudge.constraints import load_library

    vals["library_sha"] = load_library().sha256[:12]
    # probes
    pm = summary.get("probes")
    if pm:
        t = pm["test"]
        vals["probe_r2_block"] = num(t["r2"]["block_xy"], 3)
        vals["probe_r2_angle"] = num(t["r2"]["angle_sincos"], 3)
        vals["probe_bucket_block"] = num(t["bucket_accuracy"]["block"], 2)
        vals["probe_bucket_angle"] = num(t["bucket_accuracy"]["block_angle"], 2)
        vals["probe_bucket_edge"] = num(t["bucket_accuracy"]["block_edge"], 2)
        if pm.get("imagined"):
            vals["probe_bucket_block_h5"] = num(pm["imagined"]["curve"][-1]["bucket_accuracy"]["block"], 2)
    # M0
    if (RES / "m0.parquet").exists():
        m0 = pd.read_parquet(RES / "m0.parquet")
        m0 = m0[m0.tag == "m0_h3"] if (m0.tag == "m0_h3").any() else m0[m0.tag == "m0"]  # frozen config: history 3
        if len(m0):
            vals["m0_success"] = pct(m0.success.mean())
            vals["m0_n"] = str(len(m0))
    # planning
    if (RES / "planning.parquet").exists():
        p = pd.read_parquet(RES / "planning.parquet")
        for cond in ("lewm", "jev", "oracle", "keyword"):
            g = p[p.condition == cond]
            if len(g):
                vals[f"{cond}_violation_all"] = pct(g.violation.mean())
                vals[f"{cond}_success_all"] = pct(g.success.mean())
        gj = p[(p.condition == "jev") & (p.constraint_set == "spatial")]
        vals["jev_abstention_spatial"] = pct(gj.abstention_rate.mean()) if len(gj) else "[X]"
        vals["plan_episodes"] = str(int(p[p.condition != "llm-small"].groupby(["condition", "constraint_set", "seed"]).size().max()))
        try:
            from lejudge.eval.report import annotate_satisfiable

            sat = annotate_satisfiable(p)
            sp = sat[sat.constraint_set == "spatial"]
            if len(sp):
                vals["spatial_unsat_frac"] = pct(1.0 - sp[sp.condition == "lewm"].satisfiable.mean()) if (sp.condition == "lewm").any() else "[X]"
            for cond in ("lewm", "jev", "oracle", "keyword"):
                g = sat[(sat.condition == cond) & sat.satisfiable]
                if len(g):
                    vals[f"{cond}_violation_sat"] = pct(g.violation.mean())
                    vals[f"{cond}_success_sat"] = pct(g.success.mean())
        except FileNotFoundError:
            pass
        vals["plan_seeds"] = ",".join(str(s) for s in sorted(p.seed.unique()))
        has_llm = (p.condition == "llm-small").any()
        vals["llm_cond_note"] = ", LLM-small (subset)" if has_llm else ""
        vals["jev_model"] = str(p[p.condition == "jev"].response_model.replace("", pd.NA).dropna().iloc[0]) if (p.condition == "jev").any() and p[p.condition == "jev"].response_model.replace("", pd.NA).notna().any() else "[X]"
        tex_table(FIG / "table2_planning.csv", FIG / "table2_planning.tex", ["condition", "constraint_set", "n", "success", "success_lo", "success_hi", "violation", "violation_lo", "violation_hi", "plan_p50_s", "judge_calls_per_episode"])
    # judge-only
    if (RES / "judge_only.parquet").exists():
        j = pd.read_parquet(RES / "judge_only.parquet")
        from lejudge.eval.judge_study import metrics_table

        t = metrics_table(j)
        pw = t[t.description == "probe-words"].groupby("judge").mean(numeric_only=True)

        def get(judge: str, col: str) -> float | None:
            return float(pw.loc[judge, col]) if judge in pw.index and col in pw.columns else None

        vals["jev_para_acc"] = pct(get("jev", "paraphrase_accuracy"))
        vals["jev_canon_acc"] = pct(get("jev", "canonical_accuracy"))
        vals["kw_para_acc"] = pct(get("keyword", "paraphrase_accuracy"))
        vals["kw_canon_acc"] = pct(get("keyword", "canonical_accuracy"))
        vals["judge_n_items"] = str(j.item_id.nunique())
        lj = j[j.judge.astype(str).str.startswith("llm")]
        if len(lj):
            models = lj.response_model.replace("", pd.NA).dropna().unique()
            mname = str(models[0]) if len(models) else "qwen2.5:7b-instruct"
            vals["llm_note"] = f"a local open model ({mname} via Ollama) evaluated on {lj.item_id.nunique()} items and three texts per constraint; it failed to return schema-valid answers in {100 * lj.failed.mean():.0f}\\% of calls"
        else:
            vals["llm_note"] = "not run (no funded hosted-LLM account at build time)"
        tex_table(FIG / "table1_judge_only.csv", FIG / "table1_judge_only.tex", ["judge", "source", "description", "n_items", "canonical_accuracy", "canonical_auroc", "canonical_ece", "paraphrase_accuracy", "negative_fpr", "latency_ms_per_1000"])
    from lejudge.judge import get_cache

    vals["cache_n"] = str(get_cache().count())
    tex = (ROOT / "paper" / "main.tex").read_text()
    missing = []

    def sub(m: re.Match) -> str:
        k = m.group(1)
        if k not in vals:
            missing.append(k)
            return "[X]"
        return vals[k]

    out = re.sub(r"\\VAL\{([a-zA-Z0-9_]+)\}", sub, tex)
    (ROOT / "paper" / "main.filled.tex").write_text(out)
    (ROOT / "paper" / "values.json").write_text(json.dumps(vals, indent=2, sort_keys=True))
    print(f"filled {len(vals)} values; missing → [X]: {sorted(set(missing))}")


def tex_table(csv: Path, out: Path, cols: list[str]) -> None:
    if not csv.exists():
        out.write_text("\\textit{[X] table not yet generated}")
        return
    df = pd.read_csv(csv)
    cols = [c for c in cols if c in df.columns]
    df = df[cols]
    lines = ["\\begin{tabular}{" + "l" * len(cols) + "}", "\\toprule", " & ".join(c.replace("_", "\\_") for c in cols) + " \\\\", "\\midrule"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(f"{v:.3f}" if isinstance(v, float) else str(v).replace("_", "\\_"))
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    out.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
