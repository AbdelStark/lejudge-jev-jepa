"""Fill every \\VAL{...} placeholder in paper/main.tex from artifacts/results and paper/figures.
Numbers that are not available become the literal ``[X]`` so nothing is fabricated."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "paper" / "figures"
RES = ROOT / "artifacts" / "results"
COND_TEX = {"lewm": "LeWM (unconstrained)", "keyword": "keyword checker", "oracle": "oracle-on-probes", "jev": "LeJudge / Jev ($\\tau{=}0.5$)", "jev:tau0.5": "LeJudge / Jev ($\\tau{=}0.5$)", "jev:tau0.75": "Jev ($\\tau{=}0.75$)", "jev:tau1.0": "Jev, gate off ($\\tau{=}1$)"}
SET_TEX = {"spatial": "spatial", "spatial+temporal": "spatial+temporal", "implicit": "implicit"}


def pct(x) -> str:
    return "[X]" if x is None or pd.isna(x) else f"{100 * float(x):.0f}\\%"


def pct1(x) -> str:
    return "[X]" if x is None or pd.isna(x) else f"{100 * float(x):.1f}\\%"


def num(x, d: int = 2) -> str:
    return "[X]" if x is None or pd.isna(x) else f"{float(x):.{d}f}"


def ci(m, lo, hi, d: int = 2) -> str:
    return "[X]" if any(pd.isna(v) for v in (m, lo, hi)) else f"{float(m):.{d}f} [{float(lo):.{d}f}, {float(hi):.{d}f}]"


def pts(x) -> str:
    return "[X]" if x is None or pd.isna(x) else f"{100 * float(x):+.0f}"


def main() -> None:
    from lejudge.constraints import load_library
    from lejudge.eval.report import annotate_satisfiable, paired_tests, planning_table
    from lejudge.judge import get_cache

    vals: dict[str, str] = {}
    cfg = yaml.safe_load((ROOT / "configs" / "pusht.yaml").read_text())
    vals["ckpt_rev"] = cfg["checkpoint"]["revision"][:12]
    vals["cem_iters"] = str(cfg["solver"]["n_steps"])
    vals["cem_samples"] = str(cfg["solver"]["num_samples"])
    lib = load_library()
    vals["library_sha"] = lib.sha256[:12]
    vals["n_constraints"] = str(len(lib.constraints))
    prereg = (ROOT / "docs" / "PREREG.md").read_text()
    vals["prereg_words"] = str(len(prereg.split()))
    summary = json.loads((FIG / "summary.json").read_text()) if (FIG / "summary.json").exists() else {}

    # ---- probes -------------------------------------------------------------------------
    pm = summary.get("probes")
    if pm:
        t = pm["test"]
        vals["probe_n_test"] = str(t["n"])
        vals["probe_r2_block"] = num(t["r2"]["block_xy"], 3)
        vals["probe_r2_agent"] = num(t["r2"]["agent_xy"], 3)
        vals["probe_r2_angle"] = num(t["r2"]["angle_sincos"], 3)
        vals["probe_bucket_block"] = num(t["bucket_accuracy"]["block"], 2)
        vals["probe_bucket_agent"] = num(t["bucket_accuracy"]["agent"], 2)
        vals["probe_bucket_angle"] = num(t["bucket_accuracy"]["block_angle"], 2)
        vals["probe_bucket_edge"] = num(t["bucket_accuracy"]["block_edge"], 2)
        vals["probe_bucket_contact"] = num(t["bucket_accuracy"]["contact"], 2)
        vals["probe_contact_auroc"] = num(t["contact"]["auroc"], 3)
        vals["probe_block_err_px"] = num(t["position_error_px"]["block"], 1)
        vals["probe_angle_err_deg"] = num(t["angle_error_deg"], 1)
        if pm.get("imagined"):
            c = pm["imagined"]["curve"]
            vals["probe_bucket_block_h5"] = num(c[-1]["bucket_accuracy"]["block"], 2)
            vals["probe_bucket_edge_h5"] = num(c[-1]["bucket_accuracy"]["block_edge"], 2)
            vals["probe_bucket_contact_h5"] = num(c[-1]["bucket_accuracy"]["contact"], 2)
            vals["probe_block_err_h5_px"] = num(c[-1]["block_error_px"], 1)
            vals["probe_imagined_n"] = str(pm["imagined"]["n_starts"])
    pmeta = json.loads((ROOT / "artifacts/probes/pusht/linear@1/meta.json").read_text())
    vals["probe_train_episodes"] = str(pmeta["dataset"]["episodes"])
    vals["probe_train_frames"] = str(pmeta["dataset"]["frames"])
    mm = json.loads((ROOT / "artifacts/probes/pusht/mlp@1/meta.json").read_text()) if (ROOT / "artifacts/probes/pusht/mlp@1/meta.json").exists() else None
    if mm:
        vals["mlp_r2_angle"] = num(mm["metrics"]["test"]["r2"]["angle_sincos"], 3)
        vals["mlp_bucket_angle"] = num(mm["metrics"]["test"]["bucket_accuracy"]["block_angle"], 2)
        vals["mlp_bucket_block"] = num(mm["metrics"]["test"]["bucket_accuracy"]["block"], 2)

    # ---- M0 --------------------------------------------------------------------------------
    if (RES / "m0.parquet").exists():
        m0 = pd.read_parquet(RES / "m0.parquet")
        for tag, key in (("m0_h3", "m0_success_h3"), ("m0", "m0_success_h1")):
            g = m0[m0.tag == tag]
            if len(g):
                vals[key] = pct(g.success.mean())
                vals[key + "_n"] = str(len(g))
                vals[key + "_p50"] = num(g.plan_time_p50_s.median(), 2)
        vals["m0_weak_success"] = pct(m0[m0.tag == "m0_weakdata_superseded"].success.mean()) if (m0.tag == "m0_weakdata_superseded").any() else "[X]"

    # ---- planning studies ----------------------------------------------------------------
    studies = {"s1": RES / "planning.parquet", "s2": RES / "planning_filtered.parquet"}
    frames: dict[str, pd.DataFrame] = {}
    for key, path in studies.items():
        if not path.exists():
            continue
        p = pd.read_parquet(path)
        frames[key] = p
        t = planning_table(p)
        for _, r in t.iterrows():
            ck = str(r.condition).replace(":", "_").replace(".", "")
            sk = str(r.constraint_set).replace("+", "_")
            vals[f"{key}_{ck}_{sk}_viol"] = ci(r.violation, r.violation_lo, r.violation_hi)
            vals[f"{key}_{ck}_{sk}_viol_pct"] = pct(r.violation)
            vals[f"{key}_{ck}_{sk}_succ"] = ci(r.success, r.success_lo, r.success_hi)
            vals[f"{key}_{ck}_{sk}_succ_pct"] = pct(r.success)
            vals[f"{key}_{ck}_{sk}_n"] = str(int(r.n))
            vals[f"{key}_{ck}_{sk}_p50"] = num(r.plan_p50_s, 2)
            vals[f"{key}_{ck}_{sk}_calls"] = num(r.judge_calls_per_episode, 1)
            vals[f"{key}_{ck}_{sk}_held"] = pct(r.abstention_rate)
        for cond, g in p.groupby("condition"):
            ck = str(cond).replace(":", "_").replace(".", "")
            vals[f"{key}_{ck}_all_viol_pct"] = pct(g.violation.mean())
            vals[f"{key}_{ck}_all_succ_pct"] = pct(g.success.mean())
            vals[f"{key}_{ck}_all_held"] = pct(g.abstention_rate.mean())
            vals[f"{key}_{ck}_all_calls"] = num(g.judge_calls.mean(), 1)
            vals[f"{key}_{ck}_all_tokens"] = f"{g.tokens_in.mean():,.0f}"
            vals[f"{key}_{ck}_all_p50"] = num(np.nanmedian(g.plan_time_p50_s), 2)
        tests = pd.concat([paired_tests(p, "lewm", "violation"), paired_tests(p, "lewm", "success")])
        for _, r in tests.iterrows():
            ck = str(r.condition).replace(":", "_").replace(".", "")
            sk = str(r.constraint_set).replace("+", "_")
            vals[f"{key}_{ck}_{sk}_{r.metric}_rd"] = pts(r.risk_difference)
            vals[f"{key}_{ck}_{sk}_{r.metric}_rdci"] = f"[{100 * r.rd_lo:+.0f}, {100 * r.rd_hi:+.0f}]"
            vals[f"{key}_{ck}_{sk}_{r.metric}_p"] = num(r.p, 3)
            vals[f"{key}_{ck}_{sk}_{r.metric}_pholm"] = num(r.p_holm, 3)
        vals[f"{key}_episodes_per_cell"] = str(int(p[p.condition != "llm-small"].groupby(["condition", "constraint_set", "seed"]).size().max()))
        vals[f"{key}_seeds"] = str(p.seed.nunique())
        vals[f"{key}_n_significant"] = str(int((tests[tests.metric == "violation"].p_holm < 0.05).sum()))
        write_planning_table(t, FIG / f"table_planning_{key}.tex", p)
    if "s1" in frames:
        sat = annotate_satisfiable(frames["s1"])
        sp = sat[(sat.constraint_set == "spatial") & (sat.condition == "lewm")]
        vals["s1_spatial_unsat_pct"] = pct(1.0 - sp.satisfiable.mean()) if len(sp) else "[X]"
        for cset in ("spatial", "spatial+temporal", "implicit"):
            g = sat[(sat.constraint_set == cset) & (sat.condition == "lewm")]
            vals[f"s1_{cset.replace('+', '_')}_sat_pct"] = pct(g.satisfiable.mean()) if len(g) else "[X]"
        for cond in ("lewm", "jev", "oracle", "keyword"):
            g = sat[(sat.condition == cond) & sat.satisfiable & (sat.constraint_set == "spatial")]
            vals[f"s1_{cond}_spatial_sat_viol_pct"] = pct(g.violation.mean()) if len(g) else "[X]"
        vals["s1_spatial_sat_n"] = str(int(sat[(sat.condition == "lewm") & (sat.constraint_set == "spatial")].satisfiable.sum()))
    # ---- Study 3 ---------------------------------------------------------------------------
    if (RES / "planning_gate.parquet").exists() and "s2" in frames:
        g3 = pd.read_parquet(RES / "planning_gate.parquet")
        g3["condition"] = "jev:tau" + g3.tau.astype(str)
        sets = sorted(g3.constraint_set.unique())
        base = frames["s2"][frames["s2"].constraint_set.isin(sets)].copy()
        base["condition"] = base.condition.where(base.condition != "jev", "jev:tau0.5")
        both = pd.concat([base, g3], ignore_index=True)
        t3 = planning_table(both)
        for _, r in t3.iterrows():
            ck = str(r.condition).replace(":", "_").replace(".", "")
            sk = str(r.constraint_set).replace("+", "_")
            vals[f"s3_{ck}_{sk}_viol"] = ci(r.violation, r.violation_lo, r.violation_hi)
            vals[f"s3_{ck}_{sk}_viol_pct"] = pct(r.violation)
            vals[f"s3_{ck}_{sk}_succ_pct"] = pct(r.success)
            vals[f"s3_{ck}_{sk}_held"] = pct(r.abstention_rate)
        for cond, g in both.groupby("condition"):
            ck = str(cond).replace(":", "_").replace(".", "")
            vals[f"s3_{ck}_all_succ_pct"] = pct(g.success.mean())
            vals[f"s3_{ck}_all_viol_pct"] = pct(g.violation.mean())
        for ref, rk in (("lewm", "vslewm"), ("jev:tau0.5", "vsgate")):
            tt = paired_tests(both, ref, "violation")
            for _, r in tt.iterrows():
                ck = str(r.condition).replace(":", "_").replace(".", "")
                sk = str(r.constraint_set).replace("+", "_")
                vals[f"s3_{ck}_{sk}_{rk}_rd"] = pts(r.risk_difference)
                vals[f"s3_{ck}_{sk}_{rk}_rdci"] = f"[{100 * r.rd_lo:+.0f}, {100 * r.rd_hi:+.0f}]"
                vals[f"s3_{ck}_{sk}_{rk}_pholm"] = num(r.p_holm, 3)
        write_planning_table(t3, FIG / "table_planning_s3.tex", both, conds=["lewm", "keyword", "oracle", "jev:tau0.5", "jev:tau0.75", "jev:tau1.0"])
    # ---- ablations -------------------------------------------------------------------------
    if (RES / "ablations.parquet").exists():
        a = pd.read_parquet(RES / "ablations.parquet")
        ta = a.groupby("tag").agg(n=("success", "size"), success=("success", "mean"), violation=("violation", "mean"), calls=("judge_calls", "mean"), held=("abstention_rate", "mean")).reset_index()
        vals["abl_viol_min"] = pct(ta.violation.min())
        vals["abl_viol_max"] = pct(ta.violation.max())
        vals["abl_succ_min"] = pct(ta.success.min())
        vals["abl_succ_max"] = pct(ta.success.max())
        vals["abl_n_runs"] = str(len(ta))
        vals["abl_episodes"] = str(int(ta.n.max()))
        write_ablation_table(ta, FIG / "table_ablations.tex")
    if (RES / "ablations_filtered.parquet").exists():
        af = pd.read_parquet(RES / "ablations_filtered.parquet")
        vals["abl_filt_viol_min"] = pct(af.groupby("lam").violation.mean().min())
        vals["abl_filt_viol_max"] = pct(af.groupby("lam").violation.mean().max())
    # ---- diagnostics -----------------------------------------------------------------------
    dpath = FIG / "paper" / "table_imagined_vs_executed.csv"
    if dpath.exists():
        d = pd.read_csv(dpath)
        for _, r in d[d.table == "planning_filtered"].iterrows():
            ck, sk = str(r.condition), str(r.constraint_set).replace("+", "_")
            vals[f"diag_{ck}_{sk}_imag"] = pct(r.imagined_violation)
            vals[f"diag_{ck}_{sk}_exec"] = pct(r.executed_violation)
            vals[f"diag_{ck}_{sk}_exec_given_clean"] = pct(r.executed_given_imagined_clean)
            vals[f"diag_{ck}_{sk}_exec_given_viol"] = pct(r.executed_given_imagined_violating)
        write_diag_table(d[d.table.isin(["planning_filtered"])], FIG / "table_imagined_vs_executed.tex")
    # ---- judge-only ------------------------------------------------------------------------
    if (RES / "judge_only.parquet").exists():
        j = pd.read_parquet(RES / "judge_only.parquet")
        from lejudge.eval.judge_study import consistency_table, metrics_table, per_constraint_table
        from lejudge.eval.stats import auroc, prf

        t = metrics_table(j)
        for _, r in t.iterrows():
            jk = str(r.judge).split(":")[0]
            key = f"jo_{jk}_{r.source}_{'gt' if r.description == 'gt-words' else 'pw'}"
            vals[key + "_n"] = str(int(r.n_items))
            vals[key + "_acc"] = num(r.canonical_accuracy, 2)
            vals[key + "_f1"] = num(r.canonical_f1, 2)
            vals[key + "_auroc"] = num(r.canonical_auroc, 2)
            vals[key + "_ece"] = num(r.canonical_ece, 3)
            vals[key + "_para_acc"] = num(r.paraphrase_accuracy, 2)
            vals[key + "_para_auroc"] = num(r.paraphrase_auroc, 2)
            vals[key + "_fpr"] = num(r.negative_fpr, 2)
            vals[key + "_lat"] = num(r.latency_ms_per_1000 / 1000.0, 1)
        vals["jo_items"] = str(j.item_id.nunique())
        vals["jo_rows"] = f"{len(j):,}"
        vals["jo_n_imagined"] = str(j[j.source == "imagined"].item_id.nunique())
        vals["jo_n_executed"] = str(j[j.source == "executed"].item_id.nunique())
        ct = consistency_table(j)
        if len(ct[ct.judge == "jev"]):
            vals["jo_jev_std_mean"] = num(ct[ct.judge == "jev"].score_std_mean.iloc[0], 3)
            vals["jo_jev_std_max"] = num(ct[ct.judge == "jev"].score_std_max.iloc[0], 3)
            vals["jo_repeat_pairs"] = str(int(ct[ct.judge == "jev"].n_pairs.iloc[0]))
        pc = per_constraint_table(j[j.description == "probe-words"])
        for jk in ("jev", "keyword"):
            g = pc[(pc.judge == jk) & (pc.variant == "canonical")].set_index("constraint").accuracy
            for cid, v in g.items():
                vals[f"pc_{jk}_{cid}"] = num(v, 2)
        # pooled Jev numbers over probe-word items
        g = j[(j.judge == "jev") & (j.description == "probe-words") & (j.repeat == 0)]
        canon, para = g[g.variant == "canonical"], g[g.variant.str.startswith("p")]
        vals["jo_jev_pw_acc_pooled"] = num(prf(canon.label.to_numpy(), canon.score.to_numpy())["accuracy"], 2)
        vals["jo_jev_pw_para_pooled"] = num(prf(para.label.to_numpy(), para.score.to_numpy())["accuracy"], 2)
        vals["jo_jev_pw_auroc_pooled"] = num(auroc(canon.label.to_numpy(), canon.score.to_numpy()), 2)
        g = j[(j.judge == "keyword") & (j.description == "probe-words") & (j.repeat == 0)]
        canon, para = g[g.variant == "canonical"], g[g.variant.str.startswith("p")]
        vals["jo_kw_pw_acc_pooled"] = num(prf(canon.label.to_numpy(), canon.score.to_numpy())["accuracy"], 2)
        vals["jo_kw_pw_para_pooled"] = num(prf(para.label.to_numpy(), para.score.to_numpy())["accuracy"], 2)
        lj = j[j.judge.astype(str).str.startswith("llm")]
        if len(lj):
            ok = lj[~lj.failed]
            vals["llm_items"] = str(lj.item_id.nunique())
            vals["llm_failed_pct"] = pct(lj.failed.mean())
            vals["llm_valid_acc"] = num(prf(ok.label.to_numpy(), ok.score.to_numpy())["accuracy"], 2) if len(ok) else "[X]"
            vals["llm_valid_auroc"] = num(auroc(ok.label.to_numpy(), ok.score.to_numpy()), 2) if len(ok) else "[X]"
            vals["llm_lat"] = num(lj.latency_ms_per_q.mean() / 1000.0 * 1000.0, 0)  # s per 1,000 = ms per judgment
            models = lj.response_model.replace("", pd.NA).dropna().unique()
            vals["llm_model"] = str(models[0]) if len(models) else "qwen2.5:7b-instruct"
        write_judge_table(t, FIG / "table_judge_only.tex")
    write_library_table(lib, FIG / "table_library.tex")
    write_bank_table(FIG / "table_bank.tex")
    c = get_cache()
    vals["cache_n"] = f"{c.count():,}"
    vals["cache_steps"] = f"{c.count_steps():,}"
    tex = (ROOT / "paper" / "main.tex").read_text()
    missing: list[str] = []

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


# ----------------------------------------------------------------------------- tables
def _esc(s: str) -> str:
    return str(s).replace("_", "\\_").replace("+", "+")


def write_planning_table(t: pd.DataFrame, out: Path, raw: pd.DataFrame, conds: list[str] | None = None) -> None:
    from lejudge.eval.report import paired_tests

    conds = conds or ["lewm", "keyword", "oracle", "jev"]
    pt = paired_tests(raw, "lewm", "violation")
    stars = {(r.condition, r.constraint_set) for _, r in pt.iterrows() if r.p_holm < 0.05}
    sets = [s for s in ("spatial", "spatial+temporal", "implicit") if s in set(t.constraint_set)]
    lines = ["\\begin{tabular}{l" + "cc" * len(sets) + "}", "\\toprule"]
    lines.append("& " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{_esc(SET_TEX[s])}}}" for s in sets) + " \\\\")
    lines.append(" ".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(sets))))
    lines.append("condition & " + " & ".join("violation $\\downarrow$ & success $\\uparrow$" for _ in sets) + " \\\\")
    lines.append("\\midrule")
    for c in conds:
        if c not in set(t.condition):
            continue
        cells = []
        for s in sets:
            r = t[(t.condition == c) & (t.constraint_set == s)]
            if len(r):
                r = r.iloc[0]
                star = "$^{\\star}$" if (c, s) in stars else ""
                cells += [f"{r.violation:.2f}{star} {{\\scriptsize[{r.violation_lo:.2f}, {r.violation_hi:.2f}]}}", f"{r.success:.2f} {{\\scriptsize[{r.success_lo:.2f}, {r.success_hi:.2f}]}}"]
            else:
                cells += ["--", "--"]
        lines.append(f"{COND_TEX.get(c, _esc(c))} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    out.write_text("\n".join(lines))


def write_ablation_table(ta: pd.DataFrame, out: Path) -> None:
    names = {"lam_sweep_0.25": "$\\lambda=0.25$", "lam_sweep_0.5": "$\\lambda=0.5$", "lam_sweep_2": "$\\lambda=2$", "lam_sweep_4": "$\\lambda=4$", "K4": "$K=4$ (soft shortlist)", "K32": "$K=32$", "K64": "$K=64$", "K32_lam4": "$K=32$, $\\lambda=4$", "final_only": "judge final iteration only", "last_n3": "judge last 3 iterations", "per_iter": "judge every iteration", "unjudged_none": "no prior for unjudged (RFC draft)", "vocab_pusht2": "vocabulary $4\\times4$ (pusht@2)", "steps_half": "judge first 2 of 5 steps", "hard_reject": "hard rejection ($+\\infty$ if $p>0.7$)", "mlp_probe": "MLP probe"}
    order = ["lam_sweep_0.25", "lam_sweep_0.5", "lam_sweep_2", "lam_sweep_4", "K4", "K32", "K64", "K32_lam4", "final_only", "last_n3", "per_iter", "unjudged_none", "vocab_pusht2", "steps_half", "hard_reject", "mlp_probe"]
    lines = ["\\begin{tabular}{lccccc}", "\\toprule", "variant & $n$ & violation & success & Jev calls / ep & held \\\\", "\\midrule"]
    for tag in order:
        r = ta[ta.tag == tag]
        if len(r):
            r = r.iloc[0]
            lines.append(f"{names.get(tag, _esc(tag))} & {int(r.n)} & {r.violation:.2f} & {r.success:.2f} & {r.calls:.1f} & {100 * r.held:.0f}\\% \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    out.write_text("\n".join(lines))


def write_diag_table(d: pd.DataFrame, out: Path) -> None:
    lines = ["\\begin{tabular}{llcccc}", "\\toprule", "set & judge & imagined & executed & executed $\\mid$ imagined clean & executed $\\mid$ imagined violating \\\\", "\\midrule"]
    for s in ("spatial", "spatial+temporal", "implicit"):
        for c in ("oracle", "keyword", "jev"):
            r = d[(d.constraint_set == s) & (d.condition == c)]
            if len(r):
                r = r.iloc[0]
                lines.append(f"{_esc(SET_TEX[s])} & {COND_TEX[c].split(' (')[0]} & {r.imagined_violation:.2f} & {r.executed_violation:.2f} & {r.executed_given_imagined_clean:.2f} & {r.executed_given_imagined_violating:.2f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    out.write_text("\n".join(lines))


def write_judge_table(t: pd.DataFrame, out: Path) -> None:
    rows = [("jev", "imagined", "probe-words"), ("jev", "executed", "probe-words"), ("jev", "executed", "gt-words"), ("keyword", "imagined", "probe-words"), ("keyword", "executed", "probe-words"), ("llm", "executed", "probe-words"), ("oracle", "imagined", "probe-words"), ("oracle", "executed", "gt-words")]
    names = {"jev": "LeJudge / Jev", "keyword": "keyword checker", "llm": "local LLM (7B)", "oracle": "oracle-on-probes"}
    lines = ["\\begin{tabular}{llcccccccc}", "\\toprule", "judge & items & $n$ & acc. & F1 & AUROC & ECE & acc. (paraphr.) & near-miss FPR & s / 1{,}000 \\\\", "\\midrule"]
    for j, src, desc in rows:
        r = t[(t.judge == j) & (t.source == src) & (t.description == desc)]
        if not len(r):
            continue
        r = r.iloc[0]

        def f(v, d=2):
            return "--" if pd.isna(v) else f"{v:.{d}f}"

        items = f"{src}, {'GT words' if desc == 'gt-words' else 'probe words'}"
        lines.append(f"{names[j]} & {items} & {int(r.n_items)} & {f(r.canonical_accuracy)} & {f(r.canonical_f1)} & {f(r.canonical_auroc)} & {f(r.canonical_ece, 3)} & {f(r.paraphrase_accuracy)} & {f(r.negative_fpr)} & {f(r.latency_ms_per_1000 / 1000.0, 1)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    out.write_text("\n".join(lines))



def write_library_table(lib, out: Path) -> None:
    lines = ["\\begin{tabular}{p{3.1cm}lp{4.9cm}p{5.6cm}}", "\\toprule", "id & family & canonical text & one paraphrase / one near-miss negative \\\\", "\\midrule"]
    for c in lib.constraints:
        lines.append(f"\\texttt{{{_esc(c.id)}}} & {c.family} & {c.text} & \\emph{{{c.paraphrases[0]}}} / {c.negatives[0]} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    out.write_text("\n".join(lines))


def write_bank_table(out: Path) -> None:
    from lejudge.judge.bank import BANK_VERSION, always_question, never_question, soft_question

    qs = [("never", never_question("k1", "c1", 2)), ("always", always_question("k1", "c1", 2)), ("soft", soft_question("k1", "c1"))]
    lines = [f"\\small Bank \\texttt{{{_esc(BANK_VERSION)}}}. Rendered for candidate \\texttt{{k1}}, constraint \\texttt{{c1}}, step $t=2$.\\par\\medskip", "\\footnotesize\\begin{tabular}{p{1.4cm}p{4.8cm}p{8.6cm}}", "\\toprule", "family & instructions & criteria \\\\", "\\midrule"]
    for fam, q in qs:
        crit = q.criteria
        if isinstance(crit, dict):
            ctext = f"\\textbf{{true:}} {crit['true']} \\textbf{{false:}} {crit['false']}"
        else:
            ctext = " ".join(f"\\textbf{{{i}:}} {c}" for i, c in enumerate(crit))
        lines.append(f"{fam} ({'Noul' if fam != 'soft' else 'Score'}) & {q.instructions} & {ctext} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    out.write_text("\n".join(lines).replace("_", "\\_"))


if __name__ == "__main__":
    main()
