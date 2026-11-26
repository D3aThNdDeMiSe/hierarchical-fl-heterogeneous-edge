"""Turn results/runs/*.json into the paper's figures, tables and key numbers.

Outputs
  paper/figures/fig_rq{1..4}_*.pdf   figures (PDF, IEEE column width)
  paper/tables/tab_main.tex          main-configuration table (tabular only)
  paper/tables/tab_scaling.tex       scaling table (tabular only)
  results/summary.txt                every configuration, mean +- std over seeds
  results/stats.json                 numbers quoted in the text (targets, t-tests, ratios)

Works on partial results: missing configurations are skipped.
"""
from __future__ import annotations

import glob
import json
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(ROOT, "results", "runs")
FIG = os.path.join(ROOT, "paper", "figures")
TAB = os.path.join(ROOT, "paper", "tables")
OUT = os.path.join(ROOT, "results")

ORDER = ["centralised", "local", "flat_fedavg", "hier_fedavg", "hier_fedprox", "hier_fedprox_dl", "hier_proposed"]
LABEL = {"centralised": "Centralised", "local": "Local-only", "flat_fedavg": "Flat FedAvg",
         "hier_fedavg": "H-FedAvg", "hier_fedprox": "H-FedProx", "hier_fedprox_dl": "H-FedProx+DL",
         "hier_proposed": "Proposed"}
# Categorical slots in fixed order (validated: scripts/validate_palette.js, light mode).
COLOR = {"hier_fedavg": "#2a78d6", "flat_fedavg": "#eb6834", "hier_fedprox": "#1baf7a",
         "hier_fedprox_dl": "#eda100", "hier_proposed": "#e87ba4",
         "centralised": "#52514e", "local": "#8f8e89"}
MARKER = {"hier_fedavg": "o", "flat_fedavg": "s", "hier_fedprox": "^", "hier_fedprox_dl": "D",
          "hier_proposed": "*", "centralised": "", "local": "v"}
ALPHAS = ["0.1", "0.5", "1"]
TARGET_FRAC = 0.9  # time-to-target: 90% of the centralised final accuracy (proposal, Section 8)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5, "legend.fontsize": 6.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.linewidth": 0.6,
    "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e",
    "ytick.color": "#52514e", "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.4,
    "lines.markersize": 3.5, "legend.frameon": False, "pdf.fonttype": 42, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})
COL_W = 3.45  # inches, IEEE single column


# ---------------------------------------------------------------- loading
def load():
    rows, hist = [], {}
    for p in sorted(glob.glob(os.path.join(RUNS, "*.json"))):
        with open(p) as f:
            r = json.load(f)
        c, fin, s = r["config"], r["final"], r["setup"]
        rows.append(dict(
            name=r["name"], method=c["method"], alpha="iid" if c["alpha"] is None else f"{c['alpha']:g}",
            E=c["gateways"], K=c["clients_per_gateway"], N=c["gateways"] * c["clients_per_gateway"],
            tiers=c["tiers"], seed=c["seed"], rounds=c["rounds"], acc=fin["acc"], p10=fin["client_acc_p10"],
            cmean=fin["client_acc_mean"], sim_time=fin["sim_time"], bytes_server=fin["bytes_server"],
            bytes_edge=fin["bytes_edge"], dropped=fin["dropped"], updates=fin["updates"], steps=fin["steps"],
            wasted=fin["wasted_steps"], edge_rt=s.get("edge_round_time_mean_s"), agg=s.get("gw_agg_time_mean_s"),
            deadline=s.get("deadline_s"), wall=r["wall_time_s"]))
        hist[r["name"]] = r["history"]
    return pd.DataFrame(rows), hist


def sel(df, **kw):
    m = pd.Series(True, index=df.index)
    for k, v in kw.items():
        m &= df[k].isin(v) if isinstance(v, (list, tuple, set)) else (df[k] == v)
    return df[m]


def time_to_target(h, target):
    for e in h:
        if e["acc"] >= target:
            return e["sim_time"], e["round"]
    return np.nan, np.nan


def mean_curve(hist, names, x="sim_time"):
    """Average acc and x across seeds, aligned on evaluation index."""
    hs = [hist[n] for n in names if n in hist]
    if not hs:
        return None, None
    L = min(len(h) for h in hs)
    xs = np.mean([[e[x] for e in h[:L]] for h in hs], axis=0)
    ys = np.mean([[e["acc"] for e in h[:L]] for h in hs], axis=0)
    return xs, ys


def pm(series, scale=100, digits=1):
    v = series.dropna()
    if len(v) == 0:
        return "--"
    if len(v) == 1:
        return f"{v.iloc[0] * scale:.{digits}f}"
    return f"{v.mean() * scale:.{digits}f}$\\pm${v.std(ddof=1) * scale:.{digits}f}"


# ---------------------------------------------------------------- analysis
def main():
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(TAB, exist_ok=True)
    df, hist = load()
    if df.empty:
        print("no results yet")
        return
    main_cfg = sel(df, E=4, K=5, tiers="mixed")

    # targets per alpha from centralised runs
    targets = {}
    for a in ALPHAS + ["iid"]:
        c = sel(main_cfg, method="centralised", alpha=a)
        if len(c):
            targets[a] = TARGET_FRAC * c.acc.mean()
    if targets:
        fallback = float(np.mean(list(targets.values())))
        for a in ALPHAS + ["iid"]:
            targets.setdefault(a, fallback)
    df["t_target"], df["r_target"] = np.nan, np.nan
    for i, row in df.iterrows():
        tgt = targets.get(row.alpha)
        if tgt is not None and row.method not in ("centralised", "local"):
            df.loc[i, ["t_target", "r_target"]] = time_to_target(hist[row["name"]], tgt)
    main_cfg = sel(df, E=4, K=5, tiers="mixed")

    stats_out = {"targets": targets, "n_runs": int(len(df)), "target_frac": TARGET_FRAC}
    write_summary(df)
    table_main(main_cfg)
    table_scaling(df)
    fig_rq1(df, main_cfg)
    fig_rq2(main_cfg, hist)
    fig_rq3(main_cfg, hist)
    fig_rq4(df)
    stats_out.update(key_numbers(df, main_cfg))
    with open(os.path.join(OUT, "stats.json"), "w") as f:
        json.dump(stats_out, f, indent=1, default=float)
    print(f"{len(df)} runs analysed; figures in paper/figures, tables in paper/tables")


def write_summary(df):
    g = df.groupby(["method", "alpha", "E", "K", "tiers"])
    agg = g.agg(n=("seed", "count"), acc=("acc", "mean"), acc_sd=("acc", "std"), p10=("p10", "mean"),
                p10_sd=("p10", "std"), t_target=("t_target", "mean"), sim_time=("sim_time", "mean"),
                server_MB=("bytes_server", lambda v: v.mean() / 1e6), dropped=("dropped", "mean"),
                updates=("updates", "mean"), wasted=("wasted", "mean"), steps=("steps", "mean"),
                edge_rt=("edge_rt", "mean"), agg_ms=("agg", lambda v: v.mean() * 1e3), wall=("wall", "mean"))
    with open(os.path.join(OUT, "summary.txt"), "w") as f:
        with pd.option_context("display.width", 250, "display.max_rows", 500, "display.float_format", "{:.4f}".format):
            f.write(agg.reset_index().to_string(index=False))


def table_main(m):
    lines = [r"\begin{tabular}{l cc cc cc ccc}", r"\toprule",
             r" & \multicolumn{2}{c}{$\alpha = 0.1$} & \multicolumn{2}{c}{$\alpha = 0.5$} & "
             r"\multicolumn{2}{c}{$\alpha = 1.0$} & \multicolumn{3}{c}{$\alpha = 0.5$} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-10}",
             r"Method & Acc & P10 & Acc & P10 & Acc & P10 & $T_{\mathrm{tgt}}$ (s) & Server (MB) & Dropped (\%) \\",
             r"\midrule"]
    fl = [x for x in ORDER if x not in ("centralised", "local")]
    best = {}
    for a in ALPHAS:
        for col in ("acc", "p10"):
            means = {meth: sel(m, method=meth, alpha=a)[col].mean() for meth in fl}
            means = {k: v for k, v in means.items() if not np.isnan(v)}
            if means:
                best[(a, col)] = max(means, key=means.get)
    for meth in ORDER:
        cells = [LABEL[meth]]
        for a in ALPHAS:
            r = sel(m, method=meth, alpha=a)
            for col in ("acc", "p10"):
                s = pm(r[col])
                cells.append(f"\\textbf{{{s}}}" if best.get((a, col)) == meth and s != "--" else s)
        r = sel(m, method=meth, alpha="0.5")
        if meth in ("centralised", "local") or r.empty:
            cells += ["--", "--", "--"]
        else:
            tt = r.t_target
            cells.append(f"{tt.mean():.0f}" if tt.notna().all() else f"n.r.\\,({tt.notna().sum()}/{len(tt)})")
            cells.append(f"{r.bytes_server.mean() / 1e6:.1f}")
            cells.append(f"{100 * (r.dropped / r.updates).mean():.1f}")
        lines.append(" & ".join(cells) + r" \\")
        if meth == "local":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(TAB, "tab_main.tex"), "w") as f:
        f.write("% generated by analyze.py -- do not edit\n" + "\n".join(lines) + "\n")


SCALING = [(4, 5), (4, 10), (4, 25), (2, 5), (10, 5)]


def table_scaling(df):
    d = sel(df, alpha="0.5", tiers="mixed", seed=[0, 1])
    lines = [r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{lr ccc cc r}", r"\toprule",
             r" & & \multicolumn{3}{c}{Accuracy (\%)} & \multicolumn{2}{c}{$\bar T_r$ (s)} & \\",
             r"\cmidrule(lr){3-5}\cmidrule(lr){6-7}",
             r"$E{\times}K$ & $N$ & Flat & H-Avg & Prop. & H-Avg & Prop. & $A_e$ (ms) \\", r"\midrule"]
    for E, K in SCALING:
        r = sel(d, E=E, K=K)
        if r.empty:
            continue
        cells = [f"{E}$\\times${K}", str(E * K)]
        for meth in ("flat_fedavg", "hier_fedavg", "hier_proposed"):
            v = sel(r, method=meth).acc
            cells.append(f"{100 * v.mean():.1f}" if len(v) else "--")
        for meth in ("hier_fedavg", "hier_proposed"):
            v = sel(r, method=meth)
            cells.append(f"{(v.sim_time / v.rounds).mean():.2f}" if len(v) else "--")
        v = sel(r, method="hier_proposed").agg
        cells.append(f"{1e3 * v.mean():.0f}" if len(v) and v.notna().any() else "--")
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(TAB, "tab_scaling.tex"), "w") as f:
        f.write("% generated by analyze.py -- do not edit\n" + "\n".join(lines) + "\n")


def _legend_below(fig, axes, ncol):
    handles, labels = [], []
    for ax in np.ravel(axes):
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, loc="lower center", ncol=ncol, bbox_to_anchor=(0.5, -0.02),
               handlelength=1.6, columnspacing=1.0)


def fig_rq1(df, m):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL_W, 1.95))
    cats = ALPHAS + ["iid"]
    x = np.arange(len(cats))
    for meth in ("flat_fedavg", "hier_fedavg", "hier_proposed", "local"):
        mu = [sel(m, method=meth, alpha=a).acc.mean() for a in cats]
        sd = [sel(m, method=meth, alpha=a).acc.std() for a in cats]
        if np.all(np.isnan(mu)):
            continue
        a1.errorbar(x, mu, yerr=np.nan_to_num(sd), color=COLOR[meth], marker=MARKER[meth], capsize=1.5,
                    elinewidth=0.6, label=LABEL[meth], linestyle=":" if meth == "local" else "-")
    c = sel(m, method="centralised").acc
    if len(c):
        a1.axhline(c.mean(), color=COLOR["centralised"], linestyle="--", linewidth=1.0, label="Centralised")
    a1.set_xticks(x, ["0.1", "0.5", "1.0", "IID"])
    a1.set_xlabel(r"label skew $\alpha$ (smaller = more skewed)")
    a1.set_ylabel("final test accuracy")
    a1.set_title("(a) statistical heterogeneity", loc="left")

    tiers = ["mid", "mixed", "low_heavy"]
    meths = ["hier_fedavg", "hier_fedprox_dl", "hier_proposed"]
    d = sel(df, alpha="0.5", E=4, K=5)
    w = 0.26
    for i, meth in enumerate(meths):
        vals = [sel(d, method=meth, tiers=t).t_target.mean() for t in tiers]
        pos = np.arange(len(tiers)) + (i - 1) * w
        a2.bar(pos, np.nan_to_num(vals), width=w - 0.04, color=COLOR[meth], label=LABEL[meth])
        for p, v in zip(pos, vals):
            if np.isnan(v):
                a2.text(p, 0, "n.r.", ha="center", va="bottom", fontsize=5.5, color="#52514e")
    a2.set_xticks(np.arange(len(tiers)), ["all mid", "mixed", "low-heavy"])
    a2.set_xlabel(r"device-tier mix ($\alpha = 0.5$)")
    a2.set_ylabel("simulated time to target (s)")
    a2.set_title("(b) system heterogeneity", loc="left")
    a2.grid(axis="x", visible=False)
    fig.tight_layout(w_pad=1.2, rect=(0, 0.13, 1, 1))
    _legend_below(fig, (a1, a2), ncol=3)
    fig.savefig(os.path.join(FIG, "fig_rq1_heterogeneity.pdf"))
    plt.close(fig)


def fig_rq2(m, hist):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL_W, 1.8), sharey=True)
    for meth in ("flat_fedavg", "hier_fedavg"):
        names = sel(m, method=meth, alpha="0.5").name.tolist()
        xb, yb = mean_curve(hist, names, "bytes_server")
        xt, yt = mean_curve(hist, names, "sim_time")
        if xb is None:
            continue
        a1.plot(xb / 1e6, yb, color=COLOR[meth], label=LABEL[meth])
        a2.plot(xt, yt, color=COLOR[meth], label=LABEL[meth])
    a1.set_xscale("log")
    a1.set_xlabel("cumulative server traffic (MB, log)")
    a1.set_ylabel(r"test accuracy ($\alpha = 0.5$)")
    a1.set_title("(a) communication", loc="left")
    a2.set_xlabel("simulated time (s)")
    a2.set_title("(b) time", loc="left")
    for a in (a1, a2):
        a.set_ylim(bottom=0.6)
    fig.tight_layout(w_pad=1.0, rect=(0, 0.1, 1, 1))
    _legend_below(fig, (a1, a2), ncol=2)
    fig.savefig(os.path.join(FIG, "fig_rq2_comm.pdf"))
    plt.close(fig)


def fig_rq3(m, hist):
    meths = ["hier_fedavg", "hier_fedprox", "hier_fedprox_dl", "hier_proposed"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL_W, 1.95))
    for meth in meths:
        names = sel(m, method=meth, alpha="0.1").name.tolist()
        xt, yt = mean_curve(hist, names, "sim_time")
        if xt is not None:
            a1.plot(xt, yt, color=COLOR[meth], label=LABEL[meth], marker=MARKER[meth], markevery=5)
    a1.set_xlabel("simulated time (s)")
    a1.set_ylabel(r"test accuracy ($\alpha = 0.1$)")
    a1.set_title("(a) convergence, strong skew", loc="left")
    a1.set_ylim(bottom=0.5)
    w = 0.2
    for i, meth in enumerate(meths):
        vals = [sel(m, method=meth, alpha=a).p10.mean() for a in ALPHAS]
        sds = [sel(m, method=meth, alpha=a).p10.std() for a in ALPHAS]
        pos = np.arange(len(ALPHAS)) + (i - 1.5) * w
        a2.bar(pos, np.nan_to_num(vals), width=w - 0.03, color=COLOR[meth], label=LABEL[meth],
               yerr=np.nan_to_num(sds), error_kw=dict(elinewidth=0.6, capsize=1.2, ecolor="#52514e"))
    a2.set_xticks(np.arange(len(ALPHAS)), ["0.1", "0.5", "1.0"])
    a2.set_xlabel(r"label skew $\alpha$")
    a2.set_ylabel("10th-pct client accuracy")
    a2.set_title("(b) worst-served clients", loc="left")
    a2.grid(axis="x", visible=False)
    lo = np.nanmin([sel(m, method=x).p10.min() for x in meths] + [1.0])
    a2.set_ylim(bottom=max(0.0, np.floor((lo - 0.05) * 10) / 10))
    fig.tight_layout(w_pad=1.2, rect=(0, 0.13, 1, 1))
    _legend_below(fig, (a1, a2), ncol=4)
    fig.savefig(os.path.join(FIG, "fig_rq3_ablation.pdf"))
    plt.close(fig)


def fig_rq4(df):
    d = sel(df, alpha="0.5", tiers="mixed", seed=[0, 1])
    meths = ["flat_fedavg", "hier_fedavg", "hier_proposed"]
    fig, axes = plt.subplots(2, 2, figsize=(COL_W, 2.9))
    sweeps = [("K", [5, 10, 25], dict(E=4), r"devices per gateway $K$ ($E = 4$)"),
              ("E", [2, 4, 10], dict(K=5), r"gateways $E$ ($K = 5$)")]
    for col, (var, vals, fixed, xlabel) in enumerate(sweeps):
        aa, at = axes[0, col], axes[1, col]
        for meth in meths:
            acc, tr = [], []
            for v in vals:
                r = sel(d, method=meth, **{var: v}, **fixed)
                acc.append(r.acc.mean() if len(r) else np.nan)
                tr.append((r.sim_time / r.rounds).mean() if len(r) else np.nan)
            aa.plot(vals, acc, color=COLOR[meth], marker=MARKER[meth], label=LABEL[meth])
            at.plot(vals, tr, color=COLOR[meth], marker=MARKER[meth], label=LABEL[meth])
        for a in (aa, at):
            a.set_xscale("log")
            a.set_xticks(vals, [str(v) for v in vals])
            a.minorticks_off()
        at.set_xlabel(xlabel)
        aa.set_title(f"({'ab'[col]}) accuracy vs {var}", loc="left")
        at.set_title(f"({'cd'[col]}) time per round vs {var}", loc="left")
    axes[0, 0].set_ylabel("final test accuracy")
    axes[1, 0].set_ylabel("sim. time / round (s)")
    fig.tight_layout(h_pad=0.6, w_pad=1.0, rect=(0, 0.07, 1, 1))
    _legend_below(fig, axes, ncol=3)
    fig.savefig(os.path.join(FIG, "fig_rq4_scaling.pdf"))
    plt.close(fig)


def _paired(m, a_meth, b_meth, col):
    a = sel(m, method=a_meth, alpha=ALPHAS).set_index(["alpha", "seed"])[col]
    b = sel(m, method=b_meth, alpha=ALPHAS).set_index(["alpha", "seed"])[col]
    j = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
    if len(j) < 2:
        return None
    t, p = stats.ttest_rel(j.a, j.b)
    return dict(n=int(len(j)), mean_diff=float((j.a - j.b).mean()), t=float(t), p=float(p))


def key_numbers(df, m):
    out = {}
    out["ttest_proposed_vs_hfedavg_acc"] = _paired(m, "hier_proposed", "hier_fedavg", "acc")
    out["ttest_proposed_vs_hfedavg_p10"] = _paired(m, "hier_proposed", "hier_fedavg", "p10")
    out["ttest_proposed_vs_hfedavg_ttarget"] = _paired(m, "hier_proposed", "hier_fedavg", "t_target")
    out["ttest_hfedavg_vs_flat_acc"] = _paired(m, "hier_fedavg", "flat_fedavg", "acc")
    out["ttest_proposed_vs_fedproxdl_p10"] = _paired(m, "hier_proposed", "hier_fedprox_dl", "p10")
    for meth in ORDER:
        for a in ALPHAS + ["iid"]:
            r = sel(m, method=meth, alpha=a)
            if len(r):
                out[f"{meth}_{a}"] = dict(acc=r.acc.mean(), acc_sd=r.acc.std(), p10=r.p10.mean(),
                                          t_target=r.t_target.mean(), sim_time=r.sim_time.mean(),
                                          server_MB=r.bytes_server.mean() / 1e6,
                                          dropped_frac=float((r.dropped / r.updates).mean()) if r.updates.max() else 0.0,
                                          wasted_frac=float((r.wasted / r.steps).mean()) if r.steps.max() else 0.0,
                                          n=int(len(r)))
    f, h = sel(m, method="flat_fedavg", alpha="0.5"), sel(m, method="hier_fedavg", alpha="0.5")
    if len(f) and len(h):
        out["server_traffic_reduction"] = float(f.bytes_server.mean() / h.bytes_server.mean())
    return out


if __name__ == "__main__":
    main()
