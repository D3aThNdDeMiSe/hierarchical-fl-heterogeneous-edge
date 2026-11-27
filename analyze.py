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
    "savefig.pad_inches": 0.02, "axes.axisbelow": True,
})
PREVIEW = os.path.join(OUT, "preview")


def save(fig, name):
    """PDF for the paper, PNG preview for slides and visual checks."""
    os.makedirs(PREVIEW, exist_ok=True)
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    fig.savefig(os.path.join(PREVIEW, name + ".png"), dpi=220)
    plt.close(fig)
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
            profile=c.get("profile", "colext"), dq=c.get("deadline_quantile", 0.5), dataset=c["dataset"],
            E=c["gateways"], K=c["clients_per_gateway"], N=c["gateways"] * c["clients_per_gateway"],
            tiers=c["tiers"], seed=c["seed"], rounds=c["rounds"], acc=fin["acc"], p10=fin["client_acc_p10"],
            cmean=fin["client_acc_mean"], sim_time=fin["sim_time"], bytes_server=fin["bytes_server"],
            bytes_edge=fin["bytes_edge"], dropped=fin["dropped"], updates=fin["updates"], steps=fin["steps"],
            wasted=fin["wasted_steps"], edge_rt=s.get("edge_round_time_mean_s"), agg_s=s.get("gw_agg_time_mean_s"),
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


def acc_at_time(h, budget):
    """Accuracy of the last evaluation completed within `budget` simulated seconds."""
    acc = np.nan
    for e in h:
        if e["sim_time"] > budget:
            break
        acc = e["acc"]
    return acc


def bytes_to_target(h, target):
    for e in h:
        if e["acc"] >= target:
            return e["bytes_server"]
    return np.nan


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
    cifar = sel(df, dataset="cifar10")  # G6, reported in its own table
    df = sel(df, dataset="fmnist")
    measured = sel(df, profile="measured")
    deadlines = sel(df, profile="colext", dq=[0.75, 0.9])  # G5 deadline sensitivity
    df = sel(df, profile="colext", dq=0.5)  # main analysis; G4 and G5 are reported separately
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
    def add_targets(frame):
        frame = frame.copy()
        frame["t_target"], frame["r_target"] = np.nan, np.nan
        for i, row in frame.iterrows():
            tgt = targets.get(row.alpha)
            if tgt is not None and row.method not in ("centralised", "local"):
                frame.loc[i, ["t_target", "r_target"]] = time_to_target(hist[row["name"]], tgt)
        return frame

    df, measured, deadlines = add_targets(df), add_targets(measured), add_targets(deadlines)
    main_cfg = sel(df, E=4, K=5, tiers="mixed")

    stats_out = {"targets": targets, "n_runs": int(len(df) + len(measured) + len(deadlines)),
                 "target_frac": TARGET_FRAC}
    write_summary(pd.concat([df, measured, deadlines]))
    table_main(main_cfg)
    table_scaling(df)
    stats_out["deadline_sensitivity"] = table_deadline(main_cfg, deadlines)
    stats_out["cifar10"] = table_cifar(cifar, hist)
    stats_out["phones_step_s"] = fig_phones()
    fig_rq1(df, main_cfg, measured)
    fig_rq2(main_cfg, hist)
    fig_rq3(main_cfg, hist)
    fig_rq4(df)
    stats_out.update(key_numbers(df, main_cfg))
    stats_out["equal_time"] = equal_time(main_cfg, hist)
    stats_out["server_MB_to_target"] = {
        f"{meth}_{a}": float(np.nanmean([bytes_to_target(hist[n], targets[a]) / 1e6
                                         for n in sel(main_cfg, method=meth, alpha=a).name]))
        for meth in ("flat_fedavg", "hier_fedavg", "hier_proposed") for a in ALPHAS
        if len(sel(main_cfg, method=meth, alpha=a))}
    stats_out["measured_profile"] = measured_numbers(measured, main_cfg)
    with open(os.path.join(OUT, "stats.json"), "w") as f:
        json.dump(stats_out, f, indent=1, default=float)
    print(f"{stats_out['n_runs'] + len(cifar)} runs analysed ({len(df)} main, {len(measured)} measured profile, "
          f"{len(deadlines)} deadline sensitivity, {len(cifar)} CIFAR-10); "
          f"figures in paper/figures, tables in paper/tables")


def write_summary(df):
    g = df.groupby(["profile", "method", "alpha", "E", "K", "tiers"])
    agg = g.agg(n=("seed", "count"), acc=("acc", "mean"), acc_sd=("acc", "std"), p10=("p10", "mean"),
                p10_sd=("p10", "std"), t_target=("t_target", "mean"), sim_time=("sim_time", "mean"),
                server_MB=("bytes_server", lambda v: v.mean() / 1e6), dropped=("dropped", "mean"),
                updates=("updates", "mean"), wasted=("wasted", "mean"), steps=("steps", "mean"),
                edge_rt=("edge_rt", "mean"), agg_ms=("agg_s", lambda v: v.mean() * 1e3), wall=("wall", "mean"))
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
        v = sel(r, method="hier_proposed")["agg_s"]
        cells.append(f"{1e3 * v.mean():.0f}" if len(v) and v.notna().any() else "--")
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(TAB, "tab_scaling.tex"), "w") as f:
        f.write("% generated by analyze.py -- do not edit\n" + "\n".join(lines) + "\n")


def table_deadline(m, dl):
    """G5: proposed scheme at deadline quantiles 0.5 (from G1), 0.75 and 0.9, next to H-FedAvg."""
    rows = [("H-FedAvg (no deadline)", sel(m, method="hier_fedavg"))]
    rows.append(("Proposed, $q = 0.5$", sel(m, method="hier_proposed")))
    for q in (0.75, 0.9):
        rows.append((f"Proposed, $q = {q:g}$", sel(dl, method="hier_proposed", dq=q)))
    lines = [r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{l ccc ccc}", r"\toprule",
             r" & \multicolumn{3}{c}{$\alpha = 0.1$} & \multicolumn{3}{c}{$\alpha = 0.5$} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
             r"Setting & Acc & P10 & $T_{\mathrm{tgt}}$ & Acc & P10 & $T_{\mathrm{tgt}}$ \\", r"\midrule"]
    out = {}
    for label, frame in rows:
        cells = [label]
        for a in ("0.1", "0.5"):
            r = sel(frame, alpha=a)
            cells += [f"{100 * r.acc.mean():.1f}" if len(r) else "--",
                      f"{100 * r.p10.mean():.1f}" if len(r) else "--",
                      (f"{r.t_target.mean():.0f}" if r.t_target.notna().all() else "n.r.") if len(r) else "--"]
            if len(r):
                out[f"{label}_{a}"] = dict(acc=r.acc.mean(), p10=r.p10.mean(), t_target=r.t_target.mean(),
                                           sim_time=r.sim_time.mean(),
                                           dropped_frac=float((r.dropped / r.updates).mean()),
                                           n=int(len(r)))
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(TAB, "tab_deadline.tex"), "w") as f:
        f.write("% generated by analyze.py -- do not edit\n" + "\n".join(lines) + "\n")
    return out


def table_cifar(c, hist):
    """G6: main configuration on CIFAR-10 (target = 90% of CIFAR-10 centralised accuracy)."""
    c = c.copy()
    c["t_target"] = np.nan
    targets = {a: TARGET_FRAC * sel(c, method="centralised", alpha=a).acc.mean() for a in ("0.1", "0.5")}
    for i, row in c.iterrows():
        tgt = targets.get(row.alpha)
        if row.method != "centralised" and tgt is not None and not np.isnan(tgt):
            c.loc[i, "t_target"] = time_to_target(hist[row["name"]], tgt)[0]
    meths = ["centralised", "flat_fedavg", "hier_fedavg", "hier_fedprox_dl", "hier_proposed"]
    lines = [r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{l ccc ccc}", r"\toprule",
             r" & \multicolumn{3}{c}{$\alpha = 0.1$} & \multicolumn{3}{c}{$\alpha = 0.5$} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
             r"Method & Acc & P10 & $T_{\mathrm{tgt}}$ & Acc & P10 & $T_{\mathrm{tgt}}$ \\", r"\midrule"]
    out = {"targets": targets}
    for meth in meths:
        cells = [LABEL[meth]]
        for a in ("0.1", "0.5"):
            r = sel(c, method=meth, alpha=a)
            if r.empty:
                cells += ["--", "--", "--"]
                continue
            tt = "--" if meth == "centralised" else (f"{r.t_target.mean():.0f}" if r.t_target.notna().all()
                                                     else f"n.r.\\,({r.t_target.notna().sum()}/{len(r)})")
            cells += [f"{100 * r.acc.mean():.1f}", f"{100 * r.p10.mean():.1f}", tt]
            out[f"{meth}_{a}"] = dict(acc=r.acc.mean(), acc_sd=r.acc.std(), p10=r.p10.mean(),
                                      t_target=r.t_target.mean(), n=int(len(r)))
        lines.append(" & ".join(cells) + r" \\")
        if meth == "centralised":
            lines.append(r"\midrule")
    for a in ("0.1", "0.5"):  # accuracy of every method when the proposed scheme finishes its rounds
        budget = sel(c, method="hier_proposed", alpha=a).sim_time.mean()
        if not np.isnan(budget):
            out[f"equal_time_{a}"] = {"budget": float(budget), **{
                meth: float(np.nanmean([acc_at_time(hist[n], budget) for n in sel(c, method=meth, alpha=a).name]))
                for meth in meths[1:] if len(sel(c, method=meth, alpha=a))}}
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(TAB, "tab_cifar.tex"), "w") as f:
        f.write("% generated by analyze.py -- do not edit\n" + "\n".join(lines) + "\n")
    return out


def _legend_below(fig, axes, ncol):
    handles, labels = [], []
    for ax in np.ravel(axes):
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, loc="lower center", ncol=ncol, bbox_to_anchor=(0.5, -0.02),
               handlelength=1.6, columnspacing=1.0)


def fig_rq1(df, m, measured):
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

    # system heterogeneity: three tier mixes with literature ratios, plus the mixed split
    # with the speed ratios measured on our own phones (G4)
    cats = [("mid", df), ("mixed", df), ("low_heavy", df), ("mixed", measured)]
    meths = ["hier_fedavg", "hier_fedprox_dl", "hier_proposed"]
    w = 0.26
    for i, meth in enumerate(meths):
        vals = [sel(src, method=meth, tiers=t, alpha="0.5", E=4, K=5).t_target.mean() for t, src in cats]
        pos = np.arange(len(cats)) + (i - 1) * w
        a2.bar(pos, np.nan_to_num(vals), width=w - 0.04, color=COLOR[meth], label=LABEL[meth])
        for p, v in zip(pos, vals):
            if np.isnan(v):
                a2.text(p, 0, "n.r.", ha="center", va="bottom", fontsize=5.5, color="#52514e")
    a2.set_xticks(np.arange(len(cats)), ["all\nmid", "mixed", "low-\nheavy", "mixed\n(phones)"])
    a2.set_xlabel(r"device-tier mix ($\alpha = 0.5$)")
    a2.set_ylabel("simulated time to target (s)")
    a2.set_title("(b) system heterogeneity", loc="left")
    a2.grid(axis="x", visible=False)
    fig.tight_layout(w_pad=1.2, rect=(0, 0.13, 1, 1))
    _legend_below(fig, (a1, a2), ncol=3)
    save(fig, "fig_rq1_heterogeneity")


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
    save(fig, "fig_rq2_comm")


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
    save(fig, "fig_rq3_ablation")


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
    save(fig, "fig_rq4_scaling")


def fig_phones():
    """Measured time per training step on real phones (phone_benchmark), slowest at the bottom."""
    phones = []
    for fn, key in (("measurements.json", "phones"), ("measurements_extra.json", "phones")):
        p = os.path.join(ROOT, "phone_benchmark", fn)
        if os.path.exists(p):
            with open(p) as f:
                phones += json.load(f)[key]
    if not phones:
        return None
    phones.sort(key=lambda d: d["train_step_ms_median"])
    names = [d["phone"].replace("Samsung Galaxy ", "Galaxy ") for d in phones]
    ms = np.array([d["train_step_ms_median"] for d in phones]) / 1000
    fastest = ms.min()
    fig, ax = plt.subplots(figsize=(COL_W, 0.32 * len(phones) + 0.55))
    y = np.arange(len(phones))[::-1]
    ax.barh(y, ms, height=0.6, color="#2a78d6")
    for yi, v in zip(y, ms):
        ax.text(v + ms.max() * 0.01, yi, f"{v:.1f} s ({v / fastest:.1f}$\\times$)", va="center", fontsize=6.5,
                color="#0b0b0b")
    ax.set_yticks(y, names)
    ax.set_xlim(0, ms.max() * 1.3)
    ax.set_xlabel("time per training step, batch 32 (s, browser CPU backend)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "fig_phones")
    return {n: float(v) for n, v in zip(names, ms)}


def _paired(m, a_meth, b_meth, col):
    a = sel(m, method=a_meth, alpha=ALPHAS).set_index(["alpha", "seed"])[col]
    b = sel(m, method=b_meth, alpha=ALPHAS).set_index(["alpha", "seed"])[col]
    j = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
    if len(j) < 2:
        return None
    t, p = stats.ttest_rel(j.a, j.b)
    return dict(n=int(len(j)), mean_diff=float((j.a - j.b).mean()), t=float(t), p=float(p))


def equal_time(m, hist):
    """Accuracy every FL method has reached when the proposed scheme finishes its R rounds."""
    out = {}
    pairs = []
    for a in ALPHAS:
        budget = sel(m, method="hier_proposed", alpha=a).sim_time.mean()
        if np.isnan(budget):
            continue
        out[f"budget_{a}"] = float(budget)
        per = {}
        for meth in ("flat_fedavg", "hier_fedavg", "hier_fedprox", "hier_fedprox_dl", "hier_proposed"):
            r = sel(m, method=meth, alpha=a)
            if len(r):
                vals = {int(s): acc_at_time(hist[n], budget) for n, s in zip(r.name, r.seed)}
                per[meth] = vals
                out[f"{meth}_{a}"] = float(np.nanmean(list(vals.values())))
        if "hier_proposed" in per and "hier_fedavg" in per:
            for s in per["hier_proposed"]:
                if s in per["hier_fedavg"]:
                    pairs.append((per["hier_proposed"][s], per["hier_fedavg"][s]))
    if len(pairs) >= 2:
        a_, b_ = np.array(pairs).T
        t, p = stats.ttest_rel(a_, b_)
        out["ttest_proposed_vs_hfedavg"] = dict(n=len(pairs), mean_diff=float((a_ - b_).mean()), t=float(t), p=float(p))
    return out


def measured_numbers(meas, m):
    """G4: same methods with phone-measured speed ratios, next to the literature profile."""
    out = {}
    for meth in ("hier_fedavg", "hier_fedprox_dl", "hier_proposed"):
        for a in ALPHAS:
            r, base = sel(meas, method=meth, alpha=a), sel(m, method=meth, alpha=a)
            if len(r):
                out[f"{meth}_{a}"] = dict(
                    acc=r.acc.mean(), p10=r.p10.mean(), t_target=r.t_target.mean(), sim_time=r.sim_time.mean(),
                    dropped_frac=float((r.dropped / r.updates).mean()),
                    wasted_frac=float((r.wasted / r.steps).mean()),
                    colext_acc=base.acc.mean() if len(base) else None,
                    colext_t_target=base.t_target.mean() if len(base) else None, n=int(len(r)))
    return out


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
