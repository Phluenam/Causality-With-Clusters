"""Plot the Cluster ICP experiments from the corrected and rerun CSVs.

Bands are 95% Clopper-Pearson intervals. FWER and EMR count independent seeds. FNR,
FPR and recall use counts pooled across seeds, which treats parents as independent.

Usage: python plot_cluster_icp.py <corrected_dir> <rerun_dir> <figure_dir>
"""
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

COLORS = ["#3B86E0", "#1B9E77", "#D9582B", "#8071DE", "#B5761B"]
METRICS = ["FPR", "FNR", "FWER", "EMR"]
REFERENCES = [("SCICP_", r"$\tilde{S}_{CICP}$"), ("parents_", r"$pa_G(\mathbf{Y})$")]
TEST_STATS = ["Hotelling-Lawley trace", "Pillai's trace", "Roy's greatest root", "Wilks' lambda"]
BAND_ALPHA = 0.13


def clopper_pearson(k, n):
    """Estimate with its 95% exact interval, as (estimate, lower, upper)."""
    k, n = int(k), int(n)
    if n == 0:
        return (float("nan"),) * 3
    lower = 0.0 if k == 0 else stats.beta.ppf(0.025, k, n - k + 1)
    upper = 1.0 if k == n else stats.beta.ppf(0.975, k + 1, n - k)
    return k / n, lower, upper


def metrics(g, prefix):
    tp, fp, fn, tn = (g[prefix + c].sum() for c in ("TP", "FP", "FN", "TN"))
    return {
        "FPR": clopper_pearson(fp, fp + tn),
        "FNR": clopper_pearson(fn, fn + tp),
        "FWER": clopper_pearson((g[prefix + "FP"] > 0).sum(), len(g)),
        "EMR": clopper_pearson(((g[prefix + "FP"] == 0) & (g[prefix + "FN"] == 0)).sum(), len(g)),
    }


def draw(ax, xs, intervals, color, label, markersize=4):
    estimates, lowers, uppers = zip(*intervals)
    ax.fill_between(xs, lowers, uppers, color=color, alpha=BAND_ALPHA, linewidth=0)
    ax.plot(xs, estimates, marker="o", markersize=markersize, linewidth=1.5, color=color, label=label)


def style(ax):
    ax.grid(True, color="#E4E4E4", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def legend(fig, handles, labels):
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=10)


def plot_vary(df, column, values, symbol, path):
    """Rows are the two references, columns the metrics, x the sample size."""
    ns = [100, 500, 1000, 5000]
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), sharex=True)
    for r, (prefix, ref_label) in enumerate(REFERENCES):
        for c, metric in enumerate(METRICS):
            ax = axes[r, c]
            for i, value in enumerate(values):
                sub = df[df[column] == value]
                draw(ax, ns, [metrics(sub[sub.n == n], prefix)[metric] for n in ns], COLORS[i], f"${symbol}={value}$")
            ax.set_xscale("log")
            ax.set_ylim(*((-0.05, 1.05) if metric in ("FNR", "EMR") else (-0.01, 0.3)))
            style(ax)
            if r == 0:
                ax.set_title(metric, fontsize=12)
            else:
                ax.set_xlabel("$n$", fontsize=11)
            if c == 0:
                ax.set_ylabel(ref_label, fontsize=12)
    legend(fig, *axes[0, 0].get_legend_handles_labels())
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_test_stat(df, prefix, path):
    ns = [100, 500, 1000, 5000]
    alphas = sorted(df.Alpha.unique())
    fig, axes = plt.subplots(4, 4, figsize=(10, 10), sharex=True)
    for c, n in enumerate(ns):
        for r, metric in enumerate(METRICS):
            ax = axes[r, c]
            for i, name in enumerate(TEST_STATS):
                sub = df[(df.n == n) & (df["Test Statistic"] == name)]
                draw(ax, alphas, [metrics(sub[sub.Alpha == a], prefix)[metric] for a in alphas], COLORS[i], name, markersize=3)
            if metric == "FPR":
                ax.plot(alphas, alphas, linestyle="--", linewidth=1, color="#888888", label=r"FPR = $\alpha$")
            ax.set_ylim(*((-0.05, 1.05) if metric in ("FNR", "EMR") else (-0.01, 0.55)))
            style(ax)
            if r == 0:
                ax.set_title(f"n = {n:,}", fontsize=12)
            if r == 3:
                ax.set_xlabel(r"$\alpha$", fontsize=11)
            if c == 0:
                ax.set_ylabel(metric, fontsize=11)
    legend(fig, *axes[0, 0].get_legend_handles_labels())
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_vary_s(df, path):
    panels = ["Recall"] + METRICS
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.8), sharex=True)
    for i, q in enumerate(sorted(df.q.unique())):
        sub = df[df.q == q]
        ss = sorted(sub.s.unique())
        rows = [metrics(sub[sub.s == s], "parents_") for s in ss]
        for r, s in zip(rows, ss):
            g = sub[sub.s == s]
            r["Recall"] = clopper_pearson(g.num_parents_in_candidates.sum(), g.num_parents.sum())
        for ax, panel in zip(axes, panels):
            draw(ax, ss, [r[panel] for r in rows], COLORS[i], f"$q={q}$")
    for ax, panel in zip(axes, panels):
        ax.set_title(panel, fontsize=12)
        ax.set_xlabel("$s$", fontsize=11)
        ax.set_xticks([2, 4, 8])
        ax.set_ylim(*((-0.05, 1.05) if panel in ("Recall", "FNR", "EMR") else (-0.01, 0.2)))
        style(ax)
    legend(fig, *axes[0].get_legend_handles_labels())
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    corrected_dir, rerun_dir, figure_dir = sys.argv[1:4]
    default = pd.read_csv(os.path.join(corrected_dir, "cluster_icp_exp_Cluster.csv"))
    plot_vary(default[default.m == 5], "q", [3, 5, 8, 10], "q", os.path.join(figure_dir, "vary_q_corrected.png"))
    plot_vary(default[default.q == 5], "m", [3, 5, 10, 20, 50], "m", os.path.join(figure_dir, "vary_m_corrected.png"))
    test_stat = pd.read_csv(os.path.join(corrected_dir, "test_stat.csv"))
    plot_test_stat(test_stat, "SCICP_", os.path.join(figure_dir, "test_statistic_comparison_scicp_corrected.png"))
    plot_test_stat(test_stat, "parents_", os.path.join(figure_dir, "test_statistic_comparison_parents_corrected.png"))
    screening = pd.concat([pd.read_csv(f, index_col=0) for f in glob.glob(os.path.join(rerun_dir, "group_lasso_*.csv"))])
    plot_vary_s(screening, os.path.join(figure_dir, "vary_s_fixed.png"))
