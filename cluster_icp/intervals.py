"""95% Clopper-Pearson intervals for the Cluster ICP numbers quoted in the text.

FWER and EMR count independent seeds. FNR, FPR and recall pool counts over seeds and
treat parents as independent, which may not hold, so those intervals may be too narrow.

Usage: python intervals.py <corrected_dir> <rerun_dir> <gap_csv>
"""
import glob
import os
import sys

import pandas as pd
from scipy import stats


def cp(k, n):
    lo = 0.0 if k == 0 else stats.beta.ppf(0.025, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(0.975, k + 1, n - k)
    return f"{k}/{n} = {k / n:.3f} [{lo:.3f}, {hi:.3f}]"


def show(label, g, prefix):
    tp, fp, fn, tn = (int(g[prefix + c].sum()) for c in ("TP", "FP", "FN", "TN"))
    fwer, emr = int((g[prefix + "FP"] > 0).sum()), int(((g[prefix + "FP"] == 0) & (g[prefix + "FN"] == 0)).sum())
    print(f"{label:34s} FNR {cp(fn, fn + tp)} | FPR {cp(fp, fp + tn)} | FWER {cp(fwer, len(g))} | EMR {cp(emr, len(g))}")


if __name__ == "__main__":
    corrected_dir, rerun_dir, gap_csv = sys.argv[1:4]
    d = pd.read_csv(os.path.join(corrected_dir, "cluster_icp_exp_Cluster.csv"))
    for column, fixed, values in (("q", ("m", 5), (3, 5, 8, 10)), ("m", ("q", 5), (3, 5, 10, 20, 50))):
        for v in values:
            for n in (100, 500, 1000, 5000):
                g = d[(d[column] == v) & (d[fixed[0]] == fixed[1]) & (d.n == n)]
                show(f"{column}={v} n={n} S-tilde", g, "SCICP_")
                show(f"{column}={v} n={n} parents", g, "parents_")
            g = d[(d[column] == v) & (d[fixed[0]] == fixed[1]) & (d.n == 5000)]
            show(f"{column}={v} oracle vs parents", g, "parents_S_CICP_")
    s = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(rerun_dir, "group_lasso_*.csv"))])
    for (q, sv), g in s.groupby(["q", "s"]):
        print(f"screening q={q} s={sv}: recall {cp(int(g.num_parents_in_candidates.sum()), int(g.num_parents.sum()))}")
        show(f"screening q={q} s={sv}", g, "parents_")
        show(f"screening q={q} s={sv} oracle", g, "parents_S_CICP_")
    gap = pd.read_csv(gap_csv)
    for (q, p), g in gap.groupby(["q", "p"]):
        print(f"gap q={q} p={p}: FNR {cp(int((g.parents - g.found).sum()), int(g.parents.sum()))} | E,Y separated {cp(int((~g.E_Y_connected).sum()), len(g))}")
