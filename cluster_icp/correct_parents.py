"""Rescore the Euler CSVs against the true parents of Y.

The original runs took parents from the cluster graph, which keeps E -> Y after
the variable-level edges are removed, so E was counted as a missed parent. The
estimator's output is unaffected, so only the scores need correcting. A cluster
edge with no variable-level edge behind it is assumed not to have been predicted.

Usage: python correct_parents.py <euler_csv_dir> <output_dir>
"""
import glob
import os
import sys

import networkx as nx
import numpy as np
import pandas as pd

# Graph generators only: the experiment script parses arguments when imported.
source = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "slurm_cluster_icp_exp.py")).read()
exec(source.split("def simulate_linear_gaussian(")[0])


def parent_counts(q, m, seed):
    p_between_cluster, p_within_var, p_between_var = 3 / q, 0.9, 1 / m
    np.random.seed(seed)
    Gc = generate_cluster_dag(q + 2, p_between=p_between_cluster, seed=seed)
    G, cluster_vars = generate_variable_dag(Gc, m, p_within=p_within_var, p_between=p_between_var, seed=seed)
    E = list(nx.topological_sort(Gc))[0]
    Y = np.random.choice([c for c in cluster_vars.keys() if c != E])
    listed = set(Gc.predecessors(Y))
    for v1 in cluster_vars[E]:
        for v2 in cluster_vars[Y]:
            if G.has_edge(v1, v2):
                G.remove_edge(v1, v2)
    var2cluster = {v: c for c, vars_ in cluster_vars.items() for v in vars_}
    true_parents = {var2cluster[u] for v in cluster_vars[Y] for u in G.predecessors(v)} - {Y}
    return len(listed), len(true_parents), len(listed - true_parents)


def correct(df):
    cache = {}
    counts = []
    for row in df.itertuples():
        key = (row.q, row.m, row.seed)
        if key not in cache:
            cache[key] = parent_counts(*key)
        counts.append(cache[key])
    counts = pd.DataFrame(counts, columns=["listed", "true", "spurious"], index=df.index)
    assert (counts.listed == df.num_parents).all(), "regenerated graphs do not match the CSV"
    df = df.copy()
    df["num_parents"] = counts.true
    for prefix in ("parents_", "parents_S_CICP_"):
        df[prefix + "FN"] -= counts.spurious
        df[prefix + "TN"] += counts.spurious
    return df


if __name__ == "__main__":
    in_dir, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    default = pd.concat([pd.read_csv(f, index_col=0) for f in glob.glob(os.path.join(in_dir, "cluster_icp_exp_*-*.csv"))],
                        ignore_index=True)
    default = default[default.Setting == "Cluster"].reset_index(drop=True)
    correct(default).to_csv(os.path.join(out_dir, "cluster_icp_exp_Cluster.csv"), index=False)
    test_stat = pd.read_csv(os.path.join(in_dir, "test_stat_0-29.csv"), index_col=0)
    correct(test_stat).to_csv(os.path.join(out_dir, "test_stat.csv"), index=False)
