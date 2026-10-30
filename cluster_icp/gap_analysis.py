"""Why does S-tilde miss more of pa_G(Y) as q grows? Graphs only, 30 seeds, m=5.

Varies q and the cluster edge probability p independently; p = 3/q is the thesis grid.
Usage: python gap_analysis.py <experiment_script> <out_csv>
"""
import itertools
import sys
import time

import networkx as nx
import numpy as np
import pandas as pd

exec(open(sys.argv[1]).read().split("def simulate_linear_gaussian(")[0])

M = 5


def analyse(q, p, seed):
    np.random.seed(seed)
    Gc = generate_cluster_dag(q + 2, p_between=p, seed=seed)
    G, cv = generate_variable_dag(Gc, M, p_within=0.9, p_between=1 / M, seed=seed)
    E = list(nx.topological_sort(Gc))[0]
    Y = np.random.choice([c for c in cv if c != E])
    for a in cv[E]:
        for b in cv[Y]:
            if G.has_edge(a, b):
                G.remove_edge(a, b)
    v2c = {v: c for c, vs in cv.items() for v in vs}
    parents = {v2c[u] for v in cv[Y] for u in G.predecessors(v)} - {Y}
    others = [c for c in cv if c not in (E, Y)]
    E_v, Y_v = set(cv[E]), set(cv[Y])

    valid = []
    for r in range(len(others) + 1):
        for S in itertools.combinations(others, r):
            Z = {v for c in S for v in cv[c]}
            if nx.is_d_separator(G, E_v, Y_v, Z):
                valid.append(set(S))
    oracle = set.intersection(*valid)

    connected = not nx.is_d_separator(G, E_v, Y_v, set())
    de_E = set().union(*(nx.descendants(G, v) for v in cv[E])) - E_v
    de_E_c = {v2c[v] for v in de_E}
    row = dict(
        q=q, p=p, seed=seed, parents=len(parents), found=len(parents & oracle),
        E_Y_connected=connected, E_outdeg=Gc.out_degree(E), Y_desc_of_E=Y in de_E_c,
        missed_E_Y_separated=0 if connected else len(parents - oracle),
        missed_not_desc=len({c for c in parents - oracle if c not in de_E_c}) if connected else 0,
        missed_desc=len({c for c in parents - oracle if c in de_E_c}) if connected else 0,
    )
    return row


if __name__ == "__main__":
    t0 = time.time()
    rows = []
    for q in (3, 5, 8, 10):
        for p in (1.0, 0.6, 0.375, 0.3):
            for seed in range(30):
                rows.append(analyse(q, p, seed))
        print("q", q, "done", round(time.time() - t0), "s", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(sys.argv[2], index=False)
    g = d.groupby(["q", "p"])
    s = g.agg(parents=("parents", "sum"), found=("found", "sum"), sep=("missed_E_Y_separated", "sum"),
              not_desc=("missed_not_desc", "sum"), desc=("missed_desc", "sum"),
              E_Y_conn=("E_Y_connected", "mean"), E_outdeg=("E_outdeg", "mean"), E_isolated=("E_outdeg", lambda x: (x == 0).mean()))
    s["oracle_FNR"] = 1 - s.found / s.parents
    print(s.round(3).to_string())
    print("oracle FNR by (q, p):")
    print(s.oracle_FNR.unstack().round(3).to_string())
