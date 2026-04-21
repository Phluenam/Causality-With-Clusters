import itertools
import numpy as np
import networkx as nx
from scipy import stats
import adelie as ad
import numpy as np
import copy
import pandas as pd
from statsmodels.multivariate.manova import MANOVA
import statsmodels.api as sm
import matplotlib.pyplot as plt
import time



# -----------------------------
# 1. Generate cluster DAG
# -----------------------------
def generate_cluster_dag(q, p_between=0.3, k=None, seed=None):
    """Generate a random DAG on q clusters efficiently (topological order).
    k: maximum number of parents any node can have (None = unlimited)
    """
    rng = np.random.default_rng(seed)
    order = list(range(q))
    rng.shuffle(order)
    Gc = nx.DiGraph()
    Gc.add_nodes_from(range(q))
    for i, u in enumerate(order):
        for v in order[i+1:]:
            if rng.random() < p_between:
                Gc.add_edge(u, v)

    if k is not None:
        for v in Gc.nodes():
            parents = list(Gc.predecessors(v))
            if len(parents) > k:
                keep = rng.choice(parents, size=k, replace=False)
                for p in parents:
                    if p not in keep:
                        Gc.remove_edge(p, v)

    return Gc

# -----------------------------
# 2. Expand to variable-level DAG
# -----------------------------
def generate_variable_dag(Gc, m, p_within=0.5, p_between=0.5, seed=None):
    rng = np.random.default_rng(seed)
    q = Gc.number_of_nodes()
    cluster_vars = {c: [f"X_{c}_{i}" for i in range(m)] for c in range(q)}

    G = nx.DiGraph()
    for vars_ in cluster_vars.values():
        G.add_nodes_from(vars_)

    # Within-cluster edges
    for vars_ in cluster_vars.values():
        for i in range(len(vars_)):
            for j in range(i+1, len(vars_)):
                if rng.random() < p_within:
                    G.add_edge(vars_[i], vars_[j])

    # Between-cluster edges
    for c1, c2 in Gc.edges():
        for v1 in cluster_vars[c1]:
            for v2 in cluster_vars[c2]:
                if rng.random() < p_between:
                    G.add_edge(v1, v2)

    return G, cluster_vars

# -----------------------------
# 3. Simulate linear Gaussian SCM
# -----------------------------
def simulate_linear_gaussian(G, n=500, coef_low=0.5, coef_high=1.5, seed=None):
    rng = np.random.default_rng(seed)
    var2idx = {v: i for i, v in enumerate(G.nodes)}
    X_mat = np.zeros((n, len(G.nodes)))

    topo_order = list(nx.topological_sort(G))
    for v in topo_order:
        parents = list(G.predecessors(v))
        noise = rng.normal(size=n)
        if parents:
            coefs = rng.uniform(coef_low, coef_high, size=len(parents))
            signs = rng.choice([-1,1], size=len(parents))
            val = sum(signs[i]*coefs[i]*X_mat[:, var2idx[parents[i]]] for i in range(len(parents))) + noise
        else:
            val = noise
        X_mat[:, var2idx[v]] = val
        X_mat[:, var2idx[v]] /= X_mat[:, var2idx[v]].std()
    return X_mat, var2idx


def simulate_linear_gaussian_dependent_environment(G, E_names, n=500, coef_low=0.5, coef_high=1.5, seed=None):
    assert isinstance(E_names, list)
    rng = np.random.default_rng(seed)
    var2idx = {v: i for i, v in enumerate(G.nodes)}
    X_mat = np.zeros((n, len(G.nodes)))

    # generate E variables
    half = (len(E_names) + 1) // 2

    # first half: independent normal noise
    first_half = E_names[:half]
    for e in first_half:
        z_e = rng.normal(size=n)
        X_mat[:, var2idx[e]] = z_e / z_e.std()

    # second half: products of pairs from first half (no duplicate pairs)
    second_half = E_names[half:]
    used_pairs = set()
    for e in second_half:
        # sample a pair not yet used
        while True:
            i, j = rng.choice(len(first_half), size=2, replace=True)
            pair = tuple(sorted((i, j)))
            if pair not in used_pairs:
                used_pairs.add(pair)
                break
        val = X_mat[:, var2idx[first_half[i]]] * X_mat[:, var2idx[first_half[j]]]
        X_mat[:, var2idx[e]] = val / val.std()

    topo_order = list(nx.topological_sort(G))
    for v in topo_order:
        if E_names is not None and v in E_names:
            continue
        parents = list(G.predecessors(v))
        noise = rng.normal(size=n)
        if parents:
            coefs = rng.uniform(coef_low, coef_high, size=len(parents))
            signs = rng.choice([-1, 1], size=len(parents))
            val = sum(signs[i]*coefs[i]*X_mat[:, var2idx[parents[i]]] for i in range(len(parents))) + noise
        else:
            val = noise
        X_mat[:, var2idx[v]] = val
        X_mat[:, var2idx[v]] /= X_mat[:, var2idx[v]].std()

    return X_mat, var2idx

def simulate_linear_gaussian_categorical_environment(G, E_names, num_levels, n=500, coef_low=0.5, coef_high=1.5, seed=None):
    assert isinstance(E_names, list)
    rng = np.random.default_rng(seed)
    var2idx = {v: i for i, v in enumerate(G.nodes)}
    X_mat = np.zeros((n, len(G.nodes)))

    # draw a level for each observation, uniform in [1, num_levels]
    l = rng.integers(1, num_levels + 1, size=n)
    for e in E_names:
        X_mat[:, var2idx[e]] = l

    # random perturbation matrix: shape (n_nodes, num_levels)
    signs = rng.choice([-1, 1], size=(len(G.nodes), num_levels))
    M = signs * rng.uniform(coef_low, coef_high, size=(len(G.nodes), num_levels))

    E_names_set = set(E_names)

    topo_order = list(nx.topological_sort(G))
    for v in topo_order:
        if v in E_names_set:
            continue
        parents = list(G.predecessors(v))
        noise = rng.normal(size=n)
        if parents:
            coefs = rng.uniform(coef_low, coef_high, size=len(parents))
            signs = rng.choice([-1, 1], size=len(parents))
            val = sum(signs[i]*coefs[i]*X_mat[:, var2idx[parents[i]]]
                      for i in range(len(parents))) + noise
            # additively perturb mean if any parent is in E_names
            if any(p in E_names_set for p in parents):
                val = val + M[var2idx[v], l - 1]  # l-1 for 0-indexing into M
        else:
            val = noise
        X_mat[:, var2idx[v]] = val
        X_mat[:, var2idx[v]] /= X_mat[:, var2idx[v]].std()

    return X_mat, var2idx


# -----------------------------
# 4. Cluster-level d-separation
# -----------------------------
def cluster_d_separated(G, cluster_vars, X, Y, Z):
    """
    Check d-separation using NetworkX.
    """
    # 1. Work on a copy so we don't break the original graph
    G_check = G.copy()

    #print(X, [cluster_vars[x] for x in X])
    #print(Y)
    #print(Z)

    X_names = sum([cluster_vars[x] for x in X],[])
    Y_names = sum([cluster_vars[x] for x in Y],[])
    Z_names = sum([cluster_vars[x] for x in Z],[])

    # 2. Use the function you found
    #    Note: We wrap X, Y, Z in sets because is_d_separator expects sets/nodes
    return nx.is_d_separator(G, set(X_names), set(Y_names), set(Z_names))


# -----------------------------
# 5. MANCOVA
# -----------------------------

def mancova_test(X_mat, cluster_vars, E, Y, S, alpha=0.05, var2idx=None, test_statistic="Wilks' lambda"):
    Y_names = cluster_vars[Y]
    S_names = [v for c in S for v in cluster_vars[c]]
    E_names = cluster_vars[E]

    predictor_name = E_names[0]

    # Build hypothesis string: "X1 = 0, X2 = 0, ..."
    hyp_str = ', '.join(f"{pred} = 0" for pred in E_names)

    # Pass as a list of (name, hypothesis) tuples
    hypotheses = [('joint_test', hyp_str)]

    all_vars = Y_names + S_names + E_names
    indices = [var2idx[v] for v in all_vars]

    df = pd.DataFrame(X_mat[:, indices], columns=all_vars)

    # Define the formula
    formula_rhs = ' + '.join(E_names)
    if S_names:
        formula_rhs += f" + {' + '.join(S_names)}"

    try:
        if len(Y_names) > 1:
            formula = f"{' + '.join(Y_names)} ~ {formula_rhs}"
            maov = MANOVA.from_formula(formula, data=df)
            hyp_str = ', '.join(f"{pred} = 0" for pred in E_names)
            result = maov.mv_test([('joint_test', hyp_str)])
            stat_table = result.results['joint_test']['stat']
            assert test_statistic in ["Wilks' lambda", "Pillai's trace", "Hotelling-Lawley trace", "Roy's greatest root"]
            pval = stat_table.loc[test_statistic, 'Pr > F']
        else:
            formula = f"{Y_names[0]} ~ {formula_rhs}"
            model = sm.OLS.from_formula(formula, data=df).fit()
            f_test = model.f_test([f"{name} = 0" for name in E_names])
            pval = f_test.pvalue

        return pval > alpha

    except Exception as e:
        print(f"--- Test Failed ---")
        print(X_mat[:, indices].std(axis=0))

        print(f"Target: {Y}, Predictor: {E}, S: {S}, Error: {e}")
        return None

def categorical_mancova_test(X_mat, cluster_vars, E, Y, S, alpha=0.05, var2idx=None, test_statistic="Wilks' lambda"):
    Y_names = cluster_vars[Y]
    S_names = [v for c in S for v in cluster_vars[c]]
    E_names = cluster_vars[E]

    # use only the first E variable since they're all identical
    predictor_name = E_names[0]

    all_vars = Y_names + S_names + [predictor_name]
    indices = [var2idx[v] for v in all_vars]

    df = pd.DataFrame(X_mat[:, indices], columns=all_vars)

    # treat E as categorical
    df[predictor_name] = df[predictor_name].astype(int).astype('category')

    formula_rhs = f"C({predictor_name})"
    if S_names:
        formula_rhs += f" + {' + '.join(S_names)}"

    try:
        if len(Y_names) > 1:
            formula = f"{' + '.join(Y_names)} ~ {formula_rhs}"
            maov = MANOVA.from_formula(formula, data=df)
            hyp_str = ', '.join(f"C({predictor_name})[T.{level}] = 0"
                                for level in df[predictor_name].cat.categories[1:])
            result = maov.mv_test([('joint_test', hyp_str)])
            stat_table = result.results['joint_test']['stat']
            assert test_statistic in ["Wilks' lambda", "Pillai's trace", "Hotelling-Lawley trace", "Roy's greatest root"]
            pval = stat_table.loc[test_statistic, 'Pr > F']
        else:
            formula = f"{Y_names[0]} ~ {formula_rhs}"
            model = sm.OLS.from_formula(formula, data=df).fit()
            # joint test over all dummy levels
            constraints = [f"C({predictor_name})[T.{level}] = 0"
                           for level in df[predictor_name].cat.categories[1:]]
            f_test = model.f_test(constraints)
            pval = f_test.pvalue

        return pval > alpha
    except Exception as e:
        print(f"--- Test Failed ---")
        print(X_mat[:, indices].std(axis=0))
        print(f"Target: {Y}, Predictor: {E}, S: {S}, Error: {e}")
        return None



# 6. Cluster ICP
# -----------------------------
def run_cluster_icp(X_mat, cluster_vars, G, E, Y, alpha=0.05, var2idx=None, candidate_clusters=None, env_mode="default", test_statistic="Wilks' lambda"):
    accepted_sets = []
    gt_sets = []

    # Only consider subsets excluding both E and Y
    if candidate_clusters is None:
        clusters = list(cluster_vars.keys())
        candidate_clusters = [c for c in clusters if c != E and c != Y]

    for k in range(len(candidate_clusters)+1):
        for S_tuple in itertools.combinations(candidate_clusters, k):
            S = set(S_tuple)

            #print(S)
            # Statistical test
            if env_mode == "categorical":
                if categorical_mancova_test(X_mat, cluster_vars, E, Y, S, alpha, var2idx, test_statistic=test_statistic):
                    accepted_sets.append(S)
            else:
                if mancova_test(X_mat, cluster_vars, E, Y, S, alpha, var2idx, test_statistic=test_statistic):
                    accepted_sets.append(S)

            # Ground truth d-separation
            #print("TEST", G, E, Y, S)
            if cluster_d_separated(G, cluster_vars, {E}, {Y}, S):
                gt_sets.append(S)

    #print("GT", gt_sets)
    #print("AC", accepted_sets)

    # Intersection of all statistically accepted sets
    if accepted_sets:
        S_hat = set.intersection(*accepted_sets)
    else:
        S_hat = set()

    # Intersection of all true d-separating sets
    if gt_sets:
        S_CICP = set.intersection(*gt_sets)
    else:
        S_CICP = set()

    return S_hat, S_CICP

def get_cluster_idxs(cluster_vars, var2idx):
    cluster_idxs = {}
    for k in cluster_vars.keys():
        cluster_idxs[k] = []
        for v in cluster_vars[k]:
            cluster_idxs[k].append(var2idx[v])
    return cluster_idxs

def choose_candidate_groups(X, cluster_idxs, E, Y, s):
    exclude = {E, Y}
    candidate_clusters = {g: idxs for g, idxs in cluster_idxs.items()
                          if g not in exclude}

    cand_groups = sorted(candidate_clusters.keys())

    col_order = []
    group_starts = []
    group_sizes = []
    pos = 0
    for g in cand_groups:
        group_starts.append(pos)
        group_sizes.append(len(candidate_clusters[g]))
        col_order.extend(candidate_clusters[g])
        pos += len(candidate_clusters[g])

    X_cand = X[:, col_order]

    # use all Y cluster variables as multivariate response
    Y_cols = cluster_idxs[Y]
    Y_mat = X[:, Y_cols].astype(float)

    groups = np.array(group_starts, dtype=np.int64)

    # fit path — multivariate if multiple Y columns
    if Y_mat.shape[1] > 1:
        glm = ad.glm.multigaussian(y=Y_mat)
    else:
        glm = ad.glm.gaussian(y=Y_mat[:, 0])

    state = ad.grpnet(X_cand, glm, groups=groups, progress_bar=False)
    # count active groups at each lambda
    n_active = []
    active_group_sets = []
    for i in range(len(state.lmdas)):
        beta_i = np.asarray(state.betas[i].todense()).ravel()
        active = []
        for gi, g in enumerate(cand_groups):
            start = group_starts[gi]
            size = group_sizes[gi]
            if np.any(beta_i[start:start+size] != 0):
                active.append(g)
        n_active.append(len(active))
        active_group_sets.append(active)

    n_active = np.array(n_active)

    # find last lambda with <= s active groups
    valid = np.where(n_active <= s)[0]
    if len(valid) == 0:
        idx = np.argmin(n_active)
    else:
        idx = valid[-1]

    selected = list(active_group_sets[idx])

    # if fewer than s fill from next lambda step
    if len(selected) < s and idx + 1 < len(state.lmdas):
        next_active = active_group_sets[idx + 1]
        extras = [g for g in next_active if g not in selected]
        selected = selected + extras[:s - len(selected)]

    return selected

def single_run(q, m, n, p_between_cluster, p_within_var, p_between_var, k=None, s=None, seed=None, env_mode="default", test_statistic="Wilks' lambda", alpha=0.05, num_levels=None):
    assert env_mode in ["default", "categorical", "dependent"]
    np.random.seed(seed)
    Gc = generate_cluster_dag(q, p_between=p_between_cluster, k=k, seed=seed)
    G, cluster_vars = generate_variable_dag(Gc, m, p_within=p_within_var, p_between=p_between_var, seed=seed)

    topo = list(nx.topological_sort(Gc))
    E = topo[0]
    Y = np.random.choice([c for c in cluster_vars.keys() if c != E])

    for v1 in cluster_vars[E]:
        for v2 in cluster_vars[Y]:
            if G.has_edge(v1, v2):
                G.remove_edge(v1, v2)
    E_names = cluster_vars[E]
    if env_mode == "dependent":
        X, var2idx = simulate_linear_gaussian_dependent_environment(G, E_names, n, seed=seed)
    elif env_mode == "categorical":
        assert num_levels is not None
        X, var2idx = simulate_linear_gaussian_categorical_environment(G, E_names, num_levels, n, seed=seed)
    else:
        X, var2idx = simulate_linear_gaussian(G, n, seed=seed)

    clusters = list(cluster_vars.keys())
    candidate_clusters = [c for c in clusters if c != E and c != Y]
    if s is not None:
        cluster_idxs = get_cluster_idxs(cluster_vars, var2idx)
        candidate_clusters = choose_candidate_groups(X, cluster_idxs, E, Y, s)

    parents = set(list(Gc.predecessors(Y)))
    num_parents = len(parents)
    num_candidates = len(candidate_clusters)
    num_parents_in_candidates = len(parents & set(candidate_clusters))

    S_hat, S_CICP = run_cluster_icp(X, cluster_vars, G, E, Y, var2idx=var2idx, candidate_clusters=candidate_clusters, env_mode=env_mode, test_statistic=test_statistic, alpha=alpha)

    return {
        "num_candidates": num_candidates,
        "num_parents": num_parents,
        "num_parents_in_candidates": num_parents_in_candidates,
        "num_S_CICP": len(S_CICP),
        "num_predicted": len(S_hat),
        # S_hat vs parents
        "parents_TP": len(S_hat & parents),
        "parents_FP": len(S_hat) - len(S_hat & parents),
        "parents_FN": len(parents) - len(S_hat & parents),
        "parents_TN": len(Gc.nodes)-2 - len(S_hat & parents) - (len(parents) - len(S_hat & parents)) - (len(S_hat) - len(S_hat & parents)),
        # S_hat vs S_CICP
        "SCICP_TP": len(S_hat & S_CICP),
        "SCICP_FP": len(S_hat) - len(S_hat & S_CICP),
        "SCICP_FN": len(S_CICP) - len(S_hat & S_CICP),
        "SCICP_TN":  num_candidates - len(S_hat & S_CICP) - (len(S_CICP) - len(S_hat & S_CICP)) - (len(S_hat) - len(S_hat & S_CICP)),
        # S_CICP vs parents
        "parents_S_CICP_TP": len(S_CICP & parents),
        "parents_S_CICP_FP": len(S_CICP) - len(S_CICP & parents),
        "parents_S_CICP_FN": len(parents) - len(S_CICP & parents),
        "parents_S_CICP_TN": len(Gc.nodes)-2 - len(S_CICP & parents) - (len(parents) - len(S_CICP & parents)) - (len(S_CICP) - len(S_CICP & parents)),
    }



import argparse

# 1. Create parser
parser = argparse.ArgumentParser(description="Run experiment with a seed")

# 2. Add seed argument
parser.add_argument("-s", "--seeds", type=str, default="0", help="Seeds to run")
parser.add_argument("-o", "--output-dir", type=str, default="csvs", help="Where to output")
parser.add_argument("-e", "--experiment", type=str, default="default", help="experiment to run", choices=["default", "test_stat", "group_lasso", "dependent_env", "categorical_env"])

# 3. Parse arguments
args = parser.parse_args()
if ":" in args.seeds:
    start = args.seeds.split(":")[0]
    end = args.seeds.split(":")[-1]
    run_seeds = [int(seed) for seed in range(int(start), int(end)+1)]
else:
    run_seeds = args.seeds.split(",")
    run_seeds = [int(seed) for seed in run_seeds]
                                                                                       
output_dir = args.output_dir

if args.experiment == "default":
    c_settings = [(3,5), (5,5), (8,5), (10,5), (5,3), (5,10), (5,20), (5,50)]
    n_settings = [50, 100, 500, 1000, 5000]

    rows = []
    for q, m in c_settings:
        for n in n_settings:
            for p_between_cluster, p_within_var, p_between_var, density_name in [(3 / q, 0.9, 1 / m, "Cluster"),
                                                                                (0.8, 0.8, 0.8, "Dense"),
                                                                                (3 / q, 3 / m, 1 / m, "Sparse")]:
                for seed in run_seeds:
                    print(q, m, n, density_name, seed)
                    t0 = time.time()
                    row = single_run(q + 2, m, n, p_between_cluster, p_within_var, p_between_var, seed=seed)
                    elapsed = time.time() - t0
                    row.update({"q": q, "m": m, "n": n, "Setting": density_name, "seed": seed, "Time": elapsed})
                    rows.append(row)

    df = pd.DataFrame(rows)

    df.to_csv(f"{output_dir}/cluster_icp_exp_{args.seeds}.csv")

elif args.experiment == "test_stat":
    c_settings = [(5,5)]
    alpha_settings = [0.001, 0.005, 0.01, 0.05, 0.1]
    test_settings = ["Wilks' lambda", "Pillai's trace", "Hotelling-Lawley trace", "Roy's greatest root"]
    n_settings = [50, 100, 500, 1000, 5000]

    rows = []
    for q, m in c_settings:
        for n in n_settings:
            for alpha in alpha_settings:
                for test_statistic in test_settings:
                    for p_between_cluster, p_within_var, p_between_var, density_name in [ (3 / q, 0.9, 1 / m, "Cluster")]:
                        for seed in run_seeds:
                            print(q, m, n, alpha, test_statistic)
                            row = single_run(q + 2, m, n, p_between_cluster, p_within_var, p_between_var, seed=seed, alpha = alpha, test_statistic=test_statistic)
                            row.update({"q": q, "m": m, "n": n, "Setting": density_name, "Alpha": alpha, "Test Statistic": test_statistic, "seed": seed})
                            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(f"{output_dir}/test_stat_{args.seeds}.csv")

elif args.experiment == "group_lasso":
    c_settings = [(10, 5), (20,5), (50, 5)]
    n_settings = [500]
    k_settings = [1, 2, 4]
    s_settings = [1, 2, 4, 8]


    rows = []
    for q, m in c_settings:
        for n in n_settings:
            for k in k_settings:
                for s in s_settings:
                    for p_between_cluster, p_within_var, p_between_var, density_name in [ (3 / q, 0.9, 1 / m, "Cluster")]:
                        for seed in run_seeds:
                            print(q, m, n, k, s, density_name, seed)
                            t0 = time.time()
                            row = single_run(q + 2, m, n, p_between_cluster, p_within_var, p_between_var, k=k, s=s, seed=seed)
                            elapsed = time.time() - t0
                            row.update({"q": q, "m": m, "n": n, "k": k, "s": s, "Setting": density_name, "seed": seed, "Time": elapsed})
                            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(f"{output_dir}/group_lasso_{args.seeds}.csv")

elif args.experiment == "dependent_env":
    c_settings = [(5,3), (5,5), (5,10), (5,20)]
    n_settings = [50, 100, 500, 1000, 5000]

    rows = []
    for q, m in c_settings:
        for n in n_settings:
            for p_between_cluster, p_within_var, p_between_var, density_name in [ (3 / q, 0.9, 1 / m, "Cluster")]:
                for seed in run_seeds:
                    print(q, m, n, seed)
                    row = single_run(q + 2, m, n, p_between_cluster, p_within_var, p_between_var, seed=seed, env_mode="dependent")
                    row.update({"q": q, "m": m, "n": n, "Setting": density_name, "seed": seed})
                    rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(f"{output_dir}/dependent_env_{args.seeds}.csv")

elif args.experiment == "categorical_env":
    c_settings = [(5,5)]
    n_settings = [50, 100, 500, 1000, 5000]
    num_levels_settings = [2, 5, 10]

    rows = []
    for q, m in c_settings:
        for n in n_settings:
            for num_levels in num_levels_settings:
                for p_between_cluster, p_within_var, p_between_var, density_name in [ (3 / q, 0.9, 1 / m, "Cluster")]:
                    for seed in run_seeds:
                        print(q, m, n, num_levels, seed)
                        row = single_run(q + 2, m, n, p_between_cluster, p_within_var, p_between_var, seed=seed, env_mode="categorical", num_levels =num_levels )
                        row.update({"q": q, "m": m, "n": n, "Setting": density_name, "seed": seed, "num_levels": num_levels})
                        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(f"{output_dir}/categorical_env_{args.seeds}.csv")
