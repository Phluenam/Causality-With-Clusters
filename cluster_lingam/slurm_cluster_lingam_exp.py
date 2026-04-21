import itertools
import numpy as np
import networkx as nx
from scipy import stats
from sklearn.utils import check_array
from sklearn.decomposition import FastICA
from scipy.optimize import linear_sum_assignment
from sklearn import linear_model
from sklearn.linear_model import LassoLarsIC, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_array, check_scalar
from sklearn.linear_model._logistic import _logistic_regression_path
import pandas as pd

from lingam.group_direct_lingam import GroupDirectLiNGAM

import igraph as ig
import time

import gadjid

import matplotlib.pyplot as plt
import math


# -----------------------------
# 1. Generate cluster DAG
# -----------------------------
def generate_cluster_dag(q, p_between=0.3, seed=None):
    """Generate a random DAG on q clusters efficiently (topological order)."""
    rng = np.random.default_rng(seed)
    order = list(range(q))
    rng.shuffle(order)
    Gc = nx.DiGraph()
    Gc.add_nodes_from(range(q))
    for i, u in enumerate(order):
        for v in order[i+1:]:
            if rng.random() < p_between:
                Gc.add_edge(u, v)
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

def get_noise(rng, n, noise_type="normal_exp"):

    assert noise_type in ["normal_exp", "uniform", "exponential", "laplace", "triangular"]

    if noise_type == "normal_exp":
        normal_noise = rng.normal(size=n)
        noise_exp_range = rng.choice([[0.5, 0.8], [1.2, 2.0]])
        noise_exp = rng.uniform(noise_exp_range[0], noise_exp_range[1])
        noise_abs = np.pow(np.abs(normal_noise), noise_exp)
        noise_sign = np.sign(normal_noise)
        noise = noise_sign * noise_abs

    if noise_type == "uniform":
        noise = rng.uniform(low=-1.0, high=1.0, size=n)

    if noise_type == "exponential":
        noise = rng.exponential(size=n)

    if noise_type == "laplace":
        noise = rng.laplace(size=n)

    if noise_type == "triangular":
        noise = rng.triangular(left=-1.0, mode=0.0, right=1.0, size=n)

    return noise - np.mean(noise)

# -----------------------------
# 3. Simulate linear nongaussian SCM with constant variance. Fraction of variance from noise controlled by noise_var_frac_range
# -----------------------------
def simulate_nongaussian_const_var(G, n=500, coef_range=(0.5,1.5), noise_var_frac_range=(0.2,0.8), seed=None, noise_type="normal_exp"):
    rng = np.random.default_rng(seed)
    X = {}
    var2idx = {v: i for i, v in enumerate(G.nodes)}
    X_mat = np.zeros((n, len(G.nodes)))
    B_mat = np.zeros((len(G.nodes), len(G.nodes)))

    topo_order = list(nx.topological_sort(G))
    for v in topo_order:
        parents = list(G.predecessors(v))

        noise = get_noise(rng, n, noise_type)

        if parents:
            # choose noise fraction
            noise_var = rng.uniform(*noise_var_frac_range)

            # scale noise to desired variance
            noise = noise / np.std(noise)
            noise *= np.sqrt(noise_var)

            # sample coefficients and random signs
            coefs = rng.uniform(*coef_range, size=len(parents))
            signs = rng.choice([-1, 1], size=len(parents))

            # build parent signal
            parent_signal = sum(
                signs[i] * coefs[i] * X_mat[:, var2idx[parents[i]]]
                for i in range(len(parents))
            )

            # scale parent contribution to desired variance
            parent_std = np.std(parent_signal)
            if parent_std > 0:
                parent_signal *= np.sqrt(1 - noise_var) / parent_std

            # store final coefficients AFTER scaling
            for i, parent in enumerate(parents):
                B_mat[var2idx[v], var2idx[parent]] = (
                    signs[i] * coefs[i] * np.sqrt(1 - noise_var) / parent_std
                )

            val = parent_signal + noise
        else:
            noise = noise / np.std(noise)
            val = noise
        X_mat[:, var2idx[v]] = val
    return X_mat, var2idx, B_mat

# Group dependent errors
def simulate_nongaussian_const_var_groups(G, groups, n=500, coef_range=(0.5,1.5),
                                           noise_var_frac_range=(0.2,0.8),
                                           group_var_frac_range=(0.3,0.7),
                                           group_factor_coef_range=(0.5, 1.5),
                                           seed=None, noise_type="normal_exp"):
    rng = np.random.default_rng(seed)
    var2idx = {v: i for i, v in enumerate(G.nodes)}
    X_mat = np.zeros((n, len(G.nodes)))
    B_mat = np.zeros((len(G.nodes), len(G.nodes)))


    # --- For each group, sample k latent variables (k = group size)
    #     and assign each node a linear combination of them as its group factor ---
    node2group_factor = {}
    for g in groups:
        k = len(g)
        # k latent nongaussian variables for this group
        latents = np.stack([get_noise(rng, n, noise_type) for _ in range(k)], axis=1)  # (n, k)

        for v in g:
            # each node gets its own random linear combination of the k latents
            coefs = rng.uniform(*group_factor_coef_range, size=k)
            signs = rng.choice([-1, 1], size=k)
            factor = latents @ (coefs * signs)  # (n,)
            node2group_factor[v] = factor / np.std(factor)

    # --- Simulate SCM ---
    topo_order = list(nx.topological_sort(G))
    for v in topo_order:
        parents = list(G.predecessors(v))

        idio_noise = get_noise(rng, n, noise_type)
        group_factor = node2group_factor[v]

        if parents:
            noise_var = rng.uniform(*noise_var_frac_range)
            group_frac = rng.uniform(*group_var_frac_range)

            noise = (np.sqrt(group_frac) * group_factor +
                     np.sqrt(1 - group_frac) * idio_noise)
            noise = noise / np.std(noise)
            noise *= np.sqrt(noise_var)

            coefs = rng.uniform(*coef_range, size=len(parents))
            signs = rng.choice([-1, 1], size=len(parents))
            parent_signal = sum(
                signs[i] * coefs[i] * X_mat[:, var2idx[parents[i]]]
                for i in range(len(parents))
            )
            parent_std = np.std(parent_signal)
            if parent_std > 0:
                parent_signal *= np.sqrt(1 - noise_var) / parent_std

            for i, parent in enumerate(parents):
                B_mat[var2idx[v], var2idx[parent]] = (
                    signs[i] * coefs[i] * np.sqrt(1 - noise_var) / parent_std
                )

            val = parent_signal + noise
        else:
            group_frac = rng.uniform(*group_var_frac_range)
            noise = (np.sqrt(group_frac) * group_factor +
                     np.sqrt(1 - group_frac) * idio_noise)
            val = noise / np.std(noise)

        X_mat[:, var2idx[v]] = val

    return X_mat, var2idx, B_mat


def predict_adaptive_lasso(X, X_std, predictors, target, gamma=1.0):
    """Predict with Adaptive Lasso.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Training data, where n_samples is the number of samples
        and n_features is the number of features.
    predictors : array-like, shape (n_predictors)
        Indices of predictor variable.
    target : int
        Index of target variable.

    Returns
    -------
    coef : array-like, shape (n_features)
        Coefficients of predictor variable.
    """
    Xp = X_std[:, predictors]
    y = X_std[:, target]

    # Pruning with Adaptive Lasso
    lr = LinearRegression()
    lr.fit(Xp, y)
    weight = np.power(np.abs(lr.coef_), gamma)
    reg = LassoLarsIC(criterion="bic")
    reg.fit(Xp * weight, y)
    pruned_idx = np.abs(reg.coef_ * weight) > 0.0

    # Calculate coefficients of the original scale
    coef = np.zeros(reg.coef_.shape)
    if pruned_idx.sum() > 0:
        lr = LinearRegression()
        pred = np.array(predictors)
        lr.fit(X[:, pred[pruned_idx]], X[:, target])
        coef[pruned_idx] = lr.coef_

    return coef


def estimate_adjacency_matrix(X, groups, causal_order):
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    B = np.zeros([X.shape[1], X.shape[1]], dtype="float64")
    for i in range(1, len(causal_order)):
        target_group_idx = causal_order[i]
        predictor_groups_idxs = causal_order[:i]

        # target is exogenous groups if predictors are empty
        if len(predictor_groups_idxs) == 0:
            continue

        # Retrieve variables from groups
        target_group = groups[target_group_idx]
        predictor_groups = [groups[idx] for idx in predictor_groups_idxs]
        predictors = [var for group in predictor_groups for var in group]

        for target in target_group:
            B[target, predictors] = predict_adaptive_lasso(X, X_std, predictors, target)

    return B

def cluster_mat(groups, B_mat):
    cluster_adj = np.zeros((len(groups), len(groups)))
    for key1, g1 in enumerate(groups):
        for key2, g2 in enumerate(groups):
            if key1 == key2:
                continue
            for u in g1:
                for v in g2:
                    cluster_adj[key1, key2] += B_mat[u, v]
    return cluster_adj

def cluster_adj(groups, B_mat):
    return (cluster_mat(groups, np.abs(B_mat)) > 0).astype(int)

def cluster_cost(groups, B_mat):
    return cluster_mat(groups, np.pow(B_mat, 2))

def get_groups(cluster_vars, var2idx):
    groups = []
    for k in sorted(cluster_vars.keys()):
        current_group = []
        for v in cluster_vars[k]:
            current_group.append(var2idx[v])
        groups.append(current_group)
    return groups

def get_B_estimate(X, seed, w_init=None):
    X = check_array(X)

    # obtain a unmixing matrix from the given data
    ica = FastICA(max_iter=2000, random_state=seed, tol=1e-3, whiten_solver='svd', w_init=w_init)
    ica.fit(X)
    W_ica = ica.components_

    # obtain a permuted W_ica
    _, col_index = linear_sum_assignment(1 / np.abs(W_ica))
    PW_ica = np.zeros_like(W_ica)
    PW_ica[col_index] = W_ica

    # obtain a vector to scale
    D = np.diag(PW_ica)[:, np.newaxis]

    # estimate an adjacency matrix
    W_estimate = PW_ica / D
    B_estimate = np.eye(len(W_estimate)) - W_estimate
    return B_estimate

def compute_cost(matrix, causal_order):
    cost = 0.0
    for i in range(len(causal_order)):
        for j in range(i + 1, len(causal_order)):
            cost += matrix[causal_order[i], causal_order[j]]

    return cost

def compute_violations(matrix, causal_order):
    mistakes = 0
    denom = 0
    for i in range(len(causal_order)):
        for j in range(i + 1, len(causal_order)):
            if matrix[causal_order[i], causal_order[j]] != 0:
                mistakes += 1
                denom += 1
            if matrix[causal_order[j], causal_order[i]] != 0:
                denom += 1

    if denom == 0:
        print("matrix size ", len(matrix), " has no edges")
        print(matrix)
        return 0, 0
    return mistakes / denom, mistakes

def compute_reachability_violations(matrix, causal_order):
    M = (matrix != 0)

    while True:
        new = M | (M @ M)
        if np.array_equal(new, M):
            break
        M = new

    matrix = M.astype(int)
    return compute_violations(matrix, causal_order)


def search_causal_order(matrix):
    causal_order = []

    row_num = matrix.shape[0]
    original_index = np.arange(row_num)

    while 0 < len(matrix):
        # find a row all of which elements are zero
        row_index_list = np.where(np.sum(np.abs(matrix), axis=1) == 0)[0]
        if len(row_index_list) == 0:
            break

        target_index = row_index_list[0]

        # append i to the end of the list
        causal_order.append(original_index[target_index])
        original_index = np.delete(original_index, target_index, axis=0)

        # remove the i-th row and the i-th column from matrix
        mask = np.delete(np.arange(len(matrix)), target_index, axis=0)
        matrix = matrix[mask][:, mask]

    if len(causal_order) != row_num:
        causal_order = None

    return causal_order

def estimate_causal_order(matrix):
    causal_order = None
    initial_matrix = matrix.copy()
    matrix = matrix.copy()

    # set the m(m + 1)/2 smallest(in absolute value) elements of the matrix to zero
    pos_list = np.argsort(np.abs(matrix), axis=None)
    pos_list = np.vstack(np.unravel_index(pos_list, matrix.shape)).T

    initial_zero_num = int(matrix.shape[0] * (matrix.shape[0] + 1) / 2)
    for i, j in pos_list[:initial_zero_num]:
        matrix[i, j] = 0

    for i, j in pos_list[initial_zero_num:]:
        causal_order = search_causal_order(matrix)
        if causal_order is not None:
            break
        else:
            # set the smallest(in absolute value) element to zero
            matrix[i, j] = 0

    return causal_order

def igraph_estimate_causal_order(matrix, method):
    causal_order = None
    matrix = matrix.copy()

    g = ig.Graph.Weighted_Adjacency(
        matrix.tolist(),
        mode="directed",
        attr="weight",
        loops=False
    )
    g_weight = ig.Graph.Weighted_Adjacency(
        (matrix).tolist(),
        mode="directed",
        attr="weight",
        loops=False
    )
    fas = g_weight.feedback_arc_set(weights="weight", method=method)
    edges_to_remove = g_weight.es[fas]

    g_dag = g.copy()
    g_dag.delete_edges(fas)

    causal_order = g_dag.topological_sorting()[::-1]

    return causal_order

def heuristic_estimate_causal_order(matrix):
    return igraph_estimate_causal_order(matrix, method="eades")

def exact_estimate_causal_order(matrix):
    return igraph_estimate_causal_order(matrix, method="exact")

class bitset_cycle_detection:
    def __init__(self, n):
        self.n = n
        # Use a boolean matrix (uses 1 byte per entry, but very fast with vectorized OR)
        self.reach = np.eye(n, dtype=bool)

    def can_add_edge(self, u, v):
        # Is there a path from v to u?
        return not self.reach[v, u]

    def add_edge(self, u, v):
        if not self.can_add_edge(u,v):
            return False
        if self.reach[u, v]:
            return True

        # Rows where reach[x, u] is True need to be updated
        sources = self.reach[:, u]
        # Update those rows by ORing them with the reachable nodes from v
        self.reach[sources, :] |= self.reach[v, :]
        return True

def cycle_estimate_causal_order(matrix):
    causal_order = None
    matrix = matrix.copy()

    pos_list = np.argsort(np.abs(matrix), axis=None)
    pos_list = np.vstack(np.unravel_index(pos_list, matrix.shape)).T

    dag_matrix = np.zeros_like(matrix)

    accepted_edges = []

    C = bitset_cycle_detection(n=matrix.shape[0])
    weights = []

    for i, j in pos_list[::-1]:

        # check if adding i→j would create a cycle
        if C.add_edge(i,j):
            accepted_edges.append((i, j, matrix[i, j]))

    g = ig.Graph(n=matrix.shape[0], directed=True)
    if accepted_edges:
        # Unpack edges and weights
        es_from, es_to, es_weights = zip(*accepted_edges)
        g.add_edges(zip(es_from, es_to))
        g.es["weight"] = es_weights

    causal_order = g.topological_sorting()[::-1]

    return causal_order

def group_direct_lingam_causal_order(X, groups):
    model = GroupDirectLiNGAM()
    model.fit(X, groups)
    return model.causal_order_

def iterative_ica_lingam_causal_order_groups(X, groups):
    X = X.copy()
    groups = [list(g) for g in groups]  # defensive copy
    active_group_indices = list(range(len(groups)))
    causal_order_groups = []
    prev_B_hat = None

    for _ in range(len(groups) - 1):
        iter_B_hat = get_B_estimate(X, seed=seed, w_init=prev_B_hat)

        cost_mat = cluster_cost(groups, iter_B_hat)

        if len(groups) <= 20:
            group_order = exact_estimate_causal_order(cost_mat)
        else:
            group_order = cycle_estimate_causal_order(cost_mat)

        target_idx = group_order[-1]
        causal_order_groups.append(active_group_indices[target_idx])

        # Drop target group's columns from X
        target_cols = sorted(groups[target_idx], reverse=True)
        mask = np.ones(X.shape[1], dtype=bool)
        for col in target_cols:
            mask[col] = False
        X = X[:, mask]

        # Update prev_B_hat
        keep_cols = np.where(mask)[0]
        prev_B_hat = iter_B_hat[np.ix_(keep_cols, keep_cols)]

        # Remap column indices in remaining groups
        old_to_new = {old: new for new, old in enumerate(keep_cols)}
        groups.pop(target_idx)
        active_group_indices.pop(target_idx)
        groups = [[old_to_new[c] for c in g] for g in groups]

    causal_order_groups.append(active_group_indices[0])
    return causal_order_groups[::-1]

def get_data(q, m, n, p_cluster, p_var_within_cluster, p_var_between_cluster, seed, noise_type="normal_exp"):
    Gc = generate_cluster_dag(q, p_between=p_cluster, seed=seed)
    G, cluster_vars = generate_variable_dag(Gc, m, p_within=p_var_within_cluster, p_between=p_var_between_cluster, seed=seed)
    X, var2idx, B_mat = simulate_nongaussian_const_var(G, n=n, seed=seed, noise_type=noise_type)
    groups = get_groups(cluster_vars, var2idx)

    start = time.time()
    B_hat = get_B_estimate(X, seed=seed)
    end = time.time()

    ica_time = end - start

    return X, groups, B_mat, B_hat, ica_time

def get_group_dependent_data(q, m, n, p_cluster, p_var_within_cluster, p_var_between_cluster, seed, group_var_frac_range=(0.5,0.5), noise_type="normal_exp"):
    Gc = generate_cluster_dag(q, p_between=p_cluster, seed=seed)
    G, cluster_vars = generate_variable_dag(Gc, m, p_within=p_var_within_cluster, p_between=p_var_between_cluster, seed=seed)
    X, var2idx, B_mat = simulate_nongaussian_const_var_groups(G, [cluster_vars[k] for k in sorted(cluster_vars.keys())], n=n, seed=seed, group_var_frac_range=group_var_frac_range, noise_type=noise_type)
    groups = get_groups(cluster_vars, var2idx)

    start = time.time()
    B_hat = get_B_estimate(X, seed=seed)
    end = time.time()

    ica_time = end - start

    return X, groups, B_mat, B_hat, ica_time

import argparse
import random
import numpy as np

# 1. Create parser
parser = argparse.ArgumentParser(description="Run experiment with a seed")

# 2. Add seed argument
parser.add_argument("-s", "--seeds", type=str, default="0", help="Seeds to run")
parser.add_argument("-o", "--output-dir", type=str, default="csvs", help="Where to output")
parser.add_argument("-e", "--experiment", type=str, default="default", help="experiment to run", choices=["default", "group_dependent_noises", "noise_type"])

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
    qs = []
    ms = []
    ns = []
    settings = []
    seeds = []

    algs = []
    times = []
    causal_order_times = []
    costs = []

    violations = []
    reachability_violations = []
    sids = []
    shds = []

    for q in [3, 5, 10]:
        for m in [1, 3, 5, 10]:#[3, 5, 8, 10, 15, 20, 50]:
            for n in [500, 1000, 5000]:#, 5000]:
                for p_cluster, p_var_within_cluster, p_var_between_cluster, density_name in [ (3 / q, 0.8, 1 / m, "Cluster")]: #(3 / p, "Sparse"),
                    for seed in run_seeds:
                        print(q, m, n, seed)
                        np.random.seed(seed)

                        X, groups, B_mat, B_hat, ica_time = get_data(q, m, n, p_cluster, p_var_within_cluster, p_var_between_cluster, seed)

                        for alg, name in [(estimate_causal_order, "Current"), (exact_estimate_causal_order, "Exact"), (cycle_estimate_causal_order, "Greedy"),
                                        (group_direct_lingam_causal_order, "GroupDirectLiNGAM"), (iterative_ica_lingam_causal_order_groups, "IterativeClusterICALiNGAM")]: # (heuristic_estimate_causal_order, "ELS"),
                            if name == "Current" and q > 300:
                                continue
                            if name == "Exact" and q > 20:
                                continue
                            if name == "GroupDirectLiNGAM" and q > 10:
                                continue

                            start = time.time()
                            if name in ["GroupDirectLiNGAM", "IterativeClusterICALiNGAM"]:
                                causal_order = alg(X, groups)
                            else:
                                B_hat_group = cluster_cost(groups, B_hat)
                                causal_order = alg(B_hat_group)
                            end = time.time()

                            time_used = end - start

                            if name not in ["GroupDirectLiNGAM", "IterativeClusterICALiNGAM"]:
                                time_used += ica_time

                            B_hat_group = cluster_cost(groups, B_hat)
                            cost = compute_cost(B_hat_group, causal_order)

                            start = time.time()
                            B_guess = estimate_adjacency_matrix(X, groups, causal_order)
                            end = time.time()
                            estimate_time = end - start

                            if name not in ["GroupDirectLiNGAM"]:
                                time_used += estimate_time

                            B_adj = cluster_adj(groups, B_mat).astype(np.int8)
                            B_guess_adj = cluster_adj(groups, B_guess).astype(np.int8)

                            sid, _ = gadjid.sid(B_adj, B_guess_adj, edge_direction="from column to row")
                            shd, _ = gadjid.shd(B_adj, B_guess_adj)

                            violation, _ = compute_violations(B_adj, causal_order)
                            reachability_violation, _ = compute_reachability_violations(B_adj, causal_order)

                            causal_order_time = time_used - estimate_time - ica_time
                            if name in ["IterativeICALiNGAM", "Direct_pwling", "Direct_kernel"]:
                                causal_order_time = math.nan

                            qs.append(q)
                            ms.append(m)
                            ns.append(n)
                            settings.append(density_name)
                            seeds.append(seed)

                            algs.append(name)
                            times.append(time_used)
                            causal_order_times.append(causal_order_time)
                            costs.append(cost)

                            violations.append(violation)
                            reachability_violations.append(reachability_violation)
                            sids.append(sid)
                            shds.append(shd)


    df = pd.DataFrame({"q":qs, "m": ms, "n": ns, "Seed": seeds, "Setting": settings, "Algorithm": algs, "Time": times, "Causal Order Time": causal_order_times, "Cost": costs, "Causal Order Edge Violations": violations, "Causal Order Reachability Violations": reachability_violations, "SID": sids, "SHD": shds})

    df.to_csv(f"{output_dir}/cluster_lingam_exp_{args.seeds}.csv")

if args.experiment == "group_dependent_noises":
    qs = []
    ms = []
    ns = []
    group_var_fracs = []
    settings = []
    seeds = []

    algs = []
    causal_order_times = []
    times = []
    costs = []

    violations = []
    reachability_violations = []
    sids = []
    shds = []

    for q in [5]:
        for m in [5]:#[3, 5, 8, 10, 15, 20, 50]:
            for n in [500, 1000, 5000]:#, 5000]:
                for p_cluster, p_var_within_cluster, p_var_between_cluster, density_name in [ (3 / q, 0.9, 1 / m, "Cluster")]: #(3 / p, "Sparse"),
                    for group_var_frac in [0, 0.25, 0.5, 0.75, 1]:
                        for seed in run_seeds:
                            print(q, m, n, group_var_frac, seed)
                            np.random.seed(seed)

                            X, groups, B_mat, B_hat, ica_time = get_group_dependent_data(q, m, n, p_cluster, p_var_within_cluster, p_var_between_cluster, seed, group_var_frac_range=(group_var_frac,group_var_frac))

                            for alg, name in [(estimate_causal_order, "Current"), (exact_estimate_causal_order, "Exact"), (cycle_estimate_causal_order, "Greedy"),
                                            (group_direct_lingam_causal_order, "GroupDirectLiNGAM"), (iterative_ica_lingam_causal_order_groups, "IterativeClusterICALiNGAM")]: # (heuristic_estimate_causal_order, "ELS"),
                                if name == "Current" and q > 300:
                                    continue
                                if name == "Exact" and q > 20:
                                    continue
                                if name == "GroupDirectLiNGAM" and q > 10:
                                    continue

                                start = time.time()
                                if name in ["GroupDirectLiNGAM", "IterativeClusterICALiNGAM"]:
                                    causal_order = alg(X, groups)
                                else:
                                    B_hat_group = cluster_cost(groups, B_hat)
                                    causal_order = alg(B_hat_group)
                                end = time.time()

                                time_used = end - start

                                if name not in ["GroupDirectLiNGAM", "IterativeClusterICALiNGAM"]:
                                    time_used += ica_time

                                B_hat_group = cluster_cost(groups, B_hat)
                                cost = compute_cost(B_hat_group, causal_order)

                                start = time.time()
                                B_guess = estimate_adjacency_matrix(X, groups, causal_order)
                                end = time.time()
                                estimate_time = end - start

                                if name not in ["GroupDirectLiNGAM"]:
                                    time_used += estimate_time

                                B_adj = cluster_adj(groups, B_mat).astype(np.int8)
                                B_guess_adj = cluster_adj(groups, B_guess).astype(np.int8)

                                sid, _ = gadjid.sid(B_adj, B_guess_adj, edge_direction="from column to row")
                                shd, _ = gadjid.shd(B_adj, B_guess_adj)

                                violation, _ = compute_violations(B_adj, causal_order)
                                reachability_violation, _ = compute_reachability_violations(B_adj, causal_order)

                                causal_order_time = time_used - estimate_time - ica_time
                                if name in ["IterativeICALiNGAM", "Direct_pwling", "Direct_kernel"]:
                                    causal_order_time = math.nan

                                qs.append(q)
                                ms.append(m)
                                ns.append(n)
                                group_var_fracs.append(group_var_frac)
                                settings.append(density_name)
                                seeds.append(seed)

                                algs.append(name)
                                times.append(time_used)
                                causal_order_times.append(causal_order_time)
                                costs.append(cost)

                                violations.append(violation)
                                reachability_violations.append(reachability_violation)
                                sids.append(sid)
                                shds.append(shd)


    df = pd.DataFrame({"q":qs, "m": ms, "n": ns, "Seed": seeds, "Setting": settings, "Group Variance Fraction": group_var_fracs, "Algorithm": algs, "Time": times, "Causal Order Time": causal_order_times, "Cost": costs, "Causal Order Edge Violations": violations, "Causal Order Reachability Violations": reachability_violations, "SID": sids, "SHD": shds})

    df.to_csv(f"{output_dir}/group_dependent_noise_cluster_lingam_exp_{args.seeds}.csv")


if args.experiment == "noise_type":
    qs = []
    ms = []
    ns = []
    noise_types = []
    settings = []
    seeds = []

    algs = []
    causal_order_times = []
    times = []
    costs = []

    violations = []
    reachability_violations = []
    sids = []
    shds = []

    for q in [5]:
        for m in [5]:#[3, 5, 8, 10, 15, 20, 50]:
            for n in [500, 1000, 5000]:#, 5000]:
                for p_cluster, p_var_within_cluster, p_var_between_cluster, density_name in [ (3 / q, 0.9, 1 / m, "Cluster")]: #(3 / p, "Sparse"),
                    for noise_type in ["normal_exp", "uniform", "triangular", "laplace", "exponential"]:
                        for seed in run_seeds:
                            print(q, m, n, noise_type, seed)
                            np.random.seed(seed)

                            X, groups, B_mat, B_hat, ica_time = get_data(q, m, n, p_cluster, p_var_within_cluster, p_var_between_cluster, seed, noise_type=noise_type)

                            for alg, name in [(estimate_causal_order, "Current"), (exact_estimate_causal_order, "Exact"), (cycle_estimate_causal_order, "Greedy"),
                                            (group_direct_lingam_causal_order, "GroupDirectLiNGAM"), (iterative_ica_lingam_causal_order_groups, "IterativeClusterICALiNGAM")]: # (heuristic_estimate_causal_order, "ELS"),
                                if name == "Current" and q > 300:
                                    continue
                                if name == "Exact" and q > 20:
                                    continue
                                if name == "GroupDirectLiNGAM" and q > 10:
                                    continue

                                start = time.time()
                                if name in ["GroupDirectLiNGAM", "IterativeClusterICALiNGAM"]:
                                    causal_order = alg(X, groups)
                                else:
                                    B_hat_group = cluster_cost(groups, B_hat)
                                    causal_order = alg(B_hat_group)
                                end = time.time()

                                time_used = end - start

                                if name not in ["GroupDirectLiNGAM", "IterativeClusterICALiNGAM"]:
                                    time_used += ica_time

                                B_hat_group = cluster_cost(groups, B_hat)
                                cost = compute_cost(B_hat_group, causal_order)

                                start = time.time()
                                B_guess = estimate_adjacency_matrix(X, groups, causal_order)
                                end = time.time()
                                estimate_time = end - start

                                if name not in ["GroupDirectLiNGAM"]:
                                    time_used += estimate_time

                                B_adj = cluster_adj(groups, B_mat).astype(np.int8)
                                B_guess_adj = cluster_adj(groups, B_guess).astype(np.int8)

                                sid, _ = gadjid.sid(B_adj, B_guess_adj, edge_direction="from column to row")
                                shd, _ = gadjid.shd(B_adj, B_guess_adj)

                                violation, _ = compute_violations(B_adj, causal_order)
                                reachability_violation, _ = compute_reachability_violations(B_adj, causal_order)

                                causal_order_time = time_used - estimate_time - ica_time
                                if name in ["IterativeICALiNGAM", "Direct_pwling", "Direct_kernel"]:
                                    causal_order_time = math.nan

                                qs.append(q)
                                ms.append(m)
                                ns.append(n)
                                noise_types.append(noise_type)
                                settings.append(density_name)
                                seeds.append(seed)

                                algs.append(name)
                                times.append(time_used)
                                causal_order_times.append(causal_order_time)
                                costs.append(cost)

                                violations.append(violation)
                                reachability_violations.append(reachability_violation)
                                sids.append(sid)
                                shds.append(shd)


    df = pd.DataFrame({"q":qs, "m": ms, "n": ns, "Seed": seeds, "Setting": settings, "Noise Type": noise_types, "Algorithm": algs, "Time": times, "Causal Order Time": causal_order_times, "Cost": costs, "Causal Order Edge Violations": violations, "Causal Order Reachability Violations": reachability_violations, "SID": sids, "SHD": shds})

    df.to_csv(f"{output_dir}/noise_type_cluster_lingam_exp_{args.seeds}.csv")
