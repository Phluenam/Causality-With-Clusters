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

from lingam.direct_lingam import DirectLiNGAM
import igraph as ig
import time

import gadjid

import numpy as np
import networkx as nx
import itertools
from scipy import stats
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

# -----------------------------
# 3. Simulate linear Uniform SCM
# -----------------------------
def simulate_linear_uniform(G, n=500, coef_low=0.5, coef_high=1.5, noise_max=1.0, seed=None):
    rng = np.random.default_rng(seed)
    X = {}
    var2idx = {v: i for i, v in enumerate(G.nodes)}
    X_mat = np.zeros((n, len(G.nodes)))
    B_mat = np.zeros((len(G.nodes), len(G.nodes)))
    
    topo_order = list(nx.topological_sort(G))
    for v in topo_order:
        parents = list(G.predecessors(v))
        noise = rng.uniform(-noise_max, noise_max, size=n)
        if parents:
            coefs = rng.uniform(coef_low, coef_high, size=len(parents))
            signs = rng.choice([-1,1], size=len(parents))
            for coef, sign ,parent in zip(coefs, signs, parents):
                B_mat[var2idx[v], var2idx[parent]] = coef * sign
            val = sum(signs[i]*coefs[i]*X_mat[:, var2idx[parents[i]]] for i in range(len(parents))) + noise
        else:
            val = noise
        X_mat[:, var2idx[v]] = val
    return X_mat, var2idx, B_mat

# -----------------------------
# 3. Simulate linear Uniform SCM with constant variance. Fraction of variance from noise controlled by noise_var_frac_range
# -----------------------------
def simulate_linear_const_var(G, n=500, coef_range=(0.5,1.5), noise_var_frac_range=(0.2,0.8), seed=None):
    rng = np.random.default_rng(seed)
    X = {}
    var2idx = {v: i for i, v in enumerate(G.nodes)}
    X_mat = np.zeros((n, len(G.nodes)))
    B_mat = np.zeros((len(G.nodes), len(G.nodes)))
    
    topo_order = list(nx.topological_sort(G))
    for v in topo_order:
        parents = list(G.predecessors(v))
        
        if parents:
            noise_var = rng.uniform(noise_var_frac_range[0], noise_var_frac_range[1])
            noise_max = np.sqrt(3) * np.sqrt(noise_var)
            noise = rng.uniform(-noise_max, noise_max, size=n)
        
            coefs = rng.uniform(coef_range[0], coef_range[1], size=len(parents))
            coefs = np.sqrt((1-noise_var)/np.sum(coefs * coefs)) * coefs
            signs = rng.choice([-1,1], size=len(parents))
            for coef, sign ,parent in zip(coefs, signs, parents):
                B_mat[var2idx[v], var2idx[parent]] = coef * sign
            val = sum(signs[i]*coefs[i]*X_mat[:, var2idx[parents[i]]] for i in range(len(parents))) + noise
        else:
            noise_max = np.sqrt(3) * np.sqrt(1)
            noise = rng.uniform(-noise_max, noise_max, size=n)
            val = noise
        X_mat[:, var2idx[v]] = val
    return X_mat, var2idx, B_mat

# -----------------------------
# 3. Simulate linear nongaussian SCM with constant variance. Fraction of variance from noise controlled by noise_var_frac_range
# -----------------------------
def simulate_nongaussian_const_var(G, n=500, coef_range=(0.5,1.5), noise_var_frac_range=(0.2,0.8), seed=None):
    rng = np.random.default_rng(seed)
    X = {}
    var2idx = {v: i for i, v in enumerate(G.nodes)}
    X_mat = np.zeros((n, len(G.nodes)))
    B_mat = np.zeros((len(G.nodes), len(G.nodes)))
    
    topo_order = list(nx.topological_sort(G))
    for v in topo_order:
        parents = list(G.predecessors(v))

        normal_noise = rng.normal(size=n)
        noise_exp_range = rng.choice([[0.5, 0.8], [1.2, 2.0]])
        noise_exp = rng.uniform(noise_exp_range[0], noise_exp_range[1])
        noise_abs = np.pow(np.abs(normal_noise), noise_exp)
        noise_sign = np.sign(normal_noise)
        noise = noise_sign * noise_abs
        
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

def estimate_adjacency_matrix(X, causal_order):
    """Estimate adjacency matrix by causal order.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Training data, where n_samples is the number of samples
        and n_features is the number of features.
    prior_knowledge : array-like, shape (n_variables, n_variables), optional (default=None)
        Prior knowledge matrix.

    Returns
    -------
    self : object
        Returns the instance itself.
    """
    
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    B = np.zeros([X.shape[1], X.shape[1]], dtype="float64")
    for i in range(1, len(causal_order)):
        target = causal_order[i]
        predictors = causal_order[:i]

        # target is exogenous variables if predictors are empty
        if len(predictors) == 0:
            continue

        B[target, predictors] = predict_adaptive_lasso(X, X_std, predictors, target)

    return  B

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
            cost += matrix[causal_order[i], causal_order[j]] * matrix[causal_order[i], causal_order[j]] 

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

def fast_search_causal_order(matrix):
    causal_order = []

    matrix = (matrix != 0).astype(int)

    row_num = matrix.shape[0]
    original_index = np.arange(row_num)

    row_sums = matrix.sum(axis=1)

    while 0 < len(original_index):
        # find a row all of which elements are zero
        min_row_idx = np.argmin(row_sums)
        
        if row_sums[min_row_idx] > 0:
            break

        target_index = min_row_idx

        # append i to the end of the list
        causal_order.append(original_index[target_index])

        # remove the i-th row and the i-th column from matrix and sums
        for i in range(len(original_index)):
            row_sums[i] -= matrix[original_index[i]][original_index[target_index]]

        mask = np.delete(np.arange(len(original_index)), target_index, axis=0)
        row_sums = row_sums[mask]
        original_index = np.delete(original_index, target_index, axis=0)

    if len(causal_order) != row_num:
        causal_order = None

    return causal_order


def fast_estimate_causal_order(matrix):
    causal_order = None
    matrix = matrix.copy()

    # set the m(m + 1)/2 smallest(in absolute value) elements of the matrix to zero
    pos_list = np.argsort(np.abs(matrix), axis=None)
    pos_list = np.vstack(np.unravel_index(pos_list, matrix.shape)).T
    cost = 0.0

    min_zero_num = int(matrix.shape[0] * (matrix.shape[0] + 1) / 2)
    max_zero_num = int(matrix.shape[0] * matrix.shape[0])

    best_zero_num = max_zero_num
    best_causal_order = None

    while min_zero_num <= max_zero_num:
        zero_num = (min_zero_num + max_zero_num) // 2

        test_matrix = matrix.copy()
        for i, j in pos_list[:zero_num]:
            test_matrix[i, j] = 0
    
        causal_order = fast_search_causal_order(test_matrix)

        if causal_order is not None:
            best_zero_num = min(best_zero_num, zero_num)
            best_causal_order = causal_order
            max_zero_num = zero_num - 1
        else:
            min_zero_num = zero_num + 1

    return best_causal_order

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
        (matrix * matrix).tolist(),
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

def pairwise_lingam_causal_order(X):
    model = DirectLiNGAM()
    model.fit(X)
    return model.causal_order_

def direct_lingam_causal_order(X):
    model = DirectLiNGAM(measure='kernel')
    model.fit(X)
    return model.causal_order_

def iterative_ica_lingam_causal_order(X):
    X = X.copy()
    causal_order = []
    num_var = X.shape[1]
    original_index = np.arange(num_var)

    prev_B_hat = None
    for i in range(X.shape[1] - 1):
        iter_B_hat = get_B_estimate(X, seed=seed, w_init=prev_B_hat)

        if X.shape[1] <=20:
            target_index = exact_estimate_causal_order(iter_B_hat)[-1]
        else:
            target_index = cycle_estimate_causal_order(iter_B_hat)[-1]
        
        causal_order.append(original_index[target_index])
        original_index = np.delete(original_index, target_index, axis=0)
        mask = np.delete(np.arange(X.shape[1]), target_index, axis=0)
        X = X[:, mask]

        prev_B_hat = iter_B_hat[mask][:,mask]

    causal_order.append(original_index[0])
        
    return causal_order[::-1]

def get_data(q, m, n, p_cluster, p_var_within_cluster, p_var_between_cluster, seed):
    Gc = generate_cluster_dag(q, p_between=p_cluster, seed=seed)
    G, cluster_vars = generate_variable_dag(Gc, m, p_within=p_var_within_cluster, p_between=p_var_between_cluster, seed=seed)
    X, var2idx, B_mat = simulate_nongaussian_const_var(G, n=n, seed=seed)

    start = time.time()
    B_hat = get_B_estimate(X, seed=seed)
    end = time.time()

    ica_time = end - start

    return X, B_mat, B_hat, ica_time


import argparse
import random
import numpy as np

# 1. Create parser
parser = argparse.ArgumentParser(description="Run experiment with a seed")

# 2. Add seed argument
parser.add_argument("-s", "--seeds", type=str, default="0", help="Seeds to run")
parser.add_argument("-o", "--output-dir", type=str, default="csvs", help="Where to output")

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
ps = []
qs = []
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


p_var_within_cluster = 1.0
p_var_between_cluster = 1.0
#num_seeds = 1

for p in [3, 5, 8, 10, 20, 50, 75, 100]:
    for n in [500, 750, 1000, 2500, 5000]:
        for p_cluster, density_name in [ (0.8, "Dense"), (3 / p, "Sparse")]:
            for seed in run_seeds:
                print(p, seed, n)
                np.random.seed(seed)
                
                q = p
                m = 1
                
                X, B_mat, B_hat, ica_time = get_data(q, m, n, p_cluster, p_var_within_cluster, p_var_between_cluster, seed)

                for alg, name in [(estimate_causal_order, "Current"), (exact_estimate_causal_order, "Exact"), (cycle_estimate_causal_order, "Greedy"), (heuristic_estimate_causal_order, "ELS")]: # (heuristic_estimate_causal_order, "ELS"),
                    print(name)
                    if name == "Current" and p > 300:
                        continue
                    if name == "Exact" and p > 20:
                        continue
                    if name == "Direct_kernel" and (p > 10 or n > 1000):
                        continue
                    if name == "Direct_pwling" and (p > 50):
                        continue

                    start = time.time()
                    if name in ["Direct_pwling", "Direct_kernel", "IterativeICALiNGAM"]:
                        causal_order = alg(X)
                    else:
                        causal_order = alg(B_hat)
                    end = time.time()

                    time_used = end - start

                    if name not in ["Direct_pwling", "Direct_kernel", "IterativeICALiNGAM"]:
                        time_used += ica_time

                    cost = compute_cost(B_hat, causal_order)

                    causal_order_time = time_used - ica_time
                    if name in ["IterativeICALiNGAM", "Direct_pwling", "Direct_kernel"]:
                        causal_order_time = math.nan
                    
                    ps.append(p)
                    ns.append(n)
                    settings.append(density_name)
                    seeds.append(seed)

                    algs.append(name)
                    times.append(time_used)
                    causal_order_times.append(causal_order_time)
                    costs.append(cost)

                    violations.append(0)
                    reachability_violations.append(0)
                    sids.append(0)
                    shds.append(0)


df = pd.DataFrame({"p":ps, "n": ns, "Seed": seeds, "Setting": settings, "Algorithm": algs, "Time": times, "Causal Order Time": causal_order_times, "Cost": costs, "Causal Order Edge Violations": violations, "Causal Order Reachability Violations": reachability_violations, "SID": sids, "SHD": shds})

df.to_csv(f"{output_dir}/no_direct_lingam_exp_{args.seeds}.csv")                
