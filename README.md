# Causality-With-Clusters

A repo containing code for my Master's Thesis on Causality with Clusters.

## Cluster ICP results

`cluster_icp/slurm_cluster_icp_exp.py` differs from the code used for the original Euler runs (see the repo history) in two ways:

- Parents of Y are read from the variable-level DAG after edges from E to Y are removed, so E is never counted as a parent.
- Multi-response group LASSO coefficients are read per predictor, then per response.

`cluster_icp/correct_parents.py` rescored the Euler CSVs against the true parents, since outside screening only the scoring was affected. Its input CSVs are not in this repo; its output is in `cluster_icp/corrected_csvs`. `cluster_icp/rerun_csvs` holds the screening rerun, and `cluster_icp/plot_cluster_icp.py` makes the figures from both, with 95% Clopper-Pearson bands.

`cluster_icp/gap_analysis.py` computes the exact oracle output while varying the number of clusters and the edge probability independently; its output is `cluster_icp/corrected_csvs/gap_analysis.csv`. `cluster_icp/intervals.py` prints the intervals quoted in the text. Intervals for FNR, FPR and recall use pooled counts, which treats parents as independent.
