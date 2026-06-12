# src/gcrl/grn/stability.py
# -*- coding: utf-8 -*-
"""
Clustering stability assessment utilities.

This module provides functions for assessing the stability of clustering results
across multiple random initializations using Variation of Information (VI) and
Adjusted Rand Index (ARI).

Key functions:
--------------
filter_trivial_partitions : Filter out degenerate/uninformative partitions
is_trivial_partition : Check if a single partition is trivial
variation_of_information : Compute VI between two partitions
partition_stability_diagnostics : Compute pairwise stability metrics
summarize_all_stabilities : Summarize stability across hyperparameters
"""

from __future__ import annotations
import numpy as np
from itertools import combinations
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.cluster import contingency_matrix
from scipy.optimize import linear_sum_assignment
import warnings


def align_labels_to_reference(reference, labels):
    """
    Remap cluster labels in `labels` to best match `reference` using the
    Hungarian algorithm on the contingency matrix.

    Any label in `labels` that has no counterpart in `reference` (because
    the two partitions have different numbers of clusters) is left unmapped
    and assigned a new unique label beyond the reference's range.

    Parameters
    ----------
    reference : array-like of int
        Reference partition (labels are not changed)
    labels : array-like of int
        Partition to remap

    Returns
    -------
    aligned : np.ndarray of int
        Remapped version of `labels` whose cluster indices best correspond
        to those of `reference`
    """
    reference = np.asarray(reference)
    labels    = np.asarray(labels)

    ref_ids = np.unique(reference)
    lab_ids = np.unique(labels)

    # Contingency matrix: rows = reference clusters, cols = labels clusters
    cont = contingency_matrix(reference, labels)  # shape (n_ref, n_lab)

    # Hungarian algorithm maximises overlap (minimise negative overlap)
    row_ind, col_ind = linear_sum_assignment(-cont)

    # Build remapping: labels cluster -> reference cluster
    remap = {}
    for r, c in zip(row_ind, col_ind):
        remap[lab_ids[c]] = ref_ids[r]

    # Unmapped labels (more clusters in labels than in reference) get new ids
    next_id = int(ref_ids.max()) + 1
    for lab_id in lab_ids:
        if lab_id not in remap:
            remap[lab_id] = next_id
            next_id += 1

    return np.array([remap[l] for l in labels], dtype=int)


def is_trivial_partition(
    membership,
    min_clusters: int = 3,
    max_cluster_ratio: float = 0.5,
    min_mean_size: float = 3.0,
):
    """
    Check if a partition is trivial (degenerate/uninformative).

    A partition is considered trivial if it meets any of these criteria:
    1. Too few clusters (everything lumped into 1-2 groups)
    2. Too many clusters (over-fragmented network)
    3. Mean cluster size is too small (mostly singleton or tiny clusters)

    Parameters
    ----------
    membership : array-like
        Cluster labels for nodes
    min_clusters : int, default=3
        Minimum number of clusters required for non-trivial partition
    max_cluster_ratio : float, default=0.5
        Maximum ratio of (n_clusters / n_nodes). If exceeded, partition
        is considered over-fragmented
    min_mean_size : float, default=3.0
        Minimum mean cluster size required

    Returns
    -------
    is_trivial : bool
        True if partition is trivial, False otherwise

    Examples
    --------
    >>> membership = np.array([0, 0, 1, 1, 2, 2])  # 3 clusters of size 2
    >>> is_trivial_partition(membership, min_mean_size=3.0)
    True  # mean size = 2 < 3.0

    >>> membership = np.array([0]*98 + [1, 2])  # 1 giant cluster + 2 singletons
    >>> is_trivial_partition(membership, min_clusters=3)
    False  # has 3 clusters, but check mean size
    >>> is_trivial_partition(membership, min_mean_size=3.0)
    True  # mean size = 33.3, but this would fail max_cluster_ratio for other reasons
    """
    membership = np.asarray(membership)
    n = len(membership)
    n_clusters = len(np.unique(membership))

    if n_clusters == 0:
        return True

    mean_size = n / n_clusters

    # Too few clusters (everything in 1-2 groups)
    if n_clusters < min_clusters:
        return True

    # Too many clusters (over-fragmented)
    if n_clusters > n * max_cluster_ratio:
        return True

    # Mean cluster size too small
    if mean_size < min_mean_size:
        return True

    return False


def filter_trivial_partitions(
    memberships_list,
    min_clusters: int = 3,
    max_cluster_ratio: float = 0.5,
    min_mean_size: float = 3.0,
    verbose: bool = False,
):
    """
    Filter out trivial (degenerate) partitions from a list of clustering results.

    This is useful before computing stability metrics to avoid including
    outlier runs that produced uninformative clusterings.

    Parameters
    ----------
    memberships_list : list of array-like
        List of membership arrays from multiple clustering runs
    min_clusters : int, default=3
        Minimum number of clusters required for non-trivial partition
    max_cluster_ratio : float, default=0.5
        Maximum ratio of (n_clusters / n_nodes)
    min_mean_size : float, default=3.0
        Minimum mean cluster size required
    verbose : bool, default=False
        If True, print filtering statistics

    Returns
    -------
    filtered_memberships : list of np.ndarray
        List containing only non-trivial partitions
    n_removed : int
        Number of trivial partitions removed

    Examples
    --------
    >>> # Generate some clustering results
    >>> results = {
    ...     (1.0, 1.0, 0.1): [
    ...         np.array([0, 0, 1, 1, 2, 2]),  # good partition
    ...         np.array([0, 1, 2, 3, 4, 5]),  # over-fragmented
    ...         np.array([0, 0, 0, 0, 1, 1]),  # good partition
    ...     ]
    ... }
    >>> filtered, n_removed = filter_trivial_partitions(
    ...     results[(1.0, 1.0, 0.1)],
    ...     min_mean_size=2.0,
    ...     verbose=True
    ... )
    Removed 1 trivial partitions out of 3 (33.3%)

    >>> # Use before stability analysis
    >>> filtered, _ = filter_trivial_partitions(memberships_list)
    >>> summary, aris, vis = partition_stability_diagnostics(filtered)
    """
    filtered = []
    n_removed = 0

    for mem in memberships_list:
        if not is_trivial_partition(
            mem,
            min_clusters=min_clusters,
            max_cluster_ratio=max_cluster_ratio,
            min_mean_size=min_mean_size,
        ):
            filtered.append(np.asarray(mem))
        else:
            n_removed += 1

    if verbose:
        total = len(memberships_list)
        pct = 100 * n_removed / total if total > 0 else 0
        print(f"Removed {n_removed} trivial partitions out of {total} ({pct:.1f}%)")

    return filtered, n_removed


def variation_of_information(labels1, labels2):
    """
    Compute Variation of Information (VI) between two partitions.

    VI(X, Y) = H(X) + H(Y) - 2 I(X;Y)
    where H is the entropy and I is mutual information.
    Uses natural logarithms (units: nats).

    Parameters
    ----------
    labels1 : array-like
        First partition (cluster labels)
    labels2 : array-like
        Second partition (cluster labels)

    Returns
    -------
    vi : float
        Variation of Information (lower is more similar)
    """
    labels1 = np.asarray(labels1)
    labels2 = np.asarray(labels2)
    if labels1.shape != labels2.shape:
        raise ValueError("labels1 and labels2 must have the same shape")

    # Contingency table
    cont = contingency_matrix(labels1, labels2)  # shape (n_X, n_Y)
    n = cont.sum()
    if n == 0:
        return 0.0

    pij = cont / n                     # joint distribution P(X=i, Y=j)
    pi = pij.sum(axis=1)               # P(X=i)
    pj = pij.sum(axis=0)               # P(Y=j)

    def entropy(p):
        p = p[p > 0]
        return -np.sum(p * np.log(p))

    HX = entropy(pi)
    HY = entropy(pj)

    # Mutual information I(X;Y) = sum_ij pij * log( pij / (pi * pj) )
    pi_pj = np.outer(pi, pj)           # same shape as pij
    nonzero = pij > 0
    I = np.sum(pij[nonzero] * np.log(pij[nonzero] / pi_pj[nonzero]))

    VI = HX + HY - 2.0 * I
    return float(VI)


def partition_stability_diagnostics(
    memberships_list,
    compute_ari: bool = False,
    max_pairs: int | None = None,
):
    """
    Given a list of partitions (membership arrays),
    compute pairwise VI and (optionally) ARI, and summarise.

    Parameters
    ----------
    memberships_list : list of 1D np.ndarray
        Each element is a partition (cluster labels for all TFs)
    compute_ari : bool, default=False
        If True, also compute Adjusted Rand Index (ARI) using sklearn.
        This is slower and may trigger sklearn warnings; we silence the
        specific regression-like warning internally
    max_pairs : int or None, default=None
        If not None, randomly sample at most `max_pairs` distinct pairs
        of runs instead of using all C(R, 2) pairs. Useful for speed
        when you have many seeds

    Returns
    -------
    summary : dict
        Summary statistics for VI (and ARI if requested)
    aris : np.ndarray or None
        All sampled ARI values, or None if compute_ari=False
    vis : np.ndarray
        All sampled VI values
    """
    R = len(memberships_list)
    if R < 2:
        raise ValueError("Need at least 2 partitions to assess stability.")

    # All partitions should have same length
    n = len(memberships_list[0])
    for lab in memberships_list[1:]:
        if len(lab) != n:
            raise ValueError("All membership arrays must have the same length.")

    # All run pairs
    all_pairs = list(combinations(range(R), 2))

    # Optionally subsample pairs for speed
    if (max_pairs is not None) and (len(all_pairs) > max_pairs):
        rng = np.random.default_rng(0)
        idx = rng.choice(len(all_pairs), size=max_pairs, replace=False)
        pairs = [all_pairs[i] for i in idx]
    else:
        pairs = all_pairs

    vi_values = []
    ari_values = [] if compute_ari else None

    for i, j in pairs:
        l1 = np.asarray(memberships_list[i])
        l2 = np.asarray(memberships_list[j])

        # VI (our own implementation, no sklearn)
        vi = variation_of_information(l1, l2)
        vi_values.append(vi)

        if compute_ari:
            # Silence the "could represent regression" warning from sklearn
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="The number of unique classes is greater than 50% of the number of samples. `y` could represent a regression problem, not a classification problem.",
                )
                ari = adjusted_rand_score(l1, l2)
            ari_values.append(ari)

    vis = np.array(vi_values, dtype=float)
    if compute_ari:
        aris = np.array(ari_values, dtype=float)
        summary = {
            "VI_mean": vis.mean(),
            "VI_std": vis.std(),
            "ARI_mean": aris.mean(),
            "ARI_std": aris.std(),
            "n_pairs": len(pairs),
        }
    else:
        aris = None
        summary = {
            "VI_mean": vis.mean(),
            "VI_std": vis.std(),
            "n_pairs": len(pairs),
        }

    return summary, aris, vis


def summarize_all_stabilities(
    results,
    compute_ari: bool = False,
    max_pairs: int | None = 200,
):
    """
    Compute stability summaries for all hyperparameter combinations.

    Parameters
    ----------
    results : dict
        Dictionary where results[(gamma_reg, gamma_cot, w_cot)] = list of membership arrays
    compute_ari : bool, default=False
        Whether to compute ARI in addition to VI
    max_pairs : int or None, default=200
        Maximum number of pairs to sample per hyperparameter combination

    Returns
    -------
    stability : dict
        Dictionary where key -> stability summary (VI stats and optionally ARI stats)
    """
    stability = {}
    for key, membs in results.items():
        summary, aris, vis = partition_stability_diagnostics(
            membs,
            compute_ari=compute_ari,
            max_pairs=max_pairs,
        )
        stability[key] = summary
    return stability
