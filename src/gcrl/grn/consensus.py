# src/gcrl/grn/consensus.py
# -*- coding: utf-8 -*-
"""
Consensus clustering utilities for TF communities.

This module provides functions for building consensus partitions from multiple
clustering runs using co-association matrices.

Key functions:
--------------
build_coassociation_matrix : Build co-association matrix from multiple partitions
consensus_partition_from_coassoc : Build consensus partition from co-association matrix
community_stats : Compute statistics for communities on a graph
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from collections import defaultdict
from itertools import combinations
import igraph as ig
import leidenalg as la


def build_coassociation_matrix(memberships_list):
    """
    Build co-association matrix from multiple clustering runs.

    The co-association matrix C_ij represents the fraction of runs in which
    nodes i and j were assigned to the same cluster.

    Parameters
    ----------
    memberships_list : list of 1D arrays
        Each element is a membership array (length n) from one clustering run

    Returns
    -------
    C : np.ndarray
        Co-association matrix (n x n), where C_ij ∈ [0, 1]
        C_ij = 1 means nodes i and j were always clustered together
    """
    R = len(memberships_list)
    n = len(memberships_list[0])
    C = np.zeros((n, n), dtype=float)

    for labels in memberships_list:
        labels = np.array(labels)
        # group nodes by community in this run
        comm_to_nodes = defaultdict(list)
        for i, c in enumerate(labels):
            comm_to_nodes[c].append(i)
        # for each community, increment all pairs
        for nodes in comm_to_nodes.values():
            for i, j in combinations(nodes, 2):
                C[i, j] += 1
                C[j, i] += 1

    # diagonal = always co-clustered with itself
    np.fill_diagonal(C, R)
    C /= R
    return C


def consensus_partition_from_coassoc(C,
                                     min_coassoc: float = 0.5,
                                     resolution: float = 0.5,
                                     seed: int | None = None):
    """
    Build a consensus partition from co-association matrix C.

    Steps:
      - threshold weak co-associations
      - build weighted graph
      - run Leiden on this graph

    Parameters
    ----------
    C : np.ndarray
        Co-association matrix (n x n) from build_coassociation_matrix
    min_coassoc : float, default=0.5
        Minimum co-association threshold to include an edge
    resolution : float, default=0.5
        Resolution parameter for Leiden clustering on consensus graph
    seed : int or None, default=None
        Random seed for the Leiden optimiser

    Returns
    -------
    membership : np.ndarray
        Consensus cluster labels
    g : igraph.Graph
        Consensus graph (edges = strong co-associations)
    """
    n = C.shape[0]
    edges = []
    weights = []
    for i in range(n):
        for j in range(i + 1, n):
            if C[i, j] >= min_coassoc:
                edges.append((i, j))
                weights.append(float(C[i, j]))

    g = ig.Graph(n=n, edges=edges, directed=False)
    g.es['weight'] = weights

    part = la.RBConfigurationVertexPartition(
        g,
        weights='weight',
        resolution_parameter=resolution
    )
    optimiser = la.Optimiser()
    if seed is not None:
        optimiser.set_rng_seed(seed)
    optimiser.optimise_partition(part)
    return np.array(part.membership, dtype=int), g


def consensus_partition_majority_vote(memberships_list):
    """
    Build a consensus partition by majority vote after aligning all runs to
    a common reference (seed 0) via the Hungarian algorithm.

    Steps:
      1. Use the first partition as reference.
      2. For each subsequent partition, remap its labels to best match the
         reference using the Hungarian algorithm on the contingency matrix.
      3. Take the element-wise mode across all aligned partitions.

    This is fully deterministic and has no free parameters.

    Parameters
    ----------
    memberships_list : list of 1D array-like
        Each element is a membership array (length n) from one clustering run.

    Returns
    -------
    consensus : np.ndarray of int
        Majority-vote consensus labels (length n).
    agreement : np.ndarray of float
        For each node, the fraction of runs that agreed with the consensus
        label (after alignment). Values in (0, 1]; low values flag nodes
        that were unstably assigned.
    """
    from .stability import align_labels_to_reference
    from scipy.stats import mode as scipy_mode

    reference = np.asarray(memberships_list[0])
    aligned   = [reference]
    for mem in memberships_list[1:]:
        aligned.append(align_labels_to_reference(reference, mem))

    aligned = np.stack(aligned, axis=0)          # shape (n_runs, n_nodes)
    result  = scipy_mode(aligned, axis=0)
    consensus  = result.mode.ravel().astype(int)
    agreement  = result.count.ravel() / len(memberships_list)
    return consensus, agreement


def consensus_partition_hierarchical(C, min_coassoc: float = 0.5):
    """
    Build a deterministic consensus partition via average-linkage hierarchical
    clustering on the co-association matrix.

    Distance between TFs is defined as (1 - C[i,j]).  Cutting the dendrogram
    at distance (1 - min_coassoc) groups together all TFs that were co-clustered
    in at least min_coassoc fraction of runs.  No random seed, no free parameters
    beyond min_coassoc.

    Parameters
    ----------
    C : np.ndarray
        Co-association matrix (n x n) from build_coassociation_matrix.
    min_coassoc : float, default=0.5
        Cut threshold in co-association space.  Pairs with C[i,j] >= min_coassoc
        end up in the same cluster.

    Returns
    -------
    consensus : np.ndarray of int
        Cluster labels (length n), zero-indexed and contiguous.
    """
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform

    D = 1.0 - C
    np.fill_diagonal(D, 0.0)
    condensed = squareform(D, checks=False)
    Z = linkage(condensed, method='average')
    labels = fcluster(Z, t=1.0 - min_coassoc, criterion='distance')
    # fcluster returns 1-indexed labels; make 0-indexed and contiguous
    _, contiguous = np.unique(labels, return_inverse=True)
    return contiguous.astype(int)


def community_stats(graph: ig.Graph, membership):
    """
    Compute diagnostics for a partition on a given graph.

    For each community:
      - size                : number of nodes
      - density             : num_internal_edges / (size*(size-1)/2)  — fraction of
                              possible edges that are present (pure topology, no weights)
      - num_internal_edges  : count of edges with both endpoints inside
      - num_boundary_edges  : count of edges with exactly one endpoint inside
      - w_internal          : sum of weights of internal edges
      - w_boundary          : sum of weights of boundary edges
      - w_ratio             : w_internal / (w_internal + w_boundary)

    Parameters
    ----------
    graph : igraph.Graph
        Undirected weighted graph on which the partition was computed
    membership : array-like
        Cluster labels for each node

    Returns
    -------
    stats : pd.DataFrame
        Columns: ['community', 'size', 'density', 'num_internal_edges',
                  'num_boundary_edges', 'w_internal', 'w_boundary', 'w_ratio']
        Sorted by size descending
    """
    membership = np.array(membership)
    edge_weights = (np.array(graph.es['weight']) if 'weight' in graph.es.attributes()
                    else np.ones(graph.ecount()))

    comm_to_nodes = defaultdict(set)
    for i, c in enumerate(membership):
        comm_to_nodes[c].add(i)

    w_internal      = defaultdict(float)
    w_boundary      = defaultdict(float)
    n_internal      = defaultdict(int)
    n_boundary      = defaultdict(int)

    for e, w in zip(graph.es, edge_weights):
        s, t = e.source, e.target
        cs, ct = membership[s], membership[t]
        if cs == ct:
            w_internal[cs] += w
            n_internal[cs] += 1
        else:
            w_boundary[cs] += w
            w_boundary[ct] += w
            n_boundary[cs] += 1
            n_boundary[ct] += 1

    stats = []
    for c, nodes in comm_to_nodes.items():
        size = len(nodes)
        wi   = w_internal[c]
        wb   = w_boundary[c]
        ni   = n_internal[c]
        nb   = n_boundary[c]
        max_edges = size * (size - 1) / 2
        density   = ni / max_edges if max_edges > 0 else 0.0
        w_ratio   = wi / (wi + wb) if (wi + wb) > 0 else 0.0
        stats.append({
            'community':          c,
            'size':               size,
            'density':            density,
            'num_internal_edges': ni,
            'num_boundary_edges': nb,
            'w_internal':         wi,
            'w_boundary':         wb,
            'w_ratio':            w_ratio,
        })

    return pd.DataFrame(stats).sort_values('size', ascending=False).reset_index(drop=True)
