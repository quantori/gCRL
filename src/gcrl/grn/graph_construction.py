# src/gcrl/grn/graph_construction.py
# -*- coding: utf-8 -*-
"""
Graph construction utilities for gene regulatory networks.

This module provides functions for building TF-TF multiplex networks from GRN data:
- Regulatory layer: direct TF-TF regulatory interactions
- Co-target layer: TF-TF similarity based on shared target genes

Key functions:
--------------
build_tf_tf_regulatory_layer : Build TF-TF regulatory graph from GRN
build_tf_tf_cotarget_layer : Build TF-TF co-target similarity graph
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from collections import defaultdict
import igraph as ig
from sklearn.preprocessing import normalize


def build_tf_tf_regulatory_layer(current_grn: pd.DataFrame, tf_names):
    """
    Build an undirected weighted TF–TF regulatory graph from the GRN.
    Uses absolute coefficients, symmetrized between TF pairs.

    Parameters
    ----------
    current_grn : pd.DataFrame
        GRN with columns: 'source' (TF), 'target' (gene), 'coef_abs' (absolute coefficient)
    tf_names : list or set
        List of TF names to include in the graph

    Returns
    -------
    g_reg : igraph.Graph
        Undirected weighted graph of TF-TF regulatory interactions
    tf_index : dict
        Mapping from TF name to vertex index
    """
    tf_names = list(tf_names)
    tf_index = {tf: i for i, tf in enumerate(tf_names)}
    n_tf = len(tf_names)

    # Filter for TF->TF edges
    mask_tf_tf = current_grn['source'].isin(tf_names) & current_grn['target'].isin(tf_names)
    grn_tf_tf = current_grn[mask_tf_tf]

    edge_weights = defaultdict(float)  # (i,j) -> weight

    for _, row in grn_tf_tf.iterrows():
        s, t, w = row['source'], row['target'], row['coef_abs']
        i, j = tf_index[s], tf_index[t]
        if i == j:
            continue
        a, b = sorted((i, j))
        edge_weights[(a, b)] += float(w)

    edges = list(edge_weights.keys())
    weights = [edge_weights[e] for e in edges]

    g_reg = ig.Graph(n=n_tf, edges=edges, directed=False)
    g_reg.vs['name'] = tf_names
    g_reg.es['weight'] = weights

    return g_reg, tf_index


def build_tf_tf_cotarget_layer(current_grn: pd.DataFrame,
                               tf_names,
                               tf_index,
                               min_similarity: float = 0.15):
    """
    Build an undirected weighted TF–TF co-target similarity graph.

    Similarity = cosine similarity between TF rows of the TF–TG coefficient matrix.
    Targets that are TFs are excluded: we only use "true" TGs here.

    Parameters
    ----------
    current_grn : pd.DataFrame
        GRN with columns: 'source' (TF), 'target' (gene), 'coef_abs' (absolute coefficient)
    tf_names : list or set
        List of TF names
    tf_index : dict
        Mapping from TF name to vertex index (from build_tf_tf_regulatory_layer)
    min_similarity : float, default=0.15
        Minimum cosine similarity to include an edge

    Returns
    -------
    g_cot : igraph.Graph
        Undirected weighted graph of TF-TF co-target similarity
    sim : np.ndarray
        Full similarity matrix (n_tf x n_tf)
    """
    tf_names = list(tf_names)
    n_tf = len(tf_names)

    # Keep only edges from TFs to non-TFs
    mask = current_grn['source'].isin(tf_names) & (~current_grn['target'].isin(tf_names))
    grn_tf_tg = current_grn[mask].copy()

    if grn_tf_tg.empty:
        raise ValueError("No TF->TG edges (non-TF targets) found; cannot build co-target layer.")

    # Build list of unique TGs
    tg_names = grn_tf_tg['target'].unique().tolist()
    tg_index = {g: i for i, g in enumerate(tg_names)}
    n_tg = len(tg_names)

    # Build TF x TG matrix of absolute coefficients
    B = np.zeros((n_tf, n_tg), dtype=float)
    for _, row in grn_tf_tg.iterrows():
        tf = row['source']
        tg = row['target']
        i = tf_index[tf]
        j = tg_index[tg]
        B[i, j] = float(row['coef_abs'])

    # Cosine similarity between TF rows
    # avoid division by zero by adding a small epsilon
    B_norm = normalize(B, norm='l2', axis=1)
    sim = np.dot(B_norm, B_norm.T)  # n_tf x n_tf

    # Build sparse similarity graph (undirected)
    edges = []
    weights = []
    n = n_tf
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= min_similarity:
                edges.append((i, j))
                weights.append(float(sim[i, j]))

    g_cot = ig.Graph(n=n_tf, edges=edges, directed=False)
    g_cot.vs['name'] = tf_names
    g_cot.es['weight'] = weights

    return g_cot, sim
