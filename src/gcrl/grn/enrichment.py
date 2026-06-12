# src/gcrl/grn/enrichment.py
# -*- coding: utf-8 -*-
"""
Gene set enrichment analysis utilities for TF communities.

This module provides functions for performing GO term enrichment analysis
on TF communities using hypergeometric tests (Over-Representation Analysis).

Key functions:
--------------
build_cluster_gene_sets : Build gene sets for each TF cluster
filter_gene_sets_by_size_and_level : Filter GO gene sets by size and hierarchy level
run_ora_for_clusters : Run hypergeometric ORA for clusters
compute_go_levels : Compute GO hierarchy levels from OBO file
extract_go_ids_from_terms : Extract GO IDs from term names
"""

from __future__ import annotations
from typing import Dict, List, Iterable, Optional, Union
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.stats import hypergeom
from statsmodels.stats.multitest import fdrcorrection
from goatools.obo_parser import GODag


def build_cluster_gene_sets(
    tf_names: List[str],
    consensus_labels: np.ndarray,
    current_grn: pd.DataFrame,
) -> Dict[int, Dict[str, Iterable[str]]]:
    """
    Build, for each TF cluster, the set of TFs and target genes to use for enrichment.

    Parameters
    ----------
    tf_names : list of str
        List of TF names in the same order as consensus_labels
    consensus_labels : np.ndarray of shape (n_tfs,)
        Cluster labels, one per TF
    current_grn : pd.DataFrame
        GRN with at least columns: 'source' (TF), 'target' (regulated gene)

    Returns
    -------
    cluster2genes : dict
        {
          cluster_id: {
              "tfs": set([...]),
              "targets": set([...]),
              "all_genes": set([...])  # tfs ∪ targets
          },
          ...
        }
    """
    tf_names = np.asarray(tf_names)
    if len(tf_names) != len(consensus_labels):
        raise ValueError("tf_names and consensus_labels must have the same length.")

    required_cols = {"source", "target"}
    missing = required_cols.difference(current_grn.columns)
    if missing:
        raise ValueError(f"current_grn is missing columns: {missing}")

    # Pre-group GRN by source TF
    grn_by_source = defaultdict(list)
    for _, row in current_grn[["source", "target"]].iterrows():
        grn_by_source[row["source"]].append(row["target"])

    cluster2genes = {}
    for cluster_id in np.unique(consensus_labels):
        cluster_tfs = tf_names[consensus_labels == cluster_id]

        cluster_targets = set()
        for tf in cluster_tfs:
            if tf in grn_by_source:
                cluster_targets.update(grn_by_source[tf])

        all_genes = set(cluster_tfs) | cluster_targets

        cluster2genes[cluster_id] = {
            "tfs": set(cluster_tfs),
            "targets": cluster_targets,
            "all_genes": all_genes,
        }

    return cluster2genes


def filter_gene_sets_by_size_and_level(
    gene_sets: Dict[str, List[str]],
    min_size: int = 10,
    max_size: int = 500,
    go_term_levels: Optional[Dict[str, int]] = None,
    min_level: Optional[int] = None,
    max_level: Optional[int] = None,
) -> Dict[str, set]:
    """
    Filter GO BP gene sets by size and (optionally) GO level.

    Parameters
    ----------
    gene_sets : dict
        {term: [genes...]} as returned by gseapy.get_library.
        'term' can be something like 'GO_Biological_Process_xxx_GO:0001234'
    min_size, max_size : int
        Keep only terms with size in [min_size, max_size]
    go_term_levels : dict or None
        Optional mapping {go_id: level}, e.g. {"GO:0001234": 4, ...}.
        If provided, terms are also filtered by min_level / max_level
    min_level, max_level : int or None
        GO levels to keep (inclusive). Ignored if go_term_levels is None

    Returns
    -------
    filtered_gene_sets : dict
        {term: set(genes)}
    """

    filtered = {}

    for term, genes in gene_sets.items():
        geneset = set(genes)
        size = len(geneset)
        if size < min_size or size > max_size:
            continue

        # If levels are not requested, just keep by size
        if go_term_levels is None or (min_level is None and max_level is None):
            filtered[term] = geneset
            continue

        # Extract GO ID from the term string, if present
        # Many libraries encode it as '..._GO:0001234'
        go_id = None
        for token in term.split("("):
            if token.startswith("GO:"):
                go_id = token.strip(")")
                break

        if go_id is None:
            # No GO ID in term string; skip level filtering for this term
            continue

        level = go_term_levels.get(go_id, None)
        if level is None:
            # GO term not in level dict
            continue

        if (min_level is not None and level < min_level) or \
           (max_level is not None and level > max_level):
            continue

        filtered[term] = geneset

    return filtered


def run_ora_for_clusters(
    cluster2genes: Dict[int, Dict[str, Iterable[str]]],
    gene_sets: Dict[str, set],
    universe_mode: str = "grn",
    custom_universe: Optional[Iterable[str]] = None,
    min_genes_in_cluster: int = 5,
) -> pd.DataFrame:
    """
    Run hypergeometric ORA for each cluster against GO BP gene sets.

    Parameters
    ----------
    cluster2genes : dict
        Output of build_cluster_gene_sets
    gene_sets : dict
        {go_term: set(genes)} (already filtered by size/level)
    universe_mode : {'grn', 'go', 'custom'}, default='grn'
        Controls the background population N used in the hypergeometric test:

        'grn'    — all genes (TFs + targets) present in the GRN, intersected
                   with genes present in at least one GO term. Biologically
                   honest for within-GRN comparisons, but can be too small
                   (N~700) if cluster gene-sets cover >50% of the universe.

        'go'     — all genes present in the GO BP library (~11,000–13,000).
                   Maximises statistical power but inflates enrichment of
                   cell-line housekeeping terms (e.g. myeloid differentiation
                   in K562), as the cluster genes are a very small fraction.

        'custom' — an arbitrary gene set supplied via ``custom_universe``.
                   Use this to pass e.g. the 5,000 HVGs used to build the GRN:
                   a balanced choice larger than 'grn', smaller than 'go',
                   and tied to the genes that were actually measured.

    custom_universe : iterable of str or None
        Required when ``universe_mode='custom'``. Ignored otherwise.
    min_genes_in_cluster : int, default=5
        Minimum number of genes in a cluster (after intersecting with universe)
        to perform enrichment

    Returns
    -------
    results : pd.DataFrame
        Long table with columns:
        ['cluster_id', 'go_term', 'k', 'K', 'n', 'N', 'pval', 'pval_adj', 'overlap_genes']
    """
    valid_modes = {"grn", "go", "custom"}
    if universe_mode not in valid_modes:
        raise ValueError(f"universe_mode must be one of {valid_modes}, got '{universe_mode}'.")

    go_universe = set().union(*gene_sets.values())

    # Determine universe
    if universe_mode == "grn":
        universe = set()
        for d in cluster2genes.values():
            universe.update(d["all_genes"])
        # Restrict to genes that appear in at least one GO set
        universe = universe & go_universe
    elif universe_mode == "go":
        universe = go_universe
    else:  # custom
        if custom_universe is None:
            raise ValueError("universe_mode='custom' requires custom_universe to be supplied.")
        universe = set(custom_universe) & go_universe

    print(f"Universe mode: '{universe_mode}'  →  N = {len(universe):,} genes")

    N = len(universe)
    if N == 0:
        raise ValueError("Universe is empty after intersection; check gene IDs and mapping.")

    # Precompute for each GO term the set of genes within the universe
    go_sets_universe = {
        term: genes & universe
        for term, genes in gene_sets.items()
    }

    records = []

    for cluster_id, d in cluster2genes.items():
        # genes in this cluster, restricted to universe
        cluster_genes = set(d["all_genes"]) & universe
        n = len(cluster_genes)
        if n < min_genes_in_cluster:
            print(f"Skipping cluster {cluster_id}: only {n} genes in universe.")
            continue

        for term, term_genes in go_sets_universe.items():
            K = len(term_genes)
            if K == 0:
                continue

            overlap = cluster_genes & term_genes
            k = len(overlap)
            if k == 0:
                continue

            # Hypergeometric p-value: P(X >= k)
            pval = hypergeom.sf(k - 1, N, K, n)

            records.append({
                "cluster_id": cluster_id,
                "go_term": term,
                "k": k,
                "K": K,
                "n": n,
                "N": N,
                "pval": pval,
                "overlap_genes": ";".join(sorted(overlap)),
            })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame.from_records(records)

    # =========================
    # Per-cluster FDR correction
    # =========================
    df_list = []
    for cid, sub in df.groupby("cluster_id", sort=False):
        # Keep original order or sort by pval; fdrcorrection does not require sorting
        pvals = sub["pval"].values
        rejected, pvals_adj = fdrcorrection(pvals, alpha=0.05, method='indep')
        sub = sub.copy()
        sub["pval_adj"] = pvals_adj
        sub["significant"] = rejected
        df_list.append(sub)

    df_out = pd.concat(df_list, axis=0, ignore_index=True)

    return df_out


def compute_go_levels(
    go_obo_path: Optional[Union[str, Path]] = None,
    go_ids: Optional[Iterable[str]] = None,
    namespace: str = "biological_process",
) -> Dict[str, int]:
    """
    Compute GO levels using GOATOOLS.

    Parameters
    ----------
    go_obo_path : str, Path, or None
        Path to go-basic.obo (or another GO OBO file).
        If None, uses the default GO OBO file from gCRL/data/reference/ontologies/
    go_ids : iterable of str or None
        Optional list of GO IDs to restrict to (e.g. those present in your GO BP library).
        If None, all GO terms in the DAG for the given namespace are considered
    namespace : {'biological_process', 'molecular_function', 'cellular_component'}
        Restrict GO terms to this namespace

    Returns
    -------
    go_term_levels : dict
        {go_id: level}, where 'level' is the GOATOOLS 'level' attribute
        (distance from the root term in the given namespace)
    """
    # Use default GO OBO path if not provided
    if go_obo_path is None:
        from gcrl.data import get_go_obo_path
        go_obo_path = get_go_obo_path()

    print(f"Loading GO DAG from {go_obo_path} ...")
    go_dag = GODag(str(go_obo_path))

    if go_ids is None:
        # All GO IDs in the specified namespace
        go_ids = [
            go_id for go_id, rec in go_dag.items()
            if rec.namespace == namespace
        ]

    go_term_levels = {}
    for go_id in go_ids:
        rec = go_dag.get(go_id, None)
        if rec is None:
            continue
        if rec.namespace != namespace:
            continue
        # rec.level is the graph level (0 = root 'all')
        go_term_levels[go_id] = rec.level

    return go_term_levels


def extract_go_ids_from_terms(gene_sets: Dict[str, list]) -> set:
    """
    Extract GO IDs (GO:xxxxxxx) from term names in a gseapy library.

    Parameters
    ----------
    gene_sets : dict
        Dictionary with term names as keys (e.g., from gseapy.get_library)

    Returns
    -------
    go_ids : set
        Set of GO IDs found in term names
    """
    go_ids = set()
    for term in gene_sets.keys():
        for token in term.split("("):
            if token.startswith("GO:"):
                token = token.strip(")")
                go_ids.add(token)
                break
    return go_ids


def filter_cluster_specific_terms(
    ora_results: pd.DataFrame,
    fdr_col: str = "pval_adj",
    fdr_threshold: float = 0.05,
    max_cluster_fraction: float = 0.5,
    term_col: str = "go_term",
) -> pd.DataFrame:
    """
    Flag and remove GO terms that are significantly enriched across too many
    clusters — these are likely cell-line background terms rather than
    cluster-specific biology.

    Parameters
    ----------
    ora_results : pd.DataFrame
        Output of run_ora_for_clusters.
    fdr_col : str
        Column containing adjusted p-values (default: 'pval_adj').
    fdr_threshold : float
        FDR threshold for calling a term significant in a cluster (default: 0.05).
    max_cluster_fraction : float
        Terms significant in more than this fraction of clusters are flagged
        as background (default: 0.5 → terms significant in >50% of clusters).
    term_col : str
        Column containing GO term identifiers (default: 'go_term').

    Returns
    -------
    pd.DataFrame
        The input dataframe with an added boolean column ``background_term``
        (True = likely cell-line background, should be excluded from plots).
        Rows where ``background_term`` is True can be dropped with
        ``df[~df['background_term']]``.
    """
    n_clusters = ora_results["cluster_id"].nunique()
    sig = ora_results[ora_results[fdr_col] < fdr_threshold]

    # Count in how many clusters each term is significant
    term_cluster_counts = (
        sig.groupby(term_col)["cluster_id"]
        .nunique()
        .rename("n_sig_clusters")
    )

    threshold_count = max_cluster_fraction * n_clusters
    background_terms = set(
        term_cluster_counts.index[term_cluster_counts > threshold_count]
    )

    ora_out = ora_results.copy()
    ora_out["background_term"] = ora_out[term_col].isin(background_terms)

    n_bg = len(background_terms)
    print(
        f"filter_cluster_specific_terms: {n_clusters} clusters, "
        f"FDR < {fdr_threshold}, max_cluster_fraction = {max_cluster_fraction}\n"
        f"  → {n_bg} background terms flagged "
        f"(significant in > {max_cluster_fraction*100:.0f}% of clusters)\n"
        f"  → {(~ora_out['background_term']).sum():,} rows retained"
    )
    return ora_out
