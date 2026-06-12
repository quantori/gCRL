import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from itertools import combinations

import igraph as ig
import leidenalg as la

from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.cluster import contingency_matrix
from sklearn.preprocessing import normalize
from sklearn.metrics.cluster import contingency_matrix
from itertools import combinations
from sklearn.metrics import adjusted_rand_score
import warnings

from collections import defaultdict
from typing import Dict, List, Iterable, Optional
from scipy.stats import hypergeom
from statsmodels.stats.multitest import fdrcorrection
from goatools.obo_parser import GODag
import re

def build_tf_tf_regulatory_layer(current_grn: pd.DataFrame, tf_names):
    """
    Build an undirected weighted TF–TF regulatory graph from the GRN.
    Uses absolute coefficients, symmetrized between TF pairs.
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
                               min_similarity: float = 0.1):
    """
    Build an undirected weighted TF–TF co-target similarity graph.

    Similarity = cosine similarity between TF rows of the TF–TG coefficient matrix.
    Targets that are TFs are excluded: we only use "true" TGs here.
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


def run_multiplex_leiden(
    g_reg,
    g_cot,
    gamma_reg: float = 1.0,
    gamma_cot: float = 0.5,
    layer_weight_reg: float = 1.0,
    layer_weight_cot: float = 1.0,
    n_iterations: int = -1,
    seed: int | None = None,
):
    """
    Single multiplex Leiden run on two layers:

    - Layer 1: regulatory TF–TF graph (RBConfigurationVertexPartition)
    - Layer 2: co-target TF–TF graph (CPMVertexPartition)

    Parameters
    ----------
    gamma_reg : float
        Resolution parameter for RBConfiguration on the regulatory layer.
    gamma_cot : float
        Resolution parameter for CPM on the co-target layer.
    layer_weight_reg : float
        Weight of the regulatory layer in the combined objective.
    layer_weight_cot : float
        Weight of the co-target layer in the combined objective.
    n_iterations : int
        Number of Leiden iterations; -1 means until convergence.
    seed : int or None
        Random seed for reproducibility.
    """
    optimiser = la.Optimiser()
    if seed is not None:
        optimiser.set_rng_seed(seed)

    # Layer 1: RBConfiguration (modularity-like) on regulatory graph
    part_reg = la.RBConfigurationVertexPartition(
        g_reg,
        weights="weight",
        resolution_parameter=gamma_reg,
    )

    # Layer 2: CPM on co-target graph
    part_cot = la.RBConfigurationVertexPartition(
        g_cot,
        weights="weight",
        resolution_parameter=gamma_cot,
    )

    partitions = [part_reg, part_cot]
    layer_weights = [float(layer_weight_reg), float(layer_weight_cot)]

    # Joint optimisation – note: no interslice_weight here
    diff = optimiser.optimise_partition_multiplex(
        partitions,
        layer_weights=layer_weights,
        n_iterations=n_iterations,
    )

    # All layers share the same membership after optimisation
    membership = np.array(partitions[0].membership, dtype=int)
    return membership, partitions, diff


def sweep_hyperparams(
    g_reg,
    g_cot,
    gamma_reg_list,
    gamma_cot_list,
    cot_weight_list,
    n_seeds: int = 20,
    n_iterations: int = -1,
):
    """
    Run multiplex Leiden for multiple (gamma_reg, gamma_cot, layer_weight_cot) combinations
    and multiple seeds for each combination.

    We keep the regulatory layer weight fixed at 1.0 and vary the co-target
    layer weight to control its influence.

    Returns
    -------
    results : dict
        results[(gamma_reg, gamma_cot, layer_weight_cot)] = list of membership arrays
    """
    results = {}

    for gamma_reg in gamma_reg_list:
        for gamma_cot in gamma_cot_list:
            for w_cot in cot_weight_list:
                key = (gamma_reg, gamma_cot, w_cot)
                memberships = []
                for seed in range(n_seeds):
                    mem, _, _ = run_multiplex_leiden(
                        g_reg,
                        g_cot,
                        gamma_reg=gamma_reg,
                        gamma_cot=gamma_cot,
                        layer_weight_reg=1.0,
                        layer_weight_cot=w_cot,
                        n_iterations=n_iterations,
                        seed=seed,
                    )
                    memberships.append(mem)
                results[key] = memberships
                print(f"Done γ_reg={gamma_reg}, γ_cot={gamma_cot}, w_cot={w_cot}")

    return results


def variation_of_information(labels1, labels2):
    """
    Compute Variation of Information (VI) between two partitions.

    VI(X, Y) = H(X) + H(Y) - 2 I(X;Y)
    where H is the entropy and I is mutual information.
    Uses natural logarithms (units: nats).
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

import numpy as np
from itertools import combinations
from sklearn.metrics import adjusted_rand_score
import warnings

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
        Each element is a partition (cluster labels for all TFs).
    compute_ari : bool, default False
        If True, also compute Adjusted Rand Index (ARI) using sklearn.
        This is slower and may trigger sklearn warnings; we silence the
        specific regression-like warning internally.
    max_pairs : int or None, default None
        If not None, randomly sample at most `max_pairs` distinct pairs
        of runs instead of using all C(R, 2) pairs. Useful for speed
        when you have many seeds.

    Returns
    -------
    summary : dict
        Summary statistics for VI (and ARI if requested).
    aris : np.ndarray or None
        All sampled ARI values, or None if compute_ari=False.
    vis : np.ndarray
        All sampled VI values.
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
    results[(gamma_reg, gamma_cot, w_cot)] = list of membership arrays

    Returns
    -------
    stability : dict
        key -> stability summary (VI stats and optionally ARI stats).
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


def build_coassociation_matrix(memberships_list):
    """
    memberships_list: list of 1D arrays of length n (membership per run)

    Returns:
      C: n x n co-association matrix, C_ij in [0,1]
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
                                     resolution: float = 0.5):
    """
    Build a consensus partition from co-association matrix C.
    Steps:
      - threshold weak co-associations
      - build weighted graph
      - run Leiden on this graph
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

    #part = la.CPMVertexPartition( #XXX
    part = la.RBConfigurationVertexPartition(    
        g,
        weights='weight',
        resolution_parameter=resolution
    )
    optimiser = la.Optimiser()
    optimiser.optimise_partition(part)
    return np.array(part.membership, dtype=int), g

def community_stats(graph: ig.Graph, membership):
    """
    Compute simple diagnostics for a partition on a given graph:
      - module sizes
      - internal edge weight sums / densities
    """
    membership = np.array(membership)
    n = len(membership)
    m = len(graph.es)
    total_weight = np.sum(graph.es['weight']) if 'weight' in graph.es.attributes() else m

    comm_to_nodes = defaultdict(list)
    for i, c in enumerate(membership):
        comm_to_nodes[c].append(i)

    stats = []
    for c, nodes in comm_to_nodes.items():
        # internal edges
        sub = graph.subgraph(nodes)
        if 'weight' in sub.es.attributes():
            w_internal = float(np.sum(sub.es['weight'])) if len(sub.es) > 0 else 0.0
        else:
            w_internal = float(len(sub.es))
        size = len(nodes)
        density = w_internal / (size * (size - 1) / 2) if size > 1 else 0.0
        stats.append({
            'community': c,
            'size': size,
            'w_internal': w_internal,
            'density': density
        })

    # global summary
    return pd.DataFrame(stats).sort_values('size', ascending=False)

def run_single_layer_leiden_reg(
    g_reg,
    gamma_reg: float = 0.5,
    seed: int | None = None,
    n_iterations: int = -1,
):
    optimiser = la.Optimiser()
    if seed is not None:
        optimiser.set_rng_seed(seed)

    part = la.RBConfigurationVertexPartition(
        g_reg,
        weights="weight",
        resolution_parameter=gamma_reg,
    )
    diff = optimiser.optimise_partition(part, n_iterations=n_iterations)
    membership = np.array(part.membership, dtype=int)
    return membership, part, diff

def run_single_layer_leiden_cot(
    g_cot,
    gamma_cot: float = 0.1,
    seed: int | None = None,
    n_iterations: int = -1,
):
    optimiser = la.Optimiser()
    if seed is not None:
        optimiser.set_rng_seed(seed)

    part = la.RBConfigurationVertexPartition(
        g_cot,
        weights="weight",
        resolution_parameter=gamma_cot,
    )
    diff = optimiser.optimise_partition(part, n_iterations=n_iterations)
    membership = np.array(part.membership, dtype=int)
    return membership, part, diff

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
        List of TF names in the same order as consensus_labels.
    consensus_labels : np.ndarray of shape (n_tfs,)
        Cluster labels, one per TF.
    current_grn : pd.DataFrame
        GRN with at least columns: 'source' (TF), 'target' (regulated gene).

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
        'term' can be something like 'GO_Biological_Process_xxx_GO:0001234'.
    min_size, max_size : int
        Keep only terms with size in [min_size, max_size].
    go_term_levels : dict or None
        Optional mapping {go_id: level}, e.g. {"GO:0001234": 4, ...}.
        If provided, terms are also filtered by min_level / max_level.
    min_level, max_level : int or None
        GO levels to keep (inclusive). Ignored if go_term_levels is None.

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
    use_grn_universe: bool = True,
    min_genes_in_cluster: int = 5,
) -> pd.DataFrame:
    """
    Run hypergeometric ORA for each cluster against GO BP gene sets.

    Parameters
    ----------
    cluster2genes : dict
        Output of build_cluster_gene_sets.
    gene_sets : dict
        {go_term: set(genes)} (already filtered by size/level).
    use_grn_universe : bool
        If True, use as universe all genes (TFs + targets) in the GRN.
        If False, use as universe all genes present in the GO BP gene sets.
    min_genes_in_cluster : int
        Minimum number of genes in a cluster (after intersecting with universe)
        to perform enrichment.

    Returns
    -------
    results : pd.DataFrame
        Long table with columns:
        ['cluster_id', 'go_term', 'k', 'K', 'n', 'N', 'pval', 'pval_adj', 'overlap_genes']
    """
    # Determine universe
    if use_grn_universe:
        universe = set()
        for d in cluster2genes.values():
            universe.update(d["all_genes"])
        # Restrict to genes that appear in at least one GO set
        go_universe = set().union(*gene_sets.values())
        universe = universe & go_universe
    else:
        universe = set().union(*gene_sets.values())

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
    go_obo_path: str = "go-basic.obo",
    go_ids: Optional[Iterable[str]] = None,
    namespace: str = "biological_process",
) -> Dict[str, int]:
    """
    Compute GO levels using GOATOOLS.

    Parameters
    ----------
    go_obo_path : str
        Path to go-basic.obo (or another GO OBO file).
    go_ids : iterable of str or None
        Optional list of GO IDs to restrict to (e.g. those present in your GO BP library).
        If None, all GO terms in the DAG for the given namespace are considered.
    namespace : {'biological_process', 'molecular_function', 'cellular_component'}
        Restrict GO terms to this namespace.

    Returns
    -------
    go_term_levels : dict
        {go_id: level}, where 'level' is the GOATOOLS 'level' attribute
        (distance from the root term in the given namespace).
    """
    print(f"Loading GO DAG from {go_obo_path} ...")
    go_dag = GODag(go_obo_path)

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
    """
    go_ids = set()
    for term in gene_sets.keys():
        for token in term.split("("):
            if token.startswith("GO:"):
                token = token.strip(")")
                go_ids.add(token)
                break
    return go_ids
