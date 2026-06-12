# src/gcrl/grn/visualization.py
# -*- coding: utf-8 -*-
"""
Visualization utilities for TF community networks and enrichment results.

Key functions:
--------------
plot_tf_tf_communities  : Plot TF–TF network with community color coding
plot_enrichment_dotplot : clusterProfiler-style dotplot for GO enrichment results
prettify_go_term        : Convert raw gseapy GO term strings to readable labels
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ---------------------------------------------------------------------------
# Network visualization
# ---------------------------------------------------------------------------

def _community_layout(G, tf2cluster, seed, k):
    """
    Supervised community-aware spring layout.

    Algorithm:
      1. Place community centroids evenly on a unit circle.
      2. For each community, run a local spring layout on the subgraph,
         then translate and scale it to sit around its centroid.
      3. The inter-community spacing is governed by the circle radius
         (proportional to sqrt(n_communities)), and the intra-community
         spread by k_local (proportional to 1/sqrt(community_size)).

    This guarantees that same-community nodes cluster together regardless
    of how many inter-community edges exist.
    """
    communities = {}
    for node in G.nodes():
        cl = tf2cluster[node]
        communities.setdefault(cl, []).append(node)

    unique_clusters = sorted(communities.keys())
    n_cl = len(unique_clusters)

    # Centroid positions on a circle
    radius = max(1.5, np.sqrt(n_cl))
    angles = {cl: 2 * np.pi * i / n_cl for i, cl in enumerate(unique_clusters)}
    centroids = {
        cl: np.array([radius * np.cos(angles[cl]), radius * np.sin(angles[cl])])
        for cl in unique_clusters
    }

    pos = {}
    rng = np.random.default_rng(seed)

    for cl, nodes in communities.items():
        centroid = centroids[cl]
        n = len(nodes)

        if n == 1:
            pos[nodes[0]] = centroid
            continue

        subG = G.subgraph(nodes).copy()
        k_local = k / max(1.0, np.sqrt(n))

        # Seed positions tightly around centroid so the spring layout
        # stays localised and doesn't drift toward other communities
        init_pos = {
            node: centroid + rng.uniform(-0.1, 0.1, size=2)
            for node in nodes
        }

        sub_pos = nx.spring_layout(
            subG,
            k=k_local,
            pos=init_pos,
            fixed=None,
            seed=seed,
            weight="weight",
        )

        # Normalise and scale to a blob of radius ~0.8 around centroid
        coords = np.array([sub_pos[n] for n in nodes])
        coords -= coords.mean(axis=0)
        span = np.linalg.norm(coords, axis=1).max()
        if span > 0:
            coords = coords / span * 0.8
        coords += centroid

        for node, xy in zip(nodes, coords):
            pos[node] = xy

    return pos


def plot_tf_tf_communities(
    current_grn: pd.DataFrame,
    tf_names,
    consensus_labels,
    weight_col: str = "coef_abs",
    source_col: str = "source",
    target_col: str = "target",
    directed: bool = False,
    min_edge_weight: float = 0.0,
    hide_isolates: bool = True,
    layout: str = "community",
    k: float = 0.7,
    seed: int = 42,
    figsize=(12, 12),
    node_size: int = 250,
    with_labels: bool = False,
    label_top_n_degree: int = 0,
    edge_alpha_intra: float = 0.35,
    edge_alpha_inter: float = 0.08,
    edge_width_scale: float = 2.0,
    title: str = "TF\u2013TF network with communities",
    show: bool = True,
):
    """
    Plot a TF–TF network highlighting community assignments.

    Parameters
    ----------
    current_grn : pd.DataFrame
        GRN with at least source_col, target_col, and optionally weight_col.
    tf_names : list-like
        TF names in the same order as consensus_labels.
    consensus_labels : array-like
        Community labels for tf_names.
    weight_col : str
        Column containing edge weights. If absent, weight=1 is used.
    source_col : str
        Column name for edge sources (default: 'source').
    target_col : str
        Column name for edge targets (default: 'target').
    directed : bool
        If False, build an undirected graph (default).
    min_edge_weight : float
        Filter out edges below this threshold.
    hide_isolates : bool
        If True (default), remove TFs with no TF–TF edges from the plot.
    layout : str
        One of {'community', 'spring', 'kamada_kawai', 'circular', 'spectral'}.
        'community' (default) uses a supervised layout that places each community
        as a distinct blob on a circle — recommended for highlighting structure.
    k : float
        Spring layout spacing parameter (also controls intra-community spread
        in 'community' layout).
    seed : int
        Random seed for reproducible layouts.
    figsize : tuple
        Figure size.
    node_size : int
        Node size for drawing.
    with_labels : bool
        If True, draw all node labels.
    label_top_n_degree : int
        If > 0, label only the top N TFs by weighted degree.
    edge_alpha_intra : float
        Alpha for within-community edges.
    edge_alpha_inter : float
        Alpha for between-community edges.
    edge_width_scale : float
        Scale factor applied to edge weights for line width.
    title : str
        Figure title.
    show : bool
        If True (default), call plt.show() at the end.
        Set to False to suppress display and handle the figure yourself
        (e.g. to save it before showing).

    Returns
    -------
    G : networkx.Graph or networkx.DiGraph
        The TF–TF graph used for plotting.
    pos : dict
        Node positions {node: (x, y)}.
    """
    tf_names = np.asarray(tf_names)
    consensus_labels = np.asarray(consensus_labels)

    if len(tf_names) != len(consensus_labels):
        raise ValueError("tf_names and consensus_labels must have the same length.")

    tf_set = set(tf_names)
    tf2cluster = dict(zip(tf_names, consensus_labels))

    cols = [source_col, target_col]
    if weight_col in current_grn.columns:
        cols.append(weight_col)
    df = current_grn[cols].copy()

    # Keep only TF -> TF edges
    df = df[df[source_col].isin(tf_set) & df[target_col].isin(tf_set)].copy()

    if weight_col not in df.columns:
        df[weight_col] = 1.0

    df = df[df[weight_col] >= min_edge_weight].copy()

    # Aggregate duplicate edges
    if directed:
        df = (
            df.groupby([source_col, target_col], as_index=False)[weight_col]
            .sum()
        )
        G = nx.DiGraph()
    else:
        df["_u"] = df[[source_col, target_col]].min(axis=1)
        df["_v"] = df[[source_col, target_col]].max(axis=1)
        df = (
            df.groupby(["_u", "_v"], as_index=False)[weight_col]
            .sum()
            .rename(columns={"_u": source_col, "_v": target_col})
        )
        G = nx.Graph()

    # Add all TFs as nodes, even isolated ones
    for tf in tf_names:
        G.add_node(tf, cluster=int(tf2cluster[tf]))

    for _, row in df.iterrows():
        G.add_edge(row[source_col], row[target_col], weight=float(row[weight_col]))

    if hide_isolates:
        isolates = list(nx.isolates(G))
        if isolates:
            G.remove_nodes_from(isolates)

    # Layout
    if layout == "community":
        pos = _community_layout(G, tf2cluster, seed=seed, k=k)
    elif layout == "spring":
        pos = nx.spring_layout(G, seed=seed, k=k, weight="weight")
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G, weight="weight")
    elif layout == "spectral":
        pos = nx.spectral_layout(G, weight="weight")
    elif layout == "circular":
        pos = nx.circular_layout(G)
    else:
        raise ValueError(
            "layout must be one of {'community', 'spring', 'kamada_kawai', 'circular', 'spectral'}"
        )

    # Community colors — derived from nodes present in G (isolates may have been removed)
    unique_clusters = np.sort(np.unique([tf2cluster[n] for n in G.nodes()]))
    cmap = matplotlib.colormaps.get_cmap("tab20").resampled(len(unique_clusters))
    cluster2color = {cl: cmap(i) for i, cl in enumerate(unique_clusters)}

    node_colors = [cluster2color[tf2cluster[n]] for n in G.nodes()]

    # Edge styling
    edge_colors = []
    edge_widths = []
    for u, v, d in G.edges(data=True):
        same_cluster = tf2cluster[u] == tf2cluster[v]
        alpha = edge_alpha_intra if same_cluster else edge_alpha_inter
        r, g_ch, b, _ = cluster2color[tf2cluster[u]] if same_cluster else (0.6, 0.6, 0.6, 1.0)
        edge_colors.append((r, g_ch, b, alpha))
        edge_widths.append(max(0.2, d.get("weight", 1.0) * edge_width_scale))

    fig, ax = plt.subplots(figsize=figsize)

    nx.draw_networkx_edges(
        G, pos,
        ax=ax,
        edge_color=edge_colors,
        width=edge_widths,
    )

    nx.draw_networkx_nodes(
        G, pos,
        ax=ax,
        node_color=node_colors,
        node_size=node_size,
        linewidths=0.6,
        edgecolors="black",
    )

    # Labels
    labels_to_draw = {}
    if with_labels:
        labels_to_draw = {n: n for n in G.nodes()}
    elif label_top_n_degree > 0:
        deg = dict(G.degree(weight="weight"))
        top_nodes = sorted(deg, key=deg.get, reverse=True)[:label_top_n_degree]
        labels_to_draw = {n: n for n in top_nodes}

    if labels_to_draw:
        for node, label in labels_to_draw.items():
            x, y = pos[node]
            ax.annotate(
                label,
                xy=(x, y),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.8,
                ),
            )

    # Legend
    legend_handles = [
        Line2D(
            [0], [0],
            marker="o", color="w",
            markerfacecolor=cluster2color[cl],
            markeredgecolor="black",
            markersize=8,
            label=f"Cluster {cl}",
        )
        for cl in unique_clusters
    ]
    ax.legend(handles=legend_handles, loc="best", frameon=True, fontsize=8)

    ax.set_title(title)
    ax.set_axis_off()
    plt.tight_layout()
    if show:
        plt.show()

    return G, pos


# ---------------------------------------------------------------------------
# Enrichment visualization
# ---------------------------------------------------------------------------

def prettify_go_term(term: str) -> str:
    """
    Convert a raw gseapy GO term string to a readable label.

    Handles both formats produced by gseapy libraries:
      - "apoptotic process (GO:0006915)"  → "apoptotic process"
      - "GO_Biological_Process_apoptotic_process_GO:0006915" → "apoptotic process"

    Parameters
    ----------
    term : str
        Raw term string from the enrichment results.

    Returns
    -------
    str
        Human-readable term label with the GO ID stripped.
    """
    # Format 1: "Term name (GO:xxxxxxx)" — produced by gseapy GO BP 2021+
    if "(" in term and "GO:" in term:
        return term[:term.rfind("(")].strip()

    # Format 2: "prefix_word1_word2_GO:xxxxxxx" — older gseapy libraries
    tokens = term.split("_")
    if tokens and tokens[-1].startswith("GO:"):
        return " ".join(tokens[:-1])

    # Fallback: just replace underscores
    return term.replace("_", " ")


def plot_enrichment_dotplot(
    enrich_df: pd.DataFrame,
    cluster_id=None,
    top_n: int = 15,
    fdr_col: str = "pval_adj",
    term_col: str = "go_term",
    overlap_col: str = "k",
    query_size_col: str = "n",
    title: str = None,
    figsize=(8, 6),
    sort_by: str = "pval_adj",
    term_parser=prettify_go_term,
    show: bool = True,
):
    """
    clusterProfiler-style dotplot for GO enrichment results.

    Each dot represents one GO term:
      - x-axis : GeneRatio = overlap / cluster_size
      - y-axis : GO term label (top terms, ordered best-to-worst top-to-bottom)
      - dot size : number of overlapping genes (k)
      - dot color : -log10(FDR)

    Parameters
    ----------
    enrich_df : pd.DataFrame
        Enrichment table as returned by run_ora_for_clusters, with columns:
        'cluster_id', 'go_term', 'k' (overlap), 'n' (cluster size),
        'pval_adj' (FDR-corrected p-value).
    cluster_id : int or None
        If provided, subset to this cluster_id. If None, uses all rows.
    top_n : int
        Number of top terms to display.
    fdr_col : str
        Column name for the adjusted p-value (default: 'pval_adj').
    term_col : str
        Column name for GO term strings (default: 'go_term').
    overlap_col : str
        Column name for overlap count k (default: 'k').
    query_size_col : str
        Column name for cluster gene-set size n (default: 'n').
    title : str or None
        Plot title. Auto-generated if None.
    figsize : tuple
        Figure size.
    sort_by : str
        Column to sort terms by before selecting top_n (default: 'pval_adj').
    term_parser : callable or None
        Function mapping raw term string → display label.
        Defaults to prettify_go_term. Pass None to display raw strings.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    df = enrich_df.copy()

    if cluster_id is not None:
        df = df[df["cluster_id"] == cluster_id].copy()

    if df.empty:
        raise ValueError(f"No enrichment rows found for cluster_id={cluster_id}.")

    # Drop rows with invalid FDR values
    df = df[np.isfinite(df[fdr_col]) & (df[fdr_col] > 0)].copy()
    if df.empty:
        raise ValueError("No rows with valid (finite, positive) adjusted p-values.")

    # Derived columns
    df["GeneRatio"] = df[overlap_col] / df[query_size_col]
    df["minus_log10_fdr"] = -np.log10(df[fdr_col])

    # Term labels — fall back to raw string if parser returns NaN/None
    if term_parser is not None:
        df["term_label"] = df[term_col].map(term_parser)
        mask_missing = df["term_label"].isna() | (df["term_label"] == "")
        df.loc[mask_missing, "term_label"] = df.loc[mask_missing, term_col]
    else:
        df["term_label"] = df[term_col]

    # Select top_n terms, sorted best-to-worst by sort_by criterion
    ascending = sort_by in {fdr_col, "pval", "pval_adj", "fdr"}
    df = df.sort_values(sort_by, ascending=ascending).head(top_n).copy()

    # Re-order rows by GeneRatio ascending so that the y-axis runs
    # low GeneRatio (bottom) → high GeneRatio (top), making the plot
    # easy to read left-to-right along the x-axis.
    df = df.sort_values("GeneRatio", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=figsize)

    sc = ax.scatter(
        df["GeneRatio"],
        df["term_label"],
        s=df[overlap_col] * 25,
        c=df["minus_log10_fdr"],
        cmap="viridis",
        edgecolor="black",
        linewidth=0.5,
    )

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("-log₁₀(FDR)")

    ax.set_xlabel("GeneRatio  (overlap / cluster size)")
    ax.set_ylabel("")

    if title is None:
        title = "GO BP enrichment"
        if cluster_id is not None:
            title += f" — Cluster {cluster_id}"
    ax.set_title(title)

    plt.tight_layout()
    if show:
        plt.show()

    return fig, ax
