# src/gcrl/grn/__init__.py
# -*- coding: utf-8 -*-
"""
Gene regulatory network (GRN) analysis tools for gCRL.

This subpackage provides utilities for analyzing gene regulatory networks,
including TF community detection via multiplex clustering and functional
enrichment analysis.

Modules:
--------
graph_construction : Build TF-TF multiplex networks (regulatory + co-target layers)
clustering : Leiden clustering (single-layer and multiplex)
stability : Clustering stability assessment (VI, ARI)
consensus : Consensus clustering from multiple runs
enrichment : GO term enrichment analysis for TF communities
eigengenes : Eigengene computation for TF communities
visualization : Network plots with community color coding and enrichment dotplots
"""

# Graph construction
from .graph_construction import (
    build_tf_tf_regulatory_layer,
    build_tf_tf_cotarget_layer,
)

# Clustering
from .clustering import (
    run_single_layer_leiden_reg,
    run_single_layer_leiden_cot,
)

# Stability assessment
from .stability import (
    filter_trivial_partitions,
    partition_stability_diagnostics,
)

# Consensus clustering
from .consensus import (
    build_coassociation_matrix,
    consensus_partition_from_coassoc,
    consensus_partition_majority_vote,
    community_stats,
)

# Enrichment analysis
from .enrichment import (
    build_cluster_gene_sets,
    filter_gene_sets_by_size_and_level,
    run_ora_for_clusters,
    filter_cluster_specific_terms,
    compute_go_levels,
    extract_go_ids_from_terms,
)

# Eigengenes (already exists)
from .eigengenes import compute_eigengenes

# Visualization
from .visualization import (
    plot_tf_tf_communities,
    plot_enrichment_dotplot,
    prettify_go_term,
)

__all__ = [
    # Graph construction
    "build_tf_tf_regulatory_layer",
    "build_tf_tf_cotarget_layer",
    # Clustering
    "run_single_layer_leiden_reg",
    "run_single_layer_leiden_cot",
    # Stability
    "filter_trivial_partitions",
    "partition_stability_diagnostics",
    # Consensus
    "build_coassociation_matrix",
    "consensus_partition_from_coassoc",
    "consensus_partition_majority_vote",
    "community_stats",
    # Enrichment
    "build_cluster_gene_sets",
    "filter_gene_sets_by_size_and_level",
    "run_ora_for_clusters",
    "filter_cluster_specific_terms",
    "compute_go_levels",
    "extract_go_ids_from_terms",
    # Eigengenes
    "compute_eigengenes",
    # Visualization
    "plot_tf_tf_communities",
    "plot_enrichment_dotplot",
    "prettify_go_term",
]
