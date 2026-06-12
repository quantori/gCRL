# src/gcrl/grn/clustering.py
# -*- coding: utf-8 -*-
"""
Leiden clustering utilities for TF community detection.

This module provides functions for running Leiden clustering on TF-TF networks:
- Single-layer clustering on regulatory or co-target graphs
- Multiplex clustering combining both layers
- Hyperparameter sweeps for stability analysis

Key functions:
--------------
run_single_layer_leiden_reg : Run Leiden on regulatory layer
run_single_layer_leiden_cot : Run Leiden on co-target layer
run_multiplex_leiden : Run multiplex Leiden on both layers
sweep_hyperparams : Run multiplex Leiden across hyperparameter grid
"""

from __future__ import annotations
import numpy as np
import igraph as ig
import leidenalg as la


def run_single_layer_leiden_reg(
    g_reg,
    gamma_reg: float = 1.0,
    seed: int | None = None,
    n_iterations: int = -1,
):
    """
    Run single-layer Leiden clustering on the regulatory graph.

    Parameters
    ----------
    g_reg : igraph.Graph
        Regulatory TF-TF graph
    gamma_reg : float, default=1.0
        Resolution parameter for RBConfigurationVertexPartition
    seed : int or None
        Random seed for reproducibility
    n_iterations : int, default=-1
        Number of iterations (-1 = until convergence)

    Returns
    -------
    membership : np.ndarray
        Cluster labels for each TF
    part : leidenalg.RBConfigurationVertexPartition
        Partition object
    diff : float
        Quality improvement in final iteration
    """
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
    gamma_cot: float = 1.0,
    seed: int | None = None,
    n_iterations: int = -1,
):
    """
    Run single-layer Leiden clustering on the co-target graph.

    Parameters
    ----------
    g_cot : igraph.Graph
        Co-target TF-TF graph
    gamma_cot : float, default=1.0
        Resolution parameter for RBConfigurationVertexPartition
    seed : int or None
        Random seed for reproducibility
    n_iterations : int, default=-1
        Number of iterations (-1 = until convergence)

    Returns
    -------
    membership : np.ndarray
        Cluster labels for each TF
    part : leidenalg.RBConfigurationVertexPartition
        Partition object
    diff : float
        Quality improvement in final iteration
    """
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
    - Layer 2: co-target TF–TF graph (RBConfigurationVertexPartition)

    Parameters
    ----------
    g_reg : igraph.Graph
        Regulatory TF-TF graph
    g_cot : igraph.Graph
        Co-target TF-TF graph
    gamma_reg : float, default=1.0
        Resolution parameter for RBConfiguration on the regulatory layer
    gamma_cot : float, default=1.0
        Resolution parameter for RBConfiguration on the co-target layer
    layer_weight_reg : float, default=1.0
        Weight of the regulatory layer in the combined objective
    layer_weight_cot : float, default=1.0
        Weight of the co-target layer in the combined objective
    n_iterations : int, default=-1
        Number of Leiden iterations; -1 means until convergence
    seed : int or None
        Random seed for reproducibility

    Returns
    -------
    membership : np.ndarray
        Cluster labels for each TF (same across both layers)
    partitions : list
        List of partition objects for each layer
    diff : float
        Quality improvement in final iteration
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

    # Layer 2: RBConfiguration on co-target graph
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
    n_seeds: int = 100,
    n_iterations: int = -1,
):
    """
    Run multiplex Leiden for multiple (gamma_reg, gamma_cot, layer_weight_cot) combinations
    and multiple seeds for each combination.

    We keep the regulatory layer weight fixed at 1.0 and vary the co-target
    layer weight to control its influence.

    Parameters
    ----------
    g_reg : igraph.Graph
        Regulatory TF-TF graph
    g_cot : igraph.Graph
        Co-target TF-TF graph
    gamma_reg_list : list of float
        Resolution parameters to try for regulatory layer
    gamma_cot_list : list of float
        Resolution parameters to try for co-target layer
    cot_weight_list : list of float
        Layer weights to try for co-target layer (regulatory layer = 1.0)
    n_seeds : int, default=100
        Number of random seeds per hyperparameter combination
    n_iterations : int, default=-1
        Number of Leiden iterations (-1 = until convergence)

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
