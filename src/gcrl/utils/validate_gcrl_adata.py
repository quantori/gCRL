"""
==========================================================
Requirements for the AnnData object in gCRL-VAE (p+1 schema)
==========================================================

1. Expression matrix
   - adata.X must contain normalized, log1p-transformed expression values
   - Shape: (n_cells × n_genes)

2. Gene-level metadata (adata.var)
   - Required columns:
       kind: "TF" or "TG" (transcription factor vs target gene)
       community: community ID (categorical or integer)
   - adata.var_names must include all genes referenced in adata.obs['intervention']

3. Cell-level metadata (adata.obs)
   - Required columns:
       cell_type: categorical label for each cell’s type
       intervention: string describing the perturbation (e.g. "TF1", "TF1+TF2")
   - Allowed delimiters in intervention: + , ;
   - "control", "unperturbed", "na" (case-insensitive) are interpreted as controls
   - All TFs in intervention must exist in adata.var_names and have kind == "TF"

4. Optional precomputed eigengenes (recommended for efficiency)
   - adata.obsm["X_comm_eig"]: matrix (n_cells × M) of per-cell eigengenes
   - adata.uns["X_comm_eig_comm_ids"]: list of community IDs for eigengene columns
   - adata.uns["X_comm_eig_global_index"]: integer index of the global eigengene column
   - If metadata is missing, training assumes first p columns = communities, last column = global

==========================================================
This validator checks all required fields and raises informative errors
if something is missing or inconsistent.
==========================================================
"""

import numpy as np
import pandas as pd

def validate_gcrl_adata(adata):
    # 1. Expression matrix
    if adata.X is None:
        raise ValueError("adata.X is missing. It must contain normalized, log1p expression values.")

    # 2. Gene-level metadata
    for col in ["kind", "community"]:
        if col not in adata.var.columns:
            raise KeyError(f"adata.var must contain column '{col}'")

    if not np.all(adata.var["kind"].isin(["TF", "TG"])):
        bad = adata.var["kind"].unique()
        raise ValueError(f"adata.var['kind'] must contain only 'TF' or 'TG' (found {bad})")

    # 3. Cell-level metadata
    for col in ["cell_type", "intervention"]:
        if col not in adata.obs.columns:
            raise KeyError(f"adata.obs must contain column '{col}'")

    # Validate interventions
    import re
    def _parse_condition_to_tfset(s):
        if s is None:
            return []
        s = str(s)
        if s.strip().lower() in ("", "control", "unperturbed", "na"):
            return []
        parts = re.split(r"[+;,]", s)
        return [p.strip() for p in parts if p.strip()]

    all_intervened_tfs = set()
    for v in adata.obs["intervention"].values:
        all_intervened_tfs.update(_parse_condition_to_tfset(v))

    name2idx = {g: i for i, g in enumerate(adata.var_names.values)}
    bad = []
    for g in all_intervened_tfs:
        j = name2idx.get(g, None)
        if j is None or str(adata.var.loc[g, "kind"]) != "TF":
            bad.append(g)
    if bad:
        raise ValueError(f"Interventions reference unknown or non-TF genes: {bad}")

    # 4. Optional eigengenes
    if "X_comm_eig" in adata.obsm:
        if not isinstance(adata.obsm["X_comm_eig"], (np.ndarray, pd.DataFrame)):
            raise TypeError("adata.obsm['X_comm_eig'] must be a numpy array or DataFrame")
        if "X_comm_eig_comm_ids" not in adata.uns or "X_comm_eig_global_index" not in adata.uns:
            print("Warning: adata.obsm['X_comm_eig'] found but metadata in adata.uns is missing. "
                  "Training will assume first p columns = communities, last column = global.")

    print("✅ adata object passed all gCRL-VAE validation checks.")
    return True
