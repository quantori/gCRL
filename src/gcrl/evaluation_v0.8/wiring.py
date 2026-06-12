# src/gcrl/evaluation/wiring.py
from __future__ import annotations
import json
from typing import Dict, List, Sequence, Optional
import numpy as np
import anndata as ad

def _get_ct_names(adata: ad.AnnData, ct_key: str = "cell_type") -> List[str]:
    if ct_key not in adata.obs:
        return []
    ct = adata.obs[ct_key]
    return list(ct.cat.categories if hasattr(ct, "cat") else sorted(ct.unique()))

def _get_tf_names(
    adata: ad.AnnData,
    tf_flag_key: str = "kind",
    tf_label: str = "TF",
    tf_order_from_uns_key: Optional[str] = None,
) -> List[str]:
    """
    Returns the *ordered* list of TF names that define the c-vector (length c_dim).
    Priority:
      1) adata.uns[tf_order_from_uns_key] if provided
      2) order of adata.var_names where var[tf_flag_key]==tf_label
    """
    if tf_order_from_uns_key and tf_order_from_uns_key in adata.uns:
        names = list(adata.uns[tf_order_from_uns_key])
        return names
    if tf_flag_key not in adata.var:
        raise KeyError(f"adata.var lacks '{tf_flag_key}'; cannot infer TFs")
    mask = (adata.var[tf_flag_key] == tf_label)
    names = list(adata.var_names[mask])
    if not names:
        raise ValueError("No TFs found (var[kind]=='TF').")
    return names

def _default_tf_to_latent(
    tf_names: Sequence[str],
    adata: ad.AnnData,
    z_dim: int,
    community_key: str = "community",
) -> np.ndarray:
    """
    Default hard routing: map each TF to a latent by its 'community' (if present),
    else round-robin onto z-dims.
    """
    if community_key in adata.var:
        comm = adata.var[community_key].astype("category") if hasattr(adata.var[community_key], "astype") else adata.var[community_key]
        comm_idx = {g: i for i, g in enumerate(sorted(set(comm)))}
        out = np.zeros(len(tf_names), dtype=int)
        for i, tf in enumerate(tf_names):
            if tf in adata.var_names:
                ci = adata.var.loc[tf, community_key]
                out[i] = comm_idx.get(ci, i % z_dim)
            else:
                out[i] = i % z_dim
        return out
    # Fallback: round-robin
    return np.arange(len(tf_names), dtype=int) % max(1, z_dim)

def wire_for_eval(
    model,
    adata: ad.AnnData,
    *,
    tf_flag_key: str = "kind",
    tf_label: str = "TF",
    tf_order_from_uns_key: Optional[str] = None,
    community_key: str = "community",
    ct_key: str = "cell_type",
    save_json_to: Optional[str] = None,
):
    """
    Populate the model's evaluation wiring:
      - model.set_tf_index(tf_names)
      - if hard routing: model.set_tf_to_latent(tf_to_latent)
      - if ct_dim>0: model.set_ct_index(ct_names)
    Optionally saves a JSON manifest with the wiring for reproducibility.
    """
    # TF list in the *exact order* of the c-vector
    tf_names = _get_tf_names(adata, tf_flag_key=tf_flag_key, tf_label=tf_label, tf_order_from_uns_key=tf_order_from_uns_key)
    model.set_tf_index(tf_names)

    # Hard routing needs tf_to_latent
    if getattr(model.cfg, "intervention_mapping", "hard") == "hard":
        tf_to_latent = _default_tf_to_latent(tf_names, adata, z_dim=model.cfg.z_dim, community_key=community_key)
        model.set_tf_to_latent(tf_to_latent)

    # Optional CT conditioning
    if getattr(model.cfg, "ct_dim", 0) > 0:
        ct_names = _get_ct_names(adata, ct_key=ct_key)
        if len(ct_names) == model.cfg.ct_dim:
            model.set_ct_index(ct_names)
        elif model.cfg.ct_dim == 0:
            pass
        else:
            # Be permissive: if you didn't plan to use CT conditioning, leave it unset
            # Otherwise, enforce exact length match.
            # raise ValueError(f"ct_dim={model.cfg.ct_dim} but found {len(ct_names)} unique cell types")
            pass

    if save_json_to:
        payload = {
            "tf_names": tf_names,
            "hard_mapping": getattr(model.cfg, "intervention_mapping", "hard") == "hard",
            "tf_to_latent": (model.tf_to_latent.detach().cpu().tolist() if getattr(model, "tf_to_latent", None) is not None else None),
            "ct_dim": getattr(model.cfg, "ct_dim", 0),
            "ct_names": _get_ct_names(adata, ct_key=ct_key),
        }
        with open(save_json_to, "w") as f:
            json.dump(payload, f, indent=2)

    return model
