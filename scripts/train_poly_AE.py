#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train a polynomial autoencoder on single-cell data
with TF-only inputs (default) and an all-genes reconstruction target.

Key features:
- Input scaling: reference-based per-gene z-score on UNPERTURBED cells (same cell type).
- Inputs: TF-only by default; pass --input_mode ALL to use all genes as inputs.
- Latent dimension: default = (# communities) + 1; override via --latent_dim.
- Decoder: fixed polynomial (constant + linear + quadratic/cross terms).
- Encoder: simple MLP, unmasked.
- Output: reconstruct ALL genes (TF + TG).
"""

import argparse
import json
import os
import random
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F


# ---------------------------- Utils ---------------------------- #

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_dense(x):
    return x.toarray() if hasattr(x, "toarray") else np.asarray(x)


def zscore_on_reference(adata, reference_query: str, layer_in=None, clip=5.0, layer_out="X_zref"):
    """
    Per-gene z-score using mean/std computed on reference cells, then applied to ALL cells.
    Reference: usually intervention == 'unperturbed' within the selected cell_type.
    """
    X = adata.X if layer_in is None else adata.layers[layer_in]
    X = to_dense(X).astype(np.float32)

    ref_idx = adata.obs.query(reference_query).index
    ref_pos = adata.obs_names.get_indexer(ref_idx)
    ref_pos = ref_pos[ref_pos >= 0]
    if ref_pos.size == 0:
        raise ValueError(f"No cells matched reference_query: {reference_query}")

    Xref = X[ref_pos, :]
    mu = Xref.mean(axis=0)
    sd = Xref.std(axis=0, ddof=0)
    sd[sd == 0] = 1.0

    Xz = (X - mu) / sd
    if clip is not None:
        Xz = np.clip(Xz, -clip, clip)

    adata.layers[layer_out] = Xz.astype(np.float32)
    adata.uns["zscore_reference"] = {
        "reference_query": reference_query,
        "mean": mu.astype(np.float32),
        "std": sd.astype(np.float32),
        "clip": clip,
        "layer_in": layer_in,
        "layer_out": layer_out
    }
    return adata


# ---------------------------- Model ---------------------------- #

class PolyDecoder(nn.Module):
    def __init__(self, latent_dim: int, output_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        # 1 (constant) + linear + upper-triangular quadratic
        poly_feats = 1 + latent_dim + (latent_dim * (latent_dim + 1)) // 2
        self.fc = nn.Linear(poly_feats, output_dim)

    def forward(self, z):
        B, K = z.shape
        device = z.device
        # constant
        cols = [torch.ones(B, 1, device=device)]
        # linear
        cols.append(z)
        # quadratic/cross (i<=j)
        for i in range(K):
            zi = z[:, i:i+1]
            for j in range(i, K):
                cols.append(zi * z[:, j:j+1])
        phi = torch.cat(cols, dim=1)
        return self.fc(phi)


class AE_Poly(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, output_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        self.decoder = PolyDecoder(latent_dim, output_dim)

    def forward(self, x_in):
        z = self.encoder(x_in)
        x_rec = self.decoder(z)
        return x_rec, z


# ---------------------------- Training Script ---------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Polynomial AE with TF-only inputs and all-genes reconstruction.")
    # Data & selection
    parser.add_argument("--h5ad_file", type=str, required=True, help="Input AnnData (.h5ad).")
    parser.add_argument("--cell_type", type=str, required=True, help="Train on this single cell_type.")
    parser.add_argument("--reference_intervention", type=str, default="unperturbed",
                        help="intervention used as reference for z-scoring (default: unperturbed).")
    parser.add_argument("--input_mode", type=str, choices=["TF", "ALL"], default="TF",
                        help="Inputs to the encoder: TF (default) or ALL genes.")
    # Latent dim
    parser.add_argument("--latent_dim", type=int, default=-1,
                        help="If -1, use (# communities) + 1; else use the provided value.")
    # Training
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_step", type=int, default=50)
    parser.add_argument("--lr_gamma", type=float, default=0.5)
    parser.add_argument("--cuda_device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    # Output folder
    parser.add_argument("--outdir", type=str, default="results_polyAE",
                        help="Folder where to save all outputs (default: results_polyAE).")
    parser.add_argument("--log_interval", type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Save run configuration for reproducibility
    with open(os.path.join(args.outdir, "run_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    set_seed(args.seed)

    # Device
    if args.cuda_device >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(args.cuda_device)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # ---- Load AnnData and subset to one cell type ----
    adata = sc.read_h5ad(args.h5ad_file)
    if "cell_type" not in adata.obs.columns:
        raise ValueError("adata.obs must contain 'cell_type'")
    if "intervention" not in adata.obs.columns:
        raise ValueError("adata.obs must contain 'intervention'")

    mask_ct = (adata.obs["cell_type"].astype(str) == args.cell_type)
    if mask_ct.sum() == 0:
        raise ValueError(f"No cells with cell_type == '{args.cell_type}'")

    adata = adata[mask_ct].copy()  # now only this cell type

    # ---- Determine latent dim default = (# communities) + 1 ----
    if "community" not in adata.var.columns:
        raise ValueError("adata.var must contain 'community' (gene community assignment)")
    comm = adata.var["community"].astype("category")
    n_comm = len(comm.cat.categories)
    latent_dim = (n_comm + 1) if args.latent_dim == -1 else args.latent_dim
    print(f"Latent dim = {latent_dim} (communities={n_comm}, default+1 unless overridden)")

    # ---- Build TF mask and choose inputs ----
    if "kind" not in adata.var.columns:
        raise ValueError("adata.var must contain 'kind' with values 'TF' or 'TG'")
    # Enforce allowed values for early failure
    allowed = {"TF", "TG"}
    kinds = set(map(str, adata.var["kind"].unique()))
    if not kinds.issubset(allowed):
        raise ValueError(f"adata.var['kind'] must be in {allowed}, found: {kinds}")

    tf_mask = (adata.var["kind"].astype(str) == "TF").to_numpy()
    n_genes = adata.n_vars
    n_tfs = int(tf_mask.sum())
    print(f"Genes: {n_genes} total; TFs: {n_tfs}")

    # ---- Reference-based scaling on UNPERTURBED cells of this cell type ----
    ref_query = f'intervention == "{args.reference_intervention}"'
    adata = zscore_on_reference(adata, reference_query=ref_query, layer_in=None, clip=5.0, layer_out="X_zref")

    # Matrices (inputs and targets)
    X_all = adata.layers["X_zref"]  # already dense float32
    if args.input_mode == "TF":
        X_in = X_all[:, tf_mask]
        input_dim = n_tfs
        print("Input mode: TF-only")
    else:
        X_in = X_all
        input_dim = n_genes
        print("Input mode: ALL genes")

    output_dim = n_genes  # reconstruct all genes
    gene_names = adata.var_names.astype(str).tolist()
    tf_names = adata.var_names[tf_mask].astype(str).tolist()
    cell_ids = adata.obs_names.astype(str).tolist()

    # ---- DataLoader ----
    X_in_t = torch.tensor(X_in, dtype=torch.float32)
    X_out_t = torch.tensor(X_all, dtype=torch.float32)
    dataset = TensorDataset(X_in_t, X_out_t)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=False)

    # ---- Model/Optim ----
    model = AE_Poly(input_dim=input_dim, latent_dim=latent_dim, output_dim=output_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)

    # ---- Train ----
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        total = 0.0
        for xb_in, xb_out in loader:
            xb_in = xb_in.to(device, non_blocking=True)
            xb_out = xb_out.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            x_pred, _ = model(xb_in)
            loss = F.mse_loss(x_pred, xb_out)
            loss.backward()
            optimizer.step()
            total += loss.item() * xb_in.size(0)
        scheduler.step()
        avg = total / len(dataset)
        if epoch == 1 or epoch % args.log_interval == 0:
            lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch}/{args.num_epochs} | loss {avg:.5f} | lr {lr:.2e}")

    model_path = os.path.join(args.outdir, "model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Saved model → {model_path}")

    # ---- Extract embeddings and export ----
    model.eval()
    with torch.no_grad():
        Z = model.encoder(X_in_t.to(device)).cpu().numpy()

    embeddings_npy = os.path.join(args.outdir, "cell_latent_embeddings.npy")
    embeddings_csv = os.path.join(args.outdir, "cell_latent_embeddings.csv")
    np.save(embeddings_npy, Z)
    pd.DataFrame(Z, index=cell_ids, columns=[f"Latent_{i+1}" for i in range(Z.shape[1])]).to_csv(embeddings_csv)
    print(f"Wrote embeddings → {embeddings_npy}, {embeddings_csv}")

    # ---- Decoder weights (linear terms only) ----
    W = model.decoder.fc.weight.detach().cpu().numpy()  # (n_genes, poly_feats)
    K = latent_dim
    lin_start = 1
    gene_weights_df = pd.DataFrame(
        {f"Latent_{k+1}": W[:, lin_start + k] for k in range(K)},
        index=gene_names
    )
    weights_csv = os.path.join(args.outdir, "all_gene_weights_by_latent.csv")
    gene_weights_df.to_csv(weights_csv)
    print(f"Wrote per-gene linear weights by latent → {weights_csv}")

    # ---- Correlate TF inputs with latent (for interpretability) ----
    # Use TF inputs matrix if input_mode==TF, else correlate latents with TF columns from all genes.
    if args.input_mode == "TF":
        TF_mat = X_in  # already TF-only
    else:
        TF_mat = X_all[:, tf_mask]

    Z_df = pd.DataFrame(Z, columns=[f"Latent_{i+1}" for i in range(K)])
    TF_df = pd.DataFrame(TF_mat, columns=tf_names)

    corr_matrix = pd.DataFrame(
        {f"Latent_{k+1}": TF_df.corrwith(Z_df[f"Latent_{k+1}"]) for k in range(K)},
        index=tf_names
    )

    # Produce a compact TF→top-genes report using linear weights of the best-correlated latent
    rows = []
    n_genes_total = len(gene_names)
    n_top = max(1, n_genes_total // 10)  # top 10% by abs(weight) per TF's best latent
    for tf in tf_names:
        corrs = corr_matrix.loc[tf]
        best_latent = corrs.abs().idxmax()
        best_corr = corrs[best_latent]
        w = gene_weights_df[best_latent]
        top_genes = w.abs().nlargest(n_top).index
        for g in top_genes:
            rows.append({"TF": tf, "Best_Latent": best_latent, "TF_Corr": best_corr, "Gene": g, "Weight": w.loc[g]})

    influence_csv = os.path.join(args.outdir, "tf_gene_influence.csv")
    pd.DataFrame(rows).to_csv(influence_csv, index=False)
    print(f"Wrote TF→top-genes influence table → {influence_csv}")


if __name__ == "__main__":
    main()
