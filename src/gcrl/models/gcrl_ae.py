
# src/gcrl/models/gcrl_ae.py
# -*- coding: utf-8 -*-
"""
gCRL-AE model components.

- Polynomial decoder of latent factors (constant + linear + quadratic terms).
- GCRLAE: MLP encoder + polynomial decoder.
"""

from __future__ import annotations
from typing import Sequence
import torch
from torch import nn


class PolyDecoder(nn.Module):
    """
    Polynomial decoder Φ(z) = W · [1, z, z_i z_j (i<=j)].
    output_dim typically equals the number of genes to reconstruct.
    """
    def __init__(self, latent_dim: int, output_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        poly_feats = 1 + latent_dim + (latent_dim * (latent_dim + 1)) // 2
        self.fc = nn.Linear(poly_feats, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        batch = z.size(0)
        # Constant term
        phi = [torch.ones(batch, 1, device=z.device, dtype=z.dtype)]
        # Linear terms
        phi.append(z)
        # Quadratic and cross terms
        for i in range(self.latent_dim):
            zi = z[:, i]
            for j in range(i, self.latent_dim):
                phi.append((zi * z[:, j]).unsqueeze(1))
        phi = torch.cat(phi, dim=1)
        return self.fc(phi)


class GCRLAE(nn.Module):
    """
    gCRL Autoencoder with MLP encoder and polynomial decoder.

    Args
    ----
    input_dim : int
        Number of input features (TF-only or all genes).
    latent_dim : int
        Size of latent space.
    hidden_dims : Sequence[int]
        Hidden layer sizes for the encoder MLP.
    activation : nn.Module
        Nonlinearity used in the encoder (default ReLU).
    output_dim : int
        Number of outputs to reconstruct (typically all genes).
    """
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: Sequence[int] = (256,),
        activation: nn.Module = nn.ReLU(),
        output_dim: int | None = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim if output_dim is not None else input_dim

        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(activation)
            prev = h
        layers.append(nn.Linear(prev, latent_dim))
        self.encoder = nn.Sequential(*layers)

        self.decoder = PolyDecoder(latent_dim, self.output_dim)

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return x_rec, z
