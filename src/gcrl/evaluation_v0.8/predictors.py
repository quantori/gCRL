from __future__ import annotations
from typing import Protocol
import numpy as np
import anndata as ad
from .config import EvalConfig

class InterventionalPredictor(Protocol):
    def predict_group(
        self,
        controls_train_adata: ad.AnnData,
        intervention: str,
        cell_type: str,
        n_pred: int,
        rng: np.random.Generator,
    ) -> ad.AnnData: ...

class LinearDeltaPredictor:
    """Data-driven baseline:
    For each (CT, intervention), estimate gene-wise Δ from TRAINING data as:
        Δ = mean(training perturbed group) - mean(training controls of same CT)
    Predictions are synthesized by sampling controls of that CT and adding Δ.
    """
    def __init__(self, adata: ad.AnnData, cfg: EvalConfig):
        self.cfg = cfg
        self.adata = adata
        self._precompute()

    def _precompute(self):
        obs = self.adata.obs
        set_key = self.cfg.set_key
        int_key = self.cfg.intervention_key
        ct_key = self.cfg.cell_type_key
        ctl = self.cfg.control_label

        mask_train = (obs[set_key] == "training")
        train = self.adata[mask_train].copy()

        self.mean_ctl_ct = {}
        self.delta = {}
        ints = np.unique(train.obs[int_key])
        for ct in np.unique(train.obs[ct_key]):
            m_ctl = (train.obs[ct_key] == ct) & (train.obs[int_key] == ctl)
            if m_ctl.sum() == 0: 
                continue
            cmean = np.asarray(train[m_ctl].X).mean(axis=0)
            self.mean_ctl_ct[ct] = cmean
            for itv in ints:
                if itv == ctl: 
                    continue
                m_grp = (train.obs[ct_key] == ct) & (train.obs[int_key] == itv)
                if m_grp.sum() == 0:
                    continue
                gmean = np.asarray(train[m_grp].X).mean(axis=0)
                self.delta[(ct, itv)] = gmean - cmean

    def predict_group(
        self,
        controls_train_adata: ad.AnnData,
        intervention: str,
        cell_type: str,
        n_pred: int,
        rng: np.random.Generator,
    ) -> ad.AnnData:
        d = self.delta.get((cell_type, intervention), np.zeros(controls_train_adata.n_vars, dtype=float))
        idx = rng.integers(0, controls_train_adata.n_obs, size=n_pred)
        Xc = np.asarray(controls_train_adata.X)[idx]
        Xp = Xc + d
        pred = ad.AnnData(Xp, var=controls_train_adata.var.copy())
        pred.obs = controls_train_adata.obs.iloc[idx].copy()
        pred.obs[self.cfg.set_key] = "pred"
        pred.obs[self.cfg.cell_type_key] = cell_type
        pred.obs[self.cfg.intervention_key] = intervention
        return pred

class GcrlVaePredictor:
    """Adapter template for your src/gcrl/models/gcrl_vae.py implementation.
    Implement ONE of the following model APIs:
    1) model.predict_group(controls_adata, intervention, cell_type, n_pred, rng) -> AnnData  (preferred)
    2) Provide encode/ intervene/ decode callables via attributes:
       - model.encode_batch(X, ct=None) -> dict with 'mu' (B x z) and optionally 'logvar'
       - model.apply_intervention(mu, intervention, cell_type) -> z_int (B x z)
       - model.decode_batch(z) -> Xhat (B x G)
    If neither is available, this class will raise NotImplementedError.
    """
    def __init__(self, model, cfg: EvalConfig):
        self.model = model
        self.cfg = cfg

    def predict_group(
        self,
        controls_train_adata: ad.AnnData,
        intervention: str,
        cell_type: str,
        n_pred: int,
        rng: np.random.Generator,
    ) -> ad.AnnData:
        # Option 1: the model already exposes predict_group
        if hasattr(self.model, 'predict_group'):
            return self.model.predict_group(controls_train_adata, intervention, cell_type, n_pred, rng)

        # Option 2: compose encode → intervene → decode
        if not all(hasattr(self.model, attr) for attr in ['encode_batch','apply_intervention','decode_batch']):
            raise NotImplementedError("Your gCRL-VAE must implement either predict_group(...) or the trio encode_batch/apply_intervention/decode_batch.")

        idx = rng.integers(0, controls_train_adata.n_obs, size=n_pred)
        Xc = np.asarray(controls_train_adata.X)[idx]
        # Optional CT vector if your model expects it; here we pass None
        enc = self.model.encode_batch(Xc, ct=None)  # expected to return {'mu': ..., ...}
        mu = enc['mu']
        z_int = self.model.apply_intervention(mu, intervention, cell_type)
        Xp = self.model.decode_batch(z_int)

        pred = ad.AnnData(Xp, var=controls_train_adata.var.copy())
        pred.obs = controls_train_adata.obs.iloc[idx].copy()
        pred.obs[self.cfg.set_key] = "pred"
        pred.obs[self.cfg.cell_type_key] = cell_type
        pred.obs[self.cfg.intervention_key] = intervention
        return pred
