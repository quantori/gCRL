from __future__ import annotations
import numpy as np
import anndata as ad
from .config import EvalConfig

class BaselineProvider:
    """Provides per-cell-type control means (biased negative) and µ_all (null)."""
    def __init__(self, adata: ad.AnnData, cfg: EvalConfig):
        self.cfg = cfg
        self.adata = adata
        self._build()

    def _build(self):
        obs = self.adata.obs
        set_key = self.cfg.set_key
        int_key = self.cfg.intervention_key
        ct_key = self.cfg.cell_type_key
        ctl = self.cfg.control_label

        train = self.adata[obs[set_key] == "training"].copy()

        # CT control means (biased negative baseline)
        self.mean_ctl_ct = {}
        for ct in np.unique(train.obs[ct_key]):
            m = (train.obs[ct_key] == ct) & (train.obs[int_key] == ctl)
            if m.sum() == 0:
                continue
            self.mean_ctl_ct[ct] = np.asarray(train[m].X).mean(axis=0)

        # µ_all (null baseline)
        if self.cfg.mu_all_per_ct:
            self.mu_all_ct = {}
            for ct in np.unique(train.obs[ct_key]):
                m = (train.obs[ct_key] == ct) & (train.obs[int_key] != ctl)
                if m.sum() == 0:
                    continue
                self.mu_all_ct[ct] = np.asarray(train[m].X).mean(axis=0)
        else:
            m = (train.obs[int_key] != ctl)
            self.mu_all_global = np.asarray(train[m].X).mean(axis=0)

    def mean_by_ct(self, ct: str):
        return self.mean_ctl_ct.get(ct, None)

    def mu_all(self, ct: str | None):
        if self.cfg.mu_all_per_ct:
            return self.mu_all_ct.get(ct, None)
        return self.mu_all_global
