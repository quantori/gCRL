from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

@dataclass
class EvalConfig:
    # Column keys
    set_key: str = "set"
    intervention_key: str = "intervention"
    cell_type_key: str = "cell_type"
    control_label: str = "unperturbed"

    # Evaluation
    random_state: int = 0
    n_boot: int = 0  # bootstrap iterations (0 disables CIs)
    topk: int = 200
    deg_weight_floor: float = 1e-6

    # µ_all reference options (Diversity-style). If True, compute within cell type.
    mu_all_per_ct: bool = True

    # DEG selection (vs Rest)
    deg_n_rest: Optional[int] = None  # if None, use all remaining cells
    deg_min_cells: int = 20

    # UMAP / viz
    make_umap_plots: bool = True
    umap_neighbors: int = 20
    umap_min_dist: float = 0.2
    umap_metric: str = "cosine"
    umap_sample_train: int = 50000  # cap for fitting stability
    knn_k_overlap: int = 10

    # Output
    out_dir: str = "./experiments/generalization"
