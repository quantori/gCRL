from .config import EvalConfig
from .orchestrator import evaluate_dataset as _evaluate_dataset
from .predictors import InterventionalPredictor, LinearDeltaPredictor, GcrlVaePredictor

def evaluate_gcrl_vae(adata, model=None, predictor: InterventionalPredictor|None=None, cfg: EvalConfig|None=None):
    """Entry point used by external code.
    - If `predictor` is provided, use it.
    - Else if `model` exposes a `predict_group(...)` method, it will be wrapped.
    - Else if `model` looks like gCRL-VAE and you provide a GcrlVaePredictor, use it.
    - Else, fallback to LinearDeltaPredictor (baseline).
    Returns a dict of pandas DataFrames and writes CSVs/PNGs to cfg.out_dir.
    """
    from .metrics import DEFAULT_METRICS_SUITE
    from .baselines import BaselineProvider  # linting side-effect
    if cfg is None:
        from .config import EvalConfig as _EvalConfig
        cfg = _EvalConfig()
    if predictor is None:
        if model is not None and hasattr(model, 'predict_group'):
            class _Wrapper:
                def __init__(self, m, c): self.m=m; self.cfg=c
                def predict_group(self, controls_train_adata, intervention, cell_type, n_pred, rng):
                    return self.m.predict_group(controls_train_adata, intervention, cell_type, n_pred, rng)
            predictor = _Wrapper(model, cfg)
        elif model is not None:
            predictor = GcrlVaePredictor(model, cfg)  # may raise NotImplementedError if not integrated
        else:
            predictor = LinearDeltaPredictor(adata, cfg)
    return _evaluate_dataset(adata, predictor, DEFAULT_METRICS_SUITE, cfg)
