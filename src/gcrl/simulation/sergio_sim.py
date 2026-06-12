
# src/gcrl/simulation/sergio_sim.py
# -*- coding: utf-8 -*-
"""
SERGIO simulation utilities.

Exposes:
- run_sergio_sim(...)
- resolve SERGIO location via gcrl.config.resolve_sergio_dir().
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed
import os, sys
import numpy as np
import pandas as pd
import anndata as ad

from gcrl.config import resolve_sergio_dir

# --- SERGIO import helper (works in workers) ---
def _import_sergio(sergio_dir: Optional[str] = None):
    sd = resolve_sergio_dir(sergio_dir)
    if sd:
        repo_parent = Path(sd).resolve().parent
        sys.path.insert(0, str(repo_parent))
    from SERGIO.sergio import sergio  # will raise if not importable
    return sergio

# --- File parsing helpers ---
def _parse_input_targets_txt(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for ln, line in enumerate(f, 1):
            parts = [p.strip() for p in line.strip().split(',') if p.strip() != '']
            if len(parts) < 2:
                continue
            tgt = int(float(parts[0])); nreg = int(float(parts[1]))
            rest = parts[2:]
            if len(rest) < 3 * nreg:
                raise ValueError(f"Line {ln}: expected {3*nreg} extra fields, got {len(rest)}")
            regs  = [int(float(x)) for x in rest[:nreg]]
            Ks    = [float(x) for x in rest[nreg:2*nreg]]
            Hills = [float(x) for x in rest[2*nreg:3*nreg]]
            rows.append({'target_id': tgt, 'n_reg': nreg, 'regs': regs, 'Ks': Ks, 'Hills': Hills})
    return rows

def _write_input_targets_txt(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        for r in rows:
            n = r['n_reg']
            line = [r['target_id'], n] + r['regs'][:n] + r['Ks'][:n] + r['Hills'][:n]
            f.write(", ".join(str(x) for x in line) + "\n")

def _parse_input_regs_txt(path: Path) -> dict[int, list[float]]:
    out = {}
    with open(path) as f:
        for ln, line in enumerate(f, 1):
            parts = [p.strip() for p in line.strip().split(',') if p.strip() != '']
            if len(parts) < 2:
                continue
            mr = int(float(parts[0]))
            prods = [float(x) for x in parts[1:]]
            out[mr] = prods
    return out

def _write_input_regs_txt(path: Path, mr_to_prods: dict[int, list[float]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        for mr, prods in mr_to_prods.items():
            f.write(", ".join([str(mr)] + [str(p) for p in prods]) + "\n")

def _compute_master_regulators(nodes: pd.DataFrame, edges_tf_tf: pd.DataFrame) -> set[int]:
    tfs = nodes[nodes['kind'] == 'TF']['gene_id'].astype(int).tolist()
    indeg = {tf: 0 for tf in tfs}
    for _, r in edges_tf_tf.iterrows():
        dst = int(r['dst']); src = int(r['src'])
        if src in indeg and dst in indeg:
            indeg[dst] += 1
    return {tf for tf, d in indeg.items() if d == 0}

def _principal_regulator_community(nodes: pd.DataFrame, edges_tf_tg: pd.DataFrame) -> pd.Series:
    tf_comm = nodes.set_index('gene_id')['community'].to_dict()
    e = edges_tf_tg[['src', 'dst']].copy()
    e['src_comm'] = e['src'].map(tf_comm)
    grp = e.groupby(['dst', 'src_comm']).size().reset_index(name='count')
    dom = grp.sort_values(['dst', 'count'], ascending=[True, False]).drop_duplicates('dst')
    return dom.set_index('dst')['src_comm']

# --- worker ---
def _worker_simulate(job: dict) -> dict:
    sergio_cls = _import_sergio(job['sergio_dir'])

    grn_dir = Path(job['grn_dir'])
    out_dir = Path(job['out_dir'])

    if job['mode'] == 'unperturbed':
        input_targets = grn_dir / 'input_targets.txt'
        input_regs    = grn_dir / 'input_regs.txt'
        name_prefix   = 'unperturbed'
    else:
        c = job['community']; tf_id = int(job['tf_id'])
        tmp_comm = out_dir / 'interventions' / f'comm{c}'
        targets_rows = _parse_input_targets_txt(grn_dir / 'input_targets.txt')
        regs_map     = _parse_input_regs_txt(grn_dir / 'input_regs.txt')
        nodes = pd.read_csv(grn_dir / 'nodes.csv').sort_values('gene_id')
        edges_tf_tf = pd.read_csv(grn_dir / 'edges_tf_tf.csv')
        mr_set = _compute_master_regulators(nodes, edges_tf_tf)
        if tf_id in mr_set:
            regs_map[tf_id] = [0.0] * job['n_bins']
        else:
            targets_rows = [r for r in targets_rows if int(r['target_id']) != tf_id]
            regs_map[tf_id] = [0.0] * job['n_bins']

        input_targets = tmp_comm / 'input_targets.txt'
        input_regs    = tmp_comm / 'input_regs.txt'
        _write_input_targets_txt(input_targets, targets_rows)
        _write_input_regs_txt(input_regs, regs_map)
        name_prefix   = f'pert_c{c}'

    sim = sergio_cls(
        number_genes = int(job['n_genes']),
        number_bins  = int(job['n_bins']),
        number_sc    = int(job['n_cells']),
        noise_params = float(job['noise_params']),
        decays       = float(job['decays']),
        sampling_state = int(job['sampling_state']),
        noise_type   = job['noise_type'],
    )
    sim.build_graph(
        input_file_taregts = str(input_targets),
        input_file_regs    = str(input_regs),
        shared_coop_state  = 2
    )
    sim.simulate()
    expr_list = sim.getExpressions()
    bins_k = [arr.shape[1] for arr in expr_list]
    X = np.concatenate(expr_list, axis=1).T

    if job['mode'] == 'unperturbed':
        obs = pd.DataFrame({
            'intervention': 'unperturbed',
            'bin': np.concatenate([np.full(k, b+1) for b, k in enumerate(bins_k)]),
            'intervention_tf_id': np.nan,
            'intervention_tf_name': None,
            'intervention_comm': np.nan,
        })
        obs.index = [f'{name_prefix}_{i}' for i in range(X.shape[0])]
        order = 0
    else:
        obs = pd.DataFrame({
            'intervention': 'perturbed',
            'bin': np.concatenate([np.full(k, b+1) for b, k in enumerate(bins_k)]),
            'intervention_tf_id': int(job['tf_id']),
            'intervention_tf_name': str(job['tf_name']),
            'intervention_comm': int(job['community']),
        })
        obs.index = [f'{name_prefix}_{i}' for i in range(X.shape[0])]
        order = 1 + int(job['community'])

    return {'order': order, 'mode': job['mode'], 'community': job.get('community', None), 'X': X, 'obs': obs}

# --- public API ---
def run_sergio_sim(
    grn_dir: str,
    out_h5ad: str = "adata.h5ad",
    n_unperturbed: int = 2000,
    n_perturbed_per_comm: int = 1000,
    seed: int = 123,
    sergio_dir: Optional[str] = None,   # resolved via config if None
    noise_params: float = 1.0,
    decays: float = 0.8,
    sampling_state: int = 15,
    noise_type: str = "dpd",
    jobs: Optional[int] = None,
) -> ad.AnnData:
    """
    Simulate SERGIO expression using a GRN folder.

    Resolves SERGIO location via argument / env var / project config / user config.
    Returns the AnnData object and writes `out_h5ad` to disk.
    """
    from numpy.random import default_rng
    rng = default_rng(seed)

    grn = Path(grn_dir)
    out_path = Path(out_h5ad)
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "interventions").mkdir(exist_ok=True, parents=True)

    # Load GRN metadata
    nodes = pd.read_csv(grn / 'nodes.csv').sort_values('gene_id')
    edges_tf_tf = pd.read_csv(grn / 'edges_tf_tf.csv')
    edges_tf_tg = pd.read_csv(grn / 'edges_tf_target.csv')
    n_genes = nodes.shape[0]
    communities = sorted(nodes[nodes['kind'] == 'TF']['community'].unique().tolist())

    regs_map = _parse_input_regs_txt(grn / 'input_regs.txt')
    if len(regs_map) == 0:
        raise ValueError("input_regs.txt is empty or malformed")
    n_bins = len(next(iter(regs_map.values())))

    # choose one TF per community
    tf_nodes = nodes[nodes['kind'] == 'TF'][['gene_id', 'gene_name', 'community']].copy()
    perts = []
    for c in communities:
        cand = tf_nodes[tf_nodes['community'] == c]['gene_id'].to_numpy()
        if len(cand) == 0:
            continue
        tf_id = int(rng.choice(cand))
        tf_name = str(tf_nodes.loc[tf_nodes['gene_id'] == tf_id, 'gene_name'].values[0])
        perts.append({'community': int(c), 'tf_id': tf_id, 'tf_name': tf_name})

    # prepare jobs
    jobs_list = []
    jobs_list.append({
        'mode': 'unperturbed', 'grn_dir': str(grn), 'out_dir': str(out_dir), 'sergio_dir': sergio_dir,
        'n_genes': n_genes, 'n_bins': n_bins, 'n_cells': n_unperturbed,
        'noise_params': noise_params, 'decays': decays, 'sampling_state': sampling_state, 'noise_type': noise_type,
    })
    for p in perts:
        jobs_list.append({
            'mode': 'perturbed', 'grn_dir': str(grn), 'out_dir': str(out_dir), 'sergio_dir': sergio_dir,
            'n_genes': n_genes, 'n_bins': n_bins, 'n_cells': n_perturbed_per_comm,
            'noise_params': noise_params, 'decays': decays, 'sampling_state': sampling_state, 'noise_type': noise_type,
            'community': p['community'], 'tf_id': p['tf_id'], 'tf_name': p['tf_name'],
        })

    default_jobs = min(os.cpu_count() or 1, 1 + len(perts))
    max_workers = jobs if jobs is not None else default_jobs

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_worker_simulate, job) for job in jobs_list]
        for fut in as_completed(futs):
            results.append(fut.result())

    # order and combine
    results.sort(key=lambda r: r['order'])
    X_all = np.vstack([r['X'] for r in results])
    obs_all = pd.concat([r['obs'] for r in results], axis=0, ignore_index=False)

    # var & AnnData
    var_df = nodes[['gene_id', 'gene_name', 'kind', 'community']].copy()
    dom_comm = _principal_regulator_community(nodes, edges_tf_tg)
    var_df['principal_reg_comm'] = np.nan
    mask_tg = var_df['kind'] == 'TG'
    var_df.loc[mask_tg, 'principal_reg_comm'] = var_df.loc[mask_tg, 'gene_id'].map(dom_comm)
    var_df = var_df.sort_values('gene_id').reset_index(drop=True)

    adata = ad.AnnData(X_all, obs=obs_all, var=var_df.set_index('gene_name').copy())
    adata.var['gene_id'] = var_df['gene_id'].values

    # adjacency K into varp
    gid_to_idx = {int(gid): i for i, gid in enumerate(var_df['gene_id'].tolist())}
    rows = []; cols = []; vals = []
    for df_edges in (edges_tf_tf, edges_tf_tg):
        if 'K' not in df_edges.columns:
            continue
        for _, r in df_edges.iterrows():
            src = int(r['src']); dst = int(r['dst']); K = float(r['K'])
            if src in gid_to_idx and dst in gid_to_idx:
                rows.append(gid_to_idx[src]); cols.append(gid_to_idx[dst]); vals.append(K)
    if len(vals) > 0:
        from scipy import sparse
        A = sparse.coo_matrix((vals, (rows, cols)), shape=(adata.n_vars, adata.n_vars)).tocsr()
    else:
        from scipy import sparse
        A = sparse.csr_matrix((adata.n_vars, adata.n_vars))
    adata.varp['grn_K'] = A

    # final column rename
    adata.obs.rename(
        columns={'bin': 'cell_type', 'intervention_tf_name': 'intervention_gene_name', 'intervention_tf_id': 'intervention_gene_id'},
        inplace=True, errors='ignore',
    )
    adata.obs['intervention_gene_name'] = adata.obs['intervention_gene_name'].astype(object)

    adata.write_h5ad(out_path)
    return adata
