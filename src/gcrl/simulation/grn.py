
# src/gcrl/simulation/grn.py
# -*- coding: utf-8 -*-
"""
GRN generation utilities for SERGIO inputs.

Exposes:
- Params (dataclass)
- build_grn(outdir: str, **kwargs): generates network and writes SERGIO input files
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Any
import os, json, random
import numpy as np
import pandas as pd
import networkx as nx

LAYERS = ["Top", "Core", "Bottom"]
LAYER_ORDER = {"Top": 2, "Core": 1, "Bottom": 0}

@dataclass
class Params:
    seed: int = 42
    n_communities: int = 5
    tf_per_comm_low: int = 15
    tf_per_comm_high: int = 20
    layer_props: Tuple[float,float,float] = (0.2, 0.5, 0.3)
    within_mu_level: str = "mid"              # {"low":1.5, "mid":2.5, "high":3.5}
    in_degree_cap: int = 4
    motif_fraction: float = 0.15
    inter_density: str = "five_per_pair"      # {"none","five_per_pair","half_small"}
    inter_topdown_bias: float = 0.8
    n_targets: int = 1000
    target_home_bias: float = 0.8
    max_tf_per_target: int = 10
    target_fanin_lambda: float = 3.5
    max_regulator_communities: str = "3"      # {"1","3","all"}
    n_bins: int = 3
    hill_value: float = 2.0                   # Hij fixed to 2
    K_abs_low: float = 2.0                    # K_ij ∈ [2,5]
    K_abs_high: float = 5.0
    p_repression: float = 0.30                # ~30% negative K to simulate repression
    mr_prod_low: float = 2.0
    mr_prod_high: float = 5.0
    ensure_acyclic: bool = True

def _set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)

def _split_layers(n: int, props: Tuple[float,float,float]):
    t,c,b = [max(1, int(round(p*n))) for p in props]
    while t + c + b > n:
        if c>1: c -= 1
        elif b>1: b -= 1
        else: break
    while t + c + b < n:
        c += 1
    return (["Top"]*t) + (["Core"]*c) + (["Bottom"]*b)

def _assign_sign_and_params(n_edges: int, K_lo: float, K_hi: float, p_repress: float, hill_value: float):
    K_abs = np.random.uniform(K_lo, K_hi, size=n_edges)
    signs = np.where(np.random.rand(n_edges) < p_repress, -1.0, 1.0)
    K = signs * K_abs
    hills = np.full(n_edges, float(hill_value), dtype=float)
    return K, hills

class GRNBuilder:
    def __init__(self, P: Params):
        self.P = P
        _set_seed(P.seed)
        self.G = nx.DiGraph()
        self.comm_nodes: Dict[int, List[int]] = {}
        self.node_meta: Dict[int, Dict] = {}
        self.tf_counts: List[int] = []

    def build_tf_communities(self):
        P = self.P
        self.tf_counts = [random.randint(P.tf_per_comm_low, P.tf_per_comm_high) for _ in range(P.n_communities)]
        curr_id = 0
        for cidx, n in enumerate(self.tf_counts):
            layers = _split_layers(n, P.layer_props)
            ids = list(range(curr_id, curr_id+n))
            self.comm_nodes[cidx] = ids
            for i, nid in enumerate(ids):
                layer = layers[i]
                name = f"TF_c{cidx}_n{i}"
                self.G.add_node(nid, kind="TF", community=cidx, layer=layer, name=name)
                self.node_meta[nid] = {"name": name, "kind":"TF", "community": cidx, "layer": layer}
            curr_id += n

    def wire_within_community(self):
        P = self.P
        mu = {"low":1.5, "mid":2.5, "high":3.5}[P.within_mu_level]
        for cidx, ids in self.comm_nodes.items():
            by_layer = {L: [nid for nid in ids if self.G.nodes[nid]['layer']==L] for L in LAYERS}
            N = len(ids)
            target_edges = int(round(mu * N))
            motif_budget = int(round(P.motif_fraction * target_edges))

            def can_edge(u,v):
                if u==v or self.G.has_edge(u,v): return False
                if self.G.nodes[u]['community'] != cidx or self.G.nodes[v]['community'] != cidx: return False
                if LAYER_ORDER[self.G.nodes[u]['layer']] <= LAYER_ORDER[self.G.nodes[v]['layer']]: return False
                if self.G.in_degree(v) >= P.in_degree_cap: return False
                return True

            added = 0
            # motifs (FFL / bi-fan)
            attempts = 0
            while added < motif_budget and attempts < 2000:
                attempts += 1
                if np.random.rand() < 0.7:
                    if not (by_layer["Top"] and by_layer["Core"] and by_layer["Bottom"]):
                        continue
                    X = np.random.choice(by_layer["Top"])
                    Y = np.random.choice(by_layer["Core"])
                    Z = np.random.choice(by_layer["Bottom"])
                    edges = [(X,Y),(Y,Z),(X,Z)]
                else:
                    src_pool = by_layer["Top"] + by_layer["Core"]
                    dst_pool = by_layer["Core"] + by_layer["Bottom"]
                    if len(src_pool) < 2 or len(dst_pool) < 2:
                        continue
                    X1,X2 = np.random.choice(src_pool, 2, replace=False)
                    Y1,Y2 = np.random.choice(dst_pool, 2, replace=False)
                    def ok_pair(u,v):
                        return LAYER_ORDER[self.G.nodes[u]['layer']] > LAYER_ORDER[self.G.nodes[v]['layer']]
                    cand = [(X1,Y1),(X1,Y2),(X2,Y1),(X2,Y2)]
                    edges = [e for e in cand if ok_pair(*e)]
                    if len(edges) < 3: continue
                for (u,v) in edges:
                    if added >= motif_budget: break
                    if can_edge(u,v):
                        self.G.add_edge(u,v, kind="TF_TF"); added += 1

            # random fill
            valid_targets = by_layer["Core"] + by_layer["Bottom"]
            attempts = 0
            while added < target_edges and attempts < 5000:
                attempts += 1
                if not valid_targets: break
                v = int(np.random.choice(valid_targets))
                higher_sources = [u for u in ids if LAYER_ORDER[self.G.nodes[u]['layer']] > LAYER_ORDER[self.G.nodes[v]['layer']]]
                if not higher_sources: continue
                u = int(np.random.choice(higher_sources))
                if self.G.in_degree(v) >= P.in_degree_cap: continue
                if can_edge(u,v):
                    self.G.add_edge(u,v, kind="TF_TF"); added += 1

    def wire_between_communities(self):
        P = self.P
        if P.inter_density == "none": return
        for i in range(P.n_communities):
            for j in range(i+1, P.n_communities):
                srcs = self.comm_nodes[i]; dsts = self.comm_nodes[j]
                if P.inter_density == "five_per_pair":
                    n_edges = 5
                elif P.inter_density == "half_small":
                    n_edges = int(np.ceil(0.75 * min(len(srcs), len(dsts))))
                else:
                    n_edges = 0
                if n_edges <= 0: continue
                pairs_topdown = []; pairs_lateral = []
                for u in srcs:
                    for v in dsts:
                        if LAYER_ORDER[self.G.nodes[u]['layer']] > LAYER_ORDER[self.G.nodes[v]['layer']]:
                            pairs_topdown.append((u,v))
                        elif LAYER_ORDER[self.G.nodes[u]['layer']] == LAYER_ORDER[self.G.nodes[v]['layer']]:
                            pairs_lateral.append((u,v))
                cnt=0; attempts=0
                while cnt < n_edges and attempts < 50*n_edges:
                    attempts += 1
                    pool = pairs_topdown if (np.random.rand() < P.inter_topdown_bias and pairs_topdown) else (pairs_lateral if pairs_lateral else pairs_topdown)
                    if not pool: break
                    u,v = pool[np.random.randint(0, len(pool))]
                    if self.G.has_edge(u,v): continue
                    if self.G.in_degree(v) >= P.in_degree_cap: continue
                    self.G.add_edge(u,v, kind="TF_TF"); cnt += 1

    def add_targets_and_params(self):
        P = self.P
        comm_sizes = np.array([len(self.comm_nodes[c]) for c in range(P.n_communities)], dtype=float)
        comm_probs = comm_sizes / comm_sizes.sum()
        self.target_ids = []
        base = max(self.G.nodes()) + 1 if self.G.nodes() else 0
        for t in range(P.n_targets):
            nid = base + t
            home = int(np.random.choice(np.arange(P.n_communities), p=comm_probs))
            name = f"TG_{t}"
            self.G.add_node(nid, kind="TG", community=home, layer="NA", name=name)
            self.node_meta[nid] = {"name": name, "kind":"TG", "community": home, "layer": "NA"}
            self.target_ids.append(nid)

        # connect TF->TG
        for tg in self.target_ids:
            r = np.random.poisson(lam=P.target_fanin_lambda)
            r = max(1, min(P.max_tf_per_target, r))
            home = self.G.nodes[tg]['community']
            # regulator community selection
            if P.max_regulator_communities == "all":
                allowed_comms = list(range(P.n_communities))
            else:
                K = int(P.max_regulator_communities)
                other = [c for c in range(P.n_communities) if c != home]
                draw = np.random.choice(other, size=max(0, K-1), replace=False).tolist()
                allowed_comms = [home] + draw
            pool = []
            for c in allowed_comms:
                pool.extend(self.comm_nodes[c])
            if not pool:
                continue
            regs = np.random.choice(pool, size=min(r, len(pool)), replace=False).tolist()
            for tf in regs:
                self.G.add_edge(int(tf), int(tg), kind="TF_TG")

        # parameterize edges
        edges = list(self.G.edges())
        K, hills = _assign_sign_and_params(len(edges), P.K_abs_low, P.K_abs_high, P.p_repression, P.hill_value)
        for e,(u,v) in enumerate(edges):
            self.G.edges[(u,v)]['K'] = float(K[e])
            self.G.edges[(u,v)]['hill'] = float(hills[e])

    def ensure_mr_per_community(self):
        for cidx, ids in self.comm_nodes.items():
            top_candidates = [n for n in ids if self.G.nodes[n]['layer'] == 'Top']
            candidates = top_candidates if top_candidates else list(ids)
            if not candidates:
                continue
            def tf_in_deg(n):
                return sum(1 for u in self.G.predecessors(n) if self.G.nodes[u]['kind'] == 'TF')
            best = min(candidates, key=tf_in_deg)
            for u in list(self.G.predecessors(best)):
                if self.G.nodes[u]['kind'] == 'TF' and self.G.has_edge(u, best):
                    self.G.remove_edge(u, best)

    def export(self, outdir: str):
        os.makedirs(outdir, exist_ok=True)
        # nodes
        node_rows = []
        for nid, meta in self.node_meta.items():
            node_rows.append({"gene_id": nid, "gene_name": meta["name"], "kind": meta["kind"],
                              "community": meta["community"], "layer": meta["layer"]})
        pd.DataFrame(node_rows).sort_values("gene_id").to_csv(os.path.join(outdir, "nodes.csv"), index=False)
        # edges
        tf_tf, tf_tg = [], []
        for u,v,data in self.G.edges(data=True):
            row = {"src": u, "dst": v, "K": data.get("K", None), "hill": data.get("hill", None)}
            if self.G.nodes[v]['kind']=="TF": tf_tf.append(row)
            else: tf_tg.append(row)
        pd.DataFrame(tf_tf).to_csv(os.path.join(outdir, "edges_tf_tf.csv"), index=False)
        pd.DataFrame(tf_tg).to_csv(os.path.join(outdir, "edges_tf_target.csv"), index=False)

        # SERGIO inputs
        TFs = [n for n in self.G.nodes() if self.G.nodes[n]['kind']=="TF"]
        indeg_from_TF = {n:0 for n in self.G.nodes()}
        for u,v in self.G.edges():
            if self.G.nodes[u]['kind']=="TF": indeg_from_TF[v] += 1
        MRs = [n for n in TFs if indeg_from_TF[n]==0]

        # input_targets.txt
        with open(os.path.join(outdir, "input_targets.txt"), "w") as f:
            for n in self.G.nodes():
                parents = [u for u in self.G.predecessors(n) if self.G.nodes[u]['kind']=="TF"]
                if len(parents)==0: 
                    continue
                Ks = [self.G.edges[(u,n)]['K'] for u in parents]
                hills = [self.G.edges[(u,n)]['hill'] for u in parents]
                row = [n, len(parents)] + parents + Ks + hills
                f.write(", ".join(str(x) for x in row) + "\n")

        # input_regs.txt
        with open(os.path.join(outdir, "input_regs.txt"), "w") as f:
            for mr in MRs:
                prods = list(np.random.uniform(self.P.mr_prod_low, self.P.mr_prod_high, size=self.P.n_bins))
                row = [mr] + prods
                f.write(", ".join(str(x) for x in row) + "\n")

        # README.csv
        settings: Dict[str, Any] = {}
        settings.update(asdict(self.P))
        settings["tf_counts_per_community"] = self.tf_counts
        settings["n_tfs_total"] = int(sum(self.tf_counts))
        settings["n_targets_total"] = int(self.P.n_targets)
        settings["n_edges_tf_tf"] = int(len(tf_tf))
        settings["n_edges_tf_tg"] = int(len(tf_tg))
        settings["n_master_regulators"] = int(len(MRs))
        settings["communities"] = list(range(self.P.n_communities))
        rows = []
        for k,v in settings.items():
            if isinstance(v, (list, tuple, dict)):
                v = json.dumps(v)
            rows.append({"key": k, "value": v})
        pd.DataFrame(rows).to_csv(os.path.join(outdir, "README.csv"), index=False)

def build_grn(outdir: str, **kwargs) -> GRNBuilder:
    P = Params(**kwargs)
    b = GRNBuilder(P)
    b.build_tf_communities()
    b.wire_within_community()
    b.wire_between_communities()
    b.ensure_mr_per_community()
    b.add_targets_and_params()
    # optional acyclicity check
    try:
        _ = list(nx.topological_sort(b.G))
    except nx.NetworkXUnfeasible:
        raise RuntimeError("Cycle detected; retry with another seed or settings.")
    os.makedirs(outdir, exist_ok=True)
    b.export(outdir)
    return b
