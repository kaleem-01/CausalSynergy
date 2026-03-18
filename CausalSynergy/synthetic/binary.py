from __future__ import annotations

from typing import Dict, Any, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import networkx as nx


# ============================================================
# Helpers
# ============================================================
def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def logistic_gate_params(logic: str, *, sharpness: float = 10.0) -> Tuple[np.ndarray, float]:
    logic = logic.upper()
    s = float(sharpness)

    if logic == "OR":
        w = np.array([s, s], dtype=float)
        b = -0.5 * s
        return w, float(b)

    if logic == "AND":
        w = np.array([s, s], dtype=float)
        b = -1.5 * s
        return w, float(b)

    if logic == "NOR":
        w, b = logistic_gate_params("OR", sharpness=s)
        return -w, float(-b)

    if logic == "NAND":
        w, b = logistic_gate_params("AND", sharpness=s)
        return -w, float(-b)

    raise ValueError(f"Unsupported logic for logistic gate: {logic}")


def apply_bitflip(y: np.ndarray, rng: np.random.Generator, p_flip: float) -> np.ndarray:
    if p_flip <= 0.0:
        return y
    flip = rng.random(y.shape[0]) < p_flip
    y2 = y.copy()
    y2[flip] = 1 - y2[flip]
    return y2


def xor_sigmoid2_prob(X2: np.ndarray, s: float) -> np.ndarray:
    """
    2-layer sigmoid XOR approximation.
    X2: shape (n,2) with entries 0/1.
    Returns p(y=1|x).
    """
    x1 = X2[:, 0].astype(float)
    x2 = X2[:, 1].astype(float)

    h_or = sigmoid(-0.5 * s + s * x1 + s * x2)
    h_and = sigmoid(-1.5 * s + s * x1 + s * x2)

    p = sigmoid(-0.5 * s + s * h_or - s * h_and)
    return p


def node_to_str(i: int) -> str:
    return f"{i}"


# ============================================================
# CPD builder
# ============================================================
def build_node_cpds_from_dag(
    dag: nx.DiGraph,
    *,
    topo: Optional[Sequence[int]] = None,
    seed_params: int = 0,
    root_p: float = 0.5,
    p_copy_flip: float = 0.05,
    p_syn: float = 0.3,
    gate_types: Sequence[str] = ("OR", "AND", "NAND", "NOR"),
    gate_probs: Optional[Sequence[float]] = None,
    sharpness: float = 10.0,
) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Any]]:
    if topo is None:
        topo = list(nx.topological_sort(dag))
    topo = list(map(int, topo))

    n = dag.number_of_nodes()
    if set(dag.nodes()) != set(range(n)):
        raise ValueError("Expected nodes to be labeled 0..n_nodes-1.")

    if not (0.0 <= root_p <= 1.0):
        raise ValueError("root_p must be in [0,1].")
    if not (0.0 <= p_copy_flip <= 1.0):
        raise ValueError("p_copy_flip must be in [0,1].")
    if not (0.0 <= p_syn <= 1.0):
        raise ValueError("p_syn must be in [0,1].")
    if sharpness <= 0:
        raise ValueError("sharpness must be > 0.")

    gate_types = [g.upper() for g in gate_types]
    if gate_probs is not None:
        if len(gate_probs) != len(gate_types):
            raise ValueError("gate_probs must have same length as gate_types.")
        s = float(np.sum(gate_probs))
        if not np.isclose(s, 1.0):
            raise ValueError(f"gate_probs must sum to 1.0 (got {s}).")
        if any(p < 0 for p in gate_probs):
            raise ValueError("gate_probs must be non-negative.")

    rng = np.random.default_rng(seed_params)

    indeg = {v: int(dag.in_degree(v)) for v in dag.nodes()}
    colliders = [v for v in dag.nodes() if indeg[v] == 2]

    n_syn = int(round(p_syn * len(colliders)))
    syn_set = set(rng.choice(colliders, size=n_syn, replace=False).tolist()) if n_syn > 0 else set()

    node_cpd: Dict[int, Dict[str, Any]] = {}

    for v in topo:
        parents = sorted(list(dag.predecessors(v)))
        k = len(parents)

        if k == 0:
            node_cpd[v] = {"model": "root", "p": float(root_p), "parents": []}

        elif k == 1:
            node_cpd[v] = {"model": "copy_noise", "parents": parents, "p_flip": float(p_copy_flip)}

        elif k == 2:
            if v in syn_set:
                node_cpd[v] = {"model": "xor_sigmoid2", "parents": parents, "sharpness": float(sharpness)}
            else:
                logic = str(rng.choice(gate_types, p=gate_probs))
                w, b = logistic_gate_params(logic, sharpness=sharpness)
                node_cpd[v] = {
                    "model": "logistic_gate",
                    "logic": logic,
                    "parents": parents,
                    "w": w.tolist(),
                    "b": float(b),
                    "sharpness": float(sharpness),
                }
        else:
            raise ValueError(
                f"Node {v} has indegree {k}. Expected indegree in {{0,1,2}}. "
                "Make sure your DAG generator caps indegree at 2."
            )

    info = {
        "n_nodes": int(n),
        "n_colliders": int(len(colliders)),
        "n_syn_colliders": int(len(syn_set)),
        "syn_colliders": sorted(map(int, syn_set)),
    }
    return node_cpd, info


# ============================================================
# Simulator 
# ============================================================
def simulate_from_dag_cpds(
    dag: nx.DiGraph,
    node_cpd: Dict[int, Dict[str, Any]],
    *,
    n_samples: int,
    seed_data: int = 0,
    p_flip_global: float = 0.0,
    topo: Optional[Sequence[int]] = None,
) -> Tuple[pd.DataFrame, nx.DiGraph, Dict[str, Dict[str, Any]], pd.DataFrame]:
    """
    Returns:
      df         : DataFrame with string columns ("0","1",...)
      dag_str    : string-labeled DAG
      node_spec  : per-node CPD spec with string ids
      metadata   : like your metadata style, but now includes:
                   - indeg==1 edges: Combs = "[parent, child]" and Type="Pairwise"
                   - indeg==2 colliders: Combs = "[p0, p1, child]" and Type in {XOR, OR, AND, NAND, NOR}
    """
    if topo is None:
        topo = list(nx.topological_sort(dag))
    topo = list(map(int, topo))

    if not (0.0 <= p_flip_global <= 1.0):
        raise ValueError("p_flip_global must be in [0,1].")

    rng = np.random.default_rng(seed_data)
    n = dag.number_of_nodes()
    X = np.zeros((n_samples, n), dtype=int)

    meta_combs: List[str] = []
    meta_type: List[str] = []

    for v in topo:
        spec = node_cpd[v]
        parents = list(map(int, spec.get("parents", [])))

        if spec["model"] == "root":
            p = float(spec["p"])
            X[:, v] = (rng.random(n_samples) < p).astype(int)

        elif spec["model"] == "copy_noise":
            # indeg==1: parent -> child
            p0 = parents[0]
            y = X[:, p0].astype(int)
            y = apply_bitflip(y, rng, float(spec["p_flip"]))
            X[:, v] = y

            # ---- metadata row (pairwise edge) ----
            meta_combs.append(str([int(p0), int(v)]))  # [parent, child]
            meta_type.append("Pairwise")

        elif spec["model"] == "logistic_gate":
            # indeg==2 collider (non-synergistic)
            p0, p1 = parents
            P = np.stack([X[:, p0], X[:, p1]], axis=1).astype(float)
            w = np.array(spec["w"], dtype=float)
            b = float(spec["b"])
            p = sigmoid(b + P @ w)
            X[:, v] = (rng.random(n_samples) < p).astype(int)

            # ---- metadata row (collider) ----
            meta_combs.append(str([int(p0), int(p1), int(v)]))
            meta_type.append(str(spec.get("logic", "OR")))

        elif spec["model"] == "xor_sigmoid2":
            # indeg==2 collider (synergistic)
            p0, p1 = parents
            P = np.stack([X[:, p0], X[:, p1]], axis=1)
            s = float(spec["sharpness"])
            p = xor_sigmoid2_prob(P, s=s)
            X[:, v] = (rng.random(n_samples) < p).astype(int)

            # ---- metadata row (collider) ----
            meta_combs.append(str([int(p0), int(p1), int(v)]))
            meta_type.append("XOR")

        else:
            raise ValueError(f"Unknown model: {spec['model']}")

        # optional global measurement noise (bit flips) after node generation
        if p_flip_global > 0.0:
            X[:, v] = apply_bitflip(X[:, v], rng, p_flip_global)

    df = pd.DataFrame(X, columns=[node_to_str(i) for i in range(n)])
    dag_str = nx.relabel_nodes(dag, node_to_str, copy=True)

    node_spec_str: Dict[str, Dict[str, Any]] = {}
    for v, spec in node_cpd.items():
        spec2 = dict(spec)
        spec2["parents"] = [node_to_str(int(u)) for u in spec2.get("parents", [])]
        node_spec_str[node_to_str(int(v))] = spec2

    metadata = pd.DataFrame({"Combs": meta_combs, "Type": meta_type})
    return df, dag_str, node_spec_str, metadata

