from __future__ import annotations

from typing import Tuple, Dict, Any, Iterable, List
import numpy as np
import networkx as nx
from utils.plotting import visualize_graph, visualize_network_labels
from itertools import combinations

# ============================================================
# Shared helpers
# ============================================================
def _random_topo(n_nodes: int, seed: int) -> Tuple[Tuple[int, ...], Dict[int, int]]:
    """Sample a random permutation and return (topo_tuple, position_map)."""
    rng = np.random.default_rng(seed)
    topo_list = rng.permutation(n_nodes).tolist()
    topo = tuple(int(x) for x in topo_list)
    pos = {int(node): i for i, node in enumerate(topo_list)}
    return topo, pos

def relabel_dag_by_topo(
    dag: nx.DiGraph,
    topo: Tuple[int, ...],
    *,
    start_at: int = 1,
) -> Tuple[nx.DiGraph, Tuple[int, ...], Dict[int, int]]:
    mapping = {old: new for new, old in enumerate(topo, start=start_at)}
    dag = nx.relabel_nodes(dag, mapping, copy=True)
    topo = tuple(mapping[node] for node in topo)
    return dag, topo, mapping


# ============================================================
# Helper: cap degrees AND avoid shielded colliders
# ============================================================
def _cap_degrees_and_avoid_shielded_colliders(
    n_nodes: int,
    edges: Iterable[Tuple[int, int]],
    *,
    seed: int = 0,
    max_indeg: int = 2,
    max_outdeg: int = 2,
) -> Tuple[Tuple[Tuple[int, int], ...], Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    edges_list = [(int(u), int(v)) for u, v in edges]
    rng.shuffle(edges_list)

    dag = nx.DiGraph()
    dag.add_nodes_from(range(n_nodes))

    indeg = np.zeros(n_nodes, dtype=int)
    outdeg = np.zeros(n_nodes, dtype=int)
    kept: List[Tuple[int, int]] = []

    for u, v in edges_list:
        if indeg[v] >= max_indeg or outdeg[u] >= max_outdeg:
            continue

        # 1) Would u -> v create a shielded collider at v?
        existing_parents = list(dag.predecessors(v))
        if any(dag.has_edge(u, p) or dag.has_edge(p, u) for p in existing_parents):
            continue

        # 2) Would u -> v shield an existing collider u -> c <- v ?
        common_children = set(dag.successors(u)).intersection(dag.successors(v))
        if common_children:
            continue

        dag.add_edge(u, v)
        indeg[v] += 1
        outdeg[u] += 1
        kept.append((u, v))

    info = {
        "max_indeg": int(max(indeg)) if len(indeg) > 0 else 0,
        "max_outdeg": int(max(outdeg)) if len(outdeg) > 0 else 0,
        "candidate_edges": len(edges_list),
        "kept_edges": len(kept),
    }
    return tuple(kept), info


# ============================================================
# 1) Erdos–Renyi DAG
# ============================================================
def sample_er_dag(
    n_nodes: int,
    *,
    seed: int = 0,
    p_edge: float = 0.1,
    max_indeg: int = 2,
    max_outdeg: int = 2,
) -> Tuple[nx.DiGraph, Tuple[int, ...], Dict[str, Any]]:
    """
    Erdos–Renyi DAG:
      1) Sample random topo order π.
      2) For each i<j, add π[i] -> π[j] with prob p_edge.
      3) Cap indegree/outdegree via greedy filter.

    Returns (dag, topo, info).
    """
    if n_nodes < 2:
        raise ValueError("n_nodes must be >= 2")
    if not (0.0 <= p_edge <= 1.0):
        raise ValueError("p_edge must be in [0,1].")
    if max_indeg < 0 or max_outdeg < 0:
        raise ValueError("max_indeg/max_outdeg must be >= 0")

    rng = np.random.default_rng(seed)
    topo, _ = _random_topo(n_nodes, seed)

    # candidate edges in topo direction
    topo_list = list(topo)
    cand_edges: List[Tuple[int, int]] = []
    for i in range(n_nodes):
        u = int(topo_list[i])
        for j in range(i + 1, n_nodes):
            v = int(topo_list[j])
            if rng.random() < p_edge:
                cand_edges.append((u, v))

    kept_edges, cap_info = _cap_degrees_and_avoid_shielded_colliders(
        n_nodes,
        cand_edges,
        seed=seed + 999,
        max_indeg=max_indeg,
        max_outdeg=max_outdeg,
    )

    dag = nx.DiGraph()
    dag.add_nodes_from(range(n_nodes))
    dag.add_edges_from(kept_edges)

    assert nx.is_directed_acyclic_graph(dag)

    info: Dict[str, Any] = {
        "family": "er",
        "n_nodes": int(n_nodes),
        "seed": seed,
        "topo": topo,
        "p_edge": float(p_edge),
    }
    info.update(cap_info)

    dag, topo, mapping = relabel_dag_by_topo(dag, topo, start_at=0)
    info["relabel_mapping"] = mapping

    return dag, topo, info


# ============================================================
# 2) Barabasi–Albert DAG
# ============================================================
def sample_ba_dag(
    n_nodes: int,
    *,
    seed: int = 0,
    m: int = 2,
    max_indeg: int = 2,
    max_outdeg: int = 2,
) -> Tuple[nx.DiGraph, Tuple[int, ...], Dict[str, Any]]:
    """
    Barabasi–Albert DAG:
      1) Sample a BA undirected graph G_u with parameter m.
      2) Sample random topo order π.
      3) Orient each undirected edge a-b as min_pos->max_pos in π.
      4) Cap indegree/outdegree via greedy filter.

    Returns (dag, topo, info).
    """
    if n_nodes < 2:
        raise ValueError("n_nodes must be >= 2")
    if m < 1:
        raise ValueError("m must be >= 1")
    if m >= n_nodes:
        raise ValueError("m must be < n_nodes")
    if max_indeg < 0 or max_outdeg < 0:
        raise ValueError("max_indeg/max_outdeg must be >= 0")

    topo, pos = _random_topo(n_nodes, seed)
    G_u = nx.barabasi_albert_graph(n=n_nodes, m=m, seed=seed)

    # orient edges by topo
    cand_edges: List[Tuple[int, int]] = []
    for a, b in G_u.edges():
        a, b = int(a), int(b)
        u, v = (a, b) if pos[a] < pos[b] else (b, a)
        cand_edges.append((u, v))

    kept_edges, cap_info = _cap_degrees_and_avoid_shielded_colliders(
        n_nodes,
        cand_edges,
        seed=seed + 999,
        max_indeg=max_indeg,
        max_outdeg=max_outdeg,
    )

    dag = nx.DiGraph()
    dag.add_nodes_from(range(n_nodes))
    dag.add_edges_from(kept_edges)

    assert nx.is_directed_acyclic_graph(dag)

    info: Dict[str, Any] = {
        "family": "ba",
        "n_nodes": int(n_nodes),
        "seed": seed,
        "topo": topo,
        "m": int(m),
        "undirected_edges": int(G_u.number_of_edges()),
    }
    info.update(cap_info)


    dag, topo, mapping = relabel_dag_by_topo(dag, topo, start_at=0)
    info["relabel_mapping"] = mapping

    return dag, topo, info


# ============================================================
# 3) Small-world (Watts–Strogatz) DAG
# ============================================================
def sample_small_world_dag(
    n_nodes: int,
    *,
    seed: int = 0,
    k: int = 4,
    beta: float = 0.2,
    max_indeg: int = 2,
    max_outdeg: int = 2,
) -> Tuple[nx.DiGraph, Tuple[int, ...], Dict[str, Any]]:
    """
    Small-world (Watts–Strogatz) DAG:
      1) Sample WS undirected graph G_u with parameters (k, beta).
      2) Sample random topo order π.
      3) Orient each undirected edge a-b as min_pos->max_pos in π.
      4) Cap indegree/outdegree via greedy filter.

    Returns (dag, topo, info).
    """
    if n_nodes < 2:
        raise ValueError("n_nodes must be >= 2")
    if k % 2 != 0:
        raise ValueError("k must be even.")
    if k >= n_nodes:
        raise ValueError("k must be < n_nodes.")
    if not (0.0 <= beta <= 1.0):
        raise ValueError("beta must be in [0,1].")
    if max_indeg < 0 or max_outdeg < 0:
        raise ValueError("max_indeg/max_outdeg must be >= 0")

    topo, pos = _random_topo(n_nodes, seed)
    G_u = nx.watts_strogatz_graph(n=n_nodes, k=k, p=beta, seed=seed)

    cand_edges: List[Tuple[int, int]] = []
    for a, b in G_u.edges():
        a, b = int(a), int(b)
        u, v = (a, b) if pos[a] < pos[b] else (b, a)
        cand_edges.append((u, v))

    kept_edges, cap_info = _cap_degrees_and_avoid_shielded_colliders(
        n_nodes,
        cand_edges,
        seed=seed + 999,
        max_indeg=max_indeg,
        max_outdeg=max_outdeg,
    )

    dag = nx.DiGraph()
    dag.add_nodes_from(range(n_nodes))
    dag.add_edges_from(kept_edges)

    assert nx.is_directed_acyclic_graph(dag)

    info: Dict[str, Any] = {
        "family": "small-world",
        "n_nodes": int(n_nodes),
        "seed": seed,
        "topo": topo,
        "k": int(k),
        "beta": float(beta),
        "undirected_edges": int(G_u.number_of_edges()),
    }
    info.update(cap_info)
    
    dag, topo, mapping = relabel_dag_by_topo(dag, topo, start_at=0)
    info["relabel_mapping"] = mapping
    return dag, topo, info
    


# ============================================================
# Optional: tiny dispatcher (if you still want a single entry point)
# ============================================================
def sample_dag_family_with_caps(
    n_nodes: int,
    family: str,
    **kwargs,
) -> Tuple[nx.DiGraph, Tuple[int, ...], Dict[str, Any]]:
    family = family.lower()
    if family == "er":
        return sample_er_dag(n_nodes, **kwargs)
    if family == "ba":
        return sample_ba_dag(n_nodes, **kwargs)
    if family in {"small-world", "ws"}:
        return sample_small_world_dag(n_nodes, **kwargs)
    raise ValueError(f"Unknown family: {family!r}. Use 'er', 'ba', or 'small-world'.")


# ============================================================
# Smoke test
# ============================================================
if __name__ == "__main__":
    for fam in ["er", "ba", "small-world"]:
        dag, topo, info = sample_dag_family_with_caps(
            n_nodes=30,
            family=fam,
            seed=123,
            max_indeg=2,
            max_outdeg=2,
        )
        visualize_graph(dag, title=f"{fam.upper()} DAG")
        print(
            fam,
            "| edges:", dag.number_of_edges(),
            "| max_in:", max(dict(dag.in_degree()).values()),
            "| max_out:", max(dict(dag.out_degree()).values()),
        )
