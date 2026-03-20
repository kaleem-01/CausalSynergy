from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, Iterable, Optional, Set, Tuple
import networkx as nx
from ast import literal_eval
from itertools import combinations
from metrics.synergy import categorize_colliders
# from networkx.algorithms.dag import colliders as nx_colliders

# -----------------------------
# CPDAG encoding (single DiGraph)
# -----------------------------
# - Directed edge u->v  : only (u,v) exists
# - Undirected edge u-v : BOTH (u,v) and (v,u) exist
#
# We evaluate per unordered pair {u,v} with relation in:
#   "none", "undirected", "u->v", "v->u"


def pair_relation(g: nx.DiGraph, u, v) -> str:
    """
    Relation of unordered pair {u,v} in a CPDAG encoded as a single DiGraph.

    Returns one of: "none", "undirected", "u->v", "v->u"
    where "u->v" means u -> v (and NOT v -> u).
    """
    uv = g.has_edge(u, v)
    vu = g.has_edge(v, u)

    if not uv and not vu:
        return "none"
    if uv and vu:
        return "undirected"
    if uv:
        return "u->v"
    return "v->u"


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den != 0 else 0.0


def _f1(p: float, r: float) -> float:
    return 0.0 if (p + r) == 0 else 2.0 * p * r / (p + r)


@dataclass(frozen=True)
class CPDAGSets:
    # Skeleton adjacencies (regardless of type), as unordered pairs
    adj: Set[FrozenSet]
    # Undirected edges as unordered pairs
    undirected: Set[FrozenSet]
    # Directed edges as ordered pairs (u,v)
    directed: Set[Tuple]


def cpdag_sets(g: nx.DiGraph, nodes: Optional[Iterable] = None) -> CPDAGSets:
    """
    Build canonical sets from the single-DiGraph CPDAG encoding.

    Parameters
    ----------
    g : nx.DiGraph
        CPDAG encoding: undirected edges are bidirectional.
    nodes : optional iterable
        If provided, restrict / align to this node set.
        (Missing nodes are treated as isolated.)

    Returns
    -------
    CPDAGSets
    """
    if nodes is None:
        nodes = list(g.nodes())
    nodes = list(nodes)

    # Work with just the relevant nodes
    # (Edges to/from nodes not in `nodes` ignored)
    node_set = set(nodes)

    undirected: Set[FrozenSet] = set()
    directed: Set[Tuple] = set()

    # Determine relation per unordered pair by checking edge presence.
    for u, v in combinations(nodes, 2):
        uv = g.has_edge(u, v)
        vu = g.has_edge(v, u)
        if not uv and not vu:
            continue
        if uv and vu:
            undirected.add(frozenset((u, v)))
        else:
            # exactly one direction exists
            if uv:
                directed.add((u, v))
            else:
                directed.add((v, u))

    adj = set(undirected)
    adj.update(frozenset((u, v)) for (u, v) in directed)

    return CPDAGSets(adj=adj, undirected=undirected, directed=directed)


def shd_cpdag(true_g: nx.DiGraph, pred_g: nx.DiGraph, nodes: Optional[Iterable] = None) -> int:
    """
    CPDAG-SHD: count unordered pairs {u,v} where the relation differs between graphs.

    Relations are in: none / undirected / u->v / v->u.
    """
    if nodes is None:
        nodes = sorted(set(true_g.nodes()) | set(pred_g.nodes()))
    nodes = list(nodes)

    shd = 0
    for u, v in combinations(nodes, 2):
        r_t = pair_relation(true_g, u, v)
        r_p = pair_relation(pred_g, u, v)
        if r_t != r_p:
            shd += 1
    return shd


def skeleton_shd(true_g: nx.DiGraph, pred_g: nx.DiGraph, nodes: Optional[Iterable] = None) -> int:
    """
    Skeleton-SHD: count unordered pairs {u,v} where adjacency differs (ignore orientation/type).
    Equivalent to |E_true XOR E_pred| on the skeleton.
    """
    if nodes is None:
        nodes = sorted(set(true_g.nodes()) | set(pred_g.nodes()))
    t = cpdag_sets(true_g, nodes=nodes)
    p = cpdag_sets(pred_g, nodes=nodes)
    return len(t.adj ^ p.adj)


def compare_cpdags(true_graph: nx.DiGraph, learned_graph: nx.DiGraph, nodes: Optional[Iterable] = None) -> Dict[str, float]:
    """
    Compute global metrics comparing two CPDAGs encoded as a single nx.DiGraph:
      - cpdag_shd (type-aware SHD on CPDAG relations)
      - skeleton_shd (adjacency-only)
      - adjacency precision/recall/F1
      - directed-edge precision/recall/F1 (strict: only compelled directions count)
      - undirected-edge precision/recall/F1

    Notes
    -----
    - This is "strict" for directions: predicting u->v when truth is undirected counts as
      a directed false positive (because you asserted a compelled orientation that isn't compelled).
    - Likewise, if truth is directed but you predict undirected, you miss a directed edge.
    """
    if nodes is None:
        true_graph = nx.relabel_nodes(true_graph, lambda x: int(x))
        learned_graph = nx.relabel_nodes(learned_graph, lambda x: int(x))
        nodes = sorted(set(true_graph.nodes()) | set(learned_graph.nodes()))
    nodes = list(nodes)

    t = cpdag_sets(true_graph, nodes=nodes)
    p = cpdag_sets(learned_graph, nodes=nodes)

    # --- SHDs ---
    # cpdag_shd_val = shd_cpdag(true_graph, learned_graph, nodes=nodes)
    skel_shd_val = len(t.adj ^ p.adj)

    # --- Adjacency metrics (skeleton) ---
    adj_tp = len(t.adj & p.adj)
    adj_fp = len(p.adj - t.adj)
    adj_fn = len(t.adj - p.adj)

    adj_prec = _safe_div(adj_tp, adj_tp + adj_fp)
    adj_rec = _safe_div(adj_tp, adj_tp + adj_fn)
    adj_f1 = _f1(adj_prec, adj_rec)

    # # --- Directed metrics (ordered pairs) ---
    # dir_tp = len(t.directed & p.directed)
    # dir_fp = len(p.directed - t.directed)
    # dir_fn = len(t.directed - p.directed)

    # dir_prec = _safe_div(dir_tp, dir_tp + dir_fp)
    # dir_rec = _safe_div(dir_tp, dir_tp + dir_fn)
    # dir_f1 = _f1(dir_prec, dir_rec)

    # # --- Undirected metrics (unordered pairs) ---
    # undir_tp = len(t.undirected & p.undirected)
    # undir_fp = len(p.undirected - t.undirected)
    # undir_fn = len(t.undirected - p.undirected)

    # undir_prec = _safe_div(undir_tp, undir_tp + undir_fp)
    # undir_rec = _safe_div(undir_tp, undir_tp + undir_fn)
    # undir_f1 = _f1(undir_prec, undir_rec)

    # Optional: break down "type confusion" counts on adjacencies
    # (pairs that are adjacent in both, but differ in type/orientation)
    type_confusions = 0
    for uv in (t.adj & p.adj):
        u, v = tuple(uv)
        if pair_relation(true_graph, u, v) != pair_relation(learned_graph, u, v):
            type_confusions += 1

    return {
    "n_nodes": float(len(nodes)),
    "n_pairs": float(len(nodes) * (len(nodes) - 1) // 2),

    # "SHD [CPDAG]": float(cpdag_shd_val),
    "SHD [Skeleton]": float(skel_shd_val),
    "Type Confusions [Adjacency]": float(type_confusions),

    "n_true [Adjacency]": float(len(t.adj)),
    "n_pred [Adjacency]": float(len(p.adj)),
    "TP [Adjacency]": float(adj_tp),
    "FP [Adjacency]": float(adj_fp),
    "FN [Adjacency]": float(adj_fn),
    "Precision [Skeleton]": float(adj_prec),
    "Recall [Skeleton]": float(adj_rec),
    "F1 [Skeleton]": float(adj_f1),

    # "n_true [Directed]": float(len(t.directed)),
    # "n_pred [Directed]": float(len(p.directed)),
    # "TP [Directed]": float(dir_tp),
    # "FP [Directed]": float(dir_fp),
    # "FN [Directed]": float(dir_fn),
    # "Precision [Directed]": float(dir_prec),
    # "Recall [Directed]": float(dir_rec),
    # "F1 [Directed]": float(dir_f1),

    # "n_true [Undirected]": float(len(t.undirected)),
    # "n_pred [Undirected]": float(len(p.undirected)),
    # "TP [Undirected]": float(undir_tp),
    # "FP [Undirected]": float(undir_fp),
    # "FN [Undirected]": float(undir_fn),
    # "Precision [Undirected]": float(undir_prec),
    # "Recall [Undirected]": float(undir_rec),
    # "F1 [Undirected]": float(undir_f1),
}



# def pair_relation(g: nx.DiGraph, u, v) -> str:
#     """
#     Returns relation of unordered pair {u,v} in a CPDAG encoded as a single DiGraph.

#     One of: "none", "undirected", "u->v", "v->u"
#     """
#     uv = g.has_edge(u, v)
#     vu = g.has_edge(v, u)

#     if not uv and not vu:
#         return "none"

#     # Prefer explicit kind attribute if present
#     kind_uv = g.edges[u, v].get("kind") if uv else None
#     # kind_vu = g.edges[v, u].get("kind") if vu else None

#     # If either direction is explicitly undirected, treat as undirected adjacency
#     if kind_uv == "undirected":
#         return "undirected"

#     # Otherwise infer: both directions present => undirected
#     if uv and vu:
#         return "undirected"

#     # Single directed edge
#     if uv:
#         return "u->v"
#     return "v->u"


# def compare_cpdags(true_graph: nx.DiGraph, learned_graph: nx.DiGraph, nodes=None):
#     if nodes is None:
#         # Convert all node labels to int for consistent comparison
#         true_graph = nx.relabel_nodes(true_graph, lambda x: int(x))
#         learned_graph = nx.relabel_nodes(learned_graph, lambda x: int(x))
#         nodes = sorted(set(true_graph.nodes()) | set(learned_graph.nodes()))
#         nodes = sorted(set(true_graph.nodes()) | set(learned_graph.nodes()))

#     tp = fp = fn = 0
#     shd = 0

#     for u, v in itertools.combinations(nodes, 2):
#         rt = pair_relation(true_graph, u, v)
#         rl = pair_relation(learned_graph, u, v)

#         # SHD over marks (none/undirected/directed)
#         shd += (rt != rl)

#         true_adj = (rt != "none")
#         learned_adj = (rl != "none")

#         if learned_adj and true_adj:
#             tp += 1
#         elif learned_adj and not true_adj:
#             fp += 1
#         elif (not learned_adj) and true_adj:
#             fn += 1

#     precision = tp / (tp + fp) if tp + fp else 0.0
#     recall    = tp / (tp + fn) if tp + fn else 0.0
#     f1        = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

#     return {
#         "TP": tp,
#         "FP": fp,
#         "FN": fn,
#         "SHD": shd,
#         "Precision": round(precision, 3),
#         "Recall": round(recall, 3),
#         "F1": round(f1, 3),
#     }



def _kind(G: nx.DiGraph, u, v):
    if not G.has_edge(u, v):
        return None
    k = G.edges[u, v].get("kind", None)
    return None if k is None else str(k).strip().lower()

def _is_undirected_pair(G: nx.DiGraph, u, v) -> bool:
    uv = G.has_edge(u, v)
    vu = G.has_edge(v, u)
    if not uv and not vu:
        return False

    k_uv = _kind(G, u, v)
    k_vu = _kind(G, v, u)

    # Explicit undirected marker wins
    if k_uv == "undirected" or k_vu == "undirected":
        return True

    # If either edge has an explicit kind (and none says "undirected"),
    # do NOT auto-collapse bidirectional into "undirected".
    if (k_uv is not None) or (k_vu is not None):
        return False

    # Classic CPDAG encoding: bidirectional means undirected
    return uv and vu


def _has_directed_edge(G: nx.DiGraph, u, v) -> bool:
    return G.has_edge(u, v) and not _is_undirected_pair(G, u, v)


def colliders_directed_only(G: nx.DiGraph, *, require_unshielded: bool = True):
    """
    Returns triples (a, c, b) where a->c and b->c are directed (not undirected).
    If require_unshielded=True, also requires a and b are non-adjacent.
    """
    out = set()

    for c in G.nodes:
        parents = [p for p in G.predecessors(c) if _has_directed_edge(G, p, c)]
        for a, b in combinations(parents, 2):
            if require_unshielded and (G.has_edge(a, b) or G.has_edge(b, a)):
                # print(f"Skipping shielded collider candidate: {a} -> {c} <- {b}")
                continue

            a2, b2 = sorted((a, b), key=lambda x: str(x))
            out.add((a2, c, b2))

    return sorted(out, key=lambda t: (str(t[1]), str(t[0]), str(t[2])))

# def compare_dags(true_dag, learned_dag, tolerate_undirected=False):
#     """
#     Compare true DAG vs learned DAG with strict or tolerant matching.

#     Parameters
#     ----------
#     true_dag : nx.DiGraph
#         Ground truth graph.
#     learned_dag : nx.DiGraph
#         Learned causal graph.
#     tolerate_undirected : bool
#         If True, treat (A→B) and (B→A) as match if A—B exists.

#     Returns
#     -------
#     dict with TP, FP, FN, SHD, Precision, Recall, F1
#     """
#     # true_edges = set(true_dag.edges())
#     true_edges = {(str(u), str(v)) for u, v in true_dag.edges() if u != v}  # Exclude self-loops
#     learned_edges = {(str(u), str(v)) for u, v in learned_dag.edges() if u != v}  # Exclude self-loops
#     # learned_edges = set(learned_dag.edges())
#     # print(learned_edges)
#     tp = 0
#     fp = 0
#     fn = 0

#     # Count TP and FP
#     for edge in learned_edges:
#         if edge in true_edges:
#             tp += 1
#         elif tolerate_undirected and (edge[1], edge[0]) in true_edges:
#             # If reversed direction allowed
#             tp += 1
#         else:
#             fp += 1

#     # Count FN
#     for edge in true_edges:
#         if edge not in learned_edges:
#             if tolerate_undirected and (edge[1], edge[0]) in learned_edges:
#                 continue  # reversed match
#             fn += 1

#     precision = tp / (tp + fp) if tp + fp > 0 else 0.0
#     recall = tp / (tp + fn) if tp + fn > 0 else 0.0
#     f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
#     shd = fp + fn

#     tn = count_true_negatives(true_edges, learned_edges, nodes=set(true_dag.nodes()).union(set(learned_dag.nodes())), tolerate_undirected=tolerate_undirected)

#     if tolerate_undirected:
#         return {
#             "TP (undirected)": tp,
#             "FP (undirected)": fp,
#             "TN (undirected)": tn,
#             "FN (undirected)": fn,
#             "SHD (undirected)": shd,
#             "Precision (undirected)": round(precision, 3),
#             "Recall (undirected)": round(recall, 3),
#             "F1 (undirected)": round(f1, 3)
#         }
#     else:
#         return {
#             "TP": tp,
#             "FP": fp,
#             "TN": tn,
#             "FN": fn,
#             "SHD": shd,
#             "Precision": round(precision, 3),
#             "Recall": round(recall, 3),
#             "F1": round(f1, 3)
#         }


def count_true_negatives(true_edges, learned_edges, nodes, tolerate_undirected=False):
    """
    Count true negatives (TN) in structure learning evaluation.

    Args:
        true_edges (set): Set of true directed edges (tuples: (u,v))
        learned_edges (set): Set of learned directed edges (tuples: (u,v))
        nodes (list or set): All node names
        tolerate_undirected (bool): If True, reversed edges count as matches

    Returns:
        int: number of true negatives
    """
    # Build universe of all possible directed edges
    all_edges = {(u, v) for u in nodes for v in nodes if u != v}

    # True positives and false positives
    tp, fp, fn = 0, 0, 0

    for edge in learned_edges:
        if edge in true_edges:
            tp += 1
        elif tolerate_undirected and (edge[1], edge[0]) in true_edges:
            tp += 1
        else:
            fp += 1

    for edge in true_edges:
        if edge not in learned_edges:
            if tolerate_undirected and (edge[1], edge[0]) in learned_edges:
                continue
            fn += 1

    # TN = everything else
    tn = len(all_edges) - (tp + fp + fn)
    return tn



def _parse_comb(x):
    """Parse a comb entry from metadata into a tuple[int, ...]."""
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        return tuple(int(v) for v in x)
    if isinstance(x, str):
        t = literal_eval(x)  # safer than eval
        if isinstance(t, (list, tuple)):
            return tuple(int(v) for v in t)
    raise TypeError(f"Unsupported comb type: {type(x)} -> {x!r}")


def _expand_true_comb_to_triplets(comb):
    """
    metadata format: (*parents, collider)  (collider is LAST)
    return comparable triplets: (min(p1,p2), max(p1,p2), collider)
    If multi-parent: expand to all parent pairs.
    """
    comb = tuple(int(v) for v in comb)
    if len(comb) < 3:
        return []

    *parents, col = comb
    parents = sorted(parents)
    col = int(col)

    if len(parents) < 2:
        return []

    if len(parents) == 2:
        a, b = parents
        return [(a, b, col)]

    # multi-parent -> expand to all unordered pairs
    return [(a, b, col) for a, b in combinations(parents, 2)]


def _normalize_learned_triplet(t):
    """
    networkx.colliders format: (parent1, collider, parent2)  (collider is MIDDLE)
    normalize to: (min(parent1,parent2), max(parent1,parent2), collider)
    """
    p1, col, p2 = t
    a, b = sorted((int(p1), int(p2)))
    return (a, b, int(col))


def evaluate_colliders(metadata_df, learned_graph, dataset):
    # Decide which Type is considered "synergistic"
    # learned_graph = nx.relabel_nodes(learned_graph, lambda x: int(x))
    
    types = set(metadata_df["Type"].astype(str).values)
    syn_type = "SRV" if "SRV" in types else "XOR"

    all_combs = metadata_df["Combs"].dropna().tolist()
    pair_combs = metadata_df.loc[metadata_df["Type"] != syn_type, "Combs"].dropna().tolist()
    syn_combs = metadata_df.loc[metadata_df["Type"] == syn_type, "Combs"].dropna().tolist()

    # ---- TRUE: parse + normalize/expand to comparable triplets (parents sorted, collider last)
    ac_raw = [_parse_comb(x) for x in all_combs]
    ac_raw = [c for c in ac_raw if c is not None]

    sc_raw = [_parse_comb(x) for x in syn_combs]
    sc_raw = [c for c in sc_raw if c is not None]

    pw_raw = [_parse_comb(x) for x in pair_combs]
    pw_raw = [c for c in pw_raw if c is not None]


    ac_triplets = [t for comb in ac_raw for t in _expand_true_comb_to_triplets(comb)]
    sc_triplets = [t for comb in sc_raw for t in _expand_true_comb_to_triplets(comb)]
    pw_triplets = [t for comb in pw_raw for t in _expand_true_comb_to_triplets(comb)]


    true_sets = set(ac_triplets)
    true_sc_sets = set(sc_triplets)
    true_pw_sets = set(pw_triplets)

    # ---- LEARNED: normalize networkx colliders to (sorted parents, collider last)
    lc_list = list(colliders_directed_only(learned_graph))
    learned_sets = set(_normalize_learned_triplet(t) for t in lc_list)

    # ---- All-collider metrics
    correct = true_sets & learned_sets
    missing = true_sets - learned_sets
    extra = learned_sets - true_sets

    tp = len(correct)
    fp = len(extra)
    fn = len(missing)

    collider_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    collider_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    collider_f1 = (
        2 * collider_precision * collider_recall / (collider_precision + collider_recall)
        if (collider_precision + collider_recall) > 0 else 0.0
    )

    #----- Categorizing extra colliders into "synergistic" vs "pairwise" -----------------
    extra_sc, extra_pw = categorize_colliders(extra, dataset)


    # ---- Synergy collider recall (pairwise, comparable to learned colliders)
    correct_sc = true_sc_sets & learned_sets
    missing_sc = true_sc_sets - learned_sets



    tp_sc = len(correct_sc)
    fn_sc = len(missing_sc)
    fp_sc = len(extra_sc)
    sc_precision = tp_sc / (tp_sc + fp_sc) if (tp_sc + fp_sc) > 0 else 0.0
    sc_recall = tp_sc / (tp_sc + fn_sc) if (tp_sc + fn_sc) > 0 else 0.0
    sc_f1 = (
        2 * sc_precision * sc_recall / (sc_precision + sc_recall)
        if (sc_precision + sc_recall) > 0 else 0.0
    )


    
    # ---- Pairwise collider recall ------------------------------------
    correct_pw = true_pw_sets & learned_sets
    missing_pw = true_pw_sets - learned_sets
    tp_pw = len(correct_pw)
    fn_pw = len(missing_pw)
    fp_pw = len(extra_pw)
    pw_recall = tp_pw / (tp_pw + fn_pw) if (tp_pw + fn_pw) > 0 else 0.0
    pw_precision = tp_pw / (tp_pw + fp_pw) if (tp_pw + fp_pw) > 0 else 0.0
    pw_f1 = (
        2 * pw_precision * pw_recall / (pw_precision + pw_recall)
        if (pw_precision + pw_recall) > 0 else 0.0
    )

    return {
        # "TP (Collider Set)": correct,
        # "FP (Collider Set)": extra,
        # "FN (Collider Set)": missing,
        "Total (Colliders)": len(true_sets),
        "TP (Found Colliders)": len(correct),
        "FP (Extra Colliders)": len(extra),
        "FN (Missing Colliders )": len(missing),
        "TP (Synergy)": len(correct_sc),
        "FN (Synergy)": len(missing_sc),
        

        "Total (SC)": len(true_sc_sets),
        "Precision [Collider]": collider_precision,
        "Recall [Collider]": collider_recall,
        "F1 [Collider]": collider_f1,

        "Precision [Synergy]": sc_precision,
        "Recall [Synergy]": sc_recall,
        "F1 [Synergy]": sc_f1,
        
        "Total (PW)": len(true_pw_sets),
        "TP (Pairwise)": len(correct_pw),
        "FN (Pairwise)": len(missing_pw),
        
        "Precision [Pairwise]": pw_precision,
        "Recall [Pairwise]": pw_recall,
        "F1 [Pairwise]": pw_f1,

        # "Recall (Synergy Multi-Parent)": multi_parent_recall,
    }, correct, missing, extra


