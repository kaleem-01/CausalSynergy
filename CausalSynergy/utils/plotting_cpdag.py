# plotting_cpdag.py
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Hashable, Iterable, Optional, Tuple, Union, Set, Dict

import ast
import matplotlib.pyplot as plt
import networkx as nx

from networkx.drawing.nx_agraph import graphviz_layout


Node = Hashable
DirEdge = Tuple[Node, Node]
UndirEdge = Tuple[Node, Node]  # canonicalized


def _canon_undirected(u: Node, v: Node) -> UndirEdge:
    # stable canonicalization (works even if nodes aren't directly comparable)
    return (u, v) if str(u) <= str(v) else (v, u)


@dataclass(frozen=True)
class CPDAG:
    directed: nx.DiGraph
    undirected: nx.Graph


CPDAGLike = Union[CPDAG, Tuple[nx.DiGraph, nx.Graph], nx.DiGraph]


def _layout_dot(G: nx.Graph) -> dict:
    """Graphviz dot layout with safe fallback."""
    try:
        return graphviz_layout(G, prog="dot")
    except Exception:
        return nx.spring_layout(G, seed=0)


def to_cpdag_parts(cpdag: CPDAGLike) -> CPDAG:
    """
    Normalize CPDAG into (directed DiGraph, undirected Graph).

    Supports:
      - CPDAGParts(directed, undirected)
      - (directed, undirected)
      - legacy single DiGraph where undirected edges are encoded as u->v and v->u
    """
    if isinstance(cpdag, CPDAG):
        return cpdag

    if isinstance(cpdag, tuple) and len(cpdag) == 2:
        d, u = cpdag
        if not isinstance(d, nx.DiGraph) or not isinstance(u, nx.Graph):
            raise TypeError("Expected (nx.DiGraph, nx.Graph) for CPDAG tuple.")
        return CPDAG(directed=d, undirected=u)

    if isinstance(cpdag, nx.DiGraph):
        edges = set(cpdag.edges())
        directed = nx.DiGraph()
        directed.add_nodes_from(cpdag.nodes())

        undirected = nx.Graph()
        undirected.add_nodes_from(cpdag.nodes())

        for a, b in edges:
            if (b, a) in edges:
                undirected.add_edge(*_canon_undirected(a, b))
            else:
                directed.add_edge(a, b)

        return CPDAG(directed=directed, undirected=undirected)

    raise TypeError(f"Unsupported CPDAG type: {type(cpdag)}")


def graph_edge_sets(G: Union[nx.DiGraph, CPDAGLike]) -> Tuple[Set[DirEdge], Set[UndirEdge], Set[UndirEdge]]:
    """
    Returns:
      directed_edges: {(u,v)}
      undirected_edges: {(min,max)}
      skeleton_edges: {(min,max)} (all adjacencies ignoring direction)
    """
    if isinstance(G, (CPDAG, tuple)) or isinstance(G, nx.DiGraph) and not nx.is_directed_acyclic_graph(G):
        # Note: legacy single-DiGraph CPDAG may not be acyclic; treat as CPDAGLike via to_cpdag_parts.
        try:
            parts = to_cpdag_parts(G)  # type: ignore[arg-type]
            d = set(parts.directed.edges())
            u = {_canon_undirected(a, b) for a, b in parts.undirected.edges()}
            skel = set(u)
            for a, b in d:
                skel.add(_canon_undirected(a, b))
            return d, u, skel
        except TypeError:
            pass

    # treat as DAG/DiGraph
    if not isinstance(G, nx.DiGraph):
        raise TypeError("Expected nx.DiGraph for non-CPDAG input.")
    d = set(G.edges())
    u: Set[UndirEdge] = set()
    skel = {_canon_undirected(a, b) for a, b in d}
    return d, u, skel


def visualize_cpdag(
    cpdag: CPDAGLike,
    *,
    title: str = "CPDAG Visualization",
    labels: Optional[dict] = None,
    logic_map: Optional[dict] = None,
    ax=None,
    figsize=(10, 8),
):
    parts = to_cpdag_parts(cpdag)

    d_edges = list(parts.directed.edges())
    u_edges = list({_canon_undirected(a, b) for a, b in parts.undirected.edges()})

    if len(d_edges) == 0 and len(u_edges) == 0:
        fig, ax2 = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(6, 4))
        ax2.set_title(title)
        ax2.text(0.5, 0.5, "No edges to display.", ha="center", va="center")
        ax2.axis("off")
        return fig, ax2

    # skeleton for layout
    skel = nx.Graph()
    skel.add_nodes_from(parts.directed.nodes())
    skel.add_edges_from(u_edges)
    skel.add_edges_from((_canon_undirected(a, b) for a, b in d_edges))

    pos = _layout_dot(skel)

    # labels
    node_labels = {}
    for n in skel.nodes():
        lines = [str(n)]
        if logic_map is not None and n in logic_map:
            lines.append(str(logic_map[n]))
        if labels is not None and n in labels:
            lines.append(str(labels[n]))
        node_labels[n] = "\n".join(lines)

    fig, ax2 = (ax.figure, ax) if ax is not None else plt.subplots(figsize=figsize)

    # nodes
    nx.draw_networkx_nodes(skel, pos, ax=ax2, node_size=2200, edgecolors="black")
    nx.draw_networkx_labels(skel, pos, ax=ax2, labels=node_labels, font_size=10)

    # undirected edges (no arrows)
    if u_edges:
        nx.draw_networkx_edges(
            skel,
            pos,
            ax=ax2,
            edgelist=u_edges,
            arrows=False,
            width=2,
        )

    # directed edges (arrows)
    if d_edges:
        nx.draw_networkx_edges(
            parts.directed,
            pos,
            ax=ax2,
            edgelist=d_edges,
            arrows=True,
            width=2,
        )

    ax2.set_title(title)
    ax2.axis("off")
    fig.tight_layout()
    return fig, ax2


def compare_graphs_with_labels(
    true_graph: Union[nx.DiGraph, CPDAGLike],
    learned_graph: Union[nx.DiGraph, CPDAGLike],
    metadata=None,
    title="Learned vs True",
    logic_map=None,
    figsize=(12, 10),
    ax=None,
):
    """
    Works for:
      - DAG vs DAG (like before)
      - CPDAG vs CPDAG
      - DAG vs CPDAG (compares against DAG's skeleton/directions)
    """
    node_kind = defaultdict(set)
    if metadata is not None:
        for _, row in metadata.iterrows():
            comb = row["Combs"]
            if isinstance(comb, str):
                comb = ast.literal_eval(comb)
            kind = row["Type"]
            if len(comb) == 3:
                _, _, w = comb
                node_kind[int(w)].add(str(kind))
            elif len(comb) == 2:
                _, v = comb
                node_kind[int(v)].add(str(kind))

    true_d, true_u, true_skel = graph_edge_sets(true_graph)
    learned_d, learned_u, learned_skel = graph_edge_sets(learned_graph)

    # Build layout graph from TRUE skeleton (best for “compare to ground truth”)
    base = nx.Graph()
    # choose nodes from both to avoid missing positions
    all_nodes = set()
    if isinstance(true_graph, (nx.Graph, nx.DiGraph)):
        all_nodes |= set(true_graph.nodes())
    if isinstance(learned_graph, (nx.Graph, nx.DiGraph)):
        all_nodes |= set(learned_graph.nodes())
    # also include nodes present in edge sets
    for a, b in true_skel | learned_skel:
        all_nodes.add(a)
        all_nodes.add(b)

    base.add_nodes_from(all_nodes)
    base.add_edges_from(true_skel)

    pos = _layout_dot(base)

    # Node labels/colors (same idea as your original)
    node_labels = {}
    node_colors = []
    for n in base.nodes():
        lines = [str(n)]
        if logic_map and n in logic_map:
            lines.append(str(logic_map[n]))
        if node_kind and n in node_kind:
            lines.append("/".join(sorted(node_kind[n])))
        node_labels[n] = "\n".join(lines)

        kinds = node_kind.get(n, set())
        if "SRV" in kinds and "Pairwise" in kinds:
            node_colors.append("violet")
        elif "SRV" in kinds:
            node_colors.append("tomato")
        elif "Pairwise" in kinds:
            node_colors.append("skyblue")
        else:
            node_colors.append("lightgrey")

    # Classify learned edges for coloring
    correct_dir = set()
    wrong_dir = set()
    correct_adj_only = set()
    extra = set()

    # directed learned edges
    for u, v in learned_d:
        if (u, v) in true_d:
            correct_dir.add((u, v))
        elif (v, u) in true_d:
            wrong_dir.add((u, v))
        elif _canon_undirected(u, v) in true_skel:
            correct_adj_only.add((u, v))  # adjacency exists, direction not matching/compelled
        else:
            extra.add((u, v))

    # undirected learned edges
    learned_u_canon = {_canon_undirected(a, b) for a, b in learned_u}
    correct_undir = {e for e in learned_u_canon if e in true_skel}
    extra_undir = {e for e in learned_u_canon if e not in true_skel}

    # simple adjacency-level precision/recall (often what you want for CPDAG plots)
    tp_adj = len(learned_skel & true_skel)
    fp_adj = len(learned_skel - true_skel)
    fn_adj = len(true_skel - learned_skel)
    precision_adj = tp_adj / (tp_adj + fp_adj) if (tp_adj + fp_adj) else 0.0
    recall_adj = tp_adj / (tp_adj + fn_adj) if (tp_adj + fn_adj) else 0.0
    f1_adj = (2 * precision_adj * recall_adj / (precision_adj + recall_adj)) if (precision_adj + recall_adj) else 0.0

    fig, ax2 = (ax.figure, ax) if ax is not None else plt.subplots(figsize=figsize)

    # Draw base skeleton (TRUE) in neutral style
    nx.draw_networkx_nodes(base, pos, ax=ax2, node_color=node_colors, node_size=2500, edgecolors="black")
    nx.draw_networkx_labels(base, pos, ax=ax2, labels=node_labels, font_size=9)
    if true_skel:
        nx.draw_networkx_edges(
            base,
            pos,
            ax=ax2,
            edgelist=list(true_skel),
            arrows=False,
            width=2,
        )

    # Overlay learned undirected edges (correct vs extra)
    if correct_undir:
        nx.draw_networkx_edges(
            base,
            pos,
            ax=ax2,
            edgelist=list(correct_undir),
            edge_color="green",
            arrows=False,
            width=2,
        )
    if extra_undir:
        nx.draw_networkx_edges(
            base,
            pos,
            ax=ax2,
            edgelist=list(extra_undir),
            edge_color="red",
            arrows=False,
            width=2,
        )

    # Overlay learned directed edges with colors (like your original, plus “adjacency-only”)
    def _draw_dir(edgelist, color):
        if not edgelist:
            return
        nx.draw_networkx_edges(
            nx.DiGraph(base),  # dummy container for arrows
            pos,
            ax=ax2,
            edgelist=list(edgelist),
            edge_color=color,
            arrows=True,
            width=2,
        )

    _draw_dir(correct_dir, "green")
    _draw_dir(wrong_dir, "yellow")
    _draw_dir(correct_adj_only, "orange")
    _draw_dir(extra, "red")

    ax2.set_title(title)
    ax2.axis("off")
    fig.tight_layout()

    return fig, {
        "correct_dir": correct_dir,
        "wrong_dir": wrong_dir,
        "correct_adj_only": correct_adj_only,
        "extra_dir": extra,
        "correct_undir": correct_undir,
        "extra_undir": extra_undir,
        "precision_adj": precision_adj,
        "recall_adj": recall_adj,
        "f1_adj": f1_adj,
        "tp_adj": tp_adj,
        "fp_adj": fp_adj,
        "fn_adj": fn_adj,
    }


def visualize_network_labels(metadata, labels=None, title="Causal Network visualisation", ax=None):
    G = nx.DiGraph()
    node_kind = defaultdict(set)

    if metadata is not None:
        for _, row in metadata.iterrows():
            comb = row["Combs"]
            if isinstance(comb, str):
                comb = ast.literal_eval(comb)

            if not isinstance(comb, (list, tuple)) or len(comb) < 2:
                raise ValueError(f"Invalid comb: {comb}")

            kind = str(row["Type"])
            *parents, child = comb

            for p in parents:
                G.add_edge(int(p), int(child))
            node_kind[int(child)].add(kind)

    if G.number_of_edges() == 0:
        fig, ax2 = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(6, 4))
        ax2.set_title(title)
        ax2.text(0.5, 0.5, "No edges to display.", ha="center", va="center")
        ax2.axis("off")
        return fig, ax2

    use_nodes = {u for u, _ in G.edges()} | {v for _, v in G.edges()}
    H = G.subgraph(use_nodes).copy()

    pos = _layout_dot(nx.DiGraph(H))

    node_labels = {}
    node_colors = []
    for n in H.nodes:
        lines = [str(n)]
        if labels is not None and n in labels:
            lines.append(str(labels[n]))
        if n in node_kind:
            lines.append("/".join(sorted(node_kind[n])))
        node_labels[n] = "\n".join(lines)

        kinds = node_kind.get(n, set())
        if "SRV" in kinds and "Pairwise" in kinds:
            node_colors.append("violet")
        elif "SRV" in kinds:
            node_colors.append("tomato")
        elif "Pairwise" in kinds:
            node_colors.append("skyblue")
        else:
            node_colors.append("lightgrey")

    fig, ax2 = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(10, 8))
    nx.draw(
        H,
        pos,
        ax=ax2,
        labels=node_labels,
        node_color=node_colors,
        node_size=2200,
        edgecolors="black",
        arrows=True,
        width=2,
        font_size=10,
    )
    ax2.set_title(title)
    ax2.axis("off")
    fig.tight_layout()
    return fig, ax2


# ----------------------------
# Updated: three side-by-side (panel 2 now works for DAG or CPDAG)
# ----------------------------
def plot_three_side_by_side(
    cpdag=None,
    true_graph=None,
    learned_graph=None,
    metadata=None,
    labels=None,
    logic_map=None,
    titles=("CPDAG", "Learned vs True", "Network from Metadata"),
    figsize=(20, 8),
):
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    visualize_cpdag(cpdag, title=titles[0], ax=axes[0])

    if true_graph is None or learned_graph is None:
        axes[1].set_title(titles[1])
        axes[1].text(0.5, 0.5, "true_graph / learned_graph missing.", ha="center", va="center")
        axes[1].axis("off")
        metrics = None
    else:
        _, metrics = compare_graphs_with_labels(
            true_graph=true_graph,
            learned_graph=learned_graph,
            metadata=metadata,
            title=titles[1],
            logic_map=logic_map,
            ax=axes[1],
        )

    visualize_network_labels(metadata=metadata, labels=labels, title=titles[2], ax=axes[2])

    return fig
