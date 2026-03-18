# import ast
from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx

from networkx.drawing.nx_agraph import graphviz_layout
import ast


def visualize_graph(dag, title="Causal Network Visualization", figsize=(10, 8), ax=None):
    if dag is None or dag.number_of_edges() == 0:
        fig, ax2 = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(6, 4))
        ax2.set_title(title)
        ax2.text(0.5, 0.5, "No edges to display.", ha="center", va="center")
        ax2.axis("off")
        return fig, ax2

    use_nodes = {u for u, _ in dag.edges} | {v for _, v in dag.edges}
    H = dag.subgraph(use_nodes).copy()

    pos = graphviz_layout(H, prog="dot")

    have_labels = hasattr(dag, "labels") and isinstance(getattr(dag, "labels"), dict)
    node_labels = {}
    for n in H.nodes:
        if have_labels and n in dag.labels and str(dag.labels[n]).strip() != "":
            node_labels[n] = f"{n}\n{dag.labels[n]}"
        else:
            node_labels[n] = str(n)

    def _is_undirected_edge(G: nx.DiGraph, u, v) -> bool:
        uv = G.has_edge(u, v)
        vu = G.has_edge(v, u)

        kind_uv = G.edges[u, v].get("kind") if uv else None
        kind_vu = G.edges[v, u].get("kind") if vu else None

        if kind_uv == "undirected" or kind_vu == "undirected":
            return True

        # Common CPDAG encoding: both directions present => undirected adjacency
        if uv and vu and kind_uv != "directed" and kind_vu != "directed":
            return True

        return False

    # Split edges
    undirected_pairs = set()
    directed_edges = []

    for u, v in H.edges():
        if _is_undirected_edge(H, u, v):
            a, b = sorted((u, v), key=lambda x: str(x))
            undirected_pairs.add((a, b))
        else:
            directed_edges.append((u, v))

    # Degrees for coloring: ignore undirected adjacencies
    Hd = nx.DiGraph()
    Hd.add_nodes_from(H.nodes())
    Hd.add_edges_from(directed_edges)

    node_colors = []
    for n in H.nodes:
        indeg = Hd.in_degree(n)
        outdeg = Hd.out_degree(n)
        if indeg == 0 and outdeg > 0:
            node_colors.append("palegreen")
        elif indeg > 0 and outdeg == 0:
            node_colors.append("salmon")
        else:
            node_colors.append("skyblue")

    fig, ax2 = (ax.figure, ax) if ax is not None else plt.subplots(figsize=figsize)

    # Draw nodes + labels
    nx.draw_networkx_nodes(
        H, pos, ax=ax2,
        node_color=node_colors,
        node_size=2200,
        edgecolors="black",
        linewidths=1.5,
    )
    nx.draw_networkx_labels(H, pos, labels=node_labels, ax=ax2, font_size=10)

    # Draw undirected edges (no arrows)
    nx.draw_networkx_edges(
        H, pos, ax=ax2,
        edgelist=list(undirected_pairs),
        arrows=False,
        width=2,
    )

    # Draw directed edges (with arrows)
    nx.draw_networkx_edges(
        H, pos, ax=ax2,
        edgelist=directed_edges,
        arrows=True,
        node_size=2200,
        arrowstyle="-|>",
        arrowsize=10,
        width=2,
    )

    ax2.set_title(title)
    ax2.axis("off")
    fig.tight_layout()
    return fig, ax2


def visualize_dag(dag, title="Causal Network Visualization", results=False, ax=None):
    if dag is None or dag.number_of_edges() == 0:
        fig, ax2 = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(6, 4))
        ax2.set_title(title)
        ax2.text(0.5, 0.5, "No edges to display.", ha="center", va="center")
        ax2.axis("off")
        return fig, ax2

    use_nodes = {u for u, _ in dag.edges} | {v for _, v in dag.edges}
    H = dag.subgraph(use_nodes).copy()

    pos = graphviz_layout(H)

    have_labels = hasattr(dag, "labels") and isinstance(getattr(dag, "labels"), dict)
    node_labels = {}
    for n in H.nodes:
        if have_labels and n in dag.labels and str(dag.labels[n]).strip() != "":
            node_labels[n] = f"{n}\n{dag.labels[n]}"
        else:
            node_labels[n] = str(n)

    node_colors = []
    for n in H.nodes:
        indeg = H.in_degree(n)
        outdeg = H.out_degree(n)
        if indeg == 0 and outdeg > 0:
            node_colors.append("palegreen")
        elif indeg > 0 and outdeg == 0:
            node_colors.append("salmon")
        else:
            node_colors.append("skyblue")

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


# def compare_dags_with_labels(
#     true_graph,
#     learned_graph,
#     metadata=None,
#     title="Learned vs True DAG",
#     logic_map=None,
#     figsize=(12, 10),
#     ax=None,
# ):
#     node_kind = defaultdict(set)
#     if metadata is not None:
#         for _, row in metadata.iterrows():
#             comb = row["Combs"]
#             if isinstance(comb, str):
#                 comb = ast.literal_eval(comb)
#             kind = row["Type"]
#             if len(comb) == 3:
#                 _, _, w = comb
#                 node_kind[w].add(kind)
#             elif len(comb) == 2:
#                 _, v = comb
#                 node_kind[v].add(kind)
    
    

#     true_set = {(int(u), int(v)) for u, v in true_graph.edges()}
#     learned_set = {(int(u), int(v)) for u, v in learned_graph.edges()}
#     true_rev = {(v, u) for (u, v) in true_set}

#     correct = learned_set & true_set
#     wrong_dir = learned_set & true_rev
#     extra = learned_set - (true_set | true_rev)
#     missing = true_set - (learned_set | true_rev)

#     tp = len(correct)
#     fp = len(extra) + len(wrong_dir)
#     fn = len(missing)

#     precision = tp / (tp + fp) if tp + fp > 0 else 0.0
#     recall = tp / (tp + fn) if tp + fn > 0 else 0.0
#     f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

#     edge_list = []
#     edge_colors = []
#     for u, v in learned_set:
#         edge_list.append((u, v))
#         if (u, v) in correct:
#             edge_colors.append("green")
#         elif (u, v) in wrong_dir:
#             edge_colors.append("yellow")
#         else:
#             edge_colors.append("red")

#     use_nodes = {u for u, _ in true_graph.edges} | {v for _, v in true_graph.edges}
#     H = true_graph.subgraph(use_nodes).copy()
#     pos = graphviz_layout(H, prog="dot")

#     node_labels = {}
#     node_colors = []
#     for n in true_graph.nodes():
#         lines = [str(n)]
#         if logic_map and n in logic_map:
#             lines.append(str(logic_map[n]))
#         if node_kind and n in node_kind:
#             lines.append("/".join(sorted(node_kind[n])))
#         node_labels[n] = "\n".join(lines)

#         kinds = node_kind.get(n, set())
#         if "SRV" in kinds and "Pairwise" in kinds:
#             node_colors.append("violet")
#         elif "SRV" in kinds:
#             node_colors.append("tomato")
#         elif "Pairwise" in kinds:
#             node_colors.append("skyblue")
#         else:
#             node_colors.append("lightgrey")

#     fig, ax2 = (ax.figure, ax) if ax is not None else plt.subplots(figsize=figsize)
#     nx.draw(
#         true_graph,
#         pos,
#         ax=ax2,
#         node_color=node_colors,
#         labels=node_labels,
#         node_size=2500,
#         edgecolors="black",
#         arrows=False,
#         width=2,
#         font_size=9,
#     )
#     nx.draw_networkx_edges(
#         learned_graph,
#         pos,
#         ax=ax2,
#         edgelist=edge_list,
#         edge_color=edge_colors,
#         arrows=True,
#         width=2,
#     )
#     ax2.set_title(title)
#     ax2.axis("off")
#     fig.tight_layout()

#     return fig, {
#         "correct": correct,
#         "wrong_dir": wrong_dir,
#         "extra": extra,
#         "missing": missing,
#         "precision": precision,
#         "recall": recall,
#         "f1": f1,
#     }


def visualize_network_labels(metadata, labels=None, title="Causal Network visualisation", ax=None):
    G = nx.DiGraph()
    node_kind = defaultdict(set)

    if metadata is not None:
        # for _, row in metadata.iterrows():
        #     comb = row["Combs"]
        #     if isinstance(comb, str):
        #         comb = ast.literal_eval(comb)
        #     kind = row["Type"]

        #     if len(comb) == 3:
        #         u, v, w = comb
        #         G.add_edge(u, w)
        #         G.add_edge(v, w)
        #         node_kind[w].add(kind)
        #     elif len(comb) == 2:
        #         u, v = comb
        #         G.add_edge(u, v)
        #         node_kind[v].add(kind)
        #     else:
        #         raise ValueError(f"Unexpected comb length: {comb}")

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

    use_nodes = {u for u, _ in G.edges} | {v for _, v in G.edges}
    H = G.subgraph(use_nodes).copy()

    pos = graphviz_layout(H, prog="dot")

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


def plot_three_side_by_side(
    dag=None,
    true_graph=None,
    learned_graph=None,
    metadata=None,
    labels=None,
    logic_map=None,
    titles=("DAG", "Learned vs True DAG", "Network from Metadata"),
    figsize=(20, 8),
):
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    
    visualize_graph(dag, title=titles[0], ax=axes[0])

    if true_graph is None or learned_graph is None:
        axes[1].set_title(titles[1])
        axes[1].text(0.5, 0.5, "true_graph / learned_graph missing.", ha="center", va="center")
        axes[1].axis("off")
        metrics = None
    else:
        _, metrics = compare_dags_with_labels(
            true_graph=true_graph,
            learned_graph=learned_graph,
            metadata=metadata,
            title=titles[1],
            logic_map=logic_map,
            ax=axes[1],
        )

    visualize_network_labels(metadata=metadata, labels=labels, title=titles[2], ax=axes[2])

    # fig.tight_layout()
    return fig

