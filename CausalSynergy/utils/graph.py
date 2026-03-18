from __future__ import annotations


import ast
from collections import defaultdict

# from metrics.graph import compare_dags

import itertools
import networkx as nx
from pgmpy.base import DAG


# from dataclasses import dataclass

# @dataclass(frozen=True)
# class CPDAG:
#     directed: nx.DiGraph
#     undirected: nx.Graph


# def dag_to_cpdag_pgmpy(dag: nx.DiGraph) -> tuple[nx.DiGraph, nx.Graph]: 
#     dag = DAG(dag.edges())
#     cpdag = dag.to_pdag()  # CPDAG/PDAG

#     directed = cpdag.directed_edges
#     undirected = cpdag.undirected_edges
#     # print("directed:", directed)
#     # print("undirected:", undirected)
#     return directed, undirected

def create_graph(meta_df):
    G = nx.DiGraph()
    node_kind = defaultdict(set)          # collects the kinds per *target* node

    for _, row in meta_df.iterrows():
        comb = row["Combs"]
        if isinstance(comb, str):
            comb = ast.literal_eval(comb)  # turn "[1, 2]" into [1, 2]
        kind = row["Type"]

        if len(comb) == 3:                 # [u, v, w] → u→w, v→w
            u, v, w = comb
            G.add_edge(u, w)
            G.add_edge(v, w)
            node_kind[w].add(kind)
        elif len(comb) == 2:               # [u, v] → u→v
            u, v = comb
            G.add_edge(u, v)
            node_kind[v].add(kind)
        else:
            raise ValueError(f"Unexpected comb length: {comb}")

    # keep only nodes with at least one edge
    use_nodes = {u for u, _ in G.edges} | {v for _, v in G.edges}
    H = G.subgraph(use_nodes).copy()

    return H


def dag_to_cpdag(dag: nx.DiGraph) -> tuple[nx.DiGraph, nx.Graph]:
    """
    Convert a DAG (nx.DiGraph) to its CPDAG (essential graph).

    Returns
    -------
    directed : nx.DiGraph
        The directed part of the CPDAG (compelled edges).
    undirected : nx.Graph
        The undirected part of the CPDAG (reversible edges).
    """
    # if not nx.is_directed_acyclic_graph(dag):
    #     raise ValueError("Input must be a DAG (directed and acyclic).")

    nodes = list(dag.nodes())
    directed = nx.DiGraph()
    directed.add_nodes_from(nodes)

    # Start with the skeleton: all DAG adjacencies as undirected edges
    undirected = nx.Graph()
    undirected.add_nodes_from(nodes)
    for u, v in dag.edges():
        undirected.add_edge(u, v)

    # Helper to represent CPDAG as single DiGraph
    def as_single_networkx_digraph(directed: nx.DiGraph, undirected: nx.Graph) -> nx.DiGraph:
        """
        Optional: represent the CPDAG as a single DiGraph by encoding undirected edges
        as two opposite directed edges with attribute `kind="undirected"`.
        """
        g = nx.DiGraph()
        g.add_nodes_from(directed.nodes())
        g.add_edges_from(directed.edges(), kind="directed")
        for u, v in undirected.edges():
            # print(f"undirected edge: {u} -- {v}")
            g.add_edge(u, v, kind="undirected")
            g.add_edge(v, u, kind="undirected")
        return g

    def adjacent(a, b) -> bool:
        return (
            undirected.has_edge(a, b)
            or directed.has_edge(a, b)
            or directed.has_edge(b, a)
        )

    def would_create_cycle(u, v) -> bool:
        # adding u->v creates a directed cycle iff v reaches u already
        return nx.has_path(directed, v, u)

    def orient(u, v) -> bool:
        """Orient undirected edge u-v into u->v. Returns True if changed."""
        if not undirected.has_edge(u, v):
            return False
        if directed.has_edge(v, u):
            print(ValueError(f"Conflict: trying to orient {u}->{v} but {v}->{u} already directed."))
            # raise ValueError(f"Conflict: trying to orient {u}->{v} but {v}->{u} already directed.")
        if would_create_cycle(u, v):
            # raise ValueError(f"Would create directed cycle by orienting {u}->{v}.")
            print(ValueError(f"Would create directed cycle by orienting {u}->{v}."))
            
        undirected.remove_edge(u, v)
        directed.add_edge(u, v)
        return True

    # 1) Orient v-structures: for each node b, for each pair of parents a,c
    for b in nodes:
        parents = list(dag.predecessors(b))
        for a, c in itertools.combinations(parents, 2):
            # "unshielded": a and c not adjacent in skeleton
            if not undirected.has_edge(a, c):
                # collider a -> b <- c is compelled in the CPDAG
                if undirected.has_edge(a, b):
                    orient(a, b)
                if undirected.has_edge(c, b):
                    orient(c, b)

    # 2) Close under Meek rules (R1–R4) until convergence :contentReference[oaicite:1]{index=1}
    changed = True
    while changed:
        changed = False

        # R1: a->b and b-c and a not adjacent c  =>  orient b->c
        for a, b in list(directed.edges()):
            for c in list(undirected.neighbors(b)):
                if not adjacent(a, c):
                    changed |= orient(b, c)

        # Helper: try orienting x-y either direction if a rule fires
        def try_orient_either(x, y, predicate_xy) -> bool:
            # try x->y first
            if predicate_xy(x, y):
                return orient(x, y)
            # else try y->x
            if predicate_xy(y, x):
                return orient(y, x)
            return False

        # Precompute successor sets for faster checks
        succ = {u: set(directed.successors(u)) for u in directed.nodes()}

        # R2: a-b and a->c->b  => orient a->b
        for x, y in list(undirected.edges()):
            def r2(a, b):
                return any((c in succ[a]) and directed.has_edge(c, b) for c in succ[a])
            changed |= try_orient_either(x, y, r2)

        # R3: a-b and (a-k -> b) and (a-l -> b) with k nonadjacent l  => orient a->b
        for x, y in list(undirected.edges()):
            def r3(a, b):
                # candidates k where a-k undirected and k->b directed
                ks = [k for k in undirected.neighbors(a) if directed.has_edge(k, b)]
                if len(ks) < 2:
                    return False
                for k, l in itertools.combinations(ks, 2):
                    if not adjacent(k, l):
                        return True
                return False
            changed |= try_orient_either(x, y, r3)

        # R4: a-b and a-k and k->l->b with k nonadjacent b  => orient a->b
        for x, y in list(undirected.edges()):
            def r4(a, b):
                for k in undirected.neighbors(a):  # a-k (undirected)
                    if adjacent(k, b):
                        continue
                    # chain k->l->b
                    for l in succ.get(k, ()):
                        if directed.has_edge(l, b):
                            return True
                return False
            changed |= try_orient_either(x, y, r4)

    return as_single_networkx_digraph(directed, undirected)


