import random
import copy
import numpy as np
import pandas as pd
import networkx as nx
import jointpmf.jointpmf as jp


def construct_jpmf_bn_from_dag(
    dag: nx.DiGraph,
    _precomputed_dependent_vars_one_source,
    _precomputed_dependent_vars_two_sources,
    _precomputed_srvs,
    *,
    numvalues=3,
    prob_pairwise=[0.75, 0.25],
):
    """
    Construct a JPMF Bayesian Network using an EXISTING DAG structure.

    Parents are determined by the DAG, while CPDs are sampled from
    the precomputed JPMF dependency banks.

    Parameters
    ----------
    dag : nx.DiGraph
        Structure to instantiate.

    prob_pairwise :
        Probability of selecting pairwise vs SRV for 2-parent nodes.

    Returns
    -------
    bn
    metadata dataframe
    """

    if not nx.is_directed_acyclic_graph(dag):
        raise ValueError("Input graph must be a DAG")

    topo = list(nx.topological_sort(dag))

    bn = jp.BayesianNetwork()

    # map DAG node → BN variable index
    node_map = {}

    pair_tri_combs = []
    pair_tri_combs_type = []

    for node in topo:

        parents = list(dag.predecessors(node))
        parent_ix = [node_map[p] for p in parents]

        # ROOT
        if len(parents) == 0:
            bn.append_independent_variable("dirichlet", numvalues)
            node_map[node] = bn.numvariables - 1

        # SINGLE PARENT
        elif len(parents) == 1:
            cond = random.choice(_precomputed_dependent_vars_one_source)
            bn.append_conditional_variable(
                cond,
                parent_ix,
                numvalues=numvalues
            )
            new_ix = bn.numvariables - 1
            node_map[node] = new_ix
            pair_tri_combs.append(parent_ix + [new_ix])
            pair_tri_combs_type.append("Pairwise")

        # TWO PARENTS
        elif len(parents) == 2:
            parent_ix.sort()
            # choose pairwise or synergy
            mode = np.random.choice(
                ["pairwise", "srv"],
                p=prob_pairwise
            )

            if mode == "pairwise":
                cond = random.choice(_precomputed_dependent_vars_two_sources)
                pair_tri_combs_type.append("Pairwise")

            else:
                cond = random.choice(_precomputed_srvs)
                pair_tri_combs_type.append("SRV")

            bn.append_conditional_variable(
                cond,
                parent_ix,
                numvalues=numvalues
            )

            new_ix = bn.numvariables - 1
            node_map[node] = new_ix
            pair_tri_combs.append(parent_ix + [new_ix])
        else:
            raise ValueError(
                f"Node {node} has {len(parents)} parents but "
                "JPMF construction supports ≤2."
            )
    metadata = pd.DataFrame(
        {"Combs": pair_tri_combs, "Type": pair_tri_combs_type}
    )
    return bn, metadata