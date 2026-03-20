import numpy as np
import networkx as nx
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import chisq, gsq
from pgmpy.estimators import BIC, GES, HillClimbSearch
from algorithms.ea import GeneticBNSearchMatrix as GeneticBNSearch
from utils.graph import dag_to_cpdag
from algorithms.notears import notears_linear

import logging
import warnings
logging.getLogger('pgmpy').setLevel(logging.WARNING)
warnings.filterwarnings("ignore")


def run_ea(df, **kwargs):
    """Run the evolutionary algorithm for structure learning."""
    learned_dag, score = GeneticBNSearch(
            df,
            **kwargs
        ).run()
    
    return dag_to_cpdag(learned_dag), score



def run_pc_gsq(df, alpha=0.05):
    """
    Run the PC algorithm using chi-square tests on binary data.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset.
    alpha : float
        Significance level for conditional independence tests.

    Returns
    -------
    nx.DiGraph
        Learned DAG structure.
    """
    # Run PC algorithm from causal-learn
    cg = pc(data=df.values, 
            alpha=alpha, 
            indep_test=gsq, 
            labels=list(df.columns))

    # Extract learned DAG
    cg.to_nx_graph()
    learned_dag = nx.DiGraph()
    learned_dag.add_nodes_from(df.columns)
    # # cg.G.graph is an adjacency matrix, cg.G.get_graph_type() == 'dag'
    for i, src in enumerate(df.columns):
        for j, tgt in enumerate(df.columns):
            if cg.G.graph[i, j] == 1:
                learned_dag.add_edge(src, tgt)

    scorer = BIC(df)
    score = scorer.score(learned_dag)    
    return dag_to_cpdag(learned_dag), score


def run_pc(df, alpha=0.05):
    """
    Run the PC algorithm using chi-square tests on binary data.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset.
    alpha : float
        Significance level for conditional independence tests.

    Returns
    -------
    nx.DiGraph
        Learned DAG structure.
    """
    # Run PC algorithm from causal-learn
    cg = pc(data=df.values, 
            alpha=alpha, 
            indep_test=chisq, 
            labels=list(df.columns))

    # Extract learned DAG
    cg.to_nx_graph()
    learned_dag = nx.DiGraph()
    learned_dag.add_nodes_from(df.columns)
    # # cg.G.graph is an adjacency matrix, cg.G.get_graph_type() == 'dag'
    for i, src in enumerate(df.columns):
        for j, tgt in enumerate(df.columns):
            if cg.G.graph[i, j] == 1:
                learned_dag.add_edge(src, tgt)

    scorer = BIC(df)
    score = scorer.score(learned_dag)    
    return dag_to_cpdag(learned_dag), score


def run_hc(df, restarts=5):
    """
    Run Hill Climb structure learning using BIC scoring with random restarts.
    """

    hc = HillClimbSearch(df)
    scorer = BIC(df)

    best_model = None
    best_score = -np.inf

    for _ in range(max(1, restarts)):        
        model = hc.estimate(scoring_method="bic-d", show_progress=False)
        score = scorer.score(model)
        if score > best_score:
            best_score = score
            best_model = model

    learned_dag = nx.DiGraph(best_model.edges())
    return dag_to_cpdag(learned_dag), best_score


def run_ges(df, cpdag=True):
    est = GES(df)
    estimated_model = est.estimate(scoring_method="bic-d")
    scorer = BIC(df)
    learned_dag = nx.DiGraph(estimated_model.edges())
    score = scorer.score(learned_dag)
    return dag_to_cpdag(learned_dag), score


def run_notears(
    df,
    lambda1=0.1,
    loss_type="logistic",
    max_iter=100,
    h_tol=1e-8,
    rho_max=1e16,
    w_threshold=0.3,
):
    """
    Run linear NOTEARS on a dataframe and return a CPDAG + score,
    matching the style of run_pc().

    Parameters
    ----------
    df : pd.DataFrame
        Dataset.
    lambda1 : float
        L1 sparsity penalty.
    loss_type : {"l2", "logistic", "poisson"}
        For binary data, use "logistic".
    max_iter : int
        Max NOTEARS dual ascent steps.
    h_tol : float
        Acyclicity tolerance.
    rho_max : float
        Maximum penalty parameter.
    w_threshold : float
        Absolute threshold for pruning small edge weights.

    Returns
    -------
    cpdag : nx.DiGraph
        CPDAG converted from the learned NOTEARS DAG.
    score : float
        BIC score of the learned DAG on df.
    """
    X = df.to_numpy(dtype=float)
    W_est = notears_linear(
        X=X,
        lambda1=lambda1,
        loss_type=loss_type,
        max_iter=max_iter,
        h_tol=h_tol,
        rho_max=rho_max,
        w_threshold=w_threshold,
    )

    learned_dag = nx.DiGraph()
    learned_dag.add_nodes_from(df.columns)

    for i, src in enumerate(df.columns):
        for j, tgt in enumerate(df.columns):
            if i != j and W_est[i, j] != 0:
                learned_dag.add_edge(src, tgt, weight=float(W_est[i, j]))

    if not nx.is_directed_acyclic_graph(learned_dag):
        raise ValueError(
            "NOTEARS output is not a DAG after thresholding. "
            "Try increasing w_threshold or tightening h_tol."
        )

    scorer = BIC(df)
    score = scorer.score(learned_dag)

    return dag_to_cpdag(learned_dag), score