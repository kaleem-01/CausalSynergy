import numpy as np
import networkx as nx
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import chisq, gsq
from pgmpy.estimators import BIC, GES, HillClimbSearch
from algorithms.ea import GeneticBNSearchMatrix as GeneticBNSearch
from utils.graph import dag_to_cpdag

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
