from dataclasses import dataclass


@dataclass
class AlgorithmResults:
    """Class to store results of the algorithm runs."""
    folder: str
    algorithm: str
    replications: int
    experiment1_results: dict = None
    experiment2_results: dict = None
    true_graphs: dict = None
    max_score: dict = None
    learned_graphs: dict = None
    bic_scores: dict = None
    eval_metrics: dict = None
    metadata_by_id: dict = None
    df_by_id: dict = None
    ea_configurations: dict = None


@dataclass
class ExperimentResults:
    """Class to store results of a single experiment."""
    dataset: str
    true_dag: any
    learned_dag: any
    metrics: dict
    samples: int


@dataclass
class Config:
    """Configuration class for pipeline settings."""
    folder: str = None
    algorithm: str = None
    replications: int = 1
    samples: int = 2_000