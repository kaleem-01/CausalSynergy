# clear_console_soft()

# from algorithms.benchmark import run_hc
from motifs.helpers import load_biff_data, load_motif_data
from metrics.graph import compare_cpdags, evaluate_colliders
from utils.graph import dag_to_cpdag
from pipeline.dataclasses import AlgorithmResults, Config, ExperimentResults
from scoring.bicSynergy import BICSynergy
from utils.console import clear_console_soft, print_banner
from utils.graph import create_graph
import networkx as nx

import matplotlib.pyplot as plt
from algorithms.benchmark import run_hc
import pandas as pd
from networkx.algorithms.dag import colliders
from pgmpy.estimators import BIC
from tqdm import tqdm
import numpy as np

import multiprocessing as mp
from algorithms.runtime import measure_call
import os
import pickle
from multiprocessing import Pool
from pathlib import Path
from datetime import datetime
from metrics.synergy import find_synergistic_triplets

from utils.data import *
from functools import partial



class Pipeline:
    """Pipeline for running structure learning experiments on Bayesian Networks (with checkpointing)."""

    def __init__(self, algorithm_func, configs=None):
        """Initialize the Pipeline with configurations and algorithm function."""
        self.algorithm_func = algorithm_func
        self.configs = configs
        self.data_reps = self.configs.replications if self.configs and hasattr(self.configs, "replications") else 1
        # self.samples = self.configs.samples if self.configs and hasattr(self.configs, "samples") else None

        # -----------------------------
        # Checkpoint root (NEW)
        # -----------------------------
        self._init_checkpoint_paths()

        if self.configs:
            try:
                self.df_by_id, self.metadata_by_id = read_data_files(self.configs.folder)
   
            except Exception as e:
                print(f"Data files read not read successfully. Experiments can still be run. Error: {e}")
        
        self.get_true_graph()
        self.max_possible_score()

    # -----------------------------
    # Checkpoint helpers (NEW)
    # -----------------------------
    def _init_checkpoint_paths(self):
        # Default: results/checkpoints/<algorithm>/<folder_name>/
        algo = getattr(self.configs, "algorithm", "algorithm") if self.configs else "algorithm"
        folder_name = Path(getattr(self.configs, "folder", "experiment")).name if self.configs else "experiment"
        base = Path("results") / "checkpoints" / algo / folder_name 
        base.mkdir(parents=True, exist_ok=True)
        self.checkpoint_root = base

    def _rep_dir(self, replication: int) -> Path:
        d = self.checkpoint_root / f"rep_{replication:03d}"
        (d / "per_dataset").mkdir(parents=True, exist_ok=True)
        return d

    def _dataset_ckpt_path(self, replication: int, dataset_id) -> Path:
        return self._rep_dir(replication) / "per_dataset" / f"{str(dataset_id)}.pkl"

    def _bundle_path(self, replication: int) -> Path:
        return self._rep_dir(replication) / "bundle.pkl"

    def _done_path(self, replication: int) -> Path:
        return self._rep_dir(replication) / "DONE"

    @staticmethod
    def _atomic_pickle_dump(obj, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)  # atomic on same filesystem

    @staticmethod
    def _pickle_load(path: Path):
        with open(path, "rb") as f:
            return pickle.load(f)

    # -----------------------------
    # Existing methods
    # -----------------------------
    def get_true_graph(self):
        """Load true dag from metadata for all datasets."""
        self.true_graphs = {idx: dag_to_cpdag(create_graph(meta_df=self.metadata_by_id[idx])) for idx in self.metadata_by_id.keys()}
        return self.true_graphs

    def _process_one(self, idx):
        """Worker-safe: process a single dataset id and return (idx, learned_graph, bic, runtime_dict)."""
        df = self.df_by_id[idx]

        if self.configs.algorithm.startswith("ea"):
            triplets = self.triplets_by_id.get(idx, [])
            
            (learned_graph, bic), rt = measure_call(self.algorithm_func,
                df,
                synergy_triplets=triplets,
                tracemalloc=True,   # set True if you want Python alloc peak (extra overhead)
                gc_collect=False,
            )
        else:
            (learned_graph, bic), rt = measure_call(
                self.algorithm_func,
                df,
                tracemalloc=True,   # set True if you want Python alloc peak (extra overhead)
                gc_collect=False,
            )

        return idx, learned_graph, bic, rt.to_dict()


    @staticmethod
    def _score(df, dag):
        """Compute the BIC score for a given DAG and dataset."""
        bic_scorer = BIC(df)
        return bic_scorer.score(dag)

    # -----------------------------
    # Run algorithm with resume (UPDATED)
    # -----------------------------
    def run_algorithm(self, replication: int = 0, resume: bool = True, n_jobs: int | None = None):
        """
        Run the algorithm on all datasets in parallel and store the learned DAGs and BIC scores.

        Checkpointing:
          - per dataset: results/checkpoints/<alg>/<folder>/rep_XXX/per_dataset/<id>.pkl
          - replication bundle: .../rep_XXX/bundle.pkl
          - DONE marker: .../rep_XXX/DONE

        Resume:
          - if DONE exists, loads bundle and skips computation
          - otherwise loads per-dataset ckpts and computes only missing datasets
        """
        done_path = self._done_path(replication)
        bundle_path = self._bundle_path(replication)

        # Fast path: already finished
        if resume and done_path.exists() and bundle_path.exists():
            bundle = self._pickle_load(bundle_path)
            self.learned_graphs = bundle["learned_graphs"]
            self.bic_scores = bundle["bic_scores"]
            self.runtime_by_id = bundle.get("runtime_by_id", {})  # NEW

            return self.learned_graphs

        learned_graphs = {}
        bic_scores = {}
        runtime_by_id = {}  # NEW

        if resume:
            for dataset_id in self.df_by_id.keys():
                ckpt = self._dataset_ckpt_path(replication, dataset_id)
                if ckpt.exists():
                    payload = self._pickle_load(ckpt)

                    # Backward compatible with old checkpoints that stored (dag, bic)
                    if isinstance(payload, tuple) and len(payload) == 2:
                        dag, bic = payload
                        rt = None
                    else:
                        dag, bic, rt = payload

                    learned_graphs[dataset_id] = dag
                    bic_scores[dataset_id] = bic
                    if rt is not None:
                        runtime_by_id[dataset_id] = rt

        pending = [dataset_id for dataset_id in self.df_by_id.keys() if dataset_id not in learned_graphs]

        # Get synergistic triplets if needed
        if self.configs.algorithm.startswith("ea"):
            # self.triplets_by_id = {}
            with open(f"data/triplets/{Path(self.configs.folder).name}.pkl", "rb") as f:    # renamed to _pid.pkl to for newer simulation results
                triplet_data = pickle.load(f)
            self.triplets_by_id = {result["File"]: result["Triplets"] for result in triplet_data}
            print(f"data/triplets/{Path(self.configs.folder).name}.pkl")
            print(triplet_data[0]["Metric"])

        if pending:
            workers = n_jobs or mp.cpu_count()
            if n_jobs == 1: # For algorithms that are implemented with parallelism internally (e.g., PC)
                for dataset_id in tqdm(pending, desc=f"Processing CSV files (rep={replication})"):
                    idx, dag, bic, rt = self._process_one(dataset_id)
                    learned_graphs[idx] = dag
                    bic_scores[idx] = bic
                    runtime_by_id[idx] = rt  # NEW

                    ckpt = self._dataset_ckpt_path(replication, idx)
                    self._atomic_pickle_dump((dag, bic, rt), ckpt)  # NEW: include rt
            else:
                with Pool(workers) as p:
                    for idx, dag, bic, rt in tqdm(
                        p.imap_unordered(self._process_one, pending),
                        total=len(pending),
                        desc=f"Processing CSV files (rep={replication})"
                    ):
                        learned_graphs[idx] = dag
                        bic_scores[idx] = bic
                        runtime_by_id[idx] = rt  # NEW

                        ckpt = self._dataset_ckpt_path(replication, idx)
                        self._atomic_pickle_dump((dag, bic, rt), ckpt)  # NEW: include rt

        self.learned_graphs = learned_graphs
        self.bic_scores = bic_scores
        self.runtime_by_id = runtime_by_id  # NEW

        self._atomic_pickle_dump(
            {"learned_graphs": learned_graphs, "bic_scores": bic_scores, "runtime_by_id": runtime_by_id},
            bundle_path
        )
        done_path.write_text(datetime.now().isoformat() + "\n", encoding="utf-8")

        return self.learned_graphs

    def evaluate_colliders(self, idx):
        """Evaluate the learned DAG for triplet (collider) discovery and compute related metrics."""
        return evaluate_colliders(self.metadata_by_id[idx], self.learned_graphs[idx], self.df_by_id[idx])

    @staticmethod
    def condition_number(X):
        """Compute the condition number of a matrix (for multicollinearity analysis)."""
        X = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
        _, s, _ = np.linalg.svd(X, full_matrices=False)
        cond_num = s.max() / s.min()
        return cond_num

    def get_metrics(self, save=None):
        """Compute evaluation metrics for all datasets and optionally save to CSV."""
        self.eval_metrics = {}

        # IMPORTANT FIX: your ids might not be 0..K-1
        for idx in range(len(self.true_graphs.keys())):
            self.eval_metrics[idx] = compare_cpdags(true_graph=self.true_graphs[idx], learned_graph=self.learned_graphs[idx])
            collider_metrics, _, _, _ = self.evaluate_colliders(idx)
            self.eval_metrics[idx].update(collider_metrics)
            self.eval_metrics[idx].update(self.df_by_id[idx].attrs)
            # self.eval_metrics[idx]["ID"] = idx
            self.eval_metrics[idx]["TrueBIC"] = self.max_score[idx]
            self.eval_metrics[idx]["BIC"] = self.bic_scores[idx]
            # self.eval_metrics[idx]["Nodes"] = len(self.df_by_id[idx].columns)
            # self.eval_metrics[idx]["Samples"] = len(self.df_by_id[idx])

                # NEW: runtime/memory (per dataset)
            rt = getattr(self, "runtime_by_id", {}).get(idx)
            if rt is None:
                self.eval_metrics[idx]["WallTime_s"] = np.nan
                self.eval_metrics[idx]["CPUTime_s"] = np.nan
                self.eval_metrics[idx]["RSS_before_MB"] = np.nan
                self.eval_metrics[idx]["RSS_after_MB"] = np.nan
                self.eval_metrics[idx]["RSS_delta_MB"] = np.nan
                self.eval_metrics[idx]["PeakRSS_MB"] = np.nan
                self.eval_metrics[idx]["PyAllocPeak_MB"] = np.nan
            else:
                self.eval_metrics[idx]["WallTime_s"] = rt.get("wall_time_s", np.nan)
                self.eval_metrics[idx]["CPUTime_s"] = rt.get("cpu_time_s", np.nan)

                rb = rt.get("rss_before_bytes", None)
                ra = rt.get("rss_after_bytes", None)
                rd = rt.get("rss_delta_bytes", None)
                pk = rt.get("ru_maxrss_bytes", None)
                tm = rt.get("tracemalloc_peak_bytes", None)

                self.eval_metrics[idx]["RSS_before_MB"] = (rb / (1024**2)) if rb is not None else np.nan
                self.eval_metrics[idx]["RSS_after_MB"]  = (ra / (1024**2)) if ra is not None else np.nan
                self.eval_metrics[idx]["RSS_delta_MB"]  = (rd / (1024**2)) if rd is not None else np.nan
                self.eval_metrics[idx]["PeakRSS_MB"]    = (pk / (1024**2)) if pk is not None else np.nan
                self.eval_metrics[idx]["PyAllocPeak_MB"]= (tm / (1024**2)) if tm is not None else np.nan


        # Better orientation: rows=datasets (makes downstream easier)
        self.eval_metrics = pd.DataFrame(self.eval_metrics).T

        if save:
            out = Path("results") / f"{Path(self.configs.folder).name}.csv"
            out.parent.mkdir(parents=True, exist_ok=True)
            self.eval_metrics.to_csv(out, index=False)

        return self.eval_metrics

    def plot_metrics(self, metric="SHD"):
        """Plot a metric (e.g., SHD) across all datasets."""
        if isinstance(self.eval_metrics, pd.DataFrame):
            y = self.eval_metrics[metric].to_numpy()
        else:
            y = [i[metric] for i in self.eval_metrics]
        plt.plot(y)

    def max_possible_score(self):
        """Compute the maximum possible BIC score for the true DAGs."""
        self.max_score = {}
        for idx in range(len(self.true_graphs.keys())):
            df = self.df_by_id[idx]
            # print(f"Processing dataset {idx} for max BIC score.")
            mapping = {n: str(n) for n in self.true_graphs[idx].nodes()}
            G = nx.relabel_nodes(self.true_graphs[idx], mapping)

            self.max_score[idx] = self._score(df, G)

            if self.configs.algorithm == "genetic-synergy":
                bic_scorer = BICSynergy(df)
                self.max_score[idx] = bic_scorer.score(G)

        return self.max_score

    def create_results(self):
        results = AlgorithmResults(
            folder=self.configs.folder,
            algorithm=self.configs.algorithm,
            replications=self.configs.replications,
            metadata_by_id=self.metadata_by_id,
            df_by_id=self.df_by_id,
            eval_metrics=self.get_metrics(),
            true_graphs=self.true_graphs,
            learned_graphs=self.learned_graphs,
            bic_scores=self.bic_scores,
            max_score=self.max_score
        )
        return results

    # -----------------------------
    # Run replications with resume (UPDATED)
    # -----------------------------
    def run(self, resume: bool = True, n_jobs: int | None = None, save_each_rep: bool = False):
        """
        Run the pipeline for all replications, collect results.

        If resume=True:
          - skips replications that have DONE marker
          - resumes partial replications using per-dataset ckpts

        If save_each_rep=True:
          - saves AlgorithmResults.pkl inside each rep folder (optional but convenient)
        """
        self.all_results = []

        for r in range(self.configs.replications):
            # run/restore learned graphs for this replication
            self.run_algorithm(replication=r, resume=resume, n_jobs=n_jobs)

            # build AlgorithmResults for this replication
            rep_results = self.create_results()
            self.all_results.append(rep_results)

            if save_each_rep:
                rep_dir = self._rep_dir(r)
                self._atomic_pickle_dump(rep_results, rep_dir / "AlgorithmResults.pkl")
                # also save metrics for quick inspection
                rep_results.eval_metrics.to_csv(rep_dir / "eval_metrics.csv", index=False)

        return self.all_results

    # -----------------------------
    # Save aggregated results (UPDATED)
    # -----------------------------
    def save(self):
        """
        Save the list of AlgorithmResults (one per replication) in the original location:
          results/<algorithm>/<folder_name>.pkl
        """
        save_path = Path("results") / self.configs.algorithm
        save_path.mkdir(parents=True, exist_ok=True)

        save_file = Path("results") / self.configs.algorithm / f"{Path(self.configs.folder).name}.pkl" 
        save_file.parent.mkdir(parents=True, exist_ok=True)
        with save_file.open("wb") as f:
            pickle.dump(self.all_results, f, protocol=pickle.HIGHEST_PROTOCOL)

        return save_file

    # -----------------------------
    # Experiments
    # -----------------------------
    def experiment_process(self, true_cpdag, df, name):
        learned_cpdag, bic = self.algorithm_func(df)
        metrics = compare_cpdags(true_cpdag=true_cpdag, learned_cpdag=learned_cpdag)
        metrics["BIC"] = bic
        metrics["TrueBIC"] = self._score(df, true_cpdag)

        results = ExperimentResults(
            dataset=name,
            true_cpdag=true_cpdag,
            learned_cpdag=learned_cpdag,
            metrics=metrics,
            samples=self.samples if self.samples is not None else len(df)
        )
        return results

    def experiment1_process(self, name):
        true_cpdag, df = load_motif_data(name)
        results = self.experiment_process(true_cpdag=true_cpdag, df=df, name=name)
        return results

    def experiment1(self):
        datasets = ["Chain","ChainSynergy","ChainD","Fork","ForkSynergy","Collider","ForkD","ForkSynergyD",
                    "ColliderD","LargeDiscrete","Multi-ParentThree","Multi-ParentFour","Multi-ParentFive","MediatorSynergy"]

        with Pool(mp.cpu_count()) as p:
            results = list(tqdm(
                p.imap(self.experiment1_process, datasets),
                total=len(datasets),
                desc="Experiment 1"
            ))

        results = {result.dataset: result for result in results}
        return results

    def experiment2_process(self, dataset):
        true_cpdag, df = load_biff_data(folder='data/bnlearn/', dataset=dataset)
        df = df.sample(n=self.samples, random_state=42) if self.samples is not None else df
        print(f"Dataset: {dataset}, Samples used: {len(df)}")
        name = dataset.capitalize()
        results = self.experiment_process(true_cpdag=true_cpdag, df=df, name=name)
        return results

    def experiment2(self, num=None):
        datasets = os.listdir('data/bnlearn/')
        datasets = [f for f in datasets if f.endswith('.bif')]
        if num is not None:
            datasets = [f[:-4] for f in datasets][:num]
        else:
            datasets = [f[:-4] for f in datasets]

        with Pool(mp.cpu_count()) as p:
            results = list(tqdm(
                p.imap(self.experiment2_process, datasets),
                total=len(datasets),
                desc="Experiment 2"
            ))

        results = {result.dataset: result for result in results}
        return results


# -----------------------------
# Main 
# -----------------------------
if __name__ == "__main__":

    from algorithms.benchmark import run_hc

    def main():
        configs = Config(
            folder='data/syntheticBinary/binaryLowNoise',
            algorithm='hc',
            replications=1
        )

        # You had Pipeline(run_hc, configs) but run_hc isn't imported in this file snippet.
        # Replace `run_hc` with your actual algorithm function.
        pipeline = Pipeline(algorithm_func=run_hc, configs=configs)

        # This will resume if checkpoints exist:
        pipeline.run(resume=True, n_jobs=mp.cpu_count() // 2, save_each_rep=True)

        # Aggregate save (same as your original intent):
        pipeline.save()

        return pipeline

    main()
