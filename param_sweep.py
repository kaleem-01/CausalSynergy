# ===============================
# Parameter Sweep (true resume + per-iteration save + tqdm ETA)
# ===============================
from __future__ import annotations
import time
from itertools import product
from pathlib import Path
from typing import Sequence
import pandas as pd
import numpy as np
from algorithms.ea import GeneticBNSearchMatrix as GeneticBNSearch
import networkx as nx
from tqdm.auto import tqdm

import warnings


import logging
logging.getLogger('pgmpy').setLevel(logging.WARNING)

# logging.getLogger('causallearn').setLevel(logging.WARNING)

import warnings
warnings.filterwarnings("ignore")


# ---- Configurable grid ----
PARAM_GRID = {
    "crossover_proportion": [0.2, 0.3, 0.4, 0.5, 0.6], 
    "crossover_edges": [1, 2, 3, 4, 5],
    "mutation_prob": [0.6, 0.7, 0.8, 0.9, 1.0],    
    "elitism_proportion": [0.1, 0.2, 0.3, 0.4, 0.5],
    # "max_parents": [None, 2, 3],
    # "connect_disconnected": [True, False],
    # "informed_ratio": [0.6, 0.8],
}

N_REPEATS = 5
POPULATION_SIZE = 20
GENERATIONS = 150
N_JOBS = 1
OUTPUT_DIR = Path("results/param_sweeps")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Columns that uniquely identify a trial
TRIAL_KEY_FIELDS = [
    "crossover_proportion",
    "crossover_edges",
    "mutation_prob",
    "elitism_proportion",
]

def _all_param_dicts(grid: dict) -> list[dict]:
    keys = list(grid.keys())
    vals = [grid[k] for k in keys]
    out = []
    for combo in product(*vals):
        out.append({k: v for k, v in zip(keys, combo)})
    return out

def _is_nan(x) -> bool:
    try:
        return pd.isna(x)
    except Exception:
        return False

def _stable_val_str(v, colname: str | None = None) -> str:
    """
    Normalize values to strings so CSV reloads match in resume logic.
    - Treat NaN as "None" (esp. for columns that may be None, like max_parents)
    - Normalize floats to a canonical string
    """
    if _is_nan(v):
        return "None"  # critical fix: CSV None -> NaN, map back to "None"
    if v is None:
        return "None"
    if isinstance(v, float):
        return f"{v:.12g}"
    return str(v)

def _trial_key_from_rowlike(row_or_dict) -> tuple[str, ...]:
    return tuple(_stable_val_str(row_or_dict.get(k, np.nan), k) for k in TRIAL_KEY_FIELDS)

def _load_completed_keys(csv_path: Path) -> set[tuple[str, ...]]:
    if not csv_path.exists():
        return set()
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return set()
    # ensure all key columns are present
    for col in TRIAL_KEY_FIELDS:
        if col not in df.columns:
            df[col] = np.nan
    keys: set[tuple[str, ...]] = set()
    for _, row in df.iterrows():
        keys.add(_trial_key_from_rowlike(row))
    return keys

def _append_row(csv_path: Path, row: dict):
    # Append one row immediately (header only if new file)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_row = pd.DataFrame([row])
    exists = csv_path.exists()
    df_row.to_csv(csv_path, mode="a", header=not exists, index=False)

def _run_single_sweep(
    df: pd.DataFrame,
    base_kwargs: dict,
    param_cfg: dict,
    repeat_idx: int,
    seed_base: int = 12345,
) -> dict:
    """Run a single GA with given params & repeat index, return a flat result row."""
    seed = seed_base + (hash(tuple(sorted(param_cfg.items(), key=lambda kv: kv[0]))) % 10_000) + int(repeat_idx)
    t0 = time.perf_counter()
    try:
        ga = GeneticBNSearch(
            data=df,
            population_size=base_kwargs.get("population_size", POPULATION_SIZE),
            generations=base_kwargs.get("generations", GENERATIONS),
            crossover_proportion=param_cfg["crossover_proportion"],
            crossover_edges=param_cfg["crossover_edges"],
            mutation_prob=param_cfg["mutation_prob"],
            elitism_proportion=param_cfg["elitism_proportion"],
            score_fn=base_kwargs.get("score_fn", "bic"),
            random_state=seed,
            hillclimb_seed=base_kwargs.get("hillclimb_seed", False),
        )
        best_g, best_score = ga.run()
        best_score = best_score[-1]
        dt = time.perf_counter() - t0

        
        row = {
            **param_cfg,
            "repeat": int(repeat_idx),
            "seed": int(seed),
            "best_score": float(best_score),
            "elapsed_sec": float(dt),
            "error": "",
        }
    except Exception as e:
        dt = time.perf_counter() - t0
        row = {
            **param_cfg,
            "repeat": int(repeat_idx),
            "seed": int(seed),
            "best_score": np.nan,
            "n_nodes": np.nan,
            "n_edges": np.nan,
            "in_degree_max": np.nan,
            "out_degree_max": np.nan,
            "elapsed_sec": float(dt),
            "error": str(e),
        }
    return row

def sweep_parameters(
    df: pd.DataFrame,
    param_grid: dict = PARAM_GRID,
    n_repeats: int = N_REPEATS,
    population_size: int = POPULATION_SIZE,
    generations: int = GENERATIONS,
    score_fn: str = "bic",
    n_jobs: int = N_JOBS,
    outdir: Path = OUTPUT_DIR,
    tag: str | None = None,
) -> pd.DataFrame:
    """
    Run a grid search over GA hyperparameters with repeats.
    - True resume: stable filename, robust key normalization (NaN<->None).
    - Durable: each trial is appended immediately.
    - Progress: tqdm over pending trials with ETA from completed rows.
    """
    tag_part = (f"_{tag}" if tag else "")
    csv_trials = outdir / f"sweep_trials{tag_part}.csv"   # stable filename for resume
    csv_agg = outdir / f"sweep_agg{tag_part}.csv"

    param_dicts = _all_param_dicts(param_grid)
    base_kwargs = dict(
        population_size=population_size,
        generations=generations,
        score_fn=score_fn,
        n_jobs=n_jobs,
    )

    # Build full job list
    jobs: list[tuple[dict, int]] = []
    for p in param_dicts:
        for r in range(n_repeats):
            jobs.append((p, r))

    # Figure out which trials are already done
    completed_keys = _load_completed_keys(csv_trials)

    def _job_key(p: dict, r: int) -> tuple[str, ...]:
        row = {**p, "repeat": r}
        return _trial_key_from_rowlike(row)

    pending = [(p, r) for (p, r) in jobs if _job_key(p, r) not in completed_keys]

    # For ETA: average elapsed of completed rows, if any
    if csv_trials.exists():
        try:
            hist = pd.read_csv(csv_trials)
            mean_elapsed = float(hist["elapsed_sec"].dropna().mean()) if "elapsed_sec" in hist.columns else np.nan
        except Exception:
            mean_elapsed = np.nan
    else:
        mean_elapsed = np.nan

    if completed_keys:
        print(f"[resume] Found {len(completed_keys)} completed trials in {csv_trials.name}.")
    print(f"[plan] Total trials: {len(jobs)} | Pending: {len(pending)} | Skipped: {len(jobs) - len(pending)}")

    # tqdm progress bar
    bar_desc = "Param sweep"
    if not np.isnan(mean_elapsed):
        est_total_sec = mean_elapsed * max(len(pending), 0)
        bar_desc = f"{bar_desc} (~{int(est_total_sec)}s remaining est.)"
    pbar = tqdm(total=len(pending), desc=bar_desc, leave=True)

    # Run pending trials in parallel using multiprocessing.Pool.
    # Each worker runs _run_single_sweep; the main process appends rows to CSV as results arrive.
    if not pending:
        pbar.close()
    else:
        # Use up to (cpu_count - 1) workers but not more than number of pending jobs
        workers = mp.cpu_count() - 1

        def _handle_result(row):
            # callback executed in parent process when a worker returns a result
            _append_row(csv_trials, row)
            pbar.set_postfix(
                score=(None if pd.isna(row.get("best_score")) else f"{row['best_score']:.3f}"),
                secs=f"{row.get('elapsed_sec', np.nan):.2f}",
                err=("yes" if row.get("error") else "no"),
            )
            pbar.update(1)

        if workers == 1:
            # Fallback to sequential execution (simple and reliable)
            for p, r in pending:
                row = _run_single_sweep(df, base_kwargs, p, r)
                _append_row(csv_trials, row)
                pbar.set_postfix(
                    score=(None if pd.isna(row.get("best_score")) else f"{row['best_score']:.3f}"),
                    secs=f"{row.get('elapsed_sec', np.nan):.2f}",
                    err=("yes" if row.get("error") else "no"),
                )
                pbar.update(1)
            pbar.close()
        else:
            pool = mp.Pool(processes=workers)
            async_results = []
            try:
                for p, r in pending:
                    # schedule task; callback will append & update progress
                    ar = pool.apply_async(_run_single_sweep, args=(df, base_kwargs, p, r), callback=_handle_result)
                    async_results.append(ar)
                pool.close()
                # wait for all tasks to finish (callbacks run as results arrive)
                for ar in async_results:
                    ar.wait()
            finally:
                pool.join()
                pbar.close()

    # Load all trials (old + new) to return and to compute aggregate
    all_trials = pd.read_csv(csv_trials) if csv_trials.exists() else pd.DataFrame(columns=TRIAL_KEY_FIELDS + ["best_score"])
    if not all_trials.empty and "best_score" in all_trials.columns:
        agg = (
            all_trials.groupby(
                ["crossover_proportion", "crossover_edges", "mutation_prob",
                 "elitism_proportion"],
                dropna=False
            )["best_score"]
            .agg(["mean", "std", "count"])
            .reset_index()
            .sort_values("mean", ascending=False)
        )
        agg.to_csv(csv_agg, index=False)
        print("\nTop 10 aggregated configs (by mean best_score):")
        print(agg.head(10).to_string(index=False))
    else:
        print("\nNo completed trials to aggregate yet.")

    return all_trials

# -------------------------------
# Example CLI entry point
# -------------------------------
if __name__ == "__main__":
    import multiprocessing as mp

    GENERATIONS = 150
    POPULATION_SIZE = 20
    N_REPEATS = 5
    N_JOBS = 1

    
    df = pd.read_csv("data/datasets/jpmf_data/Data_Graph_160.csv")
    print("Data shape:", df.shape)

    _ = sweep_parameters(
        df=df,
        n_repeats=N_REPEATS,
        population_size=POPULATION_SIZE,
        generations=GENERATIONS,
        score_fn="bic",
        tag="er",
    )

    
    df = pd.read_csv("data/datasets/jpmf_data/Data_Graph_600.csv")
    print("Data shape:", df.shape)

    _ = sweep_parameters(
        df=df,
        n_repeats=N_REPEATS,
        population_size=POPULATION_SIZE,
        generations=GENERATIONS,
        score_fn="bic",
        tag="ws",
    )


    df = pd.read_csv("data/datasets/jpmf_data/Data_Graph_350.csv")
    print("Data shape:", df.shape)

    _ = sweep_parameters(
        df=df,
        n_repeats=N_REPEATS,
        population_size=POPULATION_SIZE,
        generations=GENERATIONS,
        score_fn="bic",
        tag="ba",
    )
