import os
import pickle
import numpy as np
from joblib import Parallel, delayed
import jointpmf.jointpmf as jp

import logging


import os
os.environ["PYTHONWARNINGS"] = "ignore"  # warnings, not logging


# -------------------------
# Pickle helpers
# -------------------------
def save_pickle(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

def load_all_precomputes(precomp_dir: str):
    """Load SRVs, one-source, two-source together."""
    srvs = load_pickle(os.path.join(precomp_dir, "precomputed_srvs.pkl"))
    one  = load_pickle(os.path.join(precomp_dir, "precomputed_one_source.pkl"))
    two  = load_pickle(os.path.join(precomp_dir, "precomputed_two_source.pkl"))
    return srvs, one, two

# -------------------------
# Precompute banks
# -------------------------
import os
import time
import pickle
import numpy as np
from joblib import Parallel, delayed
import inspect

import jointpmf.jointpmf as jp  # adjust import if you use a different alias/location


def save_pickle(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)  # atomic on most OS/filesystems


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def precompute_srv_bank(
    *,
    numvalues: int,
    max_evals_srv: int,
    n_srv_bank: int,
    n_jobs: int,
    save_path: str,
    checkpoint_every: int = 1000,
    resume: bool = True,
):
    """
    Precompute a bank of SRVs and checkpoint to disk every `checkpoint_every` evals.

    - If `resume=True` and `save_path` exists, it loads and continues from there.
    - After each checkpoint save, it immediately reloads from disk to verify/ensure
      the saved state is what the process continues from (so you can safely resume).

    Returns dict with:
      {
        "meta": {...},
        "srvs": [ {"wms_norm": float, "cpd": np.ndarray}, ... ]
      }
    """

    def _gen_one(_):
        bn = jp.BayesianNetwork()
        bn.append_independent_variable("uniform", numvalues)
        bn.append_independent_variable("uniform", numvalues)

        bn.append_synergistic_variable((0, 1), numvalues=numvalues, max_evals=max_evals_srv)

        wms_norm = float(bn.wms([0, 1], [2]) / np.log2(numvalues))
        cpd = bn.conditional_probabilities([2], [0, 1])  # P(Z|X,Y)
        return {"wms_norm": wms_norm, "cpd": cpd}

    # ---- resume/load or initialize ----
    if resume and os.path.exists(save_path):
        bank = load_pickle(save_path)
        if not isinstance(bank, dict) or "srvs" not in bank:
            raise ValueError(f"Existing pickle at {save_path} is not a valid srv bank.")
        srvs = list(bank.get("srvs", []))
        meta = dict(bank.get("meta", {}))
    else:
        srvs = []
        meta = {}

    # Keep meta consistent with current run config
    meta.update(
        {
            "numvalues": int(numvalues),
            "max_evals_srv": int(max_evals_srv),
            "n_srv_bank": int(n_srv_bank),
            "checkpoint_every": int(checkpoint_every),
            "save_path": str(save_path),
            "created_at_unix": meta.get("created_at_unix", int(time.time())),
            "last_updated_unix": int(time.time()),
            "complete": False,
        }
    )

    # If the loaded bank already satisfies the target, just mark complete and return
    if len(srvs) >= n_srv_bank:
        srvs = srvs[:n_srv_bank]
        meta["complete"] = True
        meta["last_updated_unix"] = int(time.time())
        bank = {"meta": meta, "srvs": srvs}
        save_pickle(bank, save_path)
        return bank

    # ---- generate remaining in chunks ----
    start_idx = len(srvs)
    remaining = n_srv_bank - start_idx
    
    print(f"Precomputing SRV bank: starting from {start_idx}, need {remaining} more...")

    while remaining > 0:
        batch_n = min(checkpoint_every, remaining)

        new_items = Parallel(n_jobs=n_jobs)(
            delayed(_gen_one)(i) for i in range(batch_n)
        )

        srvs.extend(new_items)

        meta["last_updated_unix"] = int(time.time())
        meta["generated"] = int(len(srvs))
        meta["remaining"] = int(max(0, n_srv_bank - len(srvs)))
        meta["complete"] = bool(len(srvs) >= n_srv_bank)

        bank = {"meta": meta, "srvs": srvs}
        print(f"  Checkpoint: generated {len(srvs)}/{n_srv_bank} SRVs, saving to {save_path}...")
        save_pickle(bank, save_path)

        # "saved and loaded again" so the continuation always uses on-disk state
        bank = load_pickle(save_path)
        srvs = list(bank["srvs"])
        meta = dict(bank["meta"])

        remaining = n_srv_bank - len(srvs)

    # Final tidy
    srvs = srvs[:n_srv_bank]
    meta["generated"] = int(len(srvs))
    meta["remaining"] = 0
    meta["complete"] = True
    meta["last_updated_unix"] = int(time.time())

    bank = {"meta": meta, "srvs": srvs}
    save_pickle(bank, save_path)
    return bank


def precompute_one_source_bank(
    *,
    numvalues: int,
    target_mi_values: list[float],
    n_one_per_mi: int,
    n_jobs: int,
):
    """
    Returns dict with:
      {
        "meta": {...},
        "by_target_mi": { float(tmi): [cpd_ndarray, ...], ... }
      }
    """
    def _gen_one(target_mi: float, _):
        bn = jp.BayesianNetwork()
        bn.append_independent_variable("uniform", numvalues)
        bn.append_dependent_variable([0], numvalues=numvalues, target_mi=float(target_mi))
        return bn.conditional_probabilities([1], [0])  # P(Y|X)

    by = {}
    for tmi in target_mi_values:
        cps = Parallel(n_jobs=n_jobs)(delayed(_gen_one)(float(tmi), i) for i in range(n_one_per_mi))
        by[float(tmi)] = cps

    return {
        "meta": {
            "numvalues": int(numvalues),
            "target_mi_values": [float(x) for x in target_mi_values],
            "n_one_per_mi": int(n_one_per_mi),
        },
        "by_target_mi": by,
    }

def precompute_two_source_bank(
    *,
    numvalues: int,
    target_mi_values: list[float],
    n_two_per_mi: int,
    n_jobs: int,
):
    """
    Returns dict with:
      {
        "meta": {...},
        "by_target_mi": { float(tmi): [cpd_ndarray, ...], ... }
      }
    """
    def _gen_two(target_mi: float, _):
        bn = jp.BayesianNetwork()
        bn.append_independent_variable("uniform", numvalues)
        bn.append_independent_variable("uniform", numvalues)
        bn.append_dependent_variable([0, 1], numvalues=numvalues, target_mi=float(target_mi))
        return bn.conditional_probabilities([2], [0, 1])  # P(Z|X,Y)

    by = {}
    for tmi in target_mi_values:
        cps = Parallel(n_jobs=n_jobs)(delayed(_gen_two)(float(tmi), i) for i in range(n_two_per_mi))
        by[float(tmi)] = cps

    return {
        "meta": {
            "numvalues": int(numvalues),
            "target_mi_values": [float(x) for x in target_mi_values],
            "n_two_per_mi": int(n_two_per_mi),
        },
        "by_target_mi": by,
    }

# -------------------------
# Selection at generation-time
# -------------------------
def select_cpds_for_point(
    srvs_bank: dict,
    one_bank: dict,
    two_bank: dict,
    *,
    syn_cutoff: float,
    target_mi: float,
    n_one_source: int,
    n_two_source: int,
    rng: np.random.Generator | None = None,
):
    rng = rng or np.random.default_rng()

    # SRVs by cutoff
    srvs_pool = [
        d["cpd"]
        for d in srvs_bank["srvs"]
        if float(syn_cutoff) < float(d["wms_norm"]) < float(syn_cutoff) + 0.1
    ]
    precomputed_srvs = list(srvs_pool)

    # Choose closest MI key available
    keys = np.array(sorted(one_bank["by_target_mi"].keys()), dtype=np.float64)
    idx = int(np.argmin(np.abs(keys - float(target_mi))))
    used_tmi = float(keys[idx])

    one_pool = one_bank["by_target_mi"][used_tmi]
    two_pool = two_bank["by_target_mi"][used_tmi]

    def _sample(pool, k):
        if len(pool) == 0:
            return []
        replace = len(pool) < k
        inds = rng.choice(len(pool), size=k, replace=replace)
        return [pool[i] for i in inds]

    precomputed_one = _sample(one_pool, n_one_source)
    precomputed_two = _sample(two_pool, n_two_source)

    return precomputed_one, precomputed_two, precomputed_srvs, used_tmi

if __name__ == "__main__":
    OUT_FOLDER = os.path.join("data")

    TARGET_MI_GRID = None  # or e.g., [0.1, 0.3, 0.5]
    NUMVALUES = 3
    MAX_EVALS_SRV = 100
    n_jobs = -1
    
    target_mi_grid = TARGET_MI_GRID
    numvalues = NUMVALUES
    max_evals_srv = MAX_EVALS_SRV

    PRECOMP_DIR = os.path.join(OUT_FOLDER, "_precomputed_cpds")
    os.makedirs(PRECOMP_DIR, exist_ok=True)

    # Same default MI grid you had
    if target_mi_grid is None:
        multipliers = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
        target_mi_values = [np.log2(numvalues) * m for m in multipliers]
    else:
        target_mi_values = list(target_mi_grid)

    N_ONE_PER_MI = 50   # per MI bucket
    N_TWO_PER_MI = 50   # per MI bucket

    print(f"Saving precomputes to: {PRECOMP_DIR}")

    # PRECOMP_DIR = "precomputed"
    save_path = os.path.join(PRECOMP_DIR, "precomputed_srvs.pkl")

    import logging
    logging.getLogger("pySOT.strategy.srbf_strategy").setLevel(logging.WARNING)
    import warnings
    warnings.filterwarnings("ignore")

    srvs_bank = precompute_srv_bank(
        numvalues=3,
        max_evals_srv=100,
        n_srv_bank=5_000,
        n_jobs=15,
        save_path=save_path,
        checkpoint_every=50,
        resume=True,
    )


    one_bank = precompute_one_source_bank(
        numvalues=numvalues,
        target_mi_values=target_mi_values,
        n_one_per_mi=N_ONE_PER_MI,
        n_jobs=n_jobs,
    )
    
    save_pickle(one_bank, os.path.join(PRECOMP_DIR, "precomputed_one_source.pkl"))
    print("Saved precomputed_one_source.pkl")

    two_bank = precompute_two_source_bank(
        numvalues=numvalues,
        target_mi_values=target_mi_values,
        n_two_per_mi=N_TWO_PER_MI,
        n_jobs=n_jobs,
    )
    save_pickle(two_bank, os.path.join(PRECOMP_DIR, "precomputed_two_source.pkl"))
    print("Saved precomputed_two_source.pkl")

    print("Done.")
    # Optional:
    # run_grid()
