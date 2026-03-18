import numpy as np
from metrics.information import *
from itertools import combinations
from idtxl.bivariate_pid import BivariatePID
from idtxl.data import Data

import pandas as pd
from typing import List, Sequence, Tuple, Union, Callable, Any
import pandas as pd

try:
    from joblib import Parallel, delayed
except ImportError as e:
    raise ImportError(
        "joblib is required for parallel scoring. Install with: pip install joblib"
    ) from e



def redundancy_Imin(s1, s2, t):
    """
    Redundant information between sources S1, S2 about target T
    using the I_min redundancy function.
    """
    n = len(t)

    # Global marginals of the sources
    p_s1 = np.bincount(s1, minlength=s1.max() + 1) / n
    p_s2 = np.bincount(s2, minlength=s2.max() + 1) / n

    red = 0.0
    for y_state, n_y in enumerate(np.bincount(t)):
        if n_y == 0:
            continue

        mask = (t == y_state)
        p_s1_y = np.bincount(s1[mask], minlength=p_s1.size) / n_y
        p_s2_y = np.bincount(s2[mask], minlength=p_s2.size) / n_y

        # Local MI(S;T=y)  for each source
        idx1 = p_s1_y > 0
        mi1 = (p_s1_y[idx1] * np.log2(p_s1_y[idx1] / p_s1[idx1])).sum()

        idx2 = p_s2_y > 0
        mi2 = (p_s2_y[idx2] * np.log2(p_s2_y[idx2] / p_s2[idx2])).sum()

        red += (n_y / n) * min(mi1, mi2)

    return red


def pid_synergy_Imin(source1, source2, target):
    """
    Williams-&-Beer PID synergy (non-negative, in bits).

    Parameters
    ----------
    source1, source2, target : 1-D integer NumPy arrays of equal length.

    Returns
    -------
    float
        Synergy ≥ 0 (==0 when the sources contribute no information
        beyond their uniques and redundancy).
    """
    red = redundancy_Imin(source1, source2, target)
    I12_t = mutual_information_joint(source1, source2, target)
    I1_t  = mutual_information(source1, target)
    I2_t  = mutual_information(source2, target)
    return I12_t - I1_t - I2_t + red



def _interaction_information(x, y, z) -> float:
    
    def entropy(x: Iterable[int] | np.ndarray | pd.Series) -> float:
        x_arr = np.asarray(x)
        if x_arr.dtype.kind not in {"i", "u"}:
            x_arr, _ = pd.factorize(x_arr, sort=False)
        probs = np.bincount(x_arr) / x_arr.size
        probs = probs[probs > 0]
        return float(-(probs * np.log2(probs)).sum())


    def joint_entropy(*vars_: Iterable[int]) -> float:
        mi = pd.MultiIndex.from_arrays(vars_)
        codes, _ = pd.factorize(mi, sort=False)
        return entropy(codes)


    Hx, Hy, Hz = entropy(x), entropy(y), entropy(z)
    return (
        Hx
        + Hy
        + Hz
        - joint_entropy(x, y)
        - joint_entropy(x, z)
        - joint_entropy(y, z)
        + joint_entropy(x, y, z)
    )


def find_synergistic_triplets(
    df: pd.DataFrame,
    func: Callable[[Any, Any, Any], float] = _interaction_information,
    inflection_point: bool = True,
    all_results: bool = False,
    all_triplets: bool = False,
    n_jobs: int = 20,          # -1 = use all cores
    prefer: str = "processes", # "processes" or "threads"
    batch_size: int = 256,     # tune for overhead vs throughput
) -> Union[
    List[Tuple[str, str, str]],
    List[Tuple[Tuple[str, str, str], float]]
]:
    """
    Parallel version: scores all (a,b,c) triplets concurrently.

    Returns
    -------
    - if all_results: List[((a,b,c), ii)]
    - elif all_triplets: List[(a,b,c)] (sorted by ii ascending)
    - elif inflection_point: List[(a,b,c)] up to inflection
    - else: List[(a,b,c)] of top_n (currently == len(df.columns), matching your code)
    """
    cols = df.columns.tolist()

    # Compute all possible
    triplets = list(combinations(cols, 3))

    def _score_triplet(a: str, b: str, c: str):
        ii = func(df[a].values, df[b].values, df[c].values)
        return (a, b, c), ii

    # Parallel scoring
    results: List[Tuple[Tuple[str, str, str], float]] = Parallel(
        n_jobs=n_jobs,
        prefer=prefer,
        batch_size=batch_size,
    )(delayed(_score_triplet)(a, b, c) for a, b, c in triplets)

    # sort by score (most synergistic first if "negative II" => ascending)
    results.sort(key=lambda t: t[1], reverse=False)

    if all_results:
        candidates = results

    if all_triplets:
        candidates =  [triplet for triplet, _ in results]

    top_n = len(df.columns)  # Keep top n triplets if inflection point is beyond that
    
    if inflection_point:
        threshold_index = find_inflection_point([ii for _, ii in results])
        candidates = [triplet for triplet, _ in results[:threshold_index]]
        if len(candidates) > len(df.columns):
            # print(f"Warning: inflection point at {threshold_index} triplets, which exceeds number of columns ({len(df.columns)}).")
            candidates = [triplet for triplet, _ in results[:top_n]]

    return candidates


def classify_collider_by_pid(
    df: pd.DataFrame,
    triplet: Sequence[str | int],
    *,
    alph_s1: int | None = None,
    alph_s2: int | None = None,
    alph_t: int | None = None,
    alpha: float = 0.05,
    n_perm: int = 1_000,
):
    """
    Classify a fixed collider candidate (source1, source2 -> target)
    without a manually chosen effect-size threshold.

    Decision rule:
        synergistic <=> syn_s1_s2 is significant under surrogate testing
    """
    if len(triplet) != 3:
        raise ValueError(f"triplet must have length 3, got {len(triplet)}")

    source1, source2, target = triplet

    if not all(isinstance(x, str) for x in (source1, source2, target)):
        source1, source2, target = map(str, triplet)

    x = df[source1].to_numpy()
    y = df[source2].to_numpy()
    z = df[target].to_numpy()

    if alph_s1 is None:
        alph_s1 = int(df[source1].nunique())
    if alph_s2 is None:
        alph_s2 = int(df[source2].nunique())
    if alph_t is None:
        alph_t = int(df[target].nunique())

    data = Data(np.vstack((x, y, z)), "ps", normalise=False)

    settings = {
        "pid_estimator": "TartuPID",
        "alph_s1": alph_s1,
        "alph_s2": alph_s2,
        "alph_t": alph_t,
        # "max_unsuc_swaps_row_parm": 60,
        # "num_reps": 100,
        # "max_iters": 100_000,
        "verbose": False,
        "lags_pid": [0, 0],
        "alpha": alpha,
        "n_perm": n_perm,
        "permute_in_time": True,
    }

    pid = BivariatePID()
    pid._initialise(settings, data, target=2, sources=[0, 1])
    
    pid = pid.analyse_single_target(
        settings=settings,
        data=data,
        target=2,
        sources=[0, 1],
    ).get_single_target(2)
 
    syn = float(pid["syn_s1_s2"])
    shd = float(pid["shd_s1_s2"])
    unq_s1 = float(pid["unq_s1"])
    unq_s2 = float(pid["unq_s2"])

    total_info = syn + shd + unq_s1 + unq_s2
    synergy_fraction = syn / total_info if total_info > 0 else 0.0

    label = "synergistic" if synergy_fraction > 0.5 else "pairwise"

    return {
        "triplet": (source1, source2, target),
        "label": label,
        "syn_s1_s2": syn,
        "shd_s1_s2": shd,
        "unq_s1": unq_s1,
        "unq_s2": unq_s2,
        "total_info": total_info,
        "synergy_fraction": synergy_fraction,
    }


def categorize_colliders(extra_colliders, df):
    # This function would implement the logic to categorize extra colliders into "synergistic" vs "pairwise"
    extra_syn = []
    extra_pairwise = []
    sorted_triplets = Parallel(n_jobs=5, backend="loky")(
    delayed(classify_collider_by_pid)(df, triplet) for triplet in extra_colliders
    )           
    for each_triplet in sorted_triplets:
        if each_triplet["label"] == "synergistic":
            extra_syn.append(each_triplet["triplet"])
        else:
            extra_pairwise.append(each_triplet["triplet"]) 
    
    extra_syn = {tuple(map(int, triplet)) for triplet in extra_syn}
    extra_pairwise = {tuple(map(int, triplet)) for triplet in extra_pairwise}
    return extra_syn, extra_pairwise


def synergy_ranker_pid(df, nbins=4, top_k=100):
    """
    Rank all 3-column triplets {i,j,k} in `df` by their
    *maximum* PID synergy (each variable is, in turn, treated as target).

    Continuous columns are discretised into `nbins` equal-frequency bins.

    Returns
    -------
    list[tuple[int,int,int]]
        Triplets sorted from highest to lowest synergy.
    """
    # 1) Discretise / ensure integer labels
    disc = pd.DataFrame()
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.integer) and df[col].nunique() <= nbins:
            disc[col] = df[col].astype(int)
        else:
            disc[col] = pd.qcut(df[col].rank(method="first"),
                                nbins, labels=False, duplicates="drop").astype(int)

    n_cols = disc.shape[1]
    scores = []

    # 2) Compute synergy for every unordered triplet
    for i, j, k in combinations(range(n_cols), 3):
        xi = disc.iloc[:, i].to_numpy()
        xj = disc.iloc[:, j].to_numpy()
        xk = disc.iloc[:, k].to_numpy()

        # three possible “(sources) → target” assignments
        s1 = pid_synergy_Imin(xj, xk, xi)
        s2 = pid_synergy_Imin(xi, xk, xj)
        s3 = pid_synergy_Imin(xi, xj, xk)

        scores.append(((i, j, k), max(s1, s2, s3)))

    # 3) Return the `top_k` triplets with largest synergy
    scores.sort(key=lambda t: t[1], reverse=True)
    return [trip for trip, _ in scores[:top_k]]


def pid_synergy_func(df, nbins=4, top_k=200):
    """
    Matches the signature expected by `triplets_found` in your script.
    Adjust `nbins` or `top_k` as needed.
    """
    return synergy_ranker_pid(df, nbins=nbins, top_k=top_k)