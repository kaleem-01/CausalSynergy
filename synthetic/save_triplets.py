from joblib import delayed, Parallel
import numpy as np
from idtxl.data import Data
from idtxl.bivariate_pid import BivariatePID
import pickle
from metrics.synergy import find_synergistic_triplets
from utils.data import read_data_files
import os
import ast
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp

def broja_synergy(df, source1, source2, target):

    s1, _ = pd.factorize(df[source1], sort=True)
    s2, _ = pd.factorize(df[source2], sort=True)
    t,  _ = pd.factorize(df[target], sort=True)

    data = Data(np.vstack((s1, s2, t)), 'ps', normalise=False)

    settings = {
        'alpha': 0.1,
        'alph_s1': int(s1.max() + 1),
        'alph_s2': int(s2.max() + 1),
        'alph_t': int(t.max() + 1),
        'max_unsuc_swaps_row_parm': 60,
        'num_reps': 63,
        'max_iters': 1000,
        'pid_estimator': 'TartuPID',
        'verbose': False,
        'lags_pid': [0, 0],   # or remove if you do not want lagged PID
    }

    pid_analysis = BivariatePID()
    results = pid_analysis.analyse_single_target(
        settings=settings,
        data=data,
        target=2,
        sources=[0, 1],
    )
    return results.get_single_target(2)['syn_s1_s2']



def triplets_found(df, metadata_df, top_n=10):
    # triplets, scores = zip(*find_synergistic_triplets(df, inflection_point=True))
    ii_triplets = find_synergistic_triplets(df, inflection_point=True, n_jobs=mp.cpu_count())
    results = Parallel(n_jobs=mp.cpu_count(), backend="loky")(
            delayed(evaluate_triplet)(df, triplet) for triplet in ii_triplets
        )
    
    triplets = [r[0] for r in results if r[1] > 0.1]  # Filter triplets with positive synergy

    triplets_dict = {}  # Dictionary to store results
    
    if "SRV" in metadata_df["Type"].values:
        all_colliders = metadata_df["Combs"] 
        synergistic_colliders = metadata_df[metadata_df["Type"] == "SRV"]["Combs"]
    else:
        all_colliders = metadata_df["Combs"]
        synergistic_colliders = metadata_df[metadata_df["Type"] == "XOR"]["Combs"]

    found = 0
    triplets_found = []
    for check in synergistic_colliders:
        # print("Checking triplet:", check)
        check_list = ast.literal_eval(check)
        for i in triplets[:top_n]:
            candidate_triplet = [int(j) for j in i]
            # print("Candidate triplet:", candidate_triplet)
            if candidate_triplet == check_list:
                # print(f"Found triplet {check} in results.")
                triplets_found.append(check_list)
                found += 1
    # print(f"Total synergistic triplets found: {found} out of {len(synergistic_colliders)}")
    triplets_ratio_found = found / len(synergistic_colliders) if len(synergistic_colliders) > 0 else 0
    
    triplets_dict["Ratio"] = triplets_ratio_found
    triplets_dict["Total"] = len(synergistic_colliders)
    triplets_dict["Found"] = found
    triplets_dict["N_Vars"] = df.shape[1]
    triplets_dict["Metric"] = "PID_Broja"
    triplets_dict["Directory"] = directory.split("\\")[-1]
    return triplets_dict, triplets


def get_triplet_results(directory, csv_by_id, metadata_by_id):
    triplets_dict_list = []
    results_path = f"data/triplets/{directory.split(os.sep)[-1]}_pid.pkl"
    # Try to load intermediate results if they exist
    if os.path.isfile(results_path):
        with open(results_path, "rb") as f:
            triplets_dict_list = pickle.load(f)
        processed_files = {d["File"] for d in triplets_dict_list if "File" in d}
    else:
        print(f"Could not load intermediate results from {results_path}. Starting fresh.")
        triplets_dict_list = []
        processed_files = set()


    for i in tqdm(list(csv_by_id.keys()), desc=f"Processing files in {directory}"):
        if i in processed_files:
            continue
        df = csv_by_id[i]
        metadata_df = metadata_by_id[i]
        triplets_dict, triplets = triplets_found(df, metadata_df)

        triplets_dict["File"] = i
        triplets_dict["Directory"] = directory.split(os.sep)[-1]
        triplets_dict["Triplets"] = triplets
        triplets_dict_list.append(triplets_dict)
        # Save intermediate results after every 20 files or if file does not exist
        if len(triplets_dict_list) % 20 == 0:
            with open(results_path, "wb") as f:
                pickle.dump(triplets_dict_list, f)

    # Final Save File
    with open(results_path, "wb") as f:
        pickle.dump(triplets_dict_list, f)

    results_df = pd.DataFrame(triplets_dict_list)

    return results_df

def save_triplet_results(directory):
    csv_by_id, metadata_by_id, = read_data_files(directory)
    results_df = get_triplet_results(directory, csv_by_id, metadata_by_id)
    # print(f"Saved results to {os.path.join(child, 'triplet_synergy_results.csv')}")
    results_df.to_csv(f"results/{directory.split(os.sep)[-1]}_triplets.csv", index=False)
    

def evaluate_triplet(df, triplet):
    highest_pid = 0

    triplet = list(map(str, triplet))

    for col in triplet:
        target = col
        source1, source2 = [c for c in triplet if c != target]
        pid = broja_synergy(df, source1, source2, target)
        # print(f"Evaluating triplet {source1}, {source2} -> {target}: PID = {pid}")

        if pid > highest_pid:
            highest_pid = pid
            final_triplet = (source1, source2, target)

    # print(final_triplet)
    # print(f"\nHighest PID for triplet {triplet}: {highest_pid} with target {final_triplet[2]}\n")
    return (final_triplet, highest_pid)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    directory  = "data/datasets/jpmf_data"
    # get_triplet_results(directory, csv_by_id, metadata_by_id)
    save_triplet_results(directory)