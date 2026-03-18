import pickle
import pandas as pd
from tqdm import tqdm
import os
import dill
from multiprocessing import Pool
import jointpmf.central_driver_methods as cdm



# This is necessary to ensure compatibility with the renamed class in the jointpmf module. (From 'jointpmf.jointpmf' to 'jointpmf')
class RenameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Map 'jointpmf.jointpmf' to 'jointpmf'
        # if module == "jointpmf.jointpmf":
        #     module = "jointpmf"
        return super().find_class(module, name)

with open("jointpmf/precomputed_srvs.pkl", "rb") as f:
    _precomputed_srvs = RenameUnpickler(f).load()

# read in the precomputed dependent variables with two sources
with open('jointpmf/precomputed_dependent_vars_two_sources.pkl', 'rb') as f:
    _precomputed_dependent_vars_two_sources = RenameUnpickler(f).load()

# read in the precomputed dependent variables with one source
with open('jointpmf/precomputed_dependent_vars_one_source.pkl', 'rb') as f:
    _precomputed_dependent_vars_one_source = RenameUnpickler(f).load()


def create_and_save_one_bn(ii, job_name, num_vars, num_roots, pair_probs,
                             cardinality, num_samples, bn_dir):
    bn, appended_edges_weight_df = cdm.construct_jpmf_bn(
        num_vars, num_roots, pair_probs,
        _precomputed_dependent_vars_one_source,
        _precomputed_dependent_vars_two_sources,
        _precomputed_srvs,
        numvalues=cardinality,
    )

    bn_path = os.path.join(bn_dir, f"JOB_{job_name}_Nodes_{num_vars}_Graph_{ii}.dill")
    with open(bn_path, 'wb') as f:
        dill.dump(bn, f)

    edge_path = os.path.join(bn_dir, f"JOB_{job_name}_Nodes_{num_vars}_Graph_{ii}_edges.dill")
    with open(edge_path, 'wb') as f:
        dill.dump(appended_edges_weight_df, f)

    samples = bn.generate_samples(num_samples)
    df_samples = pd.DataFrame(samples)
    csv_path = os.path.join(bn_dir, f"Data_Graph_{ii}.csv")
    df_samples.to_csv(csv_path, index=False)

    return f"Saved BN {ii} with {num_vars} nodes and {num_samples} samples to {bn_dir}"


def create_and_save_bn_samples_parallel(
    number_BNs=30,
    job_name="Initial_Test",
    num_vars=20,
    num_roots=5,
    pair_probs=0.6,
    cardinality=3,
    num_samples=10000,
    bn_dir="BNs_Directory",
    processes=None
):
    os.makedirs(bn_dir, exist_ok=True)

    args = [
        (ii, job_name, num_vars, num_roots, pair_probs,
         cardinality, num_samples, bn_dir)
        for ii in range(number_BNs)
    ]

    with Pool(processes=processes) as pool:
        for msg in tqdm(pool.starmap(create_and_save_one_bn, args),
                        total=number_BNs, desc="Creating BNs"):
            print(msg)


def process_bn_samples_singular(
    bn_dir="../data/precomputed_cpds/",
    job_name="Initial_Test",
    num_vars=20,
    number_BNs=30,
    process_fn=None,
    results_csv="processed_results.csv"
):
    """
    Reads BN sample CSVs, applies a processing function, and saves results.

    Parameters
    ----------
    bn_dir : str
        Directory containing BN sample CSVs.
    job_name : str
        Job name used in file naming.
    num_vars : int
        Number of variables in BN (used for file naming).
    number_BNs : int
        Number of BN sample files to process.
    process_fn : callable
        Function to apply to each sample DataFrame. Should accept a DataFrame and return a dict or Series.
    results_csv : str
        Path to save the processed results CSV.
    """
    results = []
    for ii in range(number_BNs):
        csv_path = os.path.join(bn_dir, f"Data_Graph_{ii}.csv")
        if not os.path.exists(csv_path):
            print(f"File not found: {csv_path}")
            continue
        df_samples = pd.read_csv(csv_path)
        if process_fn is not None:
            metric = process_fn(df_samples)
        result = {"Graph": ii, "NumSamples": len(df_samples)}
        if isinstance(metric, float) or isinstance(metric, int):
            result["Metric"] = metric
        elif isinstance(metric, pd.Series):
            result.update(metric.to_dict())
        elif isinstance(metric, dict):
            result.update(metric)
        else:
            print(f"Unexpected metric type: {type(metric)} for file {csv_path}")
            continue
        print(f"Processed {csv_path}: {result}")
        results.append(result)
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(bn_dir, results_csv), index=False)
    print(f"Processed results saved to {os.path.join(bn_dir, results_csv)}")
    return results_df


def create_and_save_bn_samples(
    number_BNs=30,
    job_name="Initial_Test",
    num_vars=20,
    num_roots=5,
    pair_probs=0.6,
    cardinality=3,
    num_samples=10000,
    bn_dir="BNs_Directory"
):
    """
    Creates BN objects, generates samples, and saves samples as CSV files.

    Parameters
    ----------
    number_BNs : int
        Number of BN objects to create.
    job_name : str
        Job name for file naming.
    num_vars : int
        Number of variables in BN.
    num_roots : int
        Number of root nodes.
    pair_probs : float
        Probability for edge creation.
    cardinality : int
        Cardinality of variables.
    num_samples : int
        Number of samples to generate for each BN.
    bn_dir : str
        Directory to save BN objects and samples.
    """
    os.makedirs(bn_dir, exist_ok=True)

    for ii in range(number_BNs):
        bn, appended_edges_weight_df = cdm.construct_jpmf_bn(
            num_vars, num_roots, pair_probs,
            _precomputed_dependent_vars_one_source,
            _precomputed_dependent_vars_two_sources,
            _precomputed_srvs, numvalues=cardinality,
        )

        # Save BN object
        bn_path = os.path.join(bn_dir, f"JOB_{job_name}_Nodes_{num_vars}_Graph_{ii}.dill")
        with open(bn_path, 'wb') as f:
            dill.dump(bn, f)

        # Save edges/metadata
        edge_path = os.path.join(bn_dir, f"JOB_{job_name}_Nodes_{num_vars}_Graph_{ii}_edges.dill")
        with open(edge_path, 'wb') as f:
            dill.dump(appended_edges_weight_df, f)

        # Generate samples and save as CSV
        samples = bn.generate_samples(num_samples)
        df_samples = pd.DataFrame(samples)
        csv_path = os.path.join(bn_dir, f"Data_Graph_{ii}.csv")
        df_samples.to_csv(csv_path, index=False)
        print(f"Saved BN {ii} with {num_vars} nodes and {num_samples} samples to {bn_dir}")





def _process_bn_creation(num_vars, num_roots, pair_probs, cardinality, job_name, save_dir, ii):
    # Simulate creation of BN object (pmf_toy) ...
    bn, appended_edges_weight_df = cdm.construct_jpmf_bn(
        num_vars, num_roots, pair_probs,
        _precomputed_dependent_vars_one_source,
        _precomputed_dependent_vars_two_sources,
        _precomputed_srvs, numvalues=cardinality,
    )


    # Save the BN to BN_temp for reference
    with open(os.path.join(save_dir, f"JOB_{job_name}_Nodes_{num_vars}_Graph_{ii}.dill"), 'wb') as f:
        dill.dump(bn, f)
        data = bn.generate_samples(10_000)
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(save_dir, f"Data_Graph_{ii}.csv"), index=False)

    # Save the appended edges/metadata
    # print(f"Saving metadata for Graph {ii}...")
    # print(appended_edges_weight_df)
    clean = [list(map(int, comb)) for comb in appended_edges_weight_df["Combs"]]  # -> [[1,2,3], [0,4], ...]
    appended_edges_weight_df["Combs"] = clean
    appended_edges_weight_df.to_csv(os.path.join(save_dir, f"Metadata_Graph_{ii}.csv"), index=False)

def parallel_bn_creation(num_vars, num_roots, pair_probs, cardinality, job_name, save_dir, number_BNs):
    """
    Function to parallelize the creation of Bayesian Networks.
    """
    from multiprocessing import Pool
    with Pool() as pool:
        pool.starmap(
            _process_bn_creation,
            [(num_vars, num_roots, pair_probs, cardinality, job_name, save_dir, ii) for ii in range(number_BNs)]
        )

# Example usage
