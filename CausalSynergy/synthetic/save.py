import os
import numpy as np
import pandas as pd

import jointpmf.central_driver_methods as cdm
from synthetic.binary import BinaryDataset
from synthetic.toynetworks import *

def saveBinary(folder, n_roots, n_nodes, n_top_nodes, numOfDatasets=10, n_samples=10_000, p_noise=0):

    if not os.path.exists(folder):
        os.makedirs(folder)
    
    datafilenum = 0 
    
    for i in range(datafilenum, datafilenum+numOfDatasets):
        # n_roots = 3
        # n_nodes = 10
        # n_top_nodes = 5
        
        numOfDatasets = numOfDatasets
        print(f"Saving binary datasets to {folder}")
        binary_dataset = BinaryDataset(n_roots=n_roots, n_nodes=n_nodes, n_top_nodes=n_top_nodes, p_noise=p_noise, n_samples=n_samples)
        df, true_dag, logic_map, metadata = binary_dataset.generate_binary_synergy_dataset()
        if not os.path.exists(folder):
            os.makedirs(folder)
        df.to_csv(f"{folder}/Data_Graph_{i}.csv", index=False)
        # visualize_dag(true_dag, logic_map=logic_map)
        metadata.to_csv(f"{folder}/Metadata_Graph_{i}.csv", index=False)


def saveToy(save_dir, job_name='toy', num_vars=10, number_BNs=10, n_samples=10000):
    # BN Parameters
    # num_roots = np.random.randint(2, 4)  
    num_roots = 8
    pair_probs = 0.9
    cardinality = 3


    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving toy networks to {save_dir}")
    parallel_bn_creation(num_vars, num_roots, pair_probs, cardinality, job_name, save_dir, number_BNs)    

if __name__ == "__main__":
    # # Save Binary Datasets

    # saveBinary(f"data/syntheticBinary/binaryNoNoise", numOfDatasets=20, n_samples=10000, p_noise=0)
    # saveBinary(f"data/syntheticBinary/binaryLowNoise", numOfDatasets=20, n_samples=10000, p_noise=0.1)
    # saveBinary(f"data/syntheticBinary/binaryMediumNoise", numOfDatasets=20, n_samples=10000, p_noise=0.2)
    # saveBinary(f"data/syntheticBinary/binaryHighNoise", numOfDatasets=20, n_samples=10000, p_noise=0.3)

    # saveBinary(f"data/syntheticBinary/binaryManyRoots", numOfDatasets=10, n_samples=10000, p_noise=0.1)
    
    # # Save Toy Networks
    # saveToy(f"data/syntheticToy/toySmall", job_name='small', num_vars=10, number_BNs=20, n_samples=10000)
    # saveToy(f"data/syntheticToy/toyMedium", job_name='medium', num_vars=20, number_BNs=20, n_samples=10000)
    # saveToy(f"data/syntheticToy/toyLarge", job_name='large', num_vars=30, number_BNs=20, n_samples=10000)
    # saveToy(f"data/syntheticToy/toyManyRoots", job_name='manyRoots', num_vars=20, number_BNs=20, n_samples=10000)
    # saveToy(f"data/syntheticToy/toyFewSRVs", job_name='highCardinality', num_vars=20, number_BNs=20, n_samples=10000)
    saveToy(f"data/syntheticToy/lowCorrelation", job_name='lowCorrelation', num_vars=20, number_BNs=20, n_samples=10000)