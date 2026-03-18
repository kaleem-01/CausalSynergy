import os
from algorithms.benchmark import *
from pipeline.dataclasses import Config
from pipeline.pipeline import Pipeline
import pickle
from datetime import datetime
from functools import partial
from tqdm import tqdm
from utils.console import print_banner, clear_console_soft
import time

import logging
logging.getLogger('pgmpy').setLevel(logging.WARNING)

# logging.getLogger('causallearn').setLevel(logging.WARNING)

import warnings
warnings.filterwarnings("ignore")



def run_algorithm_synthetic(directory, n_jobs=10, reps=2, samples=None, algorithm='ea', algorithm_func=run_ea):
    childdir = [x[0] for x in os.walk(directory)][1:]  # Skip the root directory

    for subdir in childdir:
        tqdm.write(f"\n▶️ Folder:  {subdir}")

        configs = Config(
            folder=subdir,
            replications=reps,
            algorithm=algorithm,
            samples=samples
        )

        pipeline = Pipeline(algorithm_func, configs=configs)
        # try:
        pipeline.run(n_jobs=n_jobs)
        pipeline.save()
        tqdm.write(f"✅ Results saved for {subdir}")
        # except Exception as e:
        # tqdm.write(f"❌ Error processing {subdir}: {e}")


def run_experiments(reps, algorithm, algorithm_func, configs):        
    pipeline = Pipeline(algorithm_func, configs=configs)         
    exp1 = []
    exp2 = []

    for i in tqdm(range(reps), desc="🔁 Running Experiments", unit="rep"):
        start = time.time()
        try:
            exp1.append(pipeline.experiment1())
            exp2.append(pipeline.experiment2())
            elapsed = time.time() - start
            tqdm.write(f"✅ Rep {i+1}/{reps} done — {elapsed:.2f}s")
        except Exception as e:
            tqdm.write(f"❌ Rep {i+1}/{reps} failed: {e}")

    save_dir = os.path.join("results", algorithm)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # save results to pickle files
    with open(os.path.join(save_dir, f'exp1.pkl'), 'wb') as f:
        pickle.dump(exp1, f)
    with open(os.path.join(save_dir, f'exp2.pkl'), 'wb') as f:
        pickle.dump(exp2, f)



if __name__ == "__main__":
    algorithms = {
        'pc': run_pc,   # original PC algorithm
        'pc_gsq':run_pc_gsq,  # PC algorithm with G-square test
        'ges': run_ges,
        # 'hc': run_hc,
        # "ea_ues":  partial(run_ea, population_size=30, generations=None, crossover_method='edge_swap', informed_ratio=0),
        # "ea_ies":  partial(run_ea, population_size=30, generations=None, crossover_method='edge_swap', informed_ratio=0.5),
        # "ea_fes":  partial(run_ea, population_size=30, generations=None, crossover_method='edge_swap', informed_ratio=1),
        # "ea_fg":  partial(run_ea, population_size=30, generations=20, informed_ratio=1, final_greedy=True),
    }


    
    for algo_name, algo_func in algorithms.items():
        reps = 1
        
        banner = f"""
        🚀 Benchmarking Datasets Process  🚀
        
        Algorithm: {algo_name}
        Replications: {reps}
        """
        

        run_algorithm_synthetic(directory=os.path.join("data", "datasets"), n_jobs=20, reps=reps, algorithm=algo_name, algorithm_func=algo_func)
        # run_algorithm_synthetic(directory=os.path.join("data", "syntheticBinary"), reps=reps, algorithm=algo_name, algorithm_func=algo_func)
        # run_experiments(reps=reps, algorithm=algo_name, algorithm_func=algo_func, configs=configs)
        
        

        # pipeline = Pipeline(algo_func, configs=configs)
        # pipeline.run()
        # pipeline.save()
        # tqdm.write(f"✅ Results saved for {configs.folder}")

        
        # time.sleep(2)  # Pause for 2 seconds between different algorithm runs
        clear_console_soft()
        # print
        
        results_banner = f"""
        ✅ Completed Benchmarking for {algo_name}  ✅

        results saved in results/{algo_name}/
        """
        print_banner(results_banner, color="green")