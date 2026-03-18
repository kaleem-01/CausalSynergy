import shutil
import networkx as nx
from motifs.motifs import *

import os
import pickle

import pandas as pd


func_map = {
    "Chain": generate_chain_no_synergy,
    "ChainSynergy": generate_chain_with_synergy,
    "ChainD": generate_chain_discrete_d,
    "Fork": generate_fork_discrete,
    "ForkSynergy": generate_fork_syn_discrete,
    "Collider": generate_collider_discrete,
    "ForkD": generate_fork_discrete_d,
    "ForkSynergyD": generate_fork_syn_discrete_d,
    "ColliderD": generate_collider_discrete_d,
    "LargeDiscrete": generate_large_discrete,
    "Multi-ParentThree": generate_multi_parent_synergy,
    "Multi-ParentFour": lambda: generate_multi_parent_synergy(parents=4),
    "Multi-ParentFive": lambda: generate_multi_parent_synergy(parents=5),
    "MediatorSynergy": generate_mediator_chain_with_synergy,
}

def load_motif_data(dataset):
    true_dag, df = func_map[dataset]()
    return true_dag, df

def _bif2bayesian(pathname, verbose=3):
    from pgmpy.models import BayesianNetwork
    from pgmpy.factors.discrete import TabularCPD

    """Return the fitted bayesian model.

    Example
    -------
    >>> from pgmpy.readwrite import BIFReader
    >>> reader = BIFReader("bif_test.bif")
    >>> reader.get_model()
    <pgmpy.models.BayesianNetwork object at 0x7f20af154320>
    """
    from pgmpy.readwrite import BIFReader
    if verbose>=3: print('[bnlearn] >Loading bif file <%s>' %(pathname))

    bifmodel = BIFReader(path=pathname)

    try:
        model = BayesianNetwork(bifmodel.variable_edges)
        model.name = bifmodel.network_name
        model.add_nodes_from(bifmodel.variable_names)

        tabular_cpds = []
        for var in sorted(bifmodel.variable_cpds.keys()):
            values = bifmodel.variable_cpds[var]
            cpd = TabularCPD(var, len(bifmodel.variable_states[var]), values,
                             evidence=bifmodel.variable_parents[var],
                             evidence_card=[len(bifmodel.variable_states[evidence_var])
                                            for evidence_var in bifmodel.variable_parents[var]])
            tabular_cpds.append(cpd)

        model.add_cpds(*tabular_cpds)
#        for node, properties in bifmodel.variable_properties.items():
#            for prop in properties:
#                prop_name, prop_value = map(lambda t: t.strip(), prop.split('='))
#                model.node[node][prop_name] = prop_value

        return model

    except AttributeError:
        raise AttributeError('[bnlearn] >First get states of variables, edges, parents and network names')


# You need to install bnlearn in a separate environment to avoid conflicts (uses an older version of pgmpy)
def bif_data(folder, dataset, n=10_000):
    """
    Save csv and metadata for bif datasets
    """
    import bnlearn as bn
    import pickle
    import os

    # model = bn.import_DAG(dataset)
    model = _bif2bayesian(os.path.join(folder, f"{dataset}.bif"))

    # Setup adjacency matrix
    adjmat = bn.dag2adjmat(model)

    # Store
    sample = {}
    sample['model']=model
    sample['adjmat']=adjmat
    # print(model["model"])
    df = bn.sampling(sample, n=n)
    true_graph = nx.DiGraph(model.edges())
    pickle.dump(true_graph, open(os.path.join(folder, f'{dataset}.pickle'), 'wb'))
    df.to_csv(os.path.join(folder, f'{dataset}.csv'), index=False)
    print(f"Graph and data saved to {folder}")
    return true_graph, df


def load_biff_data(folder = 'data/bnRep/', dataset="asia"):
    # folder = 'data/bnRep/'
    true_graph = pickle.load(open(os.path.join(folder, f'{dataset}.pickle'), 'rb'))
    df = pd.read_csv(os.path.join(folder, f'{dataset}.csv'))
    return true_graph, df


def process_bif_file(f):
        if f.endswith('.bif'):
            try:
                dataset = f[:-4]
                print(f"Processing {dataset}...")
                bif_data(folder, dataset, n=10_000)
            except Exception as e:
                print(f"Error processing {f}: {e}")
                error_folder = os.path.join(folder, "errors")
                os.makedirs(error_folder, exist_ok=True)
                shutil.move(os.path.join(folder, f), os.path.join(error_folder, f))
                print(f"Moved {f} to {error_folder}")


if __name__ == "__main__":
    folder = 'data/bnlearn/'
    files = os.listdir(folder)
    for f in files:
        process_bif_file(f)