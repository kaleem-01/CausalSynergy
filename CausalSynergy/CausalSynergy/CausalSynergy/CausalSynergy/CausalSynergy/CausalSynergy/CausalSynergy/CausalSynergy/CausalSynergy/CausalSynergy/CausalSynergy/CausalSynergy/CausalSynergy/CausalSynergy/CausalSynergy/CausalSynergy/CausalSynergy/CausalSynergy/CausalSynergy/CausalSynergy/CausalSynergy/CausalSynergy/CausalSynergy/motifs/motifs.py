# ---------- Utility functions ----------
import networkx as nx
import numpy as np
import pandas as pd


def bernoulli(p, size):
    return (np.random.rand(size) < p).astype(int)


def noisy_copy(x, flip_prob=0.05):
    flips = bernoulli(flip_prob, len(x))
    return x ^ flips  # XOR flips

NUM_SAMPLES = 10_000

# ---------- Data generation functions ----------
def generate_chain_no_synergy(n=NUM_SAMPLES):
    """
    Generate a simple causal chain A -> B -> C -> D without synergy.
    """
    noise=0.05
    np.random.seed(42)
    A = bernoulli(0.5, n)
    B = noisy_copy(A, noise)
    C = noisy_copy(B, noise)

    true_graph = nx.DiGraph()
    true_graph.add_edges_from([('A', 'B'), ('B', 'C')])
    return true_graph, pd.DataFrame({'A': A, 'B': B, 'C': C})


def generate_chain_with_synergy(n=NUM_SAMPLES):
    """
    Generate a causal chain with synergy at B: (A XOR C) -> B -> D.
    """
    noise=0.05
    np.random.seed(42)
    A = bernoulli(0.5, n)
    C = bernoulli(0.5, n)
    B_clean = A ^ C  # XOR synergy
    B = noisy_copy(B_clean, noise)
    D = noisy_copy(B, noise)

    true_graph = nx.DiGraph()
    true_graph.add_edges_from([('A', 'B'), ('C', 'B'), ('B', 'D')])
    return true_graph, pd.DataFrame({'A': A, 'C': C, 'B': B, 'D': D})


## Chain Causal Structure (A → B → C)
# def chain_discrete(N):
#     A = np.random.randint(0, 5, N)
#     B = (A + 1) % 5
#     C = (B + 1) % 5
#     return pd.DataFrame({'A': A, 'B': B, 'C': C})


# Fork Structure (A → B ← C)
def generate_fork_discrete(N=NUM_SAMPLES):
    A = np.random.randint(0, 3, N)
    C = np.random.randint(0, 3, N)
    B = (A + C)

    true_graph = nx.DiGraph()
    true_graph.add_edges_from([('A', 'B'), ('C', 'B')])
    return true_graph, pd.DataFrame({'A': A, 'B': B, 'C': C})


# Fork Structure (A → B ← C)
def generate_fork_syn_discrete(N=NUM_SAMPLES):
    A = np.random.randint(0, 3, N)
    C = np.random.randint(0, 3, N)
    B = (A ^ C)

    true_graph = nx.DiGraph()
    true_graph.add_edges_from([('A', 'B'), ('C', 'B')])
    return true_graph, pd.DataFrame({'A': A, 'B': B, 'C': C})


# Collider Structure (A → B → C and A → C)
def generate_collider_discrete(N=NUM_SAMPLES):
    A = np.random.randint(0, 3, N)
    B = (A + 1) % 3
    C = (A + B) % 3

    true_graph = nx.DiGraph()
    true_graph.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'C')])
    return true_graph, pd.DataFrame({'A': A, 'B': B, 'C': C})


## Chain Causal Structure (A → B → C) + D
def generate_chain_discrete_d(N=NUM_SAMPLES):
    A = np.random.randint(0, 5, N)
    B = (A + 1) % 5
    C = (B + 1) % 5
    D = np.random.randint(0, 3, N)

    true_graph = nx.DiGraph()
    true_graph.add_edges_from([('A', 'B'), ('B', 'C')])
    return true_graph, pd.DataFrame({'A': A, 'B': B, 'C': C, 'D': D})


# Fork Structure (A → B ← C) + D
def generate_fork_discrete_d(N=NUM_SAMPLES):
    A = np.random.randint(0, 3, N)
    C = np.random.randint(0, 3, N)
    B = (A + C)
    D = np.random.randint(0, 3, N)

    true_graph = nx.DiGraph()
    true_graph.add_edges_from([('A', 'B'), ('C', 'B')])
    return true_graph, pd.DataFrame({'A': A, 'B': B, 'C': C, 'D': D})


# Fork Structure (A → B ← C) + D
def generate_fork_syn_discrete_d(N=NUM_SAMPLES):
    A = np.random.randint(0, 3, N)
    C = np.random.randint(0, 3, N)
    B = (A ^ C)
    D = np.random.randint(0, 3, N)

    true_graph = nx.DiGraph()
    true_graph.add_edges_from([('A', 'B'), ('C', 'B')])
    return true_graph, pd.DataFrame({'A': A, 'B': B, 'C': C, 'D': D})


# Collider Structure (A → B → C and A → C) + D
def generate_collider_discrete_d(N=NUM_SAMPLES):
    A = np.random.randint(0, 3, N)
    B = (A + 1) % 3
    C = (A + B) % 3
    D = np.random.randint(0, 3, N)

    true_graph = nx.DiGraph()
    true_graph.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'C')])
    return true_graph, pd.DataFrame({'A': A, 'B': B, 'C': C, 'D': D})


# Discrete Causal Chain with Two Synergistic Mechanisms
def generate_large_discrete(n_samples=10_000):

    X1 = np.random.binomial(1, 0.5, n_samples)  + np.random.binomial(1, 0.1, n_samples) # Add noise
    X2 = np.random.binomial(1, 0.5, n_samples) + np.random.binomial(1, 0.1, n_samples)  # Add noise
    X3 = X1 ^ X2 + np.random.binomial(1, 0.1, n_samples)  # Add noise
    X4 = np.random.binomial(1, 0.1, n_samples)
    X5 = X4 + X3 + np.random.binomial(1, 0.1, n_samples) # Add noise
    X6 = X4*X3 + np.random.binomial(1, 0.1, n_samples)  # Add noise
    X7 = X1 + X4 + np.random.binomial(1, 0.1, n_samples)  # Add noise
    X8 = np.random.binomial(1, 0.5, n_samples)
    X9 =  np.random.binomial(1, 0.5, n_samples)

    true_graph = nx.DiGraph()
    true_graph.add_edges_from([('X1', 'X3'), ('X2', 'X3'), ('X3', 'X5'), ('X4', 'X5'), ('X3', 'X6'), ('X4', 'X6'), ('X1', 'X7'), ('X4', 'X7')])
    return true_graph, pd.DataFrame({'X1': X1, 'X2': X2, 'X3': X3, 'X4': X4, 'X5': X5, 'X6': X6, 'X7': X7, 'X8': X8, 'X9': X9})


def generate_multi_parent_synergy(N=1000, num_states=2, parents=3):
    """
    Generate synthetic data for a multi-parent synergy (collider) structure.

    Y = XOR-like function of multiple parents (binary or multinomial).

    Args:
        N (int): Number of samples
        num_states (int): Number of discrete states for each parent (2 for binary, >=3 for multinomial)
        parents (int): Number of parent variables influencing Y

    Returns:
        true_graph (nx.DiGraph)
        df (pd.DataFrame)
    """
    data = {}
    parent_names = [f"X{i+1}" for i in range(parents)]

    # Generate parent variables (uniform discrete)
    for p in parent_names:
        data[p] = np.random.randint(0, num_states, N)

    # Compute synergy: XOR for binary, generalized "mod-sum" for multinomial
    stacked = np.vstack([data[p] for p in parent_names])
    if num_states == 2:
        # Classic XOR (binary)
        Y = np.bitwise_xor.reduce(stacked, axis=0)
    else:
        # Generalized XOR-like: sum mod num_states
        Y = np.sum(stacked, axis=0) % num_states

    data["Y"] = Y

    # Create true DAG
    true_graph = nx.DiGraph()
    edges = [(p, "Y") for p in parent_names]
    true_graph.add_edges_from(edges)

    return true_graph, pd.DataFrame(data)


def generate_mediator_chain_with_synergy(N=1000, num_states=2, noise_prob=0.1):
    """
    Chain structure with intermediate synergy:
    X1, X2 -> M -> Y
    - M = XOR(X1, X2) + noise (binary)
    - or generalized mod-sum for multinomial

    Args:
        N (int): number of samples
        num_states (int): discrete states (2 = binary XOR, >=3 = multinomial mod-sum)
        noise_prob (float): probability of flipping M to noise

    Returns:
        true_graph (nx.DiGraph)
        df (pd.DataFrame)
    """
    # Generate parents
    X1 = np.random.randint(0, num_states, N)
    X2 = np.random.randint(0, num_states, N)

    # Compute synergy at intermediate node M
    if num_states == 2:
        M = np.bitwise_xor(X1, X2)
    else:
        M = (X1 + X2) % num_states

    # Add noise to M
    flip_mask = np.random.rand(N) < noise_prob
    M_noisy = M.copy()
    M_noisy[flip_mask] = np.random.randint(0, num_states, flip_mask.sum())

    # Downstream Y (simple dependency on M_noisy, e.g. identity or noisy copy)
    Y = M_noisy.copy()
    # Add a bit of noise to Y as well (optional, keeps things more realistic)
    y_flip_mask = np.random.rand(N) < (noise_prob / 2)
    Y[y_flip_mask] = np.random.randint(0, num_states, y_flip_mask.sum())

    # Build dataframe
    df = pd.DataFrame({
        "X1": X1,
        "X2": X2,
        "M": M_noisy,
        "Y": Y
    })

    # Build true graph
    true_graph = nx.DiGraph()
    true_graph.add_edges_from([
        ("X1", "M"),
        ("X2", "M"),
        ("M", "Y")
    ])

    return true_graph, df