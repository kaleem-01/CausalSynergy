import pandas as pd
import numpy as np

def fit_cpts_and_log_likelihood(df, adjacency):
    """
    Given a DataFrame `df` containing columns for each node,
    and a BN structure `adjacency` (dict of {node: [parents]}),
    this function:
      1) Learns CPTs (Conditional Probability Tables) for each node from the data.
      2) Computes the total log-likelihood of the entire dataset under these CPTs.

    Returns
    -------
    cpts : dict
        A dictionary of learned CPTs, where cpts[node] is itself a dictionary
        mapping parent_value_tuples -> distribution over node's values.
    total_log_likelihood : float
        Sum of log p(X_i | parents(X_i)) across all samples and nodes.
    """
    nodes = list(adjacency.keys())
    cpts = {}
    total_log_likelihood = 0.0
    
    # For each node, estimate its CPT given its parents
    for node in nodes:
        parents = adjacency[node]
        
        # Unique values for the node
        node_vals = df[node].unique()
        
        # Unique combos of parent values (Cartesian product in practice)
        if len(parents) == 0:
            # No parents, the CPT is just the distribution of `node`
            # We'll store it under a special key (e.g., None or ())
            cpts[node] = {}
            # Count how often each value appears
            counts = df[node].value_counts()
            total_count = len(df)
            # Probability for each node value
            prob_dist = (counts / total_count).to_dict()
            cpts[node][()] = prob_dist  # Use tuple() for "no parent" condition
        else:
            # We group by parent values
            group_cols = parents
            cpts[node] = {}
            
            # Group the data by all parent columns
            grouped = df.groupby(group_cols)
            for parent_vals, sub_df in grouped:
                # Make sure parent_vals is a tuple for dictionary keys
                if not isinstance(parent_vals, tuple):
                    parent_vals = (parent_vals,)
                
                # Count how often each node value appears in this subset
                value_counts = sub_df[node].value_counts()
                sub_total = len(sub_df)
                
                # Convert counts to probabilities
                prob_dist = (value_counts / sub_total).to_dict()
                
                # For node values not in sub_df, we might want to store 0
                for val in node_vals:
                    if val not in prob_dist:
                        prob_dist[val] = 0.0
                
                cpts[node][parent_vals] = prob_dist
        
        # -----------------------------------------
        # Compute log-likelihood contribution
        # -----------------------------------------
        # We'll iterate over each row in df and add log p(X_i | Parents_i)
        # Alternatively, a more efficient way is to sum logs in each group,
        # but for clarity we'll do row-by-row here.
    
    # Now that CPTs are learned, compute total log-likelihood
    for idx, row in df.iterrows():
        logp_row = 0.0
        for node in nodes:
            parents = adjacency[node]
            if len(parents) == 0:
                # Use the distribution under ()
                prob_dist = cpts[node][()]
            else:
                # Construct parent_vals tuple
                parent_vals = tuple(row[p] for p in parents)
                prob_dist = cpts[node][parent_vals]
            
            node_val = row[node]
            # Probability of the node's actual value
            p_val = prob_dist[node_val]
            # Add log prob
            # Avoid log(0) by adding a small epsilon or by ignoring zero-prob
            # for simplicity, let's do a small guard:
            eps = 1e-12
            logp_row += np.log(p_val + eps)
        total_log_likelihood += logp_row
    
    return cpts, total_log_likelihood

def count_parameters(cpts):
    """
    Count the total number of free parameters in a discrete BN
    from the learned CPTs.

    For each node:
      - Let R_node = number of values for the node
      - Let R_parents = product of (number of values for each parent)
      - Number of free parameters = (R_node - 1) * R_parents

    Returns
    -------
    num_params : int
    """
    num_params = 0
    for node, parent_map in cpts.items():
        # parent_map: dict from parent-value-tuple -> { node_val: probability }
        # We can find how many unique values the node can take from
        # any single distribution. Let's use the first item:
        any_dist = next(iter(parent_map.values()))
        R_node = len(any_dist)  # number of distinct values for this node
        
        # Also count how many distinct parent tuples there are:
        R_parents = len(parent_map)
        
        # # Free parameters for each parent tuple is R_node - 1
        # We multiply by the number of distinct parent configurations
        num_params_node = (R_node - 1) * R_parents
        num_params += num_params_node
    return num_params

def score_bic(df, adjacency):
    """
    Compute the BIC score for a discrete BN with a given adjacency structure.
    1) Fit CPTs and compute log-likelihood
    2) Count number of parameters
    3) BIC = logLik - 0.5 * num_params * ln(N)

    Returns
    -------
    bic_value : float
    """
    # 1) Learn CPTs and get total log-likelihood
    cpts, log_lik = fit_cpts_and_log_likelihood(df, adjacency)
    
    # 2) Count parameters
    num_params = count_parameters(cpts)
    
    # 3) Compute BIC
    n_samples = len(df)
    bic_value = log_lik - 0.5 * num_params * np.log(n_samples)
    
    return bic_value, log_lik, num_params


if __name__ == "__main__":
    import pandas as pd
    import numpy as np

        
    
    # Example DataFrame
    df_example = pd.read_csv("data/xor_synthetic_data_10.csv")


    print(df_example.head())
    


    # data = {
    #     "X0": [0, 0, 1, 1, 0, 1, 1, 0],
    #     "X1": [1, 0, 1, 0, 1, 0, 1, 1],
    #     "Y":  [0, 1, 1, 1, 0, 1, 1, 0]
    # }

    # df_example = pd.DataFrame(data)
    # print(df_example.head())
    # Example adjacency
    adjacency = {
        # "Y_(5_9)": ["X5"],
        # "Y_(5_9)": ["X9"],
        # "X5": ['Y_(5_9)'],
        # "X9": ['Y_(5_9)'],
        # "X1": ["X0"],
        "Y_(5_9)": ["X5", "X9"]
    }


    # Compute BIC
    bic_val, log_lik, num_params = score_bic(df_example, adjacency)
    print(f"BIC Score: {bic_val:.2f}")
    print(f"Log-Likelihood: {log_lik:.2f}")
    print(f"Number of Parameters: {num_params}")
