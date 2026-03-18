# import pandas as pd
# import sys
# import os

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# import numpy as np
# import itertools
# from src.synthetic import create_synthetic_xor_dataset
# # from .src.simulate_data import create_synthetic_xor_dataset

# def compute_entropy(values):
#     """
#     Computes the entropy (in bits) of a list or array of discrete values.
#     H(X) = - Sum( p(x) * log2(p(x)) ).
#     """
#     counts = pd.Series(values).value_counts()
#     probabilities = counts / len(values)
#     return -np.sum(probabilities * np.log2(probabilities))

# def information_gain(df, feature_col, target_col):
#     """
#     Computes the information gain (IG) of splitting on a discrete feature
#     (feature_col) with respect to the target variable (target_col) in df.
    
#     IG(D, feature_col) = H(Y) - 
#         sum_{v in values(feature_col)}( (|D_v| / |D|) * H(Y | feature_col=v) )
    
#     Parameters
#     ----------
#     df : pd.DataFrame
#         The dataset containing both the feature and the target columns.
#     feature_col : str
#         The name of the discrete (categorical) feature column to split on.
#     target_col : str
#         The name of the discrete (categorical) target column.
    
#     Returns
#     -------
#     float
#         The information gain value.
#     """
#     # 1) Entropy of the entire target
#     overall_entropy = compute_entropy(df[target_col])
    
#     # 2) Weighted sum of entropies for each unique value of feature_col
#     weighted_entropy_sum = 0.0
#     for value in df[feature_col].unique():
#         subset = df[df[feature_col] == value]
#         subset_entropy = compute_entropy(subset[target_col])
#         weight = len(subset) / len(df)
#         weighted_entropy_sum += weight * subset_entropy
    
#     # 3) Information Gain
#     return overall_entropy - weighted_entropy_sum

# def calculate_infomation_gain(df, col1, col2, target_col):
#     """
#     Calculate information gain for two columns and a target column.
#     """
#     # Create a new column combining col1 and col2
#     combined_col = f"{col1}_{col2}"
#     df[combined_col] = (
#         df[col1].astype(str) 
#         + "_" + 
#         df[col2].astype(str)
#     )
#     # Calculate information gain
#     ig = information_gain(df, combined_col, target_col)

#     df.drop(columns=[combined_col], inplace=True)  # Clean up
    
#     return ig

# if __name__ == "__main__":
#     # -------------------------------------------------------------------------
#     # EXAMPLE: Load your XOR synthetic data (or any dataset)
#     # -------------------------------------------------------------------------
#     # df_xor = pd.read_csv("data/xor_synthetic_data_multiple_20.csv")
    
#     synergy_nodes = [
#         [0, 1],
#         [5, 6]]

#     n_inputs = 10

#     df_xor = create_synthetic_xor_dataset(
#             n_inputs=n_inputs,
#             synergy_subsets=synergy_nodes,
#             noise_prob=0.10,
#             n_samples=1000,
#             random_state=42
#         )

#     print("First few rows of the XOR dataset:")
#     print(df_xor.head())
    
#     # Decide which column is your target (adjust to your data)]
#     target_col = [x for x in df_xor.columns if x.startswith("Y_")]
    
#     # Gather all feature columns (exclude the target from the list)
#     feature_columns = [col for col in df_xor.columns if col.startswith("X")]
    
#     # -------------------------------------------------------------------------
#     # Compute IG for single features (optional, for comparison)
#     # -------------------------------------------------------------------------
#     print("\n--- Information Gain for in dividual features ---")
#     for feature_col in feature_columns:
#         ig_value = information_gain(df_xor, feature_col, target_col)
#         print(f"IG({feature_col} -> {target_col}) = {ig_value:.4f}")
    
#     # -------------------------------------------------------------------------
#     # Compute IG for all pairs of features
#     # -------------------------------------------------------------------------
#     print("\n--- Information Gain for pairs of features ---")
#     for colA, colB in itertools.combinations(feature_columns, 2):
#         # Combine colA and colB into a single "pair" feature
#         combined_col = f"{colA}_AND_{colB}"
#         df_xor[combined_col] = (
#             df_xor[colA].astype(str) 
#             + "_" + 
#             df_xor[colB].astype(str)
#         )
        
#         # Compute IG for this pair vs. target
#         ig_value = information_gain(df_xor, combined_col, target_col)
#         print(f"IG({colA}, {colB} -> {target_col}) = {ig_value:.4f}")
        
#         # Remove the combined column (to avoid clutter / ensure unique name next time)
#         df_xor.drop(columns=[combined_col], inplace=True)
