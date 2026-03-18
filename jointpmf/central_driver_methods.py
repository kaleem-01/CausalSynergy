import numpy as np
import random
import pandas as pd
import copy
import math
from pgmpy.models import BayesianNetwork as PGM_BN
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
# import hypernetx as hnx
import scipy.stats as ss
from itertools import combinations
from tqdm import tqdm
import sys
# sys.path.append('/courses/Thesis/Repository/src/jointpmf')  # Adjust the path as needed
import jointpmf.jointpmf as jp
from jointpmf.jointpmf import JointProbabilityMatrix

from IPython.display import clear_output
from sklearn.metrics import r2_score, mutual_info_score, normalized_mutual_info_score


def count_multi_rvs(df_data: pd.DataFrame):
    """To count how often a value occurs in each column.

    Args:
        df_data (pd.DataFrame): A pandas dataframe

    Returns:
        _type_: Frequency
    """
    rvs_count = {}
    for rvs in df_data.columns.values:
        unique, cnt = np.unique(df_data[rvs], return_counts=True)
        rvs_count[rvs] = cnt
        
    return rvs_count


def cal_shannon_ent(df_data: pd.DataFrame, base=2):
    """To calculate Shannon's entropy of each variable in a DataFrame.

    Args:
        df_data (pd.DataFrame): A Pandas dataframe
        base (int, optional): The logarithmic base to use. Defaults to 2.

    Returns:
        _type_: Shannon's entropy
    """
    var_pmfs = count_multi_rvs(df_data)
    shann_ent = []
    for rv_f in var_pmfs.keys():
        shann_ent.append(ss.entropy(var_pmfs[rv_f], base=base))

    return shann_ent


def cond_mutual_info(joint_pdf: JointProbabilityMatrix, var1: list, var2: list, given_vars: list):
    """Calculate H(X;Y|W;Z) = H(X|WZ) - H(X|YWZ)
    TODO: given any number of variables.

    Args:
        joint_pdf (JointProbabilityMatrix): JointProbabilityMatrix object
        var1 (_type_): variable 1
        var2 (_type_): variable 2
        given_vars (_type_): given variables

    Returns:
        _type_: H(X;Y|W;Z)
    """
    return joint_pdf.conditional_entropy(var1, given_vars) - joint_pdf.conditional_entropy(var1, var2+given_vars)


def cond_mi_mat(df: pd.DataFrame, given_varnames: list):
    """Calculate H(X;Y|W;Z) = H(X|WZ) - H(X|YWZ) for all variables in a DataFrame given TWO variables.
    TODO: given any number of variables.

    Args:
        df (pd.DataFrame): pd.DataFrame
        given_varnames (list): two given variables in a list

    Returns:
        _type_: H(X;Y|W;Z) matrix, diagonal elelments are set to 0.
    """
    cond_mi_size = df.shape[1] - len(given_varnames)
    cond_mi_mat = np.zeros([cond_mi_size, cond_mi_size])
    df_no_given = df.drop(columns=given_varnames)
    df_varnames_no_given = df_no_given.columns.values
    num_col_df_no_given = len(df_varnames_no_given)
    for ii in range(num_col_df_no_given):
        print(ii)
        var_name_ii = df_varnames_no_given[ii]
        for jj in range(num_col_df_no_given):
            if ii < jj:
                var_name_jj = df_varnames_no_given[jj]
                jointpdf_var_names_list = [var_name_ii, var_name_jj] + given_varnames
                jointpdf_data_list_list = df[jointpdf_var_names_list].values.tolist()
                pdf = JointProbabilityMatrix(2, 2)
                pdf.estimate_from_data(jointpdf_data_list_list)
                cond_mi_mat[ii, jj] = cond_mutual_info(pdf, [0], [1], [2, 3])
            else:
                cond_mi_mat[ii, jj] = 0
            
    return cond_mi_mat


def total_correlation(joint_pdf: JointProbabilityMatrix, variable_X, base=2) -> float:
    indiv_ents = [joint_pdf.entropy([xi], base=base) for xi in variable_X]
    total_ent = joint_pdf.entropy(variable_X, base=base)

    return sum(indiv_ents) - total_ent


def dual_total_correlation(joint_pdf: JointProbabilityMatrix, variable_X, base=2) -> float:
    total_ent = joint_pdf.entropy(variable_X, base=base)
    cond_ents = [joint_pdf.conditional_entropy([xi], [xj for xj in variable_X if xj != xi]) for xi in variable_X]  # ask Rick to add parameter "base" to functions in class "JointProbabilityMatrix"

    return total_ent - sum(cond_ents)


def o_information(joint_pdf: JointProbabilityMatrix, variable_X, base=2) -> float:
    return total_correlation(joint_pdf, variable_X, base=base) - dual_total_correlation(joint_pdf, variable_X, base=base)


def o_info_comb(df_data: pd.DataFrame, n_set: int):
    num_vars = df_data.shape[1]
    combs = list(combinations(range(num_vars), n_set))
    num_combs = len(combs)
    o_info = []
    for ii in range(num_combs):
        print(str(ii)+"/"+str(num_combs))
        data_subset = df_data.iloc[:, list(combs[ii])]
        data_subset_list_list = data_subset.values.tolist()  # dataframe to list of list.
        pdf = JointProbabilityMatrix(2, 2)
        pdf.estimate_from_data(data_subset_list_list)
        o_info.append(o_information(pdf, list(range(n_set))))

    o_info_dict = {'Combs': combs, 'O_info': o_info}
    o_info_combs = pd.DataFrame(o_info_dict)

    return o_info_combs


# Calculate MI using functions in sklearn.cluster.metrics
def mi_abs_norm(df_data:pd.DataFrame, base=2):
    """Calculate MI using functions in sklearn.cluster.metrics, default in bit.
    NOTE: No matter base is 2 or e, no need to convert "normalized_mutual_info_score",
    Because normalized MI = MI/entropy, normalized MI has no longer unit.
    In tutorial of this function saying natural base for normalized MI, that is misleading.

    Args:
        df_data (_type_): A pandas dataframe

    Returns:
        _type_: Absolute and normalized MI.
    """
    num_var = df_data.shape[1]
    abs_mi = np.zeros([num_var, num_var])
    norm_mi = np.zeros([num_var, num_var])
    for rv_true in range(num_var):
        label_true = list(df_data.iloc[:, rv_true])
        for rv_pred in range(num_var):
            if rv_pred > rv_true:
                label_pred = list(df_data.iloc[:, rv_pred])
                abs_mi[rv_true, rv_pred] = mutual_info_score(label_true, label_pred) / np.log(base)
                norm_mi[rv_true, rv_pred] = normalized_mutual_info_score(label_true, label_pred, average_method='min')

    return abs_mi, norm_mi


def mutual_info_abs_norm(df_data: pd.DataFrame, base=2):
    """Calculate MI using functions in sklearn.cluster.metrics
     Can be in either bit or nat, using base 2 or e.

    Args:
        df_data (pd.DataFrame): A pandas dataframe
        base (int, optional): The logarithmic base to use. Defaults to 2.

    Returns:
        _type_: Absolute and normalized MI.
    """
    var_pmfs = count_multi_rvs(df_data)
    vars = df_data.columns.values
    num_vars = len(vars)
    abs_mi = np.zeros([num_vars, num_vars])
    norm_mi = np.zeros([num_vars, num_vars])
    for rv_f in range(num_vars):
        ent_rv_f = ss.entropy(var_pmfs[vars[rv_f]], base=base)
        for rv_t in range(num_vars):
            if rv_t > rv_f:
                ent_rv_t = ss.entropy(var_pmfs[vars[rv_t]], base=base)
                ent_min = np.min([ent_rv_f, ent_rv_t])
                mi_val = mutual_info_score(list(df_data.iloc[:, rv_f]), list(df_data.iloc[:, rv_t])) / np.log(base)
                abs_mi[rv_f, rv_t] = mi_val
                norm_mi[rv_f, rv_t] = mi_val / ent_min
    
    return abs_mi, norm_mi



def mi_p_val_chi2(df: pd.DataFrame):
    """A approach to calculate p-values for MI correlation.

    Args:
        df (pd.DataFrame): A pandas dataframe

    Returns:
        _type_: Chi square p-value matrix
    """
    num_col = df.shape[1]
    p_val = np.zeros([num_col, num_col])
    for ii in range(num_col):
        for jj in range(num_col):
            if jj > ii:
                _, p_val[ii, jj], _, _ = ss.chi2_contingency(pd.crosstab(df.iloc[:, ii], df.iloc[:, jj]),
                                                             lambda_='log-likelihood')
            else:
                continue
    return p_val


def p_val_o_info(df_data: pd.DataFrame, n_set: int, boot_num: int):
    """To calculate O-info and its p-value.

    Args:
        df_data (pd.DataFrame): A pandas dataframe
        n_set (int): the size of combination, for example, 3 means triplet.
        boot_num (int): The number of bootstrapping.

    Returns:
        _type_: A dataframe storing p-values of combinations and their p-values.
    """
    num_vars = df_data.shape[1]
    combs = list(combinations(range(num_vars), n_set))
    num_combs = len(combs)
    o_info = []
    p_val = []
    for ii in range(num_combs):
        print(str(ii)+"/"+str(num_combs))
        data_subset = df_data.iloc[:, list(combs[ii])]
        data_subset_list_list = data_subset.values.tolist()  # dataframe to list of list.
        pdf = JointProbabilityMatrix(2, 2)
        pdf.estimate_from_data(data_subset_list_list)
        o_info_val = o_information(pdf, list(range(n_set)))
        o_info.append(o_info_val)

        # boostrapping
        boot_num_o_info = []
        for _ in range(boot_num):
            boot_vars_data = {}
            for jj in range(n_set):
                data_jj = np.array(data_subset.iloc[:, jj])
                boot_data_jj = np.random.choice(data_jj, size=len(data_jj), replace=True)
                boot_vars_data[str(jj)] = boot_data_jj
            boot_vars_data_df = pd.DataFrame(boot_vars_data)
            boot_vars_data_list_list = boot_vars_data_df.values.tolist()
            boot_pdf = JointProbabilityMatrix(2, 2)
            boot_pdf.estimate_from_data(boot_vars_data_list_list)
            boot_num_o_info.append(o_information(boot_pdf, list(range(n_set))))
        
        # p_value calculation
        boot_num_o_info_np = np.array(boot_num_o_info)
        num_extreme_positive = np.count_nonzero(boot_num_o_info_np >= abs(o_info_val))
        num_extreme_negative = np.count_nonzero(boot_num_o_info_np <= -abs(o_info_val))
        p_val.append((num_extreme_positive+num_extreme_negative)/boot_num)

    o_info_p_val_dict = {'Combs': combs, 'O_info': o_info, 'P_value': p_val}
    o_info_p_val_combs = pd.DataFrame(o_info_p_val_dict)

    return o_info_p_val_combs


# Old function, using JointProbabilityMatrix object, 
# it has problems with calculating impact, Rick solved the problem, see function "node_nudge_impact"
def node_impact_nudge(joint_pdf: JointProbabilityMatrix, perturbed_vars=None, eps_norm=0.01, method='invariant'):
    all_vars_index = list(range(joint_pdf.numvariables))
    if perturbed_vars == None:
        perturbed_vars = all_vars_index
    else:
        pass
    nodes_impact = {}
    for perturbed_var in perturbed_vars:
        joint_pdf_nudged = joint_pdf.copy()
        joint_pdf_nudged.nudge_single(perturbed=perturbed_var, eps_norm=eps_norm, method=method)
        hellinger_dist = 0
        all_vars_index_copy = all_vars_index.copy()
        all_vars_index_copy.remove(perturbed_var)
        not_perturbed_vars = all_vars_index_copy
        for not_perturbed_var in not_perturbed_vars:
            hellinger_dist += joint_pdf.marginalize_distribution([not_perturbed_var]).hellinger_distance(joint_pdf_nudged.marginalize_distribution([not_perturbed_var]))
        nodes_impact[str(perturbed_var)] = hellinger_dist
    
    return nodes_impact


def infer_bn_pair_triplet_probs(bn: jp.BayesianNetwork, numvars, numroots, numvals, pair_probs: float, 
                                cond_ent_ratio=0.5, pdf_type='uniform', max_opt_evals=100):
    """Generate a more custom BN where a mix of pairwise and triplet interactions take place with certain probabilities

    Args:
        bn (jp.BayesianNetwork): a BN object.
        numvars (_type_): number of variables to be generated.
        numroots (_type_): number of root variables in the BN.
        numvals (_type_): number of states of each variable in the BN.
        pair_probs (float): pair-higher ratio = number of pairwise interactions / all
        cond_ent_ratio (float, optional): should be between 0.0 and 1.0; the higher, the more correlated connected variables will be. Defaults to 0.5.
        pdf_type (str, optional): type of distribution of generated values. Defaults to 'uniform'.
        max_opt_evals (int, optional): the higher, the slower but the more accurate the strengths of the dependencies (quantified by nMI).
          Defaults to 100.
    """
    # a few independent variables to start with (root(s) of the DAG)
    if numroots == None or numroots == 0:
        print("No roots for your DAGs.")
    else:
        for _ in range(numroots):
            bn.append_independent_variable(pdf_type, numvals)
    
    dep_num_vars = numvars - numroots
    num_pairs = round(dep_num_vars * pair_probs)
    # num_triplets = dep_num_vars - num_pairs
    pair_tri_numerate = []
    if numroots == 1:
        while pair_tri_numerate.count(2) != num_pairs or pair_tri_numerate[0] != 2:
            pair_tri_numerate = list(np.random.choice(np.array([2,3]), dep_num_vars, p=[pair_probs, 1-pair_probs]))
    else:
        while pair_tri_numerate.count(2) != num_pairs:
            pair_tri_numerate = list(np.random.choice(np.array([2,3]), dep_num_vars, p=[pair_probs, 1-pair_probs]))
    
    for order in pair_tri_numerate:
        if order == 2:
            predecessors_new_appended = sorted(list(np.random.choice(bn.numvariables, size=order-1)))
            bn.append_dependent_variable(predecessors_new_appended, numvals, np.log2(numvals) * cond_ent_ratio)
        elif order == 3:
            predecessors_new_appended = sorted(list(np.random.choice(bn.numvariables, size=order-1, replace=False)))
            # bn.append_dependent_variable(predecessors_new_appended, numvals, np.log2(numvals) * cond_ent_ratio)
            bn.append_synergistic_variable(predecessors_new_appended, numvals, max_evals=max_opt_evals)
        else:
            pass



# to Check if the DAG is all connected.
def bn_is_connected(bn: jp.BayesianNetwork):
    """_summary_

    Args:
        bn (jp.BayesianNetwork): a BN object.

    Returns:
        _type_: Boolean
    """
    is_connected = None
    if bn.numvariables == 0:
        print("Warning: This is a null graph.")
        is_connected = False
    else:
        is_connected = nx.is_connected(bn.dependency_graph.to_undirected())
    return is_connected


def node_nudge_impact(bn: jp.BayesianNetwork, intervention_types, target_nudge_norm):
    """Node impact, quantified by Hellinger distance.

    Args:
        bn (jp.BayesianNetwork): a BN object.
        intervention_types (_type_): intervention types: "hard" or "soft"
        target_nudge_norm (_type_): the smaller, the less likely you will get into errors because of going out of bounds (a probability <0 or >1)

    Returns:
        _type_: Hellinger distance
    """
    df_nudge_impacts = pd.DataFrame(columns=['nudged_variable', 'hellinger_distance_target_only', 'hellinger_distance_total', 'hellinger_distance_total_normalized',
                                    'hellinger_distance_descendants_only', 'hellinger_distance_descendants_only_normalized', 'type'])
    # go through the intervention types (hard=variable is replaced by its marginal, losing all its causal predecessors)
    for type in intervention_types:
        print(f'----- intervention type {type} -----')
        # go through all nodes
        for varix in range(len(bn)):
            bn_nudged = copy.deepcopy(bn)  # prevent changing the original `bn` because .nudge() acts in-place
            nudge_vec = bn_nudged.nudge(varix, target_nudge_norm, type=type)
            print(f'\t{nudge_vec=} was added to the (possibly conditional) probabilities of variable {varix}; it has norm {np.linalg.norm(nudge_vec)} and target norm was {target_nudge_norm}')
            marginal_nudged_bn = bn_nudged.marginal_pmf([varix])
            marginal_original_bn = bn.marginal_pmf([varix])
            # diffs_vec = marginal_nudged_bn.pdfs[0].joint_probabilities.joint_probabilities - marginal_original_bn.pdfs[0].joint_probabilities.joint_probabilities
            hellinger_distances_nudge_target_only = marginal_nudged_bn.hellinger_distance(marginal_original_bn)
            hellinger_distance_total = bn.hellinger_distance(bn_nudged)  # NOTE: this calculation is very slow
            num_descendants = len(sorted(nx.descendants(bn.dependency_graph, varix)))
            hellinger_distance_total_normalized = hellinger_distance_total / float(num_descendants + 1)  # NOTE: the node itself is also counted here
            if num_descendants > 0:
                bn_desc_orig = bn.marginal_pmf(sorted(nx.descendants(bn.dependency_graph, varix)))
                bn_desc_nudged = bn_nudged.marginal_pmf(sorted(nx.descendants(bn_nudged.dependency_graph, varix)))
                hellinger_distance_descendants_only = bn_desc_orig.hellinger_distance(bn_desc_nudged)
                hellinger_distance_descendants_only_normalized = hellinger_distance_descendants_only / num_descendants
            else:
                hellinger_distance_descendants_only = 0.0
                hellinger_distance_descendants_only_normalized = 0.0

            df_nudge_impacts.loc[len(df_nudge_impacts)] = {'nudged_variable': varix, 
                                                           'hellinger_distance_target_only': hellinger_distances_nudge_target_only, 
                                                           'hellinger_distance_total': hellinger_distance_total, 
                                                           'hellinger_distance_total_normalized': hellinger_distance_total_normalized, 
                                                           'hellinger_distance_descendants_only': hellinger_distance_descendants_only, 
                                                           'hellinger_distance_descendants_only_normalized': hellinger_distance_descendants_only_normalized, 
                                                           'type': type}
    
    return df_nudge_impacts


def node_nudge_impact_simplified(bn: jp.BayesianNetwork, intervention_types, target_nudge_norm):
    """Node impact, quantified by Hellinger distance.

    Args:
        bn (jp.BayesianNetwork): a BN object.
        intervention_types (_type_): intervention types: "hard" or "soft"
        target_nudge_norm (_type_): the smaller, the less likely you will get into errors because of going out of bounds (a probability <0 or >1)

    Returns:
        _type_: Hellinger distance
    """
    df_nudge_impacts = pd.DataFrame(columns=['nudged_variable', 'hellinger_distance_target_only', 'hellinger_distance_descendants_only', 
                                             'hellinger_distance_descendants_only_normalized', 'type'])
    # go through the intervention types (hard=variable is replaced by its marginal, losing all its causal predecessors)
    for type in intervention_types:
        print(f'----- intervention type {type} -----')
        # go through all nodes
        for varix in range(len(bn)):
            descendants_varix = sorted(nx.descendants(bn.dependency_graph, varix))
            num_descendants = len(descendants_varix)

            if num_descendants > 0:
                bn_nudged = copy.deepcopy(bn)  # prevent changing the original `bn` because .nudge() acts in-place
                nudge_vec = bn_nudged.nudge(varix, target_nudge_norm, type=type)
                print(f'\t{nudge_vec=} was added to the (possibly conditional) probabilities of variable {varix}; it has norm {np.linalg.norm(nudge_vec)} and target norm was {target_nudge_norm}')
                
                marginal_nudged_bn = bn_nudged.marginal_pmf([varix])
                marginal_original_bn = bn.marginal_pmf([varix])
                hellinger_distances_nudge_target_only = marginal_nudged_bn.hellinger_distance(marginal_original_bn)
            
                bn_desc_orig = bn.marginal_pmf(descendants_varix)
                bn_desc_nudged = bn_nudged.marginal_pmf(descendants_varix)
                hellinger_distance_descendants_only = bn_desc_orig.hellinger_distance(bn_desc_nudged)
                hellinger_distance_descendants_only_normalized = hellinger_distance_descendants_only / num_descendants
            else:
                hellinger_distance_descendants_only = 0.0
                hellinger_distance_descendants_only_normalized = 0.0

            df_nudge_impacts.loc[len(df_nudge_impacts)] = {'nudged_variable': varix, 
                                                           'hellinger_distance_target_only': hellinger_distances_nudge_target_only, 
                                                           'hellinger_distance_descendants_only': hellinger_distance_descendants_only, 
                                                           'hellinger_distance_descendants_only_normalized': hellinger_distance_descendants_only_normalized, 
                                                           'type': type}
    
    return df_nudge_impacts


def node_pin_impact_simplified(bn: jp.BayesianNetwork, pin_state=None, pin_shift=0):
    """_summary_: Marginalize a variable (no incoming edges anymore) and let it have a single state with 100% probability.
    To simplify, the calculation of "hellinger_distance_pinned_total" is canceled.

    Args:
        bn (jp.BayesianNetwork): _description_
        pin_state (_type_, optional): _description_. If desired, set the state of the variable always to this value. If not
             specified then a random choice is being made, weighted by the pre-intervention probabilities. 
             Defaults to None.
        pin_shift (int, optional): _description_. Defaults to 0. shift=1 means that if the current symptom is on (1) then it will be set to off (0) or vice versa (for binary symptoms)

    Returns:
        _type_: _description_: A DataFrame saving various hellinger distances.
    """
    df_pin_impacts = pd.DataFrame(columns=['pinned_variable', 'hellinger_distances_pinned_target_only', 'hellinger_distance_pinned_total', 
                                           'hellinger_distance_pinned_total_normalized', 'hellinger_distance_pinned_descendants_only', 
                                           'hellinger_distance_pinned_descendants_only_normalized'])
    for varix in range(len(bn)):
        descendants_varix = sorted(nx.descendants(bn.dependency_graph, varix))
        num_descendants = len(descendants_varix)

        if num_descendants > 0:
            bn_pinned = copy.deepcopy(bn)
            bn_pinned.pin(varix, shift=pin_shift)
            marginal_pinned_bn = bn_pinned.marginal_pmf([varix])
            marginal_original_bn = bn.marginal_pmf([varix])

            hellinger_distances_pinned_target_only = marginal_pinned_bn.hellinger_distance(marginal_original_bn)
            # hellinger_distance_pinned_total = bn.hellinger_distance(bn_pinned)  # NOTE: this calculation is very slow
            hellinger_distance_pinned_total = 0  # Problem with calculating Hellinger distance for this. Temporarily set it to zero.
            hellinger_distance_pinned_total_normalized = hellinger_distance_pinned_total / float(num_descendants + 1)  # NOTE: the node itself is also counted here

            bn_desc_orig = bn.marginal_pmf(descendants_varix)
            bn_desc_pinned = bn_pinned.marginal_pmf(descendants_varix)
            hellinger_distance_pinned_descendants_only = bn_desc_orig.hellinger_distance(bn_desc_pinned)
            hellinger_distance_pinned_descendants_only_normalized = hellinger_distance_pinned_descendants_only / num_descendants
        else:
            hellinger_distance_pinned_descendants_only = 0.0
            hellinger_distance_pinned_descendants_only_normalized = 0.0
        
        df_pin_impacts.loc[len(df_pin_impacts)] = {'pinned_variable': varix, 
                                                    'hellinger_distances_pinned_target_only': hellinger_distances_pinned_target_only, 
                                                    'hellinger_distance_pinned_total': hellinger_distance_pinned_total, 
                                                    'hellinger_distance_pinned_total_normalized': hellinger_distance_pinned_total_normalized, 
                                                    'hellinger_distance_pinned_descendants_only': hellinger_distance_pinned_descendants_only, 
                                                    'hellinger_distance_pinned_descendants_only_normalized': hellinger_distance_pinned_descendants_only_normalized}
        
    return df_pin_impacts


def node_n_nudge_impact_simplified(bn: jp.BayesianNetwork, intervention_types, target_nudge_norm, n_nudge=1):
    """Node impact, quantified by Hellinger distance.

    Args:
        bn (jp.BayesianNetwork): a BN object.
        intervention_types (_type_): intervention types: "hard" or "soft"
        target_nudge_norm (_type_): the smaller, the less likely you will get into errors because of going out of bounds (a probability <0 or >1)
        n_nudge: times to nudge each node.

    Returns:
        _type_: Hellinger distance
    """
    df_nudge_impacts = pd.DataFrame(columns=['nudged_variable', 
                                             'hellinger_distance_target_only_n', 
                                             'hellinger_distance_target_only', 
                                             'hellinger_distance_descendants_only_n',
                                             'hellinger_distance_descendants_only', 
                                             'hellinger_distance_descendants_only_normalized_n', 
                                             'hellinger_distance_descendants_only_normalized', 
                                             'type'])
    # go through the intervention types (hard=variable is replaced by its marginal, losing all its causal predecessors)
    for type in intervention_types:
        print(f'----- intervention type {type} -----')
        # go through all nodes
        for varix in tqdm(range(len(bn))):
            descendants_varix = sorted(nx.descendants(bn.dependency_graph, varix))
            num_descendants = len(descendants_varix)
            hellinger_distances_nudge_target_only_n = []
            hellinger_distance_descendants_only_n = []
            hellinger_distance_descendants_only_normalized_n = []

            marginal_original_bn = bn.marginal_pmf([varix])
            bn_desc_orig = bn.marginal_pmf(descendants_varix)

            if num_descendants > 0:
                for _ in (range(n_nudge)):
                    print(f"Nudging variable {varix}, nudge number {_} of {n_nudge}")
                    bn_nudged = copy.deepcopy(bn)  # prevent changing the original `bn` because .nudge() acts in-place
                    nudge_vec = bn_nudged.nudge(varix, target_nudge_norm, type=type)
                    # print(f'\t{nudge_vec=} was added to the (possibly conditional) probabilities of variable {varix}; it has norm {np.linalg.norm(nudge_vec)} and target norm was {target_nudge_norm}')
                    
                    # Calculate HD (hellinger distance) for target only (Optional)
                    marginal_nudged_bn = bn_nudged.marginal_pmf([varix])
                    hellinger_distances_nudge_target_only_each = marginal_nudged_bn.hellinger_distance(marginal_original_bn)
                    hellinger_distances_nudge_target_only_n.append(hellinger_distances_nudge_target_only_each)            

                    # Calculate HD (hellinger distance) for all descendants 
                    bn_desc_nudged = bn_nudged.marginal_pmf(descendants_varix)
                    hellinger_distance_descendants_only_each = bn_desc_orig.hellinger_distance(bn_desc_nudged)
                    hellinger_distance_descendants_only_n.append(hellinger_distance_descendants_only_each)
                    hellinger_distance_descendants_only_normalized_each = hellinger_distance_descendants_only_each / num_descendants
                    hellinger_distance_descendants_only_normalized_n.append(hellinger_distance_descendants_only_normalized_each)

                    if _%10==1:
                        print(f"Completed {_} of {n_nudge} nudges in a network")
            else:
                hellinger_distances_nudge_target_only_n.append(0)
                hellinger_distance_descendants_only_n.append(0)
                hellinger_distance_descendants_only_normalized_n.append(0)

            hellinger_distances_nudge_target_only = np.mean(hellinger_distances_nudge_target_only_n)
            hellinger_distance_descendants_only = np.mean(hellinger_distance_descendants_only_n)
            hellinger_distance_descendants_only_normalized = np.mean(hellinger_distance_descendants_only_normalized_n)
            df_nudge_impacts.loc[len(df_nudge_impacts)] = {'nudged_variable': varix, 
                                                           'hellinger_distance_target_only_n': hellinger_distances_nudge_target_only_n,
                                                           'hellinger_distance_target_only': hellinger_distances_nudge_target_only, 
                                                           'hellinger_distance_descendants_only_n': hellinger_distance_descendants_only_n,
                                                           'hellinger_distance_descendants_only': hellinger_distance_descendants_only, 
                                                           'hellinger_distance_descendants_only_normalized_n': hellinger_distance_descendants_only_normalized_n, 
                                                           'hellinger_distance_descendants_only_normalized': hellinger_distance_descendants_only_normalized, 
                                                           'type': type}
    
    return df_nudge_impacts


def node_n_pin_impact_simplified(bn: jp.BayesianNetwork, pin_state=None, pin_shift=0, n_pin=1):
    """_summary_: Marginalize a variable (no incoming edges anymore) and let it have a single state with 100% probability.
    To simplify, the calculation of "hellinger_distance_pinned_total" is canceled.

    Args:
        bn (jp.BayesianNetwork): _description_
        pin_state (_type_, optional): _description_. If desired, set the state of the variable always to this value. If not
             specified then a random choice is being made, weighted by the pre-intervention probabilities. 
             Defaults to None.
        pin_shift (int, optional): _description_. Defaults to 0. shift=1 means that if the current symptom is on (1) then it will be set to off (0) or vice versa (for binary symptoms)

    Returns:
        _type_: _description_: A DataFrame saving various hellinger distances.
    """
    df_pin_impacts = pd.DataFrame(columns=['pinned_variable', 
                                           'hellinger_distances_pinned_target_only_n',
                                           'hellinger_distances_pinned_target_only',
                                           'hellinger_distance_pinned_descendants_only_n',
                                           'hellinger_distance_pinned_descendants_only',
                                           'hellinger_distance_pinned_descendants_only_normalized_n', 
                                           'hellinger_distance_pinned_descendants_only_normalized'])
    for varix in range(len(bn)):
        descendants_varix = sorted(nx.descendants(bn.dependency_graph, varix))
        num_descendants = len(descendants_varix)
        hellinger_distances_pinned_target_only_n = []
        hellinger_distance_pinned_descendants_only_n = []
        hellinger_distance_pinned_descendants_only_normalized_n = []

        marginal_original_bn = bn.marginal_pmf([varix])
        bn_desc_orig = bn.marginal_pmf(descendants_varix)

        if num_descendants > 0:
            for _ in range(n_pin):
                bn_pinned = copy.deepcopy(bn)
                bn_pinned.pin(varix, shift=pin_shift)
                marginal_pinned_bn = bn_pinned.marginal_pmf([varix])
                
                hellinger_distances_pinned_target_only_each = marginal_pinned_bn.hellinger_distance(marginal_original_bn)
                hellinger_distances_pinned_target_only_n.append(hellinger_distances_pinned_target_only_each)

                # hellinger_distance_pinned_total = bn.hellinger_distance(bn_pinned)  # NOTE: this calculation is very slow
                # hellinger_distance_pinned_total = 0  # Problem with calculating Hellinger distance for this. Temporarily set it to zero.
                # hellinger_distance_pinned_total_normalized = hellinger_distance_pinned_total / float(num_descendants + 1)  # NOTE: the node itself is also counted here

                bn_desc_pinned = bn_pinned.marginal_pmf(descendants_varix)
                hellinger_distance_pinned_descendants_only_each = bn_desc_orig.hellinger_distance(bn_desc_pinned)
                hellinger_distance_pinned_descendants_only_n.append(hellinger_distance_pinned_descendants_only_each)
                hellinger_distance_pinned_descendants_only_normalized_each = hellinger_distance_pinned_descendants_only_each / num_descendants
                hellinger_distance_pinned_descendants_only_normalized_n.append(hellinger_distance_pinned_descendants_only_normalized_each)
        else:
            hellinger_distances_pinned_target_only_n.append(0)
            hellinger_distance_pinned_descendants_only_n.append(0)
            hellinger_distance_pinned_descendants_only_normalized_n.append(0)
        
        hellinger_distances_pinned_target_only = np.mean(hellinger_distances_pinned_target_only_n)
        hellinger_distance_pinned_descendants_only = np.mean(hellinger_distance_pinned_descendants_only_n)
        hellinger_distance_pinned_descendants_only_normalized = np.mean(hellinger_distance_pinned_descendants_only_normalized_n)
        
        df_pin_impacts.loc[len(df_pin_impacts)] = {'pinned_variable': varix, 
                                                   'hellinger_distances_pinned_target_only_n': hellinger_distances_pinned_target_only_n,
                                                    'hellinger_distances_pinned_target_only': hellinger_distances_pinned_target_only,
                                                    'hellinger_distance_pinned_descendants_only_n': hellinger_distance_pinned_descendants_only_n,
                                                    'hellinger_distance_pinned_descendants_only': hellinger_distance_pinned_descendants_only, 
                                                    'hellinger_distance_pinned_descendants_only_normalized_n': hellinger_distance_pinned_descendants_only_normalized_n,
                                                    'hellinger_distance_pinned_descendants_only_normalized': hellinger_distance_pinned_descendants_only_normalized}
        
    return df_pin_impacts


def node_nudge_direct_impact_simplify_old(bn: jp.BayesianNetwork, intervention_types, target_nudge_norm):
    """_summary_: To calculate the impact of intervention on direct descendants in the BN. Jie on Dec. 3 2024

    Args:
        bn (jp.BayesianNetwork): _description_ Bayesian Network
        pin_shift (int, optional): _description_. Defaults to 0. Defaults to 0. shift=1 means that if the current symptom is on (1) then it will be set to off (0) or vice versa (for binary symptoms)

    Returns:
        _type_: _description_ A DataFrame saving various hellinger distances.
    """

    df_nudge_direct_impacts = pd.DataFrame(columns=['nudged_variable', 'hellinger_distance_nudged_direct_children', 
                                                    'hellinger_distance_nudged_direct_children_normalized'])
    BN_model = PGM_BN(bn.dependency_graph.edges)

    for type in intervention_types:
        print(f'----- intervention type {type} -----')
        
        # go through all nodes
        for varix in range(len(bn)):
            bn_nudged = copy.deepcopy(bn)  # prevent changing the original `bn` because .nudge() acts in-place
            nudge_vec = bn_nudged.nudge(varix, target_nudge_norm, type=type)
            print(f'\t{nudge_vec=} was added to the (possibly conditional) probabilities of variable {varix}; it has norm {np.linalg.norm(nudge_vec)} and target norm was {target_nudge_norm}')
            varix_children = BN_model.get_children(varix)
            hellinger_distance_nudged_children = []
            if len(varix_children) > 0:
                for varix_child in varix_children:
                    bn_child_orig = bn.marginal_pmf([varix_child])
                    bn_child_nudged = bn_nudged.marginal_pmf([varix_child])
                    hellinger_distance_nudged_children.append(bn_child_orig.hellinger_distance(bn_child_nudged))
            else:
                hellinger_distance_nudged_children.append(0)
            
            df_nudge_direct_impacts.loc[len(df_nudge_direct_impacts)] = {'nudged_variable': varix, 
                                                                         'hellinger_distance_nudged_direct_children': np.sum(hellinger_distance_nudged_children), 
                                                                         'hellinger_distance_nudged_direct_children_normalized': np.mean(hellinger_distance_nudged_children)}
        
    return df_nudge_direct_impacts


def node_nudge_direct_impact_simplify(bn: jp.BayesianNetwork, intervention_types, target_nudge_norm):
    """_summary_: To calculate the impact of intervention on direct descendants in the BN. Jie on Dec. 3 2024
    Improved on Dec. 4, Cillian's suggestion.
    1. calculate the marginal pmf for non-nudged BN for once, no repeat.
    2. Only do the nudge for nodes with children, not for those without children.        

    Args:
        bn (jp.BayesianNetwork): _description_ Bayesian Network
        pin_shift (int, optional): _description_. Defaults to 0. Defaults to 0. shift=1 means that if the current symptom is on (1) then it will be set to off (0) or vice versa (for binary symptoms)

    Returns:
        _type_: _description_ A DataFrame saving various hellinger distances.
    """

    df_nudge_direct_impacts = pd.DataFrame(columns=['nudged_variable', 'hellinger_distance_nudged_direct_children', 
                                                    'hellinger_distance_nudged_direct_children_normalized'])
    BN_model = PGM_BN(bn.dependency_graph.edges)

    original_marginal_varix_all = {}
    for varix in range(len(bn)):
        original_marginal_varix_all[varix] = bn.marginal_pmf([varix])

    for type in intervention_types:
        print(f'----- intervention type {type} -----')
        # go through all nodes
        for varix in range(len(bn)):
            varix_children = BN_model.get_children(varix)
            hellinger_distance_nudged_children = []
            if len(varix_children) > 0:
                bn_nudged = copy.deepcopy(bn)  # prevent changing the original `bn` because .nudge() acts in-place
                nudge_vec = bn_nudged.nudge(varix, target_nudge_norm, type=type)
                print(f'\t{nudge_vec=} was added to the (possibly conditional) probabilities of variable {varix}; it has norm {np.linalg.norm(nudge_vec)} and target norm was {target_nudge_norm}')
                for varix_child in varix_children:
                    bn_child_orig = original_marginal_varix_all[varix_child]
                    bn_child_nudged = bn_nudged.marginal_pmf([varix_child])
                    hellinger_distance_nudged_children.append(bn_child_orig.hellinger_distance(bn_child_nudged))
            else:
                hellinger_distance_nudged_children.append(0)
            
            df_nudge_direct_impacts.loc[len(df_nudge_direct_impacts)] = {'nudged_variable': varix, 
                                                                         'hellinger_distance_nudged_direct_children': np.sum(hellinger_distance_nudged_children), 
                                                                         'hellinger_distance_nudged_direct_children_normalized': np.mean(hellinger_distance_nudged_children)}
        
    return df_nudge_direct_impacts


def node_pin_direct_impact_simplify_old(bn: jp.BayesianNetwork, pin_shift=0):
    """_summary_: To calculate the impact of intervention on direct descendants in the BN. Jie on Dec. 3 2024

    Args:
        bn (jp.BayesianNetwork): _description_ Bayesian Network
        pin_shift (int, optional): _description_. Defaults to 0. Defaults to 0. shift=1 means that if the current symptom is on (1) then it will be set to off (0) or vice versa (for binary symptoms)

    Returns:
        _type_: _description_ A DataFrame saving various hellinger distances.
    """

    df_pin_direct_impacts = pd.DataFrame(columns=['pinned_variable', 'hellinger_distance_pinned_direct_children', 
                                                  'hellinger_distance_pinned_direct_children_normalized'])
    BN_model = PGM_BN(bn.dependency_graph.edges)

    for varix in range(len(bn)):
        bn_pinned = copy.deepcopy(bn)
        bn_pinned.pin(varix, shift=pin_shift)
        varix_children = BN_model.get_children(varix)
        hellinger_distance_pinned_children = []
        if len(varix_children) > 0:
            for varix_child in varix_children:
                bn_child_orig = bn.marginal_pmf([varix_child])
                bn_child_pinned = bn_pinned.marginal_pmf([varix_child])
                hellinger_distance_pinned_children.append(bn_child_orig.hellinger_distance(bn_child_pinned))
        else:
            hellinger_distance_pinned_children.append(0)
        
        df_pin_direct_impacts.loc[len(df_pin_direct_impacts)] = {'pinned_variable': varix, 
                                                                 'hellinger_distance_pinned_direct_children': np.sum(hellinger_distance_pinned_children), 
                                                                 'hellinger_distance_pinned_direct_children_normalized': np.mean(hellinger_distance_pinned_children)}
    
    return df_pin_direct_impacts


def node_pin_direct_impact_simplify(bn: jp.BayesianNetwork, pin_shift=0):
    """_summary_: To calculate the impact of intervention on direct descendants in the BN. Jie on Dec. 3 2024
    Improved on Dec. 4, Cillian's suggestion.
    1. calculate the marginal pmf for non-nudged BN for once, no repeat.
    2. Only do the nudge for nodes with children, not for those without children.   

    Args:
        bn (jp.BayesianNetwork): _description_ Bayesian Network
        pin_shift (int, optional): _description_. Defaults to 0. Defaults to 0. shift=1 means that if the current symptom is on (1) then it will be set to off (0) or vice versa (for binary symptoms)

    Returns:
        _type_: _description_ A DataFrame saving various hellinger distances.
    """

    df_pin_direct_impacts = pd.DataFrame(columns=['pinned_variable', 'hellinger_distance_pinned_direct_children', 
                                                  'hellinger_distance_pinned_direct_children_normalized'])
    BN_model = PGM_BN(bn.dependency_graph.edges)

    original_marginal_varix_all = {}
    for varix in range(len(bn)):
        original_marginal_varix_all[varix] = bn.marginal_pmf([varix])

    for varix in range(len(bn)):
        varix_children = BN_model.get_children(varix)
        hellinger_distance_pinned_children = []
        if len(varix_children) > 0:
            bn_pinned = copy.deepcopy(bn)
            bn_pinned.pin(varix, shift=pin_shift)
            for varix_child in varix_children:
                bn_child_orig = original_marginal_varix_all[varix_child]
                bn_child_pinned = bn_pinned.marginal_pmf([varix_child])
                hellinger_distance_pinned_children.append(bn_child_orig.hellinger_distance(bn_child_pinned))
        else:
            hellinger_distance_pinned_children.append(0)
        
        df_pin_direct_impacts.loc[len(df_pin_direct_impacts)] = {'pinned_variable': varix, 
                                                                 'hellinger_distance_pinned_direct_children': np.sum(hellinger_distance_pinned_children), 
                                                                 'hellinger_distance_pinned_direct_children_normalized': np.mean(hellinger_distance_pinned_children)}
    
    return df_pin_direct_impacts


def norm_dict(dict_var: dict):
    """Normalize dictionary values

    Args:
        dict_var (dict): A dict

    Returns:
        _type_: A normalized dict
    """
    values = dict_var.values()
    max_ = max(values)
    min_ = min(values)
    if max_ == min_:
        norm_d = dict_var  # If the min is identical to max, then do not do norm.. 
    else:
        norm_d = {key: ((val - min_) / (max_ - min_)) for (key, val) in dict_var.items()}
    return norm_d


# def weighted_node_degree_hnx(H: hnx.Hypergraph):
#     """Weighted node degree in hypergraphs.

#     Args:
#         H (hnx.Hypergraph): a hypernetx.Hypergraph object.

#     Returns:
#         _type_: a dict storing weighted degree of nodes
#     """
#     weighted_degree = {}
#     for node in H.nodes:
#         wei_degree = 0
#         for edge in H.edges:
#             if node in H.edges[edge]:
#                 wei_degree = wei_degree + H.edges[edge].weight
#         weighted_degree[node] = wei_degree
#     return weighted_degree


# def adj_weighted_line_graph(H: hnx.Hypergraph):
#     """Calculate adjacency matrix for weighted line graph of dual hypergraph.

#     Args:
#         H (hnx.Hypergraph): a hypernetx.Hypergraph object.

#     Returns:
#         _type_: Adjacency matrix saved as a dataframe for weighted line graph of hypergraph
#     """
#     n_hy_edges = len(H.edges)
#     hy_edges = H.edges
#     line_net_adj = np.zeros([n_hy_edges, n_hy_edges])
#     df_line_adj = pd.DataFrame(line_net_adj, index=hy_edges, columns=hy_edges)
#     for h_edge_ii in hy_edges:
#         edge_ii = hy_edges[h_edge_ii]  # edge detail. h_edge_ii is only the name of the edge.
#         order_edge_ii = len(edge_ii)
#         weight_edge_ii = edge_ii.weight
#         for h_edge_jj in hy_edges:
#             if h_edge_ii != h_edge_jj:
#                 edge_jj = hy_edges[h_edge_jj]
#                 order_edge_jj = len(edge_jj)
#                 weight_edge_jj = edge_jj.weight
#                 num_share_node = len(set(edge_ii).intersection(set(edge_jj)))
#                 if num_share_node > 0:
#                     df_line_adj.at[h_edge_ii, h_edge_jj] = ((weight_edge_ii * num_share_node) / order_edge_ii +
#                                                             (weight_edge_jj * num_share_node) / order_edge_jj) / 2

#     return df_line_adj


def node_vector_centrality_hypergraph(H, weight='unweighted'):
    """Vector Centrality in Hypergraphs. https://doi.org/10.1016/j.chaos.2022.112397
    The sum of ALL values in the resulting vector should be equal to the sum of eigenvector centrality of all nodes in line graph.

    Args:
        H (_type_): HyperNetX Hypergraph
        weight (str, optional): weighted or unweighted line graph to construct. Defaults to 'unweighted'.

    Returns:
        _type_: Vector centrality of nodes in the hypergraph H, dict
    """
    n_hy_nodes = len(H.nodes)
    hyper_edges = H.edges

    # weighted and unweighted line graph, and node eigenvector centrality in "line graph"
    if weight == 'weight':
        df_line_adj = adj_weighted_line_graph(H)
        line_G = nx.from_pandas_adjacency(df_line_adj)
        eig_centrality_line_graph = nx.eigenvector_centrality(line_G, weight='weight')
    else:
        df_line_adj = adj_weighted_line_graph(H)
        line_G = nx.from_pandas_adjacency(df_line_adj)
        eig_centrality_line_graph = nx.eigenvector_centrality(line_G)

    # vector centrality of nodes in "hypergraph"
    v_centrality = {}
    for node_hy_net in H.nodes:
        node_cen_vector = [0] * (n_hy_nodes - 1)
        for h_edge in hyper_edges:
            edge = hyper_edges[h_edge]  # edge detail. h_edge is only the name of the edge.
            if node_hy_net in edge:
                order_edge = len(edge)
                c_h = eig_centrality_line_graph[h_edge] / order_edge
                node_cen_vector[order_edge - 2] = node_cen_vector[order_edge - 2] + c_h
            else:
                continue
        v_centrality[node_hy_net] = node_cen_vector

    return v_centrality, eig_centrality_line_graph


def draw_pdf_columns_subplot(df: pd.DataFrame, num_per_row: int, fig_size: tuple, num_bins='auto'):
    """plot pdf of columns in a dataframe.

    Args:
        df (pd.DataFrame): a Pandas dataframe
        num_per_row (int): number of rows in subplot.
        fig_size (tuple): the size of the figure to be created.
        num_bins (str, optional): number of bins to use for pdf estimation. Defaults to 'auto'.

    Returns:
        _type_: subplots showing pdf of each column variable.
    """
    num_cols = len(df.columns)
    num_rows = int(np.ceil(num_cols / num_per_row))
    fig, axes = plt.subplots(num_rows, num_per_row, figsize=fig_size)
    for ii in range(num_cols):
        pos_index_x = int(np.floor(ii / num_per_row))
        pos_index_y = np.mod(ii, num_per_row)
        hist_plot = sns.histplot(df.iloc[:, ii], ax=axes[pos_index_x, pos_index_y], bins=num_bins)
        hist_plot.set_xlabel(df.columns[ii], fontdict={'fontsize': 20})
        hist_plot.set_ylabel('Frequency', fontdict={'fontsize': 20})
        hist_plot.tick_params(labelsize=18)
    
    return fig, axes


def draw_reg_scatter_subplot(df1: pd.DataFrame, df2: pd.DataFrame, df1_col_name: str, 
                             num_per_row: int, fig_size: tuple, 
                             fit_line=False, x_in_log=False, y_in_log=False):
    """Plot regression figures.

    Args:
        df1 (pd.DataFrame): a Pandas dataframe
        df2 (pd.DataFrame): a Pandas dataframe
        df1_col_name (str): a column name in df1.
        num_per_row (int): number of rows in subplot.
        fig_size (tuple): the size of the figure to be created.

    Returns:
        _type_: subplots showing scatters and regression plot of each comparison.
    """
    df1_col_target = pd.DataFrame(df1[df1_col_name]) 
    df = pd.concat([df2, df1_col_target], axis=1)
    num_col_df2 = len(df2.columns)
    num_rows = int(np.ceil(num_col_df2 / num_per_row))
    fig, axes = plt.subplots(num_rows, num_per_row, figsize=fig_size)

    for ii in range(num_col_df2):
        df2_col_name = df2.columns.values[ii]
        pos_index_x = int(np.floor(ii / num_per_row))
        pos_index_y = np.mod(ii, num_per_row)
        reg_linear = sns.regplot(df, x=df2_col_name, y=df1_col_name, ax=axes[pos_index_x, pos_index_y], fit_reg=fit_line)
        if x_in_log:
            reg_linear.set(xscale="log")
        if y_in_log:
            reg_linear.set(yscale="log")
        reg_linear.tick_params(labelsize=18)
        # reg_linear.set_ylim([-0.1, 2.1])
        reg_linear.set_xlabel(df2_col_name, fontdict={'fontsize': 20})
        reg_linear.set_ylabel(df1_col_name, fontdict={'fontsize': 20})
        # slope, intercept, r, p, sterr = ss.linregress(x=reg_linear.get_lines()[0].get_xdata(), y=reg_linear.get_lines()[0].get_ydata()) 
        # axes[pos_index_x, pos_index_y].text(x=text_pos_x, y=text_pos_y, fontsize=18,
        #                                     s='y = ' + str(round(intercept, 2)) + ' + ' + str(round(slope, 2)) + 'x')
    
    return fig, axes


def pairwise_adj_2_edge_df(adj_mat):
    """Adjacency matrix to edge list in a DataFrame

    Args:
        adj_mat (_type_): Adjacency matrix for a pairwise network

    Returns:
        _type_: A DataFrame saving pairwise edges
    """
    edge_list = []
    edge_weight_list = []
    for x_idx in range(len(adj_mat)):
        for y_idx in range(len(adj_mat)):
            edge_value = adj_mat[x_idx, y_idx]
            if (x_idx < y_idx) and (edge_value > 0):
                edge_list.append((x_idx, y_idx))
                edge_weight_list.append(edge_value)
            else:
                continue
    edge_dict = {"Pairs": edge_list, "Value": edge_weight_list}
    edge_df = pd.DataFrame(edge_dict)
    return edge_df


def rescaling_array_based_on_other(arr_to_be_rescaled: np.array, the_other: np.array):
    """Rescaling the range of an array based on the range of the other array

    Args:
        arr_to_be_rescaled (np.array): the array to be rescaled
        the_other (np.array): the array on which the rescaling is based.

    Returns:
        _type_: the rescaled array
    """
    desired_max = np.max(the_other)
    desired_min = np.min(the_other)
    origin_max = np.max(arr_to_be_rescaled)
    origin_min = np.min(arr_to_be_rescaled)
    arr_rescaled = (arr_to_be_rescaled-origin_min) * (desired_max-desired_min)/(origin_max-origin_min) + desired_min
    return arr_rescaled


def corr_df_bar_plot(df_node_impact: pd.DataFrame, df_centrality: pd.DataFrame, compared_col: str, corr_method='pearson', in_log=False):
    """Correlation dataframe preparation for bar plot visualization.

    Args:
        df_node_impact (pd.DataFrame): pd.DataFrame saving node impact
        df_centrality (pd.DataFrame): pd.DataFrame saving centrality metrics
        compared_col (str): a col name in df_node_impact, specify a type of node impact to be compared.
        corr_method (str): {pearson, kendall, spearman} or callable
        in_log (bool): default False, if node impact in log scale.

    Returns:
        _type_: correlation pd.DataFrame for bar plot visualization
    """
    if in_log:
        node_impact_to_compare = np.log(df_node_impact[compared_col])
    else:
        node_impact_to_compare = df_node_impact[compared_col]
    
    df_centrality_to_combine = df_centrality.copy()
    df_centrality_to_combine.insert(0, column='Node impact', value=node_impact_to_compare)
    node_impact_centrality_df = df_centrality_to_combine.copy()
    
    if 'graph_id' in node_impact_centrality_df.columns.values:
        corr_node_impact_centrality_df = node_impact_centrality_df.drop(columns="graph_id").corr(method=corr_method)
    else:
        corr_node_impact_centrality_df = node_impact_centrality_df.corr(method=corr_method)
    
    corr_df_for_viz = pd.DataFrame({"Centrality type": corr_node_impact_centrality_df.index[1:], 
                                    "Correlation": corr_node_impact_centrality_df['Node impact'][1:]})

    return corr_df_for_viz


def hellinger_distance_btn_srvs(cond_pmf_srv1: jp.ConditionalProbabilityMatrix, cond_pmf_srv2: jp.ConditionalProbabilityMatrix) -> float:
    """Calculating hellinger distance between two SRVs

    Args:
        cond_pmf_srv1 (jp.ConditionalProbabilityMatrix): SRV 1, 
        cond_pmf_srv2 (jp.ConditionalProbabilityMatrix): SRV 2

    Returns:
        float: Hellinger distance betwee SRV1 and SRV2.
    """
    
    h_dist = 0
    cond_pmf_srv1_dict = cond_pmf_srv1.to_dict()
    cond_pmf_srv2_dict = cond_pmf_srv2.to_dict()
    num_states = len(cond_pmf_srv1_dict.keys())

    for state in cond_pmf_srv1_dict.keys():
        probarray1 = np.array(cond_pmf_srv1_dict[state], dtype=float)
        probarray2 = np.array(cond_pmf_srv2_dict[state], dtype=float)
        sqrt_diff = np.sqrt(probarray1) - np.sqrt(probarray2)
        dist_state = np.sqrt(np.sum(np.power(sqrt_diff, 2))) / np.sqrt(2)
        h_dist += dist_state

    return h_dist/num_states


def pre_computed_srvs_generator(num_pre_srvs: int, numvals: int, pdf_type='uniform', max_evals_srv=100, wms_ratio_threshold=0.5, hellinger_dist_threshold=0.5):
    """Generating pre-computed SRVs for appending synergistic variables using append_conditional_variable().

    Args:
        num_pre_srvs (int): Number of SRVs to be prepared.
        wms_ratio_threshold (float): Threshold for Whole minus Sum, the higher, the better the SRV is. Defuault to 0.5
        hellinger_dist_threshold (float): Threshold for hellinger distance between SRVs, the higher, the better the SRV is. Defuault to 0.5, Max Value is 1.

    Returns:
        _type_: A list saving prepared SRVs.
    """
    pre_computed_srvs = []
    while len(pre_computed_srvs) < num_pre_srvs:
        print(len(pre_computed_srvs))
        bn = jp.BayesianNetwork()

        # independent variables to start with (root(s) of the DAG)
        bn.append_independent_variable(pdf_type, numvals)
        bn.append_independent_variable(pdf_type, numvals)

        bn.append_synergistic_variable([0, 1], numvalues=numvals, max_evals=max_evals_srv)
        print(f'WMS of SRV: {bn.wms([0, 1], [2])} (maximum possible {np.log2(numvals)})')

        if bn.wms([0, 1], [2]) / np.log2(numvals) > wms_ratio_threshold:
            cond_pmf_srv = bn.conditional_probabilities([2], [0, 1])
            h_dist_btn_new_and_appended = [hellinger_distance_btn_srvs(cond_pmf_srv, srv_appended) for srv_appended in pre_computed_srvs]
            if np.any(np.array(h_dist_btn_new_and_appended) < hellinger_dist_threshold):
                print("The new SRV was not good, it was not appended.")
            else: 
                pre_computed_srvs.append(cond_pmf_srv)
        else:
            print(f'\t(SRV was not good enough so it was not appended)')
        
        # clear_output(wait=True)

    return pre_computed_srvs


def infer_bn_srvs_pair_triplet_probs(bn: jp.BayesianNetwork, numvars, numroots, numvals, pair_probs: float, 
                                     _precomputed_srvs, cond_ent_ratio=0.5, pdf_type='uniform'):
    """Generate a more custom BN where a mix of pairwise and triplet interactions take place with certain probabilities

    Args:
        bn (jp.BayesianNetwork): a BN object.
        numvars (_type_): number of variables to be generated.
        numroots (_type_): number of root variables in the BN.
        numvals (_type_): number of states of each variable in the BN.
        pair_probs (float): pair-higher ratio = number of pairwise interactions / all
        cond_ent_ratio (float, optional): should be between 0.0 and 1.0; the higher, the more correlated connected variables will be. Defaults to 0.5.
        pdf_type (str, optional): type of distribution of generated values. Defaults to 'uniform'.
    """
    # a few independent variables to start with (root(s) of the DAG)
    if numroots == None or numroots == 0:
        print("No roots for your DAGs.")
    else:
        for _ in range(numroots):
            bn.append_independent_variable(pdf_type, numvals)
    
    dep_num_vars = numvars - numroots
    num_pairs = round(dep_num_vars * pair_probs)
    # num_triplets = dep_num_vars - num_pairs
    pair_tri_numerate = []
    if numroots == 1:
        while pair_tri_numerate.count(2) != num_pairs or pair_tri_numerate[0] != 2:
            pair_tri_numerate = list(np.random.choice(np.array([2,3]), dep_num_vars, p=[pair_probs, 1-pair_probs]))
    else:
        while pair_tri_numerate.count(2) != num_pairs:
            pair_tri_numerate = list(np.random.choice(np.array([2,3]), dep_num_vars, p=[pair_probs, 1-pair_probs]))
    
    for order in pair_tri_numerate:
        if order == 2:
            predecessors_new_appended = sorted(list(np.random.choice(bn.numvariables, size=order-1)))
            bn.append_dependent_variable(predecessors_new_appended, numvals, np.log2(numvals) * cond_ent_ratio)
        elif order == 3:
            predecessors_new_appended = sorted(list(np.random.choice(bn.numvariables, size=order-1, replace=False)))
            # new_srv_idx_appended = bn.append_conditional_variable(_precomputed_srvs[random.randint(len(_precomputed_srvs))], predecessors_new_appended, numvalues=numvals)
            WMS_srvs = []
            for pre_computed_srv in _precomputed_srvs:
                bn_test_srvs = copy.deepcopy(bn)
                new_srv_idx = bn_test_srvs.append_conditional_variable(pre_computed_srv, predecessors_new_appended, numvalues=numvals)
                WMS_srvs.append(bn_test_srvs.wms(predecessors_new_appended, [new_srv_idx]))
            WMS_srvs_arr = np.array(WMS_srvs)
            idx_max_srv_appended = np.random.choice(np.where(WMS_srvs_arr == np.max(WMS_srvs_arr))[0], size=1)[0]
            new_srv_idx_appended = bn.append_conditional_variable(_precomputed_srvs[idx_max_srv_appended], predecessors_new_appended, numvalues=numvals)
        else:
            print("Error: the order is not 2 or 3.")


def infer_bn_srvs_pair_triplet_probs_with_return(bn: jp.BayesianNetwork, numvars, numroots, numvals, pair_probs: float,
                                                 _precomputed_srvs, cond_ent_ratio=0.5, pdf_type='uniform'):
    """Generate a more custom BN where a mix of pairwise and triplet interactions take place with certain probabilities

    Args:
        bn (jp.BayesianNetwork): a BN object.
        numvars (_type_): number of variables to be generated.
        numroots (_type_): number of root variables in the BN.
        numvals (_type_): number of states of each variable in the BN.
        pair_probs (float): pair-higher ratio = number of pairwise interactions / all
        cond_ent_ratio (float, optional): should be between 0.0 and 1.0; the higher, the more correlated connected variables will be. Defaults to 0.5.
        pdf_type (str, optional): type of distribution of generated values. Defaults to 'uniform'.
    
    Returns:
        Added interactions and their values.
    """
    # a few independent variables to start with (root(s) of the DAG)
    if numroots == None or numroots == 0:
        print("No roots for your DAGs.")
    else:
        for _ in range(numroots):
            bn.append_independent_variable(pdf_type, numvals)
    
    dep_num_vars = numvars - numroots
    num_pairs = round(dep_num_vars * pair_probs)
    # num_triplets = dep_num_vars - num_pairs
    pair_tri_numerate = []
    if numroots == 1:
        while pair_tri_numerate.count(2) != num_pairs or pair_tri_numerate[0] != 2:
            pair_tri_numerate = list(np.random.choice(np.array([2,3]), dep_num_vars, p=[pair_probs, 1-pair_probs]))
    else:
        while pair_tri_numerate.count(2) != num_pairs:
            pair_tri_numerate = list(np.random.choice(np.array([2,3]), dep_num_vars, p=[pair_probs, 1-pair_probs]))
    
    pair_tri_combs = []
    pair_tri_values = []
    for order in pair_tri_numerate:
        if order == 2:
            predecessors_new_appended = sorted(list(np.random.choice(bn.numvariables, size=order-1)))
            bn.append_dependent_variable(predecessors_new_appended, numvals, np.log2(numvals) * cond_ent_ratio)
            pair_tri_combs.append(predecessors_new_appended + [bn.numvariables-1])
            pair_tri_values.append(bn.mutual_information(predecessors_new_appended, [bn.numvariables-1]))
        elif order == 3:
            predecessors_new_appended = sorted(list(np.random.choice(bn.numvariables, size=order-1, replace=False)))
            # new_srv_idx_appended = bn.append_conditional_variable(_precomputed_srvs[random.randint(len(_precomputed_srvs))], predecessors_new_appended, numvalues=numvals)
            WMS_srvs = []
            for pre_computed_srv in _precomputed_srvs:
                bn_test_srvs = copy.deepcopy(bn)
                new_srv_idx = bn_test_srvs.append_conditional_variable(pre_computed_srv, predecessors_new_appended, numvalues=numvals)
                WMS_srvs.append(bn_test_srvs.wms(predecessors_new_appended, [new_srv_idx]))
            WMS_srvs_arr = np.array(WMS_srvs)
            idx_max_srv_appended = np.random.choice(np.where(WMS_srvs_arr == np.max(WMS_srvs_arr))[0], size=1)[0]
            new_srv_idx_appended = bn.append_conditional_variable(_precomputed_srvs[idx_max_srv_appended], predecessors_new_appended, numvalues=numvals)

            pair_tri_combs.append(predecessors_new_appended + [bn.numvariables-1])
            pair_tri_values.append(-bn.o_information(predecessors_new_appended + [bn.numvariables-1]))
        else:
            print("Error: the order is not 2 or 3.")
    
    return pd.DataFrame({"Combs": pair_tri_combs, "Weight": pair_tri_values})


def construct_jpmf(bn: jp.BayesianNetwork, numvars, numroots, prop_pairwise, 
                   _precomputed_dependent_vars_one_source, _precomputed_dependent_vars_two_sources, 
                   _precomputed_srvs, pdf_type='dirichlet', numvalues=3):
    """_summary_ Based on Cillian's codes, By Jie on Dec 12, 2024

    Args:
        numvars (_type_): _description_
        prop_pairwise (_type_): _description_
        _precomputed_depedent_vars_one_source (_type_): _description_
        _precomputed_depedent_vars_two_sources (_type_): _description_
        _precomputed_srvs (_type_): _description_
        pdf_type (str, optional): _description_. Defaults to 'dirichlet'.
        numvalues (int, optional): _description_. Defaults to 3.

    Returns:
        _type_: _description_
    """

    if numroots == None or numroots == 0:
        print("No roots for your DAGs.")
    else:
        for _ in range(numroots):
            bn.append_independent_variable(pdf_type, numvalues)

    num_vars_to_append = numvars - numroots
    num_pairwise = round(prop_pairwise * num_vars_to_append)
    num_triplets = num_vars_to_append - num_pairwise

    pair_tri_numerate = list([2] * num_pairwise + [3] * num_triplets)
    random.shuffle(pair_tri_numerate)
    if numroots == 1:
        while pair_tri_numerate[0] != 2:
            # if only 1 root, the first variable must be appended based on pariwise MI.
            random.shuffle(pair_tri_numerate)

    pair_tri_combs = []
    pair_tri_values = []
    for order in pair_tri_numerate:
        # print(f'Order: {order}')
        if order == 2:
            # randomly select the number of source variables, biased towards lower numbers
            num_sources = np.random.choice([1, 2], p=[0.75, 0.25])

            # randomly select the source variables
            predecessors_new_appended = list(np.random.choice(range(bn.numvariables), size=num_sources, replace=False))
            predecessors_new_appended.sort()

            if num_sources == 1:
                # bn.append_dependent_variable(predecessors_new_appended, numvalues, np.log2(numvalues) * 0.5)
        
                cond_dest_given_source = random.choice(_precomputed_dependent_vars_one_source)
                bn.append_conditional_variable(cond_dest_given_source, predecessors_new_appended, numvalues=numvalues)
                
                pair_tri_combs.append(predecessors_new_appended + [bn.numvariables-1])
                pair_tri_values.append(bn.mutual_information(predecessors_new_appended, [bn.numvariables-1]))
                # print("Appended a dependent variable given one source")
            else:
                # randomly select a dependent variable given the source variables
                cond_dest_given_source = random.choice(_precomputed_dependent_vars_two_sources)
                bn.append_conditional_variable(cond_dest_given_source, predecessors_new_appended, numvalues=numvalues)

                pair_tri_combs.append(predecessors_new_appended + [bn.numvariables-1])
                pair_tri_values.append(bn.mutual_information(predecessors_new_appended, [bn.numvariables-1]))
                # print("Appended a dependent variable given two sources")

        elif order == 3:
            
            SRV = random.choice(_precomputed_srvs)
            predecessors_new_appended = list(np.random.choice(range(bn.numvariables), size=2, replace=False))
            predecessors_new_appended.sort()
            new_srv_ix = bn.append_conditional_variable(SRV, predecessors_new_appended, numvalues=numvalues)

            pair_tri_combs.append(predecessors_new_appended + [bn.numvariables-1])
            pair_tri_values.append(-bn.o_information(predecessors_new_appended + [bn.numvariables-1]))

            print("Appended an SRV")
    
    return pd.DataFrame({"Combs": pair_tri_combs, "Weight": pair_tri_values})


def construct_jpmf_bn_adj(
    numvars,
    numroots,
    prop_pairwise,
    _precomputed_dependent_vars_one_source,
    _precomputed_dependent_vars_two_sources,
    _precomputed_srvs,
    pdf_type="dirichlet",
    numvalues=3,
    prob_parent_pairwise=[0.75, 0.25],
    ensure_unshielded=True,          # NEW
    max_pair_tries=2000,             # NEW
):
    """
    Code by Kaleem (30/01/2026): Construct a BN with unshielded colliders
    If ensure_unshielded=True:
      - whenever we append a variable with TWO parents (order==2 with 2 sources, or order==3 SRV),
        we force the chosen parent pair to be NON-adjacent in the current DAG structure.
    """

    def _pick_two_nonadjacent(existing_nodes, G: nx.DiGraph):
        """
        Pick (u, v) with u != v and no adjacency between them (no u->v and no v->u).
        Uses rejection sampling with a fallback enumeration if it struggles.
        """
        existing_nodes = list(existing_nodes)
        if len(existing_nodes) < 2:
            raise ValueError("Need at least 2 existing nodes to pick two parents.")

        # fast rejection sampling
        for _ in range(max_pair_tries):
            u, v = np.random.choice(existing_nodes, size=2, replace=False)
            u, v = int(u), int(v)
            if not (G.has_edge(u, v) or G.has_edge(v, u)):
                return (u, v)

        # fallback: enumerate all valid pairs
        valid = []
        for i, u in enumerate(existing_nodes):
            for v in existing_nodes[i + 1 :]:
                u, v = int(u), int(v)
                if not (G.has_edge(u, v) or G.has_edge(v, u)):
                    valid.append((u, v))

        if not valid:
            raise ValueError(
                "No non-adjacent parent pair available. "
                "Try increasing numroots, reducing numvars, or set ensure_unshielded=False."
            )
        return valid[int(np.random.randint(len(valid)))]

    # ---- init BN with roots
    bn = jp.BayesianNetwork()
    if numroots is None or numroots == 0:
        print("No roots for your DAGs.")
    else:
        for _ in range(numroots):
            bn.append_independent_variable(pdf_type, numvalues)

    bn_initialized = copy.deepcopy(bn)

    num_vars_to_append = numvars - numroots
    num_pairwise = round(prop_pairwise * num_vars_to_append)
    num_triplets = num_vars_to_append - num_pairwise

    pair_tri_numerate = list([2] * num_pairwise + [3] * num_triplets)
    random.shuffle(pair_tri_numerate)
    if numroots == 1:
        while pair_tri_numerate[0] != 2:
            random.shuffle(pair_tri_numerate)

    while not bn_is_connected(bn):
        bn = copy.deepcopy(bn_initialized)

        # Track structure so far (node indices are jointpmf variable indices)
        G_struct = nx.DiGraph()
        G_struct.add_nodes_from(range(bn.numvariables))

        pair_tri_combs = []
        pair_tri_combs_type = []

        for order in pair_tri_numerate:
            if order == 2:
                # pick number of sources (but if not enough existing nodes, force 1)
                if bn.numvariables < 2:
                    num_sources = 1
                else:
                    num_sources = int(np.random.choice([1, 2], p=prob_parent_pairwise))

                if num_sources == 1:
                    predecessors_new_appended = list(
                        np.random.choice(range(bn.numvariables), size=1, replace=False)
                    )
                    predecessors_new_appended.sort()

                    cond_dest_given_source = random.choice(_precomputed_dependent_vars_one_source)
                    bn.append_conditional_variable(cond_dest_given_source, predecessors_new_appended, numvalues=numvalues)

                    new_ix = bn.numvariables - 1
                    # update structure tracking
                    G_struct.add_node(new_ix)
                    G_struct.add_edge(int(predecessors_new_appended[0]), int(new_ix))

                    pair_tri_combs.append(predecessors_new_appended + [new_ix])
                    pair_tri_combs_type.append("Pairwise")

                else:
                    # TWO parents: enforce non-adjacent if requested
                    if ensure_unshielded:
                        u, v = _pick_two_nonadjacent(range(bn.numvariables), G_struct)
                        predecessors_new_appended = sorted([u, v])
                    else:
                        predecessors_new_appended = list(
                            np.random.choice(range(bn.numvariables), size=2, replace=False)
                        )
                        predecessors_new_appended.sort()

                    cond_dest_given_source = random.choice(_precomputed_dependent_vars_two_sources)
                    bn.append_conditional_variable(cond_dest_given_source, predecessors_new_appended, numvalues=numvalues)

                    new_ix = bn.numvariables - 1
                    # update structure tracking
                    G_struct.add_node(new_ix)
                    G_struct.add_edge(int(predecessors_new_appended[0]), int(new_ix))
                    G_struct.add_edge(int(predecessors_new_appended[1]), int(new_ix))

                    pair_tri_combs.append(predecessors_new_appended + [new_ix])
                    pair_tri_combs_type.append("Pairwise")

            elif order == 3:
                # SRV always uses TWO parents
                if bn.numvariables < 2:
                    # shouldn't happen if numroots>=1 and you force first order==2 for numroots==1
                    # but keep it safe:
                    continue

                SRV = random.choice(_precomputed_srvs)

                if ensure_unshielded:
                    u, v = _pick_two_nonadjacent(range(bn.numvariables), G_struct)
                    predecessors_new_appended = sorted([u, v])
                else:
                    predecessors_new_appended = list(
                        np.random.choice(range(bn.numvariables), size=2, replace=False)
                    )
                    predecessors_new_appended.sort()

                bn.append_conditional_variable(SRV, predecessors_new_appended, numvalues=numvalues)

                new_ix = bn.numvariables - 1
                # update structure tracking
                G_struct.add_node(new_ix)
                G_struct.add_edge(int(predecessors_new_appended[0]), int(new_ix))
                G_struct.add_edge(int(predecessors_new_appended[1]), int(new_ix))

                pair_tri_combs.append(predecessors_new_appended + [new_ix])
                pair_tri_combs_type.append("SRV")

    return bn, pd.DataFrame({"Combs": pair_tri_combs, "Type": pair_tri_combs_type})


def construct_jpmf_bn(numvars, numroots, prop_pairwise, 
                      _precomputed_dependent_vars_one_source, _precomputed_dependent_vars_two_sources, 
                      _precomputed_srvs, pdf_type='dirichlet', numvalues=3, prob_parent_pairwise=[0.75, 0.25]):
    """_summary_ Based on Cillian's codes, By Jie on Dec 12, 2024
    Updated by Jie on Dec 13, to put BN initialization inside the function.

    Args:
        numvars (_type_): _description_
        prop_pairwise (_type_): _description_
        _precomputed_depedent_vars_one_source (_type_): _description_
        _precomputed_depedent_vars_two_sources (_type_): _description_
        _precomputed_srvs (_type_): _description_
        pdf_type (str, optional): _description_. Defaults to 'dirichlet'.
        numvalues (int, optional): _description_. Defaults to 3.

    Returns:
        _type_: _description_
    """
    bn = jp.BayesianNetwork()

    if numroots == None or numroots == 0:
        print("No roots for your DAGs.")
    else:
        for _ in range(numroots):
            bn.append_independent_variable(pdf_type, numvalues)
    
    bn_initialized = copy.deepcopy(bn)
    
    num_vars_to_append = numvars - numroots
    num_pairwise = round(prop_pairwise * num_vars_to_append)
    num_triplets = num_vars_to_append - num_pairwise

    pair_tri_numerate = list([2] * num_pairwise + [3] * num_triplets)
    random.shuffle(pair_tri_numerate)
    if numroots == 1:
        while pair_tri_numerate[0] != 2:
            # if only 1 root, the first variable must be appended based on pariwise MI.
            random.shuffle(pair_tri_numerate)

    while not bn_is_connected(bn):

        bn = copy.deepcopy(bn_initialized)
        pair_tri_combs = []
        pair_tri_values = []

        pair_tri_combs_type = []
        
        for order in pair_tri_numerate:
            # print(f'Order: {order}')
            if order == 2:
                # randomly select the number of source variables, biased towards lower numbers
                num_sources = np.random.choice([1, 2], p=prob_parent_pairwise)

                # randomly select the source variables
                predecessors_new_appended = list(np.random.choice(range(bn.numvariables), size=num_sources, replace=False))
                predecessors_new_appended.sort()

                if num_sources == 1:
                    # bn.append_dependent_variable(predecessors_new_appended, numvalues, np.log2(numvalues) * 0.5)
            
                    cond_dest_given_source = random.choice(_precomputed_dependent_vars_one_source)
                    bn.append_conditional_variable(cond_dest_given_source, predecessors_new_appended, numvalues=numvalues)
                    
                    pair_tri_combs.append(predecessors_new_appended + [bn.numvariables-1])
                    pair_tri_combs_type.append( "Pairwise")
                    # pair_tri_values.append(bn.mutual_information(predecessors_new_appended, [bn.numvariables-1]))
                    # print("Appended a dependent variable given one source")
                else:
                    # randomly select a dependent variable given the source variables
                    cond_dest_given_source = random.choice(_precomputed_dependent_vars_two_sources)
                    bn.append_conditional_variable(cond_dest_given_source, predecessors_new_appended, numvalues=numvalues)

                    pair_tri_combs.append(predecessors_new_appended + [bn.numvariables-1])
                    pair_tri_combs_type.append( "Pairwise")
                    # pair_tri_values.append(bn.mutual_information(predecessors_new_appended, [bn.numvariables-1]))
                    # print("Appended a dependent variable given two sources")

            elif order == 3:
                
                SRV = random.choice(_precomputed_srvs)
                predecessors_new_appended = list(np.random.choice(range(bn.numvariables), size=2, replace=False))
                predecessors_new_appended.sort()
                new_srv_ix = bn.append_conditional_variable(SRV, predecessors_new_appended, numvalues=numvalues)

                pair_tri_combs.append(predecessors_new_appended + [bn.numvariables-1])
                pair_tri_combs_type.append( "SRV")
                # pair_tri_values.append(-bn.o_information(predecessors_new_appended + [bn.numvariables-1]))

                # print("Appended an SRV")
    
    return bn, pd.DataFrame({"Combs": pair_tri_combs, "Type":pair_tri_combs_type}) #, "Weight": pair_tri_values})


# Functioins to build BNs from Real data.
# 1. Preprocessing

def num_type(df_data, categorical_limit_num):
    var_type = []
    num_val = len(df_data)
    for col in list(df_data):
        # NOTE: integer in np.array is type "np.int64"; integer in list is type "int"
        col_value = df_data[col]
        col_uni, col_cnt = np.unique(col_value, return_counts=True)
        col_uni = col_uni[np.logical_not(np.isnan(col_uni))]  # remove nan from col_uni
        fraction_part = [math.modf(x)[0] for x in col_uni]
        if all(isinstance(x, np.int64) for x in col_uni) or not np.any(fraction_part):
            if len(col_uni) <= categorical_limit_num:
                var_type.append('c')  # categorical
            else:
                var_type.append('p')  # poisson
        else:
            var_type.append('g')  # gaussian

    return var_type


def impute_mixed_var(df_data, rv_type):
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp_med = SimpleImputer(missing_values=np.nan, strategy='median')
    imp_freq = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    for rvs in list(df_data):
        if rv_type[rvs] == 'g':
            data_impute = imp_mean.fit_transform(pd.DataFrame(df_data[rvs]))
        elif rv_type[rvs] == 'p':
            data_impute = imp_med.fit_transform(pd.DataFrame(df_data[rvs]))
        else:
            data_impute = imp_freq.fit_transform(pd.DataFrame(df_data[rvs]))
        df_data.loc[:, rvs] = data_impute
    return df_data

# Categorize a DataFrame anyway
def categorize_df_qcut(df: pd.DataFrame, num_bin: int):
    df_cat = df.copy()
    for col_name in df.columns.values:
        df_cat[col_name] = pd.qcut(df_cat[col_name], q=num_bin, labels=False, duplicates='drop')
    
    return df_cat


def net_mat_threshold(mat, top_per, is_directed=False):
    if is_directed:
        mat_sorted = sorted(np.reshape(mat, mat.size), reverse=True)  # directed graph
    else:
        mat_sorted = sorted(mat[np.triu_indices(len(mat), 1)], reverse=True)  # undirected graph
    mat_sorted = sorted([i for i in mat_sorted if i != 0 and not np.isnan(i)], reverse=True)  # Remove all zeros from the list.
    threshold = mat_sorted[round(len(mat_sorted) * top_per) - 1]
    return threshold


def mat_top_per_val(mat, top_per, is_directed=False):
    """return top correlation values in an array.

    Args:
        mat (_type_): correlation matrix
        top_per (_type_): top percent: 0.1 mean top 10%
        is_directed (bool, optional): _description_. symmetric or asymmetric, default: symmetric

    Returns:
        _type_: top correlation values in an array.
    """
    if is_directed:
        mat_sorted = sorted(np.reshape(mat, mat.size), reverse=True)  # directed graph
    else:
        mat_sorted = sorted(mat[np.triu_indices(len(mat), 1)], reverse=True)  # undirected graph
    mat_sorted = [i for i in mat_sorted if i != 0]  # Remove all zeros from the list.
    top_per_val = mat_sorted[:(round(len(mat_sorted) * top_per) - 1)]
    return top_per_val


def pick_top_percent_pairs(corr_mat, top_per):
    """Pick a pair whose corr value in the top correlation array.

    Args:
        corr_mat (_type_): correlation matrix
        top_per (_type_): top percent: 0.1 mean top 10%

    Returns:
        _type_: index in 1-d array.
    """
    high_corr = np.random.choice(mat_top_per_val(corr_mat, top_per), size=None, replace=True, p=None)
    picked_col_idx = np.where(corr_mat == high_corr)
    idx_list = np.array([picked_col_idx[0][0], picked_col_idx[1][0]])
    return idx_list



def infer_bn_real_data_pair_tri_probs_with_return(bn: jp.BayesianNetwork, numvars: int, numvals: int, 
                                                  pair_probs: float, top_per: float, real_data, 
                                                  o_info_p_val_combs, sig_level=0.05):
    """infer bn from real data. 

    Args:
        bn (jp.BayesianNetwork): initial BN
        numvars (int): number of variables
        numvals (int): number of values
        pair_probs (float): probability of which pair occurs
        top_per (float): top percent: 0.1 mean top 10%
        real_data (_type_): discrete real data
        o_info_p_val_combs (_type_): synergistic triplets.
        sig_level (float, optional): significance level used in this func. Defaults to 0.05.

    Returns:
        _type_: BN and appended pairwise and higher-order interactions in a dataframe.
    """

    abs_mi_mat, norm_mi_mat = mutual_info_abs_norm(real_data)
    p_val_mi = mi_p_val_chi2(real_data)
    p_value_mi = p_val_mi + p_val_mi.T - np.diag(np.diag(p_val_mi))

    one_minus_mi = 1 - norm_mi_mat
    np.fill_diagonal(norm_mi_mat, 0)
    np.fill_diagonal(one_minus_mi, 0)
    norm_mi_mat[p_value_mi > sig_level] = 0
    one_minus_mi[p_value_mi > sig_level] = 0

    if bn.numvariables == 0:
        source_pmf = jp.JointProbabilityMatrix(1, numvals)
        picked_col_idx = pick_top_percent_pairs(one_minus_mi, top_per)
        for col_idx in picked_col_idx:
            source_pmf.estimate_from_data(real_data.iloc[:, col_idx:col_idx+1].values.tolist())
            bn.append_independent_variable(source_pmf, numvals)  # one variable each time

    cut_rows = round(top_per * len(o_info_p_val_combs["Combs"])) - 1
    candidate_tirplets = o_info_p_val_combs["Combs"][:cut_rows]

    dep_num_vars = numvars - bn.numvariables
    num_pairs = round(dep_num_vars * pair_probs)

    pair_tri_numerate = []
    while pair_tri_numerate.count(2) != num_pairs:
        pair_tri_numerate = list(np.random.choice(np.array([2,3]), dep_num_vars, p=[pair_probs, 1-pair_probs]))  # np.array([2,3]): only pairwise and triplet.

    pair_tri_combs = []
    pair_tri_values = []
    for order in pair_tri_numerate:
        if order == 2:
            temp_bn = jp.BayesianNetwork()
            # pick source_column_ix, destination_column_ix based on pairwise MI network
            picked_col_idx = pick_top_percent_pairs(norm_mi_mat, top_per)
            temp_bn.infer_bn_on_dag(real_data.iloc[:, picked_col_idx], nx.path_graph(real_data.columns.to_numpy()[picked_col_idx], create_using=nx.DiGraph()))
            cond_dest_given_source = temp_bn.conditional_probabilities([1], [0])
            predecessors_new_appended = [np.random.choice(range(bn.numvariables), replace=False)]
            bn.append_conditional_variable(cond_dest_given_source, predecessors_new_appended, numvalues=numvals)

            pair_tri_combs.append(predecessors_new_appended + [bn.numvariables-1])
            pair_tri_values.append(bn.mutual_information(predecessors_new_appended, [bn.numvariables-1]))
        elif order == 3:
            temp_bn = jp.BayesianNetwork()
            # here: pick source1_column_ix, source2_column_ix, destination_column_ix based on negative O-info triplets
            picked_tri = random.choice(candidate_tirplets)
            if isinstance(picked_tri, str):
                # if O-info data are loaded from a .csv file. then comb will be strings
                picked_col_idx = np.array(eval(picked_tri))
            else:
                picked_col_idx = np.array(picked_tri)
            temp_bn.infer_bn_on_dag(real_data.iloc[:, picked_col_idx], nx.path_graph(real_data.columns.to_numpy()[picked_col_idx], create_using=nx.DiGraph()))
            cond_dest_given_source = temp_bn.conditional_probabilities([2], [0, 1])
                
            predecessors_new_appended = sorted(np.random.choice(range(bn.numvariables), size=2, replace=False))
            bn.append_conditional_variable(cond_dest_given_source, predecessors_new_appended, numvalues=numvals)

            pair_tri_combs.append(predecessors_new_appended + [bn.numvariables-1])
            pair_tri_values.append(-bn.o_information(predecessors_new_appended + [bn.numvariables-1]))
        else:
            print("Error: the order is not 2 or 3.")

    return pd.DataFrame({"Combs": pair_tri_combs, "Weight": pair_tri_values})


def infer_bn_real_data_pair_tri_probs_pre_computed(bn: jp.BayesianNetwork, numvars: int, numvals: int, 
                                                   pair_probs: float, top_per_root: float, top_per_corr: float,
                                                   real_data, norm_mi_mat, abs_mi_mat, p_value_mi,
                                                   candidate_tirplets, sig_level=0.05):
    """infer bn from real data: pre-computed O-info and MI.

    Args:
        bn (jp.BayesianNetwork): initial BN
        numvars (int): number of variables
        numvals (int): number of values
        pair_probs (float): probability of which pair occurs
        top_per (float): top percent: 0.1 mean top 10%
        real_data (_type_): discrete real data: NESDA
        norm_mi_mat (_type_): normalized MI matrix
        candidate_tirplets (_type_): triplets with high synergy.
        sig_level (float, optional): significance level used in this func. Defaults to 0.05.

    Returns:
        _type_: BN and appended pairwise and higher-order interactions in a dataframe.
    """
    
    # p_val_mi = mi_p_val_chi2(real_data)
    # p_value_mi = p_val_mi + p_val_mi.T - np.diag(np.diag(p_val_mi))

    one_minus_mi = 1 - norm_mi_mat
    np.fill_diagonal(norm_mi_mat, 0)
    np.fill_diagonal(one_minus_mi, 0)

    norm_mi_mat[p_value_mi > sig_level] = 0
    one_minus_mi[p_value_mi > sig_level] = 0
    abs_mi_mat[p_value_mi > sig_level] = 0

    if bn.numvariables == 0:
        source_pmf = jp.JointProbabilityMatrix(1, numvals)
        picked_col_idx = pick_top_percent_pairs(one_minus_mi, top_per_root)
        for col_idx in picked_col_idx:
            source_pmf.estimate_from_data(real_data.iloc[:, col_idx:col_idx+1].values.tolist())
            bn.append_independent_variable(source_pmf, numvals)  # one variable each time

    # In this function, synergy triplet already prepared.
    # cut_rows = round(top_per * len(o_info_p_val_combs["Combs"])) - 1
    # candidate_tirplets = o_info_p_val_combs["Combs"][:cut_rows]

    dep_num_vars = numvars - bn.numvariables
    num_pairs = round(dep_num_vars * pair_probs)

    pair_tri_numerate = []
    while pair_tri_numerate.count(2) != num_pairs:
        pair_tri_numerate = list(np.random.choice(np.array([2,3]), dep_num_vars, p=[pair_probs, 1-pair_probs]))  # np.array([2,3]): only pairwise and triplet.

    pair_tri_combs = []
    pair_tri_values = []
    for order in pair_tri_numerate:
        if order == 2:
            temp_bn = jp.BayesianNetwork()
            # pick source_column_ix, destination_column_ix based on pairwise MI network
            picked_col_idx = pick_top_percent_pairs(abs_mi_mat, top_per_corr)
            temp_bn.infer_bn_on_dag(real_data.iloc[:, picked_col_idx], nx.path_graph(real_data.columns.to_numpy()[picked_col_idx], create_using=nx.DiGraph()))
            cond_dest_given_source = temp_bn.conditional_probabilities([1], [0])
            predecessors_new_appended = [np.random.choice(range(bn.numvariables), replace=False)]
            bn.append_conditional_variable(cond_dest_given_source, predecessors_new_appended, numvalues=numvals)

            pair_tri_combs.append(predecessors_new_appended + [bn.numvariables-1])
            pair_tri_values.append(bn.mutual_information(predecessors_new_appended, [bn.numvariables-1]))
        elif order == 3:
            while True:
                temp_bn = jp.BayesianNetwork()
                # here: pick source1_column_ix, source2_column_ix, destination_column_ix based on negative O-info triplets
                picked_tri = random.choice(candidate_tirplets)
                if isinstance(picked_tri, str):
                    # if O-info data are loaded from a .csv file. then comb will be strings
                    picked_col_idx = np.array(eval(picked_tri))
                else:
                    picked_col_idx = np.array(picked_tri)
                
                # temp_bn.infer_bn_on_dag(real_data.iloc[:, picked_col_idx], nx.path_graph(real_data.columns.to_numpy()[picked_col_idx], create_using=nx.DiGraph()))

                # May 8, found a problem, changed the path_graph to edge list.
                picked_col_names = real_data.columns.to_numpy()[picked_col_idx]
                temp_edges = [(picked_col_names[0], picked_col_names[1]), (picked_col_names[0], picked_col_names[2]), (picked_col_names[1], picked_col_names[2])]
                temp_bn.infer_bn_on_dag(real_data.iloc[:, picked_col_idx], temp_edges)

                cond_dest_given_source = temp_bn.conditional_probabilities([2], [0, 1])
                predecessors_new_appended = sorted(np.random.choice(range(bn.numvariables), size=2, replace=False))
                bn.append_conditional_variable(cond_dest_given_source, predecessors_new_appended, numvalues=numvals)

                one_triplet_index = predecessors_new_appended + [bn.numvariables-1]
                info_synergy_tri = -bn.o_information(one_triplet_index)
                if info_synergy_tri>0:
                    pair_tri_combs.append(one_triplet_index)
                    pair_tri_values.append(info_synergy_tri)
                    break
        else:
            print("Error: the order is not 2 or 3.")

    return pd.DataFrame({"Combs": pair_tri_combs, "Weight": pair_tri_values})


def draw_pdf_columns_subplot(df: pd.DataFrame, num_per_row: int, fig_size: tuple, num_bins='auto'):
    """plot pdf of columns in a dataframe.

    Args:
        df (pd.DataFrame): a Pandas dataframe
        num_per_row (int): number of rows in subplot.
        fig_size (tuple): the size of the figure to be created.
        num_bins (str, optional): number of bins to use for pdf estimation. Defaults to 'auto'.

    Returns:
        _type_: subplots showing pdf of each column variable.
    """
    num_cols = len(df.columns)
    num_rows = int(np.ceil(num_cols / num_per_row))
    fig, axes = plt.subplots(num_rows, num_per_row, figsize=fig_size)
    for ii in range(num_cols):
        pos_index_x = int(np.floor(ii / num_per_row))
        pos_index_y = np.mod(ii, num_per_row)
        hist_plot = sns.histplot(df.iloc[:, ii], ax=axes[pos_index_x, pos_index_y], bins=num_bins)
        hist_plot.set_xlabel(df.columns[ii], fontdict={'fontsize': 12})
        hist_plot.set_ylabel('Frequency', fontdict={'fontsize': 12})
        hist_plot.tick_params(labelsize=12)
    
    return fig, axes



def draw_bns_subplot(BNs: list, num_per_row: int, fig_size: tuple):
    """_summary_

    Args:
        BNs (list): Precomputed BNs
        num_per_row (int): how many plots per row
        fig_size (tuple): figure size

    Returns:
        _type_: Subplot.
    """

    num_BNs = len(BNs)
    num_rows = int(np.ceil(num_BNs / num_per_row))
    fig, axes = plt.subplots(num_rows, num_per_row, figsize=fig_size)
    for ii in range(num_BNs):
        BN = BNs[ii]
        pos_index_x = int(np.floor(ii / num_per_row))
        pos_index_y = np.mod(ii, num_per_row)

        for layer, nodes in enumerate(nx.topological_generations(BN.dependency_graph)):
        # 'multipartite_layout' expects the layer as a node attribute, so add the
        # numeric layer value as a node attribute
            for node in nodes:
                BN.dependency_graph.nodes[node]["layer"] = layer

        # Compute the multipartite_layout using the "layer" node attribute
        pos = nx.multipartite_layout(BN.dependency_graph, subset_key="layer")
        nx.draw_networkx(BN.dependency_graph, pos=pos, ax=axes[pos_index_x, pos_index_y])
        axes[pos_index_x, pos_index_y].set_facecolor('white')  # Sets the axes background color
        # ax.set_title("DAG layout in topological order (left to right)")
        # fig.tight_layout()
    return fig, axes



def draw_pdf_columns_subplot(df: pd.DataFrame, num_per_row: int, fig_size: tuple, num_bins='auto'):
    """plot pdf of columns in a dataframe.

    Args:
        df (pd.DataFrame): a Pandas dataframe
        num_per_row (int): number of rows in subplot.
        fig_size (tuple): the size of the figure to be created.
        num_bins (str, optional): number of bins to use for pdf estimation. Defaults to 'auto'.

    Returns:
        _type_: subplots showing pdf of each column variable.
    """
    num_cols = len(df.columns)
    num_rows = int(np.ceil(num_cols / num_per_row))
    fig, axes = plt.subplots(num_rows, num_per_row, figsize=fig_size)
    for ii in range(num_cols):
        pos_index_x = int(np.floor(ii / num_per_row))
        pos_index_y = np.mod(ii, num_per_row)
        hist_plot = sns.histplot(df.iloc[:, ii], ax=axes[pos_index_x, pos_index_y], bins=num_bins)
        hist_plot.set_xlabel(df.columns[ii], fontdict={'fontsize': 12})
        hist_plot.set_ylabel('Frequency', fontdict={'fontsize': 12})
        hist_plot.tick_params(labelsize=12)
    
    return fig, axes


def draw_reg_scatter_subplot(df1: pd.DataFrame, df2: pd.DataFrame, df1_col_name: str, 
                             num_per_row: int, fig_size: tuple, 
                             fit_line=False, x_in_log=False, y_in_log=False):
    """Plot regression figures.

    Args:
        df1 (pd.DataFrame): a Pandas dataframe
        df2 (pd.DataFrame): a Pandas dataframe
        df1_col_name (str): a column name in df1.
        num_per_row (int): number of rows in subplot.
        fig_size (tuple): the size of the figure to be created.

    Returns:
        _type_: subplots showing scatters and regression plot of each comparison.
    """
    df1_col_target = pd.DataFrame(df1[df1_col_name]) 
    df = pd.concat([df2, df1_col_target], axis=1)
    num_col_df2 = len(df2.columns)
    num_rows = int(np.ceil(num_col_df2 / num_per_row))
    fig, axes = plt.subplots(num_rows, num_per_row, figsize=fig_size)

    for ii in range(num_col_df2):
        df2_col_name = df2.columns.values[ii]
        pos_index_x = int(np.floor(ii / num_per_row))
        pos_index_y = np.mod(ii, num_per_row)
        reg_linear = sns.regplot(df, x=df2_col_name, y=df1_col_name, ax=axes[pos_index_x, pos_index_y], fit_reg=fit_line, scatter_kws={'alpha':0.6})
        if x_in_log:
            reg_linear.set(xscale="log")
        if y_in_log:
            reg_linear.set(yscale="log")
        reg_linear.tick_params(labelsize=12)
        # reg_linear.set_ylim([-0.1, 2.1])
        reg_linear.set_xlabel(df2_col_name, fontdict={'fontsize': 12})
        reg_linear.set_ylabel('Node impact', fontdict={'fontsize': 12})
        # slope, intercept, r, p, sterr = ss.linregress(x=reg_linear.get_lines()[0].get_xdata(), y=reg_linear.get_lines()[0].get_ydata()) 
        # axes[pos_index_x, pos_index_y].text(x=text_pos_x, y=text_pos_y, fontsize=18,
        #                                     s='y = ' + str(round(intercept, 2)) + ' + ' + str(round(slope, 2)) + 'x')
    
    return fig, axes


def draw_reg_scatter_subplot_reverse(df1: pd.DataFrame, df2: pd.DataFrame, df1_col_name: str, 
                                     num_per_row: int, fig_size: tuple, fit_line=False, x_in_log=False, y_in_log=False):
    """Plot regression figures.

    Args:
        df1 (pd.DataFrame): a Pandas dataframe
        df2 (pd.DataFrame): a Pandas dataframe
        df1_col_name (str): a column name in df1.
        num_per_row (int): number of rows in subplot.
        fig_size (tuple): the size of the figure to be created.

    Returns:
        _type_: subplots showing scatters and regression plot of each comparison.
    """
    df1_col_target = pd.DataFrame(df1[df1_col_name]) 
    df = pd.concat([df2, df1_col_target], axis=1)
    num_col_df2 = len(df2.columns)
    num_rows = int(np.ceil(num_col_df2 / num_per_row))
    fig, axes = plt.subplots(num_rows, num_per_row, figsize=fig_size)

    for ii in range(num_col_df2):
        df2_col_name = df2.columns.values[ii]
        pos_index_x = int(np.floor(ii / num_per_row))
        pos_index_y = np.mod(ii, num_per_row)
        reg_linear = sns.regplot(df, x=df1_col_name, y=df2_col_name, ax=axes[pos_index_x, pos_index_y], fit_reg=fit_line, scatter_kws={'alpha':0.6})
        if x_in_log:
            reg_linear.set(xscale="log")
        if y_in_log:
            reg_linear.set(yscale="log")
        reg_linear.tick_params(labelsize=12)
        # reg_linear.set_ylim([-0.1, 2.1])
        reg_linear.set_xlabel('Node impact', fontdict={'fontsize': 12})
        reg_linear.set_ylabel(df2_col_name, fontdict={'fontsize': 12})
        # slope, intercept, r, p, sterr = ss.linregress(x=reg_linear.get_lines()[0].get_xdata(), y=reg_linear.get_lines()[0].get_ydata()) 
        # axes[pos_index_x, pos_index_y].text(x=text_pos_x, y=text_pos_y, fontsize=18,
        #                                     s='y = ' + str(round(intercept, 2)) + ' + ' + str(round(slope, 2)) + 'x')
    
    return fig, axes



def confidence_interval(data, confidence=0.95, err_scale="sem"):
    """
    Calculate the confidence interval for a given array of data.

    Parameters:
    data (array-like): Array of data points.
    confidence (float): The confidence level (default is 0.95 for 95% confidence interval).
    err_scale (string): scale -> standard deviation (std), or standard error of mean (sem)

    Returns:
    tuple: Lower bound and upper bound of the confidence interval.
    """
    # Convert the data to a numpy array
    data = np.array(data)
    
    # Calculate the mean of the data
    mean = np.mean(data)
    
    # Calculate the standard error of the mean
    sem = ss.sem(data)
    std_dev = np.std(data)
    
    # Get the critical value from the t-distribution
    # For large samples, you can use stats.norm.ppf((1 + confidence) / 2) instead of t.ppf
    n = len(data)
    if err_scale=="sem":
        # Emphasize the "accuracy" of data
        h = sem * ss.t.ppf((1 + confidence) / 2., n-1)
    elif err_scale=="std":
        # Emphasize the "dispersion" of data
        h = std_dev * ss.t.ppf((1 + confidence) / 2., n-1)
    
    # Calculate the lower and upper bounds of the confidence interval
    lower_bound = mean - h
    upper_bound = mean + h

    # Alternatively, another method to calculate lower and upper bounds. Results are the same.
    # lower_bound = stats.t.ppf((1 - confidence) / 2., n - 1, loc = mean, scale = std_dev)
    # upper_bound = stats.t.ppf((1 + confidence) / 2., n - 1, loc = mean, scale = std_dev)
    
    return lower_bound, upper_bound


# Polynomial Regression  This one is using r2_score in sklearn, but the result is the same.
def polyfit(x, y, degree):
    results = {}
    coeffs = np.polyfit(x, y, degree)

    # Polynomial Coefficients
    results['polynomial'] = coeffs.tolist()

    # r-squared
    p = np.poly1d(coeffs)
    # fit values, and mean
    yhat = p(x)
    results['determination'] = r2_score(y, yhat)

    return results


def r2_df_bar_plot(df_node_impact: pd.DataFrame, df_centrality: pd.DataFrame, compared_col: str, in_log=False):

    if in_log:
        y = np.log(df_node_impact[compared_col])
    else:
        y = df_node_impact[compared_col]
    
    r2_value = []
    for central_name in df_centrality.columns:
        x = df_centrality[central_name]
        try:
            linear_fit_results = polyfit(x, y, 1)
            r2_value.append(linear_fit_results["determination"])
        except:
            print(f"An error occurs while doing linear fitting for {central_name}. Set the r2 to 0.")
            r2_value.append(0)
        
    r2_df_for_viz = pd.DataFrame({"Centrality type": df_centrality.columns, "R_Squared": r2_value})
    
    return r2_df_for_viz


def value_to_rank_in_chunks(df: pd.DataFrame, chunk_size: int):
    """_summary_

    Args:
        df (pd.DataFrame): data in pd.DataFrame
        chunk_size (int): rank the continuous values considering each N rows. 
        For considering all rows, the chunk size should be len(df)

    Returns:
        _type_: ranked results in pd.DataFrame
    """
    ranked_df = pd.DataFrame()  # Empty DataFrame to store ranked values
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i + chunk_size]  # Get a chunk of rows
        ranked_chunk = chunk.rank(method='dense', ascending=False, axis=0).astype(int) - 1  # Rank and start from 0, normally from 0 for classification task.
        ranked_df = pd.concat([ranked_df, ranked_chunk])  # Append ranked chunk to the result
    return ranked_df
