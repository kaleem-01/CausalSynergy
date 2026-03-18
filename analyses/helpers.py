import pickle

def load_results(subset, base_dir= "results", ):
    global results_by_label

    with open(f'{base_dir}/pc/{subset}.pkl', 'rb') as f:
        results_pc = pickle.load(f)
    
    with open(f'{base_dir}/ges/{subset}.pkl', 'rb') as f:
        results_ges = pickle.load(f)
    
    with open(f'{base_dir}/hc/{subset}.pkl', 'rb') as f:
        results_hc = pickle.load(f)

    with open(f'{base_dir}/ea_fg/{subset}.pkl', 'rb') as f:
        results_ea_fg = pickle.load(f)

    # with open(f'{base_dir}/ea_hc/{subset}.pkl', 'rb') as f:
    #     results_hc_inf = pickle.load(f)

    # with open(f'results/ea_ues/{subset}.pkl', 'rb') as f:
    #     results_ea_ues = pickle.load(f)

    # with open(f'results/ea_ies/{subset}.pkl', 'rb') as f:
    #     results_ea_ies = pickle.load(f)
    
    # with open(f'results/ea_fes/{subset}.pkl', 'rb') as f:
    #     results_ea_fes = pickle.load(f)

    # with open(f'results/ea_upu/{subset}.pkl', 'rb') as f:
    #     results_ea_upu = pickle.load(f)
    # with open(f'results/ea_ipu/{subset}.pkl', 'rb') as f:
    #     results_ea_ipu = pickle.load(f)
    

    results_by_label = {
        "PC": results_pc,
        "GES": results_ges,
        "HC": results_hc,
        # "EA Informed": results_ea_ies,
        # "EA Final Greedy": results_ea_fg,
        # "HC Informed": results_hc_inf,
        # "EA Uninformed": results_ea_ues,
        # "EA Fully Informed": results_ea_fes,
        # "EA Uninformed (PU)": results_ea_upu,
        # "EA Informed (PU)": results_ea_ipu,
    }