# import numpy as np
# import os
# import matplotlib.pyplot as plt
# from src import stress
# from src.utils.utils import stress_normalizer
# from src.control import multicluster_afc_global, tracking_error, multicluster_afc_distributed
# import src.utils.geometry as geometry

# if __name__ == "__main__":

#     nominal_config = geometry.random_config(100)
#     D = nominal_config.shape[0]
#     N = nominal_config.shape[1]
#     init_config = np.random.rand(D, N)
#     subgraph_glb_idx = [list(range(0, 65)), list(range(35, 100))] 

#     hyperparam_dict = { 'alpha': 0.5, 'beta': 1, 'gamma': 0.1 }

#     print(f"Geometry is partitioned into {len(subgraph_glb_idx)} subgraphs with sizes {[len(idx) for idx in subgraph_glb_idx]}.")

#     stress_mat_list = stress.compute_stress_multicluster(nominal_config,
#                                                          subgraph_glb_idx, 
#                                                          hyperparam_dict,
#                                                          parallel=True)

#     print("Stress matrices computed.")
#     leaders_glb_idx = np.random.choice(nominal_config.shape[1], 5, replace=False).tolist()

#     TE_multicluster = tracking_error(multicluster_afc_global(
#         nominal_config.copy(),
#         subgraph_glb_idx,
#         init_config = init_config.copy(),
#         leaders_glb_idx = leaders_glb_idx,
#         # leaders_glb_idx = [0,1,2],
#         stress_mat_list = stress_mat_list,
#         gain = 80,
#         max_itr = 5000,
#         dt = 0.01
#         ), 
#         nominal_config)
    
#     TE_multicluster_distributed = tracking_error(multicluster_afc_distributed(
#         nominal_config.copy(),
#         subgraph_glb_idx,
#         leaders_glb_idx = leaders_glb_idx,
#         # leaders_glb_idx = [6,7,17,25,31,41,47,59],
#         stress_mat_list = stress_mat_list,
#         gain = 80,
#         max_itr = 5000,
#         dt = 0.01,
#         init_config = init_config.copy()
#         ),
#         nominal_config)
    

#     # print("Subgraph Global Indices:", subgraph_glb_idx)
#     plt.figure()
#     plt.plot(TE_multicluster)
#     plt.plot(TE_multicluster_distributed, linestyle='--')
#     plt.legend(["Multicluster AFC - Global", "Multicluster AFC - Distributed"])
#     plt.title("Tracking Error - Multicluster AFC")
#     plt.xlabel("Iteration")
#     plt.ylabel("Tracking Error")
#     plt.grid()
#     plt.yscale('log')
#     plt.savefig("results/multicluster_afc_tracking_error.png")
#     plt.show()
        
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # specific import for progress bars
import src.utils.geometry as geometry
from src import stress
from src.control import multicluster_afc_global, tracking_error, multicluster_afc_distributed

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def get_partition_indices(n_nodes, overlap):
    """
    Splits N nodes into 2 clusters with a specific overlap count.
    """
    midpoint = n_nodes // 2
    half_overlap = overlap // 2
    remainder = overlap % 2

    # Cluster 1: Start to (Mid + half + remainder)
    end_c1 = midpoint + half_overlap + remainder
    
    # Cluster 2: (Mid - half) to End
    start_c2 = midpoint - half_overlap

    idx1 = list(range(0, end_c1))
    idx2 = list(range(start_c2, n_nodes))
    
    return [idx1, idx2]

def generate_data(cfg):
    """
    Part 1: Generates random configs, computes stress matrices, and saves them.
    """
    print("\n--- Part 1: Generating Data ---")
    output_dir = cfg['experiment']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    overlaps = cfg['experiment']['overlaps']
    n_trials = cfg['experiment']['n_trials']
    prefix = cfg['experiment']['filename_prefix']

    nominal_config = geometry.random_config(cfg['experiment']['n_nodes'])
    
    for overlap in overlaps:
        print(f"Generating {n_trials} trials for overlap = {overlap}...")
        
        for trial in tqdm(range(n_trials), desc=f"Overlap {overlap}"):
            filename = f"{prefix}overlap-{overlap}_{trial:03d}.npy"
            filepath = os.path.join(output_dir, filename)
            
            # Skip if already exists (resume capability)
            if os.path.exists(filepath):
                continue

            
            # 2. Define Subgraphs
            subgraph_glb_idx = get_partition_indices(cfg['experiment']['n_nodes'], overlap)
            
            # 3. Compute Stress Matrices
            stress_mat_list = stress.compute_stress_multicluster(
                nominal_config,
                subgraph_glb_idx, 
                cfg['hyperparams'],
                parallel=True
            )
            
            # 4. Save Data Package
            data_package = {
                'nominal_config': nominal_config,
                'subgraph_glb_idx': subgraph_glb_idx,
                'stress_mat_list': stress_mat_list,
                'overlap': overlap,
                'trial_id': trial,
                'hyperparams': cfg['hyperparams']
            }
            np.save(filepath, data_package)

def evaluate_and_plot(cfg):
    """
    Part 2: Loads matrices, runs control sim, and plots results.
    """
    print("\n--- Part 2: Evaluating and Plotting ---")
    input_dir = cfg['experiment']['output_dir']
    plot_dir = cfg['experiment']['plot_dir']
    os.makedirs(plot_dir, exist_ok=True)
    
    overlaps = cfg['experiment']['overlaps']
    n_trials = cfg['experiment']['n_trials']
    prefix = cfg['experiment']['filename_prefix']
    
    # Storage for results: results[overlap]['global'] = [ [error_trace_1], [error_trace_2], ... ]
    results = {ov: {'global': [], 'dist': []} for ov in overlaps}

    total_runs = len(overlaps) * n_trials
    pbar = tqdm(total=total_runs, desc="Running Simulations")

    init_config = 5*np.random.rand(2, cfg['experiment']['n_nodes']) 
    leaders_idx = np.random.choice(cfg['experiment']['n_nodes'], cfg['control']['n_leaders'], replace=False).tolist()

    for overlap in overlaps:
        for trial in range(n_trials):
            # 1. Load Data
            filename = f"{prefix}overlap-{overlap}_{trial:03d}.npy"
            filepath = os.path.join(input_dir, filename)
            
            if not os.path.exists(filepath):
                print(f"[Warning] File missing: {filepath}")
                pbar.update(1)
                continue
                
            data = np.load(filepath, allow_pickle=True).item()
            nominal = data['nominal_config']
            sub_idx = data['subgraph_glb_idx']
            stress_mats = data['stress_mat_list']
            
            # 2. Setup Simulation Init Conditions
            D, N = nominal.shape
            
            
            # 3. Run Global Control
            traj_global = multicluster_afc_global(
                nominal.copy(), sub_idx,
                init_config=init_config.copy(),
                leaders_glb_idx=leaders_idx,
                stress_mat_list=stress_mats,
                gain=cfg['control']['gain'],
                max_itr=cfg['control']['max_itr'],
                dt=cfg['control']['dt']
            )
            err_global = tracking_error(traj_global, nominal)
            results[overlap]['global'].append(err_global)
            
            # # 4. Run Distributed Control
            # traj_dist = multicluster_afc_distributed(
            #     nominal.copy(), sub_idx,
            #     leaders_glb_idx=leaders_idx,
            #     stress_mat_list=stress_mats,
            #     gain=cfg['control']['gain'],
            #     max_itr=cfg['control']['max_itr'],
            #     dt=cfg['control']['dt'],
            #     init_config=init_config.copy()
            # )
            # err_dist = tracking_error(traj_dist, nominal)
            # results[overlap]['dist'].append(err_dist)
            
            # pbar.update(1)
            
    pbar.close()
    plot_results(results, cfg)

def plot_results(results, cfg):
    """
    Plots mean tracking error with std deviation shading.
    """
    print("Generating Plots...")
    plot_dir = cfg['experiment']['plot_dir']
    
    # Plot 1: Global Control - Overlap Comparison
    plt.figure(figsize=(10, 6))
    colors = ['r', 'g', 'b', 'm']
    
    for i, overlap in enumerate(results.keys()):
        data_matrix = np.array(results[overlap]['global']) # Shape: (n_trials, max_itr)
        
        mean_trace = np.mean(data_matrix, axis=0)
        std_trace = np.std(data_matrix, axis=0)
        iters = np.arange(len(mean_trace))
        
        plt.plot(iters, mean_trace, color=colors[i], label=f'Overlap {overlap}')
        plt.fill_between(iters, mean_trace - std_trace, mean_trace + std_trace, color=colors[i], alpha=0.15)
        
    plt.title("Effect of Overlap Size on Convergence (Global AFC)")
    plt.xlabel("Iteration")
    plt.ylabel("Tracking Error (Log Scale)")
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.savefig(os.path.join(plot_dir, "convergence_overlap_global.png"))
    plt.show()

    
if __name__ == "__main__":
    # Ensure you have the config file ready
    config_file = "configs/multicluster_afc.yaml"
    

    cfg = load_config(config_file)
    
    generate_data(cfg)
    
    evaluate_and_plot(cfg)