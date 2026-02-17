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
from src.utils.utils import find_global_stress_mat, stress_normalizer

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
    n_afc_runs = cfg['experiment']['n_afc_runs']
    prefix = cfg['experiment']['filename_prefix']
    
    # Storage for results: results[overlap]['global'] = [ [error_trace_1], [error_trace_2], ... ]
    # results = {ov: {'global': [], 'dist': []} for ov in overlaps}

    # total_runs = len(overlaps) * n_trials
    # pbar = tqdm(total=total_runs, desc="Running Simulations")

    # for afc_run in tqdm(range(cfg['experiment']['n_afc_runs']), desc="Running Simulations"):
    # total_files = len(overlaps) * n_trials
    # init_config = 5*np.random.rand(2, cfg['experiment']['n_nodes']) 
    # leaders_idx = np.random.choice(cfg['experiment']['n_nodes'], cfg['control']['n_leaders'], replace=False).tolist()
    # find random leaders in the first cluster
    # leaders_idx = np.random.choice(get_partition_indices(cfg['experiment']['n_nodes'], overlaps[0])[0], cfg['control']['n_leaders'], replace=False).tolist()
    # print(f"Selected Leaders (Global Indices): {leaders_idx}")
    # fix leader for plotting
    leaders_idx = [47, 40, 28, 29, 34]
    results = {ov: {'global': []} for ov in overlaps}

    # Total iterations = overlaps * trials
    total_files = len(overlaps) * n_trials
    pbar = tqdm(total=total_files, desc="Processing Trials")

    for overlap in overlaps:
        for trial in range(n_trials):
            # 1. Load Data (Only once per trial!)
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
            
            D, N = nominal.shape

            # 2. Run Multiple Random Initializations for this specific geometry
            for run_idx in range(n_afc_runs):
                
                # Generate new random initial configuration for this run
                # leaders_idx = np.random.choice(get_partition_indices(cfg['experiment']['n_nodes'], overlaps[0])[0], cfg['control']['n_leaders'], replace=False).tolist()
                init_config = 5 * np.random.rand(2, N)
                
                # Run Global Control
                traj_global = multicluster_afc_global(
                    nominal.copy(), sub_idx,
                    init_config=init_config.copy(),
                    leaders_glb_idx=leaders_idx,
                    stress_mat_list=stress_mats,
                    gain=cfg['control']['gain'],
                    max_itr=cfg['control']['max_itr'],
                    dt=cfg['control']['dt']
                )
                
                # Compute Error and Store
                err_global = tracking_error(traj_global, nominal)
                results[overlap]['global'].append(err_global)
            
            pbar.update(1)
            
    pbar.close()
    
    plot_results(results, cfg)

def plot_results(results, cfg):
    """
    Plots mean tracking error with std deviation shading.
    """
    print("Generating Plots...")
    plot_dir = cfg['experiment']['plot_dir']
    
    # Plot 1: Global Control - Overlap Comparison
    plt.figure(figsize=(4, 3))
    colors = list(cfg["line_colors"].values())

    t = np.arange(0, cfg['control']['max_itr']+1) * cfg['control']['dt']
    
    for i, overlap in enumerate(results.keys()):
        data_matrix = np.array(results[overlap]['global']) # Shape: (n_trials, max_itr)
        
        mean_trace = np.mean(data_matrix, axis=0)
        std_trace = np.std(data_matrix, axis=0)
        iters = np.arange(len(mean_trace))
        
        plt.plot(t, mean_trace, color=colors[i], label=f'Overlap {overlap}')
        plt.fill_between(t, mean_trace - std_trace, mean_trace + std_trace, color=colors[i], alpha=0.15)
        
    # plt.title("Effect of Overlap Size on Convergence (Global AFC)")
    # plt.xlabel("Time [s]")
    # plt.ylabel("Tracking Error (Log Scale)")
    plt.yscale('log')
    # plt.legend()
    # plt.grid(True, which="both", ls="--")
    save_path = os.path.join(plot_dir, "convergence_overlap.pdf")
    plt.savefig(save_path, dpi=300)
    plt.tight_layout()
    # plt.show()

def plot_configurations(cfg):
    """
    Reads one file for each overlap case and visualizes the node distribution.
    - Cluster 1 nodes: Blue Circles
    - Cluster 2 nodes: Orange Circles
    - Overlapping nodes: Red Stars (Special Marker)
    """
    print("\n--- Visualizing Configurations ---")
    input_dir = cfg['experiment']['output_dir']
    plot_dir = cfg['experiment']['plot_dir']
    overlaps = cfg['experiment']['overlaps']
    prefix = cfg['experiment']['filename_prefix']
    
    # Setup subplots: 1 row, N columns (one for each overlap case)
    n_plots = len(overlaps)
    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4), constrained_layout=True)
    
    # Ensure axes is iterable even if there's only 1 overlap case
    if n_plots == 1: axes = [axes]

    for i, (overlap, ax) in enumerate(zip(overlaps, axes)):
        # Load the first trial (trial 000) for visualization
        filename = f"{prefix}overlap-{overlap}_000.npy"
        print(f"Loading configuration for visualization: {filename}")
        filepath = os.path.join(input_dir, filename)
        
        if not os.path.exists(filepath):
            ax.text(0.5, 0.5, "Data Missing", ha='center', va='center')
            ax.set_title(f"Overlap {overlap}")
            continue

        data = np.load(filepath, allow_pickle=True).item()
        nominal = data['nominal_config']      # Shape (2, N)
        sub_idx = data['subgraph_glb_idx']    # List of lists [[idx_c1], [idx_c2]]
        
        # Identify sets of indices
        set_c1 = set(sub_idx[0])
        set_c2 = set(sub_idx[1])
        
        intersection = list(set_c1 & set_c2)
        only_c1 = list(set_c1 - set_c2)
        only_c2 = list(set_c2 - set_c1)
        
        # --- Plotting ---
        # 1. Cluster 1 (Unique nodes)
        if only_c1:
            ax.scatter(nominal[0, only_c1], nominal[1, only_c1], 
                       c=cfg["cluster_colors"]["cluster_1"], s=80, alpha=0.6, label='Cluster 1 Only')
            
        # 2. Cluster 2 (Unique nodes)
        if only_c2:
            ax.scatter(nominal[0, only_c2], nominal[1, only_c2], 
                       c=cfg["cluster_colors"]["cluster_2"], s=80, alpha=0.6, label='Cluster 2 Only')
            
        # 3. Overlap (Shared nodes) -> Special Marker
        if intersection:
            ax.scatter(nominal[0, intersection], nominal[1, intersection], 
                       c=cfg["cluster_colors"]["mutual"], marker='X', s=200, label='Overlap', zorder=10)

        # Formatting
        ax.set_title(f"Overlap Size: {overlap}")
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # # Only add legend to the first plot to avoid clutter
        # if i == 0:
        #     ax.legend(loc='upper right', fontsize='small', framealpha=0.9)

    save_path = os.path.join(plot_dir, "config-random-100-overlap.pdf")
    plt.savefig(save_path, dpi=300)
    print(f"Visualization saved to {save_path}")
    # plt.show()

def plot_overlap_eigenvalues(cfg):
    """
    Reads all saved trial data for each overlap setup, computes the global 
    stress matrix eigenvalues, and plots the mean with std deviation shading.
    """
    print("\n--- Plotting Eigenvalues for Overlap Setups ---")
    
    input_dir = cfg['experiment']['output_dir']
    plot_dir = cfg['experiment']['plot_dir']
    overlaps = cfg['experiment']['overlaps']
    n_trials = cfg['experiment']['n_trials']
    prefix = cfg['experiment']['filename_prefix']
    n_nodes = cfg['experiment']['n_nodes']

    plt.figure(figsize=(4, 3))
    
    # Use a colormap to dynamically assign colors based on the number of overlaps
    # colors = plt.cm.viridis(np.linspace(0, 0.9, len(overlaps)))
    colors = list(cfg["line_colors"].values())
    markers = list(cfg["markers"].values())

    for i, overlap in enumerate(overlaps):
        all_eigvals = []
        
        for trial in range(n_trials):
            # Ensure this matches the filename format you generated in Part 1
            filename = f"{prefix}overlap-{overlap}_{trial:03d}.npy"
            filepath = os.path.join(input_dir, filename)
            
            if not os.path.exists(filepath):
                continue
                
            data = np.load(filepath, allow_pickle=True).item()
            stress_mats = data['stress_mat_list']
            sub_idx = data['subgraph_glb_idx']
            
            # 1. Reconstruct the global stress matrix
            # (Assuming find_global_stress_mat is available from your earlier code)
            global_stress = find_global_stress_mat(stress_mats, sub_idx, n_nodes)
            
            # 2. Normalize the global stress matrix
            global_stress = stress_normalizer(global_stress)
            
            # 3. Calculate sorted eigenvalues
            eigvals = np.sort(np.real(np.linalg.eigvals(global_stress)))
            all_eigvals.append(eigvals)
            
        if not all_eigvals:
            print(f"[Warn] No data found for overlap {overlap}")
            continue
            
        # 4. Aggregate statistics across all trials
        all_eigvals = np.array(all_eigvals)
        mean_eigvals = np.mean(all_eigvals, axis=0)
        std_eigvals = np.std(all_eigvals, axis=0)
        
        # X-axis: Index from 1 to N
        x_idx = np.arange(1, len(mean_eigvals) + 1)
        
        # 5. Plot mean and shade std deviation
        plt.plot(x_idx, mean_eigvals, '-', label=f'Overlap {overlap}', color=colors[i], marker = markers[i], linewidth=2)
        plt.fill_between(x_idx, mean_eigvals - std_eigvals, mean_eigvals + std_eigvals, color=colors[i], alpha=0.15)

    # --- Formatting ---
    # plt.title("Sorted Eigenvalues by Overlap Size (Mean ± Std over 100 Trials)")
    # plt.xlabel("Eigenvalue Index")
    # plt.ylabel("Eigenvalue Magnitude")
    # plt.yscale('log') # Log scale visually separates small non-zero eigenvalues better
    # plt.xscale('log')
    plt.xlim(0.8, 10.5)
    plt.ylim(-0.01, 0.35)
    
    # # Clean up X-axis ticks (e.g., show every 10th index, plus the first one)
    # tick_positions = np.append([1], np.arange(10, n_nodes + 1, 10))
    # plt.xticks(ticks=tick_positions, labels=tick_positions)
    # plt.xlim(-0.5, 9.5)
    plt.xticks(ticks=np.arange(11), labels=np.arange(0, 11))
    
    plt.grid(True, linestyle='--', alpha=0.5)
    # plt.legend()
    plt.tight_layout()
    
    os.makedirs(plot_dir, exist_ok=True)
    save_path = os.path.join(plot_dir, "eigs_overlap_comparison.pdf")
    plt.savefig(save_path, format='pdf', dpi=300)
    print(f">>> Saved to {save_path}")
    plt.show()

    
if __name__ == "__main__":
    # Ensure you have the config file ready
    config_file = "configs/multicluster_afc.yaml"
    

    cfg = load_config(config_file)
    
    # generate_data(cfg)
    plot_configurations(cfg)
    plot_overlap_eigenvalues(cfg)

    
    evaluate_and_plot(cfg)