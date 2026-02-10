import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import argparse
from src.utils.utils import detect_num_edges, stress_normalizer
from src.control import tracking_error, affine_formation_control_global
import random
import itertools


def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def load_benchmark_data(results_dir, file_prefix):
    """
    Scans the directory for matching .npy files and loads the raw data.
    Returns: A list of dictionaries (one dict per simulation case).
    """
    pattern_string = f"{file_prefix}_*.npy"
    search_pattern = os.path.join(results_dir, pattern_string)
    
    files = sorted(glob.glob(search_pattern))
    
    if not files:
        print(f"[Error] No files found matching: {search_pattern}")
        return []

    print(f"Found {len(files)} files matching '{pattern_string}'. Loading data...")

    raw_data_list = []
    for f in files:
        try:
            # Load the dictionary saved in the npy file
            # .item() is needed because np.save wraps the dict in a 0-d array
            case_dict = np.load(f, allow_pickle=True).item()
            raw_data_list.append(case_dict)
        except Exception as e:
            print(f"[Warn] Skipping corrupt file {f}: {e}")

    return raw_data_list

def eval_runtime(raw_data):
    """
    Extracts runtime data, prints statistics.
    """
    print("\n--- Evaluating Runtimes [s] ---")
    records = []

    # 1. Extract Data
    for case in raw_data:
        if 'runtimes' in case:
            for algo_name, duration in case['runtimes'].items():
                if duration is not None:
                    records.append({
                        'Algorithm': algo_name,
                        'Time (s)': duration
                    })
    
    if not records:
        print("[Warn] No runtime data found.")
        return

    df = pd.DataFrame(records)

    # 2. Print Statistics
    stats = df.groupby('Algorithm')['Time (s)'].agg(['mean', 'std', 'min', 'max'])
    print(stats)

def eval_edge_count(raw_data, sparse_tol=1e-5):
    """
    Extracts runtime data, prints statistics.
    """
    print("\n--- Evaluating Edge Counts ---")
    records = []

    # 1. Extract Data
    for case in raw_data:
        if 'stress_matrices' in case:
            for algo_name, stress_mat in case['stress_matrices'].items():
                if stress_mat is not None:
                    stress_mat = stress_normalizer(stress_mat)
                    edge_count = detect_num_edges(stress_mat, sparse_tol)  # Using sparse_tol from config or a default
                    records.append({
                        'Algorithm': algo_name,
                        'Edge Count': edge_count
                    })
    
    if not records:
        print("[Warn] No runtime data found.")
        return

    df = pd.DataFrame(records)

    # 2. Print Statistics
    stats = df.groupby('Algorithm')['Edge Count'].agg(['mean', 'std', 'min', 'max'])
    print(stats)

def eval_eigs(raw_data, dim):
    """
    Extracts eigenvalues, prints statistics for:
      1. The smallest non-zero eigenvalue (D+2-th)
      2. The largest eigenvalue (Spectral Radius)
    """
    print("\n--- Evaluating Eigenvalues ---")
    records = []

    # 1. Extract Data
    for case in raw_data:
        if 'stress_matrices' in case:
            for algo_name, stress_mat in case['stress_matrices'].items():
                if stress_mat is None:
                    continue
                
                # KEY CHANGE: specific check for MC lists
                if isinstance(stress_mat, list):
                    # We skip the list version because we expect a separate key 
                    # (e.g., 'custom_mc_global') containing the combined matrix.
                    continue


                try:
                    
                    # Normalize to ensure fair comparison (usually sets max eig to 1)
                    stress_mat = stress_normalizer(stress_mat)
                    
                    # Compute and sort eigenvalues (Real parts only for symmetric matrices)
                    eigvals = np.linalg.eigvals(stress_mat)
                    eigvals = np.sort(np.real(eigvals)) # Ensure real and sorted

                    val_d2 = eigvals[dim + 1] # The (D+2)-th smallest       
                    val_max = eigvals[-1]     # The largest

                    records.append({
                        'Algorithm': algo_name,
                        'Eigenvalues': eigvals,
                        'lambda_(d+2)': val_d2,
                        'lambda_max': val_max
                    })
                except Exception as e:
                    print(f"[Warn] Eig calc failed for {algo_name}: {e}")

    if not records:
        print("[Warn] No eigenvalue data found.")
        return

    df = pd.DataFrame(records)

   
    print(f"\n[Statistics: {dim+2}-th Smallest Eigenvalue (Connectivity Metric)]")
    stats_d2 = df.groupby('Algorithm')['lambda_(d+2)'].agg(['mean', 'std', 'min', 'max'])
    print(stats_d2)

    # B. Statistics for the Largest Eigenvalue (Spectral Radius)
    print(f"\n[Statistics: Largest Eigenvalue (Spectral Radius)]")
    stats_max = df.groupby('Algorithm')['lambda_max'].agg(['mean', 'std', 'min', 'max'])
    print(stats_max)

def plot_eigs_circular(raw_data, cfg, sparse_tol=1e-5):
    """
    Plots sorted eigenvalues using direct config lookups.
    """
    print("\n--- Plotting Eigenvalues ---")
    if not raw_data: return

    # 1. Randomly pick ONE case
    case = random.choice(raw_data)
    
    # 2. Get Config Maps
    # Safe lookup: checks root first, then 'evaluation'
    eval_cfg = cfg.get('evaluation', cfg) 
    color_map = eval_cfg.get('line_colors', {})
    marker_map = eval_cfg.get('line_markers', {})
    
    # 3. Fallbacks
    fallback_colors = itertools.cycle(['gray', 'black', 'brown'])
    fallback_markers = itertools.cycle(['.', 'x', '+'])

    plt.figure(figsize=(4, 5))

    if 'stress_matrices' in case:
        for algo_name, stress_mat in case['stress_matrices'].items():
            if stress_mat is None: continue
            
            # --- Filtering ---
            name_lower = algo_name.lower()
            if 'mc' in name_lower or 'usi' in name_lower: continue
            if isinstance(stress_mat, list): continue

            try:
                # Process Matrix
                stress_mat = stress_normalizer(stress_mat)
                edge_count = detect_num_edges(stress_mat, sparse_tol)
                eigvals = np.sort(np.real(np.linalg.eigvals(stress_mat)))
                
                # --- DIRECT LOOKUP (Simpler!) ---
                # 1. Color
                if algo_name in color_map:
                    color = color_map[algo_name]
                else:
                    color = next(fallback_colors)
                    print(f"[Info] Key '{algo_name}' not in config. Using fallback.")

                # 2. Marker
                if algo_name in marker_map:
                    marker = marker_map[algo_name]
                else:
                    marker = next(fallback_markers)

                # 3. Line Style
                linestyle = '-' if 'custom' in name_lower else '--'

                # Plot
                plt.plot(eigvals, 
                         label=f"{algo_name} ({edge_count} edges)", 
                         color=color,
                         marker=marker, 
                         linestyle=linestyle, 
                         markersize=5, 
                         alpha=1)
                
            except Exception as e:
                print(f"[Warn] Plotting failed for {algo_name}: {e}")

    plt.title("Sorted Eigenvalues")
    plt.xlabel("Index")
    plt.ylabel("Eigenvalue (Real Part)")
    # plt.legend()
    # plt.grid(True, linestyle='--', alpha=0.5)
    # show every 1 tick on x-axis, starting from 1
    plt.xticks(ticks=np.arange(len(eigvals)), labels=np.arange(1, len(eigvals) + 1))
    plt.tight_layout()
    
    # Save
    output_dir = eval_cfg.get('output_dir')
    plt.savefig(f'{output_dir}/eigs_circular-10.pdf', format='pdf', dpi=300)
    print(f">>> Saved to {output_dir}/eigs_circular-10.pdf")
    plt.show()

def plot_afc_error(raw_data, cfg):
    """
    Runs the AFC simulation for a random case and plots the error convergence.
    """
    print("\n--- Plotting AFC Tracking Error ---")
    if not raw_data:
        print("[Warn] No data available.")
        return

    # 1. Randomly pick ONE case
    case = random.choice(raw_data)
    target_config = case['nominal_config'] 
    
    # 2. Setup Leaders (First D+1 nodes are standard for affine control)
    dim = target_config.shape[0]
    leader_ids = list(range(dim + 1)) 
    init_config = 3* np.random.rand(dim, target_config.shape[1])

    # 3. Get Config Maps
    eval_cfg = cfg.get('evaluation', cfg) 
    color_map = eval_cfg.get('line_colors', {})
    marker_map = eval_cfg.get('line_markers', {})
    
    # 4. Fallbacks
    fallback_colors = itertools.cycle(['gray', 'black', 'brown'])
    fallback_markers = itertools.cycle(['.', 'x', '+'])

    plt.figure(figsize=(4, 5))
    
    plots_generated = False
    
    # 5. Iterate Algorithms
    if 'stress_matrices' in case:
        for algo_name, stress_mat in case['stress_matrices'].items():
            if stress_mat is None: continue
            
            # --- Filtering ---
            name_lower = algo_name.lower()
            if 'mc' in name_lower or 'usi' in name_lower: continue
            if isinstance(stress_mat, list): continue

            try:
                # Normalize Stress Matrix
                stress_mat = stress_normalizer(stress_mat)
                
                
                error = tracking_error(affine_formation_control_global(
                    target_config, 
                    stress_mat, 
                    leader_ids, 
                    gain=cfg['evaluation']['afc_params']['gain'], 
                    max_itr=cfg['evaluation']['afc_params']['max_itr'], # Cap iterations for plotting speed
                    dt=cfg['evaluation']['afc_params']['dt'],
                    init_config=init_config.copy()
                ), target_config)

                # --- STYLE LOOKUP ---
                # 1. Color
                if algo_name in color_map:
                    color = color_map[algo_name]
                else:
                    color = next(fallback_colors)


                # 3. Line Style
                linestyle = '-' if 'custom' in name_lower else '--'

                # Plot (Log Scale for Y is standard for convergence)
                time = np.arange(len(error)) * cfg['evaluation']['afc_params']['dt']
                
                plt.semilogy(time, error, 
                             label=algo_name, 
                             color=color,
                             linestyle=linestyle, 
                             linewidth=1.5,
                             alpha=1)
                
                print(f"    > Plotted {algo_name} (Final Error: {error[-1]:.2e})")

            except Exception as e:
                print(f"[Warn] AFC Simulation failed for {algo_name}: {e}")


    plt.title("AFC Tracking Error Convergence")
    plt.xlabel("Time [s]")
    plt.ylabel("Tracking Error (Log Scale)")
    
    # plt.grid(True, which="both", linestyle='--', alpha=0.4)
    # plt.legend(fontsize='small')
    plt.tight_layout()
    
    # Save
    output_dir = eval_cfg.get('output_dir', 'results')
    save_path = f'{output_dir}/afc_error_circular_10.pdf'
    plt.savefig(save_path, format='pdf', dpi=300)
    print(f">>> Saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze benchmark results")
    # The config argument is now dynamic
    parser.add_argument('--config', type=str, default='configs/benchmarks/random-5.yaml', 
                        help='Path to the benchmark config file (e.g., configs/benchmarks/random-5.yaml)')
    args = parser.parse_args()
    cfg = load_config(args.config)
    input_dir = cfg['evaluation']['input_dir']
    output_dir = cfg['evaluation']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    sparse_tol = float(cfg['evaluation'].get('sparse_tol'))
    print(f"Starting Evaluation: {cfg['experiment']['name']}")
    print(f"Config: {args.config}")
    print("-" * 70)

    keywords = cfg['experiment']['name']

    raw_data = load_benchmark_data(input_dir, keywords)
    # eval_runtime(raw_data)
    # eval_edge_count(raw_data, sparse_tol=cfg['evaluation'].get('sparse_tol', 1e-5))
    # eval_eigs(raw_data, cfg['experiment']['dim'])

    # if it is circuar, make a plot
    if cfg['geometry']['type'] == 'circular':
        
        plot_eigs_circular(raw_data, cfg, sparse_tol=sparse_tol)

        plot_afc_error(raw_data, cfg)

        