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

    # Extract Data
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

    # Print Statistics
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

def eval_eigs(raw_data):
    """
    Extracts eigenvalues, prints statistics for:
      1. The smallest non-zero eigenvalue (D+2-th)
      2. The largest eigenvalue (Spectral Radius)
    """
    print("\n--- Evaluating Eigenvalues ---")
    records = []

    # Extract Data
    for case in raw_data:
        geom_type = case.get('geometry_type', 'unknown')
        num_nodes = case['nominal_config'].shape[1]
        dim = case['nominal_config'].shape[0]

        if 'stress_matrices' in case:
            for algo_name, stress_mat in case['stress_matrices'].items():
                if stress_mat is None:
                    continue
                
                if isinstance(stress_mat, list):
                    continue


                try:
                    
                    # Normalize to ensure fair comparison (usually sets max eig to 1)
                    stress_mat = stress_normalizer(stress_mat)
                    
                    # Compute and sort eigenvalues (Real parts only for symmetric matrices)
                    eigvals = np.linalg.eigvals(stress_mat)
                    eigvals = np.sort(np.real(eigvals)) # Ensure real and sorted

                    val_d2 = eigvals[dim + 1] # The (D+2)-th smallest       
                    val_max = eigvals[-1]     # The largest

                    sparse_tol = float(cfg['evaluation']['sparse_tol']) # Or pull from cfg if you pass it
                    edge_count = detect_num_edges(stress_mat, sparse_tol)

                    records.append({
                        'Algorithm': algo_name,
                        'geometry_type': geom_type,  # Added
                        'N': num_nodes,              # Added
                        'lambda_d2': val_d2,
                        'eigenvalues': eigvals,
                        'lambda_max': val_max,
                        'edge_count': edge_count,
                        'max_edge_count': num_nodes * (num_nodes - 1) // 2
                    })
                except Exception as e:
                    print(f"[Warn] Eig calc failed for {algo_name}: {e}")

    if not records:
        print("[Warn] No eigenvalue data found.")
        return

    df = pd.DataFrame(records)

   
    print(f"\n[Statistics: {dim+2}-th Smallest Eigenvalue (Connectivity Metric)]")
    stats_d2 = df.groupby('Algorithm')['lambda_d2'].agg(['mean', 'std', 'min', 'max'])
    print(stats_d2)

    # Statistics for the Largest Eigenvalue (Spectral Radius)
    print(f"\n[Statistics: Largest Eigenvalue (Spectral Radius)]")
    stats_max = df.groupby('Algorithm')['lambda_max'].agg(['mean', 'std', 'min', 'max'])
    print(stats_max)

    return df

def plot_eigs_circular(raw_data, cfg, sparse_tol=1e-5):
    """
    Plots sorted eigenvalues using direct config lookups.
    """
    print("\n--- Plotting Eigenvalues ---")
    if not raw_data: return

    # Randomly pick a case
    case = random.choice(raw_data)
    
    # Get Config Maps
    eval_cfg = cfg.get('evaluation', cfg) 
    color_map = eval_cfg.get('line_colors', {})
    marker_map = eval_cfg.get('line_markers', {})
    
    # Fallbacks
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
                
                if algo_name in color_map:
                    color = color_map[algo_name]
                else:
                    color = next(fallback_colors)
                    print(f"[Info] Key '{algo_name}' not in config. Using fallback.")

                if algo_name in marker_map:
                    marker = marker_map[algo_name]
                else:
                    marker = next(fallback_markers)

                linestyle = '-' if 'custom' in name_lower else '--'

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

    # Randomly pick 1 case
    case = random.choice(raw_data)
    target_config = case['nominal_config'] 
    
    # Setup Leaders (First D+1 nodes are standard for affine control)
    dim = target_config.shape[0]
    leader_ids = list(range(dim + 1)) 
    init_config = 3* np.random.rand(dim, target_config.shape[1])

    # Get Config Maps
    eval_cfg = cfg.get('evaluation', cfg) 
    color_map = eval_cfg.get('line_colors', {})
    marker_map = eval_cfg.get('line_markers', {})
    
    # Fallbacks
    fallback_colors = itertools.cycle(['gray', 'black', 'brown'])
    fallback_markers = itertools.cycle(['.', 'x', '+'])

    plt.figure(figsize=(4, 5))
    
    plots_generated = False
    
    # Iterate Algorithms
    if 'stress_matrices' in case:
        for algo_name, stress_mat in case['stress_matrices'].items():
            if stress_mat is None: continue
            
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
                    max_itr=cfg['evaluation']['afc_params']['max_itr'],
                    dt=cfg['evaluation']['afc_params']['dt'],
                    init_config=init_config.copy()
                ), target_config)

                if algo_name in color_map:
                    color = color_map[algo_name]
                else:
                    color = next(fallback_colors)

                linestyle = '-' if 'custom' in name_lower else '--'

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
    # plt.show()

def plot_all_metrics(df, cfg):
    """
    Plots Connectivity AND Spectral Efficiency on an ordinal (categorical) x-axis.
    Includes: Random, Circular, Cuboctahedron, Truncated Icosahedron.
    """
    if df is None or df.empty:
        print("[Warn] DataFrame is empty.")
        return

    print("\n--- Plotting Metrics on Ordinal Axis (All Geometries) ---")

    # DATA FILTERING & PREPARATION
    
    # Filter Geometries
    target_geoms = [
        'random', 'circular', 'cuboctahedron', 
        'truncated_icosahedron', 'truncated-icosahedron'
    ]
    plot_df = df[df['geometry_type'].str.lower().isin(target_geoms)].copy()
    
    if plot_df.empty:
        print("[Warn] No matching geometry cases found.")
        return

    # Filter Algorithms (Remove USI & raw lists)
    plot_df = plot_df[~plot_df['Algorithm'].str.lower().str.contains('usi')]
    
    exclude_list = [f'custom_mc_{i}' for i in range(10)]
    plot_df = plot_df[~plot_df['Algorithm'].isin(exclude_list)]

    # Rename for Legend & Linking
    # 'custom_mc_1_global' -> 'custom_mc_1'
    plot_df['Algorithm'] = plot_df['Algorithm'].str.replace('_global', '', regex=False)

    #  Calculate Spectral Efficiency
    # Formula: N^2 * lambda_{d+2} / (lambda_max * edge_count)
    valid_mask = (plot_df['lambda_max'] > 0) & (plot_df['edge_count'] > 0)
    plot_df.loc[valid_mask, 'spectral_eff'] = (
        (plot_df.loc[valid_mask, 'N']**2 * plot_df.loc[valid_mask, 'lambda_d2']) / 
        (plot_df.loc[valid_mask, 'lambda_max'] * plot_df.loc[valid_mask, 'edge_count'])
    )

    # normalized edge count 
    n_val = plot_df['N']
    max_edges = n_val * (n_val - 1) / 2
    plot_df['norm_edge_count'] = plot_df['edge_count'] / max_edges

    # average degree 
    plot_df['avg_degree'] = 2 * plot_df['edge_count'] / plot_df['N']



    # CREATE ORDINAL X-AXIS MAPPING
    # Get all unique (Geometry, N) pairs
    unique_conditions = plot_df[['geometry_type', 'N']].drop_duplicates()
    
    # Sort primarily by N, secondarily by geometry name
    unique_conditions = unique_conditions.sort_values(by=['N', 'geometry_type'])
    
    # Assign integer index (0, 1, 2...)
    unique_conditions['x_idx'] = range(len(unique_conditions))
    
    # Generate Labels
    def format_label(row):
        name = row['geometry_type'].replace('_', ' ').replace('-', ' ').title()
        if "Truncated" in name: name = "Trunc.\nIcos."
        elif "Cuboctahedron" in name: name = "Cuboct."
        elif "Circular" in name: name = "Circ."
        elif "Random" in name: name = "Rand."
        return f"{name}\n({row['N']})"

    xtick_labels = unique_conditions.apply(format_label, axis=1).tolist()
    xtick_indices = unique_conditions['x_idx'].tolist()

    # Merge index back to main DF
    plot_df = pd.merge(plot_df, unique_conditions, on=['geometry_type', 'N'])

    # PLOTTING LOOP (Metric by Metric)
    metrics_config = [
        {
            'col': 'lambda_d2',
            'title': 'Network Connectivity vs. Configuration',
            'ylabel': 'Smallest Non-Zero Eigenvalue',
            'filename': 'lambda_dplus2.pdf',
            'ylim': [1e-4, 5e0]
        },
        {
            'col': 'spectral_eff',
            'title': 'Spectral Efficiency vs. Configuration',
            'ylabel': 'Spectral Efficiency',
            'filename': 'spectral_efficiency.pdf',
            'ylim': [1e-3, 1e1]
        },
        {
            # --- Edge Count ---
            'col': 'norm_edge_count',
            'ylabel': 'Normalized Edge Count ($E / E_{max}$)',
            'filename': 'edge_density.pdf',
            'ylim': [1e-3, 5e0] # Sparsity can be very low for large N
        },
        {
            # --- Average Degree ---
            'col': 'avg_degree',
            'ylabel': 'Average Degree ($2E/N$)',
            'filename': 'avg_degree.pdf',
            'ylim': [1e0, 1e3] # Average degree can vary widely
        }
    ]

    # Load Styles
    eval_cfg = cfg.get('evaluation', cfg) 
    color_map = eval_cfg.get('line_colors', {})
    marker_map = eval_cfg.get('line_markers', {})
    output_dir = eval_cfg.get('output_dir', 'results')
    os.makedirs(output_dir, exist_ok=True)

    for metric in metrics_config:
        col = metric['col']
        print(f"  > Plotting {col}...")

        # Group by Algorithm + Ordinal Index
        sub_df = plot_df.dropna(subset=[col])
        grouped = sub_df.groupby(['Algorithm', 'x_idx'])[col].agg(['mean', 'std']).reset_index()
        grouped['std'] = grouped['std'].fillna(0)

        plt.figure(figsize=(8, 4)) # Wider for labels
        
        fallback_colors = itertools.cycle(['gray', 'black', 'brown', 'purple'])
        fallback_markers = itertools.cycle(['.', 'x', '+', '*'])
        
        # Store endpoints for bridging
        plotted_data = {}

        for algo_name in grouped['Algorithm'].unique():
            algo_data = grouped[grouped['Algorithm'] == algo_name].sort_values('x_idx')
            
            x = algo_data['x_idx'].values
            y_mean = algo_data['mean'].values
            y_std = algo_data['std'].values
            
            # Styles
            name_lower = algo_name.lower()
            color = color_map.get(algo_name, next(fallback_colors))
            marker = marker_map.get(algo_name, next(fallback_markers))
            linestyle = '-' if 'custom' in name_lower else '--'

            # Plot
            plt.plot(x, y_mean, label=algo_name, color=color, marker=marker, 
                     linestyle=linestyle, linewidth=1.5, markersize=8)
            
            if np.any(y_std > 0):
                plt.fill_between(x, y_mean - y_std, y_mean + y_std, color=color, alpha=0.2)

            # Save endpoints
            if len(x) > 0:
                plotted_data[algo_name] = {
                    'end_x': x[-1], 'end_y': y_mean[-1],
                    'start_x': x[0], 'start_y': y_mean[0],
                    'color': color
                }

        # Bridge Gaps (custom_X -> custom_mc_X)
        for i in range(1, 10):
            base_name = f"custom_{i}"
            mc_name = f"custom_mc_{i}"
            
            if base_name in plotted_data and mc_name in plotted_data:
                x1, y1 = plotted_data[base_name]['end_x'], plotted_data[base_name]['end_y']
                x2, y2 = plotted_data[mc_name]['start_x'], plotted_data[mc_name]['start_y']
                c = plotted_data[base_name]['color']
                
                plt.plot([x1, x2], [y1, y2], color=c, linestyle='-', linewidth=1.5, alpha=0.8)

        if col == 'avg_degree':
            x_vals = unique_conditions['x_idx'].values
            # Max average degree for a complete graph is N - 1
            max_avg_degree = unique_conditions['N'].values - 1 
            
            plt.plot(x_vals, max_avg_degree, label='Complete Graph', 
                     color='black', linestyle=':', linewidth=2, alpha=0.7)
        
        # Formatting
        # plt.title(metric['title'])
        plt.ylabel(metric['ylabel'])
        plt.xlabel("Configuration (Nodes)")
        
        # Apply Categorical Ticks
        plt.xticks(ticks=xtick_indices, labels=xtick_labels, fontsize=9)
        
        plt.grid(True, linestyle='--', alpha=0.3)
        # plt.legend(fontsize='small', loc='best')
        plt.tight_layout()
        plt.yscale('log')
        plt.ylim(bottom=metric['ylim'])

        import matplotlib.ticker as mtick
        if col == 'norm_edge_count':
            # Format as percentage: 0.1 -> "10%", 0.01 -> "1%"
            # remove_trailing_zeros ensures "10%" not "10.00%"
            def percent_fmt(x, pos):
                return f"{x*100:g}%"
            
            plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(percent_fmt))
        # -------------------------------------------------------

        # Save
        save_path = f'{output_dir}/{metric["filename"]}'
        plt.savefig(save_path, format='pdf', dpi=300)
        print(f">>> Saved to {save_path}")
        plt.close()

def plot_spectral_efficiency_all(df, cfg):
    """
    Plots the Spectral Efficiency vs nodes.
    Spectral Efficiency = lambda_{d+2} / (lambda_max * edge_count)
    """
    print("\n--- Plotting Spectral Efficiency vs. Number of Nodes ---")
    if df is None or df.empty:
        print("[Warn] DataFrame is empty. Cannot plot scaling.")
        return

    # Filter using Pandas
    plot_df = df[df['geometry_type'] == 'random'].copy()
    
    if plot_df.empty:
        print("[Warn] No 'random' geometry cases found in data.")
        return

    # Filter out analytical USI and the uncombined MC matrix
    plot_df = plot_df[~plot_df['Algorithm'].str.lower().str.contains('usi')]
    plot_df = plot_df[plot_df['Algorithm'].str.lower() != 'custom_mc']
    
    # Rename custom_mc_global
    plot_df['Algorithm'] = plot_df['Algorithm'].replace({'custom_mc_global': 'custom_mc', 'custom_mc_global'.upper(): 'custom_mc'})

    # Filter out zeros just to be safe from division by zero errors
    valid_mask = (plot_df['lambda_max'] > 0) & (plot_df['edge_count'] > 0)
    plot_df = plot_df[valid_mask]
    
    plot_df['spectral_eff'] = plot_df['N']**2*plot_df['lambda_d2'] / (plot_df['lambda_max'] * plot_df['edge_count'])

    # Group by Algorithm and Nodes to get Mean/Std
    grouped = plot_df.groupby(['Algorithm', 'N'])['spectral_eff'].agg(['mean', 'std']).reset_index()
    grouped['std'] = grouped['std'].fillna(0)

    # Setup Config Styles
    eval_cfg = cfg.get('evaluation', cfg) 
    color_map = eval_cfg.get('line_colors', {})
    marker_map = eval_cfg.get('line_markers', {})
    
    fallback_colors = itertools.cycle(['gray', 'black', 'brown', 'purple'])
    fallback_markers = itertools.cycle(['.', 'x', '+', '*'])

    # Plot
    plt.figure(figsize=(6, 5))
    
    for algo_name in grouped['Algorithm'].unique():
        algo_data = grouped[grouped['Algorithm'] == algo_name].sort_values('N')
        
        x = algo_data['N'].values
        y_mean = algo_data['mean'].values
        y_std = algo_data['std'].values
        
        # Styles
        name_lower = algo_name.lower()
        color = color_map.get(algo_name, next(fallback_colors))
        marker = marker_map.get(algo_name, next(fallback_markers))
        linestyle = '-' if 'custom' in name_lower else '--'

        plt.plot(x, y_mean, label=algo_name, color=color, marker=marker, 
                 linestyle=linestyle, linewidth=1.5, markersize=6)
        
        if np.any(y_std > 0):
            plt.fill_between(x, y_mean - y_std, y_mean + y_std, color=color, alpha=0.2)

    plt.title("Spectral Efficiency vs. Network Size (Random Geometries)")
    plt.xlabel("Number of Nodes ($N$)")
    plt.ylabel("Spectral Efficiency")
    
    unique_nodes = np.sort(plot_df['N'].unique())
    plt.xticks(unique_nodes)
    
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(fontsize='small')
    plt.tight_layout()
    plt.yscale('log') 
    plt.ylim(bottom=1e-4)  # Set a lower limit for better visibility of small values 
    plt.xscale('log')  
    
    output_dir = eval_cfg.get('output_dir', 'results')
    os.makedirs(output_dir, exist_ok=True)
    save_path = f'{output_dir}/spectral_efficiency_all.pdf'
    plt.savefig(save_path, format='pdf', dpi=300)
    print(f">>> Saved to {save_path}")
    # plt.show()
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Analyze benchmark results")
    parser.add_argument('--config', type=str, default='configs/benchmarks/eval_all_metrics.yaml', 
                        help='Path to the benchmark config file')
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    input_dir = cfg['evaluation']['input_dir']
    output_dir = cfg['evaluation']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    sparse_tol = float(cfg['evaluation'].get('sparse_tol', 1e-5))
    exp_name = cfg['experiment']['name']
    
    print(f"Starting Evaluation: {exp_name}")
    print(f"Config: {args.config}")
    print("-" * 70)

    # --- GLOBAL EVALUATION MODE ---
    if exp_name == "eval_all_metrics":
        print(">>> Running GLOBAL Evaluation across all generated benchmark files...")
        raw_data = load_benchmark_data(input_dir, "*")


        df_eigs = eval_eigs(raw_data)
        
        # Run global analytics
        # eval_runtime(raw_data)
        # eval_edge_count(raw_data, sparse_tol=sparse_tol)
        
        plot_all_metrics(df_eigs, cfg)
        # plot_spectral_efficiency_all(df_eigs, cfg)
        
        print(">>> Global Evaluation Complete.")

    # --- SINGLE EXPERIMENT MODE ---
    else:
        # Normal behavior for a specific experiment (e.g., "benchmark-circular-10")
        raw_data = load_benchmark_data(input_dir, exp_name)
        
        eval_runtime(raw_data)
        eval_edge_count(raw_data, sparse_tol=sparse_tol)
        eval_eigs(raw_data)

        # Geometry-specific plots
        if cfg.get('geometry', {}).get('type') == 'circular':
            plot_eigs_circular(raw_data, cfg, sparse_tol=sparse_tol)
            plot_afc_error(raw_data, cfg)

        