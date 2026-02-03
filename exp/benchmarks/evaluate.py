import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import argparse
from src.utils.utils import detect_num_edges, stress_normalizer

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

def eval_eigs(raw_data, dim, sparse_tol=1e-5):
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
                if stress_mat is not None:
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
    
    print(f"Starting Evaluation: {cfg['experiment']['name']}")
    print(f"Config: {args.config}")
    print("-" * 70)

    keywords = cfg['experiment']['name']

    raw_data = load_benchmark_data(input_dir, keywords)
    eval_runtime(raw_data)
    eval_edge_count(raw_data, sparse_tol=cfg['experiment'].get('sparse_tol', 1e-5))
    eval_eigs(raw_data, cfg['experiment']['dim'], sparse_tol=cfg['experiment'].get('sparse_tol', 1e-5))