import argparse
import yaml
import os
import numpy as np
import time
import warnings
# Filter out the specific internal MOSEK warnings
warnings.filterwarnings("ignore", message=".*Argument sub[ij] in putaijlist64.*")

import src.utils.geometry as geometry
from src import stress
from src.sota import yang2019, li2026, lin2016
from src.utils.utils import incidence_matrix_fully_connected, find_global_stress_mat, random_multicluster_indices

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def run_case(cfg, run_id):
    """
    Runs one simulation case for Linked Geometry.
    """
    
    # --- Setup Geometry ---
    np.random.seed(run_id) 
    
    # Initialize subgraph indices (will be filled for linked/MC cases)
    subgraph_glb_idx = None 

    if cfg['geometry']['type'] == 'truncated-icosahedron-linked':
        # This returns the full 3-unit geometry and the indices for each unit
        # Note: geometry.nested_football() must be the version returning (verts, indices, adj)
        # or (verts, indices) depending on your implementation. Assuming (verts, indices) here.
        # If it returns 3 items, adjust accordingly.
        nominal_config, subgraph_glb_idx = geometry.nested_football()       
        num_nodes = nominal_config.shape[1]
        print(f"Loaded Truncated Icosahedron Linked Geometry with {num_nodes} nodes and {len(subgraph_glb_idx)} subgraphs.")

    elif cfg['geometry']['type'] == 'letter-w':
        nominal_config_dict, subgraph_glb_idx = geometry.letter_w(filling_points_per_seg = cfg['geometry'].get('filling_points_per_seg'))
        nominal_config = nominal_config_dict['nominal_config_W']
        num_nodes = nominal_config.shape[1]
        print(f"Loaded Letter W Geometry with {num_nodes} nodes and {len(subgraph_glb_idx)} subgraphs.")


    # Initialize results structure
    results = {
        "geometry_type": cfg['geometry']['type'],
        "nominal_config": nominal_config,
        "stress_matrices": {},
        "runtimes": {}
    }
    
    custom_algs = cfg.get('custom_algs', {})

    # --- Custom MC (Multi-Cluster) ---
    if custom_algs.get('custom_mc'):
        hparams = cfg.get('opt_hyperparam_set')
        if hparams is not None:

            t0 = time.perf_counter()
            if cfg['geometry']['type'] == 'truncated-icosahedron-linked':
                my_stress_unit = stress.optimized_stress_symm_acc(
                    nominal_config[:, subgraph_glb_idx[0]], hparams)
            elif cfg['geometry']['type'] == 'letter-w':
                my_stress_unit = stress.optimize_stress(
                    nominal_config[:, subgraph_glb_idx[0]], hparams)
            t1 = time.perf_counter()
            my_stress_list = [my_stress_unit for _ in subgraph_glb_idx]
            
            my_stress_global = find_global_stress_mat(my_stress_list, subgraph_glb_idx, num_nodes)

            results['stress_matrices']['custom_mc'] = my_stress_list
            results['stress_matrices']['custom_mc_global'] = my_stress_global
            results['runtimes']['custom_mc'] = t1 - t0

    # ---  Run Baselines ---
    baselines = cfg.get('baselines', {})

    # Yang 2019
    if baselines.get('yang2019'):
        try:
            t0 = time.perf_counter()
            raw_mat = yang2019.sparse_nullspace_recon(nominal_config)
            t1 = time.perf_counter()          
            results['runtimes']['yang2019'] = t1 - t0
            results['stress_matrices']['yang2019'] = raw_mat
        except Exception as e:
            print(f"    [!] Yang2019 failed: {e}")
            results['stress_matrices']['yang2019'] = None
            results['runtimes']['yang2019'] = None

    # Li 2026
    if baselines.get('li2026'):
        try:
            t0 = time.perf_counter()
            raw_mat = li2026.vertex_additions(nominal_config)
            t1 = time.perf_counter()
            results['stress_matrices']['li2026'] = raw_mat
            results['runtimes']['li2026'] = t1 - t0
        except Exception as e:
             print(f"    [!] Li2026 failed: {e}")
             results['stress_matrices']['li2026'] = None
             results['runtimes']['li2026'] = None

    # Lin 2016 (With Fallback Strategy)
    if baselines.get('lin2016'):
        strategies = [
            ("Cauchy", lin2016.incidence_cauchy),
            ("Grunbaum", lin2016.incidence_grunbaum),
            ("FullyConnected", incidence_matrix_fully_connected)
        ]
        
        success = False
        for name, strategy_func in strategies:
            try:
                # 1. Generate Incidence
                incidence = strategy_func(num_nodes)
                # 2. Optimize
                t0 = time.perf_counter()
                raw_mat = lin2016.optimize_with_incidence(incidence, nominal_config)
                t1 = time.perf_counter()
                
                results['stress_matrices']['lin2016'] = raw_mat
                results['runtimes']['lin2016'] = t1 - t0
                success = True
                break
            except Exception as e:
                continue

        if not success:
            print(f"    [!] Lin2016 failed all strategies.")
            results['stress_matrices']['lin2016'] = None
            results['runtimes']['lin2016'] = None

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run linked geometry benchmarks")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = cfg['experiment']['output_dir']
    os.makedirs(out_dir, exist_ok=True)
    
    total_runs = cfg['experiment']['total_runs']
    print(f"Starting Benchmark: {cfg['experiment']['name']}")
    print(f"Config: {args.config}")
    
    # Identify algorithms to run
    print(f"Benchmarking Algorithms:")
    for algo in ['custom_mc', 'yang2019', 'li2026', 'lin2016']:
        if cfg['baselines'].get(algo, False) or cfg['custom_algs'].get(algo, False):
            print(f"  - {algo}")

    if 'num_nodes' in cfg['geometry']:
         print(f"Total Runs: {total_runs} | Nodes: {cfg['geometry']['num_nodes']}")
    else:
         print(f"Total Runs: {total_runs} | Nodes: [Auto-Detected]")
         
    print("-" * 70)

    # Warm-up (generic)
    stress.optimize_stress(np.random.rand(2, 10), {'alpha':1.0, 'beta':1, 'gamma':0.1})
    print("Warm-up run completed. Starting timed runs...")

    for i in range(total_runs):
        case_data = run_case(cfg, run_id=i)
        
        # Save individually
        save_path = os.path.join(out_dir, f"{cfg['experiment']['name']}_{i:03d}.npy")
        np.save(save_path, case_data)
        
        # Logging
        if i % 1 == 0:
            times = case_data['runtimes']
            def fmt(val): return f"{val:.4f}s" if isinstance(val, (int, float)) else "None"

            print(f"Case {i:03d} | "
                  f"Custom-MC: {fmt(times.get('custom_mc'))} | "
                  f"Yang: {fmt(times.get('yang2019'))} | "
                  f"Li: {fmt(times.get('li2026'))} | "
                  f"Lin: {fmt(times.get('lin2016'))}")

    print("-" * 70)
    print(f"All {total_runs} cases saved to: {out_dir}")