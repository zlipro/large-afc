import argparse
import yaml
import os
import numpy as np
import time
import warnings
# Filter out the specific internal MOSEK warnings
warnings.filterwarnings("ignore", message=".*Argument sub[ij] in putaijlist64.*")
# import sys

# # Ensure project root is in path
# sys.path.append(os.getcwd()) 

import src.utils.geometry as geometry
from src import stress
from src.sota import yang2019, xiao2022, li2026, lin2016
from src.utils.utils import incidence_matrix_fully_connected, find_global_stress_mat, random_multicluster_indices

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def run_case(cfg, run_id, solver_obj=None):
    """
    Runs one simulation case.
    Returns a dictionary containing:
      - 'nominal_config': The geometry used
      - 'stress_matrices': result matrices from each algorithm
      - 'runtimes': execution time in seconds for each algo
    """
    
    # --- Setup Geometry ---
    np.random.seed(run_id) 

    if cfg['geometry']['type'] == 'circular':
        num_nodes = int(cfg['geometry']['num_nodes'])
        playground_size = float(cfg['geometry']['radius'])
        nominal_config = geometry.circular_config(num_nodes, radius=playground_size)
    elif cfg['geometry']['type'] == 'random':   
        num_nodes = int(cfg['geometry']['num_nodes'])
        playground_size = float(cfg['geometry']['radius'])
        nominal_config = geometry.random_config(num_nodes, playground_size = playground_size)
    elif cfg['geometry']['type'] == 'cuboctahedron':
        nominal_config = geometry.cuboctahedron()
        num_nodes = nominal_config.shape[1]
    elif cfg['geometry']['type'] == 'truncated-icosahedron':
        nominal_config = geometry.truncated_icosahedron()
        num_nodes = nominal_config.shape[1]

    # Initialize results structure
    results = {
        "geometry_type": cfg['geometry']['type'],
        "nominal_config": nominal_config,
        "stress_matrices": {},
        "runtimes": {}
    }
    
    # --- Run Proposed Algorithms ---
    custom_algs = cfg.get('custom_algs', {})
    if custom_algs.get('custom_generic'):
        for i in range(1, 4): # Loops 1, 2, 3
            param_key = f'opt_hyperparam_set_{i}'
            hparams = cfg.get(param_key)
            
            if hparams is not None:
                # print(f"    > Running Proposed-Generic with Hparam Set {i}...")
                t0 = time.perf_counter()
                # Assuming optimize_stress accepts verbose arg
                my_stress = stress.optimize_stress(nominal_config, hparams) 
                t1 = time.perf_counter()
                
                # Save as custom_1, custom_2, custom_3
                results['stress_matrices'][f'custom_{i}'] = my_stress
                results['runtimes'][f'custom_{i}'] = t1 - t0
    if custom_algs.get('custom_usi'):
        hparams = cfg.get('opt_hyperparam_set_2')     # use a generic hyperparam set
        if hparams is not None:
            
            t0 = time.perf_counter()
            my_stress = stress.optimized_stress_symm_acc(nominal_config, hparams)
            t1 = time.perf_counter()

            results['stress_matrices']['custom_usi'] = my_stress
            results['runtimes']['custom_usi'] = t1 - t0
    if custom_algs.get('custom_mc'):
        hparams = cfg.get('opt_hyperparam_set_2')     # use a generic hyperparam set
        if hparams is not None:
            # print(f"    > Running Proposed-MC...")
            num_clusters = cfg.get('custom_mc_params').get('num_clusters')
            node_per_cluster = cfg.get('custom_mc_params').get('node_per_cluster')
            if cfg['geometry']['type'] == 'random':   
                subgraph_glb_idx = random_multicluster_indices(num_nodes, num_clusters, node_per_cluster)

                
            # print(f"        > Subgraph Indices: {subgraph_glb_idx}")
            t0 = time.perf_counter()
            my_stress = stress.compute_stress_multicluster(nominal_config, 
                                                           subgraph_glb_idx=subgraph_glb_idx, 
                                                           hyperparam_dict=hparams, 
                                                           parallel=False) 
            t1 = time.perf_counter()
            my_stress_global = find_global_stress_mat(my_stress, subgraph_glb_idx, num_nodes)

            # # print the first 5 eigenvalues of the global stress matrix
            # eigvals = np.linalg.eigvals(my_stress_global)
            # eigvals = np.sort(np.real(eigvals))
            # print(f"        > First 5 Eigenvalues of Global MC Stress Matrix: {eigvals[:5]}")


            # Store the consolidated global matrix (useful for eigenvalue analysis)
            results['stress_matrices']['custom_mc_global'] = my_stress_global
            results['stress_matrices']['custom_mc'] = my_stress
            results['runtimes']['custom_mc'] = t1 - t0

    # --- Run Baselines ---
    baselines = cfg['baselines']
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

    # Xiao 2022
    if baselines.get('xiao2022'):
        try:          
            t0 = time.perf_counter()
            raw_mat = xiao2022.misdp_cutting_plane(nominal_config, solver=solver_obj)
            t1 = time.perf_counter()
            results['stress_matrices']['xiao2022'] = raw_mat
            results['runtimes']['xiao2022'] = t1 - t0
        except Exception as e:
            print(f"    [!] Xiao2022 failed: {e}")
            results['stress_matrices']['xiao2022'] = np.zeros((num_nodes, num_nodes))
            results['runtimes']['xiao2022'] = 0.0
            results['metrics']['xiao2022_edges'] = 0

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
        # List of strategies to try in order: (Function Name, Incidence Generator)
        strategies = [
            ("Cauchy", lin2016.incidence_cauchy),
            ("Grunbaum", lin2016.incidence_grunbaum)
            # ("FullyConnected", incidence_matrix_fully_connected) # Assuming this function exists in your module
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
                
                # 3. Store Results if successful
                results['stress_matrices']['lin2016'] = raw_mat
                results['runtimes']['lin2016'] = t1 - t0
                
                # print(f"    [OK] Lin2016 succeeded with {name} graph.")
                success = True
                break # Stop trying other strategies
                
            except Exception as e:
                # print(f"    [!] Lin2016 ({name}) failed: {e}")
                continue # Try next strategy

        # If all strategies failed
        if not success:
            print(f"    [!] Lin2016 failed all strategies (Cauchy -> Grunbaum -> Full).")
            results['stress_matrices']['lin2016'] = None
            results['runtimes']['lin2016'] = None

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run random geometry benchmarks")
    parser.add_argument('--config', type=str, default='configs/benchmarks/random-5.yaml', help='Path to config file')
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = cfg['experiment']['output_dir']
    os.makedirs(out_dir, exist_ok=True)
    
    total_runs = cfg['experiment']['total_runs']
    print(f"Starting Benchmark: {cfg['experiment']['name']}")
    print(f"Config: {args.config}")
    # pring all the benchmarking algorithms
    print(f"Benchmarking Algorithms:")
    for algo in ['custom_generic', 'custom_usi', 'custom_mc', 'yang2019', 'li2026', 'lin2016', 'xiao2022']:
        if cfg['baselines'].get(algo, False) or cfg['custom_algs'].get(algo, False):
            print(f"  - {algo}")
    print(f"Total Runs: {total_runs}" + (f" | Nodes: {cfg['geometry']['num_nodes']}" if 'num_nodes' in cfg['geometry'] else ""))
    print("-" * 70)

    if cfg['baselines'].get('xiao2022'):
        print("Initializing solvers for Xiao2022...")
        detected_solver = None
        if cfg['baselines'].get('xiao2022'):
            detected_solver = xiao2022.check_solvers()
    else:
        detected_solver = None

    # a warm-up run, not recorded 
    stress.optimize_stress(np.random.rand(2, 10),{
                'alpha': 1.0,
                'beta': 1,
                'gamma': 0.1
            })
    print("Warm-up run completed. Starting timed runs...")

    for i in range(total_runs):
        case_data = run_case(cfg, run_id=i, solver_obj=detected_solver)
        
        # Save individually
        save_path = os.path.join(out_dir, f"{cfg['experiment']['name']}_{i:03d}.npy")
        np.save(save_path, case_data)
        
        # Logging with timing info
        if i % 1 == 0:
            times = case_data['runtimes']
            
            # Helper function to handle None values safely
            def fmt(val):
                return f"{val:.4f}s" if isinstance(val, (int, float)) else "None"

            print(f"Case {i:03d} | "
                  f"Custom-generic 1: {fmt(times.get('custom_1'))} | "
                  f"Custom-generic 2: {fmt(times.get('custom_2'))} | "
                  f"Custom-generic 3: {fmt(times.get('custom_3'))} | "
                  f"Custom-USI: {fmt(times.get('custom_usi'))} | "
                  f"Custom-MC: {fmt(times.get('custom_mc'))} | "
                  f"Yang: {fmt(times.get('yang2019'))} | "
                  f"Li: {fmt(times.get('li2026'))} | "
                  f"Lin: {fmt(times.get('lin2016'))} | "
                  f"Xiao: {fmt(times.get('xiao2022'))}")

    print("-" * 70)
    print(f"All {total_runs} cases saved to: {out_dir}")