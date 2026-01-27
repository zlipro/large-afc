import numpy as np
import cvxpy as cp

from src.utils.utils import incidence_matrix_fully_connected, get_symmetry_mapping_matrix

def optimize_stress(nominal_config: np.ndarray, hyperparam_dict: dict, verbose=False):

        """
        Our proposed framework to calculate the sparse stress matrix.
        cvxpy is needed for the convex optimization.
        
        nominal_config: np.ndarray
            The nominal configuration of shape (D, N) where D is the dimension and N is the number of nodes.
        hyperparam_dict: dict
            Dictionary containing the hyperparameters for the optimization
            alpha: sparsity-convergence trade-off; beta: largest eigenvalue; gamma: smallest eigenvalue 
        """

        D, N = nominal_config.shape
        incidence_full = incidence_matrix_fully_connected(N)
        
        M = incidence_full.shape[1]
        
        augmented_config = np.hstack((nominal_config.T, np.ones((N, 1))))
        U, _, _ = np.linalg.svd(augmented_config)
        Q = U[:, D+1:]
        
        E = np.vstack([augmented_config.T @ incidence_full @ np.diag(incidence_full.T[:, n]) for n in range(N)])
        Psi = Q.T @ incidence_full
        
        phi = np.diag(Psi.T @ Psi)

        ########## Convex Optimization ##########
        weights = cp.Variable(M)
        
        obj = cp.Minimize(cp.norm(weights, 1) - hyperparam_dict['alpha'] * phi.T @ weights)
        constraints = [
            cp.norm(incidence_full @ cp.diag(weights) @ incidence_full.T, 2) <= hyperparam_dict['beta'],
            Psi @ cp.diag(weights) @ Psi.T - hyperparam_dict['gamma'] * np.eye(N - D - 1) >> 0,
            E @ weights == 0
        ]
        
        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.SCS)

        weights_optimal = weights.value

        if verbose:
            print(f"suggested alpha: {1/np.max(phi)}")

        # weights_optimal[np.abs(weights_optimal) < sparse_tol] = 0   # small weights to zero for plotting purpose

        stress_mat = incidence_full @ np.diag(weights_optimal) @ incidence_full.T
        return stress_mat
    
def optimized_stress_symm(nominal_config: np.ndarray, hyperparam_dict: dict, verbose=False):
    """
    The Unique Stress Identifier (USI) method (Section IV, Algorithm P3).
    Optimizes stress using symmetry groups identified via EDM.
    
    Args:
        nominal_config: np.ndarray (D, N)
        hyperparam_dict: dict with 'alpha', 'beta', 'gamma'
        
    Returns:
        stress_mat: np.ndarray (N, N)
    """
    D, N = nominal_config.shape
    
    # 1. Setup Basic Matrices (Full Graph)
    incidence_full = incidence_matrix_fully_connected(N) # B matrix
    M = incidence_full.shape[1]
    
    # 2. Symmetry Reduction (EDM Shortcut)
    # Identify unique edge classes S based on geometry [cite: 267, 270]
    S_map, unique_dists = get_symmetry_mapping_matrix(nominal_config)
    K_groups = S_map.shape[1]
    
    if verbose:
        print(f"USI Method: Reduced variables from {M} edges to {K_groups} unique groups.")
    
    # 3. Constraint Construction
    # Nullspace basis Q
    augmented_config = np.hstack((nominal_config.T, np.ones((N, 1))))
    U, _, _ = np.linalg.svd(augmented_config)
    Q = U[:, D+1:]
    
    # E matrix for equilibrium constraints
    E_full = np.vstack([augmented_config.T @ incidence_full @ np.diag(incidence_full.T[:, n]) for n in range(N)])
    
    # Psi matrix for PSD constraints
    Psi = Q.T @ incidence_full
    
    # Pre-calculate phi for the objective function term
    # phi = diag(Psi.T @ Psi) [cite: 116]
    phi = np.diag(Psi.T @ Psi)

    ########## Convex Optimization (P3) ##########
    # Variable is reduced_weights (omega_underbar) [cite: 271]
    reduced_weights = cp.Variable(K_groups)
    
    # Map back to full weights: omega_bar = S @ omega_underbar [cite: 276]
    weights = S_map @ reduced_weights
    
    # Objective: Minimize L1 norm - alpha * convergence term [cite: 274]
    # Note: cp.norm(weights, 1) automatically handles the mapping S
    obj = cp.Minimize(cp.norm(weights, 1) - hyperparam_dict['alpha'] * phi.T @ weights)
    
    constraints = [
        # Spectral norm bound [cite: 278]
        cp.norm(incidence_full @ cp.diag(weights) @ incidence_full.T, 2) <= hyperparam_dict['beta'],
        
        # PSD Constraint (projected) [cite: 277]
        Psi @ cp.diag(weights) @ Psi.T - hyperparam_dict['gamma'] * np.eye(N - D - 1) >> 0,
        
        # Equilibrium Constraint [cite: 279]
        E_full @ weights == 0
    ]
    
    prob = cp.Problem(obj, constraints)
    
    # Using SCS or CLARABEL as they handle PSD cones well
    try:
        prob.solve(solver=cp.SCS, verbose=verbose)
    except:
        prob.solve(solver=cp.CVXOPT, verbose=verbose)

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        print(f"Warning: Problem status is {prob.status}")
        return None

    # Recover full weights
    weights_optimal = weights.value
    
    # Construct Stress Matrix
    stress_mat = incidence_full @ np.diag(weights_optimal) @ incidence_full.T
    
    return stress_mat


def compute_stress_multicluster(target_config, subgraph_glb_idx, hyperparam_dict, parallel=True):
    """
    loop over subgraphs and compute stress matrices and use parallel when specified.

    """
    
    if not parallel:
        stress_mat_list = []
        for subgraph in subgraph_glb_idx:
            nominal_subconfig = target_config[:, subgraph]

            stress_mat = optimize_stress(nominal_subconfig, hyperparam_dict)
            stress_mat_list.append(stress_mat)
        return stress_mat_list
    
    else:
        import psutil, multiprocessing
        from threadpoolctl import threadpool_limits
        
        N_CORES = psutil.cpu_count(logical=False)
        processes_needed = len(subgraph_glb_idx)
        
        print(f"Starting parallel optimization on {processes_needed} tasks...")

        # --- PREPARE THE DATA ---
        # We move the data preparation (slicing) here, into the "big function".
        # We create a list of tuples: [(config_1, params), (config_2, params), ...]
        tasks_args = [
            (target_config[:, subgraph], hyperparam_dict) 
            for subgraph in subgraph_glb_idx
        ]
        
        # --- EXECUTE ---
        # threadpool_limits here ensures that if the system forks (Linux), 
        # the children inherit the thread limits.
        with threadpool_limits(limits=6, user_api='blas'):
            
            # We use min() to avoid creating more processes than necessary
            with multiprocessing.Pool(processes=min(processes_needed, N_CORES)) as pool:
                
                # starmap is the key: it unpacks the tuple arguments automatically
                # equivalent to: optimize_stress(arg1, arg2)
                stress_mat_list = pool.starmap(optimize_stress, tasks_args)

        return stress_mat_list
    

    
