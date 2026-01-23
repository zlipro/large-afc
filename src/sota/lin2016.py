import numpy as np
import cvxpy as cp

def incidence_grunbaum(N):
    """
    Constructs the oriented incidence matrix for a Grunbaum-Shephard polygon framework.
    
    The framework consists of:
    - N boundary edges (cycle).
    - 1 diagonal (0, 2).
    - A fan of diagonals from node 1 to nodes 3, 4, ..., N-1.
    
    Total Edges M = 2N - 2.
    
    Parameters:
    -----------
    N : int
        Number of nodes (must be >= 4 to form the specific Grunbaum structure).
        
    Returns:
    --------
    incidence_matrix : numpy.ndarray
        An (N x M) matrix where rows are nodes and columns are edges.
        Entries are 1 (start node) and -1 (end node).
    """
    
    if N < 4:
        raise ValueError("N must be at least 4 to define a Grunbaum polygon structure.")

    # 1. Define the Edges
    edges = []
    
    # Boundary Cycle: (0,1), (1,2), ..., (N-1,0)
    for i in range(N):
        u, v = i, (i + 1) % N
        edges.append((u, v))
        
    # Diagonal 1: Connects node 0 to node 2
    edges.append((0, 2))
    
    # Diagonal Fan: Connects node 1 to nodes 3, 4, ..., N-1
    # Note: If N=4, this loop runs for k=3 only.
    for k in range(3, N):
        edges.append((1, k))
        
    M = len(edges)
    
    # 2. Construct the Matrix
    # Shape: (N nodes, M edges)
    inc_mat = np.zeros((N, M), dtype=int)
    
    for edge_index, (u, v) in enumerate(edges):
        inc_mat[u, edge_index] = 1
        inc_mat[v, edge_index] = -1
        
    return inc_mat

def optimize_as_feasibility(incidence_mat, nominal_config):
    """
    Computes the stress matrix for a given nominal configuration using SDP.
    
    Args:
        incidence_mat (np.ndarray): (N x M) incidence matrix.
        nominal_config (np.ndarray): (D x N) nominal configuration matrix.
        
    Returns:
        np.ndarray: (N x N) Normalized stress matrix.
    """
    # Ensure inputs are numpy arrays
    incidence_mat = np.array(incidence_mat)
    nominal_config = np.array(nominal_config)
    
    # Get dimensions: N agents, M edges, D dimension
    N, M = incidence_mat.shape
    D, N_conf = nominal_config.shape
    
    if N != N_conf:
        raise ValueError(f"Shape mismatch: Incidence matrix has {N} nodes, config has {N_conf}.")

    # --- Pre-processing: Null Space Basis (Q) ---
    # Construct augmented configuration: [p_1 ... p_N; 1 ... 1]^T
    # Shape: N x (D+1)
    augmented_config = np.hstack([nominal_config.T, np.ones((N, 1))])
    
    # Compute SVD to get the basis Q for the kernel of the augmented configuration
    # MATLAB: [U,~,~] = svd(augmented_config); Q = U(:,D+2:end);
    # Python: U is NxN. Columns D+1 to end span the null space.
    U, _, _ = np.linalg.svd(augmented_config)
    Q = U[:, D+1:]  # Shape: N x (N - D - 1)

    # --- CVXPY Optimization ---
    # Variable: Edge weights
    weights = cp.Variable(M)
    
    # Define Stress Matrix: Omega = H * diag(w) * H^T
    # Note: cp.diag(weights) creates a sparse diagonal variable
    stress = incidence_mat @ cp.diag(weights) @ incidence_mat.T
    
    # Constraints
    constraints = []
    
    # 1. Equilibrium condition: Omega * p = 0
    # Equivalent to stress * nominal_config' == 0 in MATLAB
    constraints.append(stress @ nominal_config.T == 0)
    
    # 2. Universal Rigidity/Stabilizability Constraint
    # Enforce Positive Definiteness on the null space basis Q
    # MATLAB: Q'*stress*Q >= 1e-3*eye(N-D-1)
    target_eye = np.eye(N - D - 1) * 1e-3
    constraints.append(Q.T @ stress @ Q >> target_eye)
    
    # Optimization Problem (Feasibility)
    # We minimize 0 just to find a feasible point satisfying constraints
    prob = cp.Problem(cp.Minimize(0), constraints)
    
    # Solve using an SDP-capable solver (SCS is a common open-source choice)
    try:
        prob.solve(solver=cp.SCS, verbose=False)
    except cp.SolverError:
        prob.solve(verbose=False) # Fallback to default solver
        
    if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        raise RuntimeError(f"Optimization failed. Status: {prob.status}")

    # --- Post-processing ---
    # Extract weights
    w_val = weights.value
    
    # Normalize weights (as done in the MATLAB script)
    w_norm = w_val / np.linalg.norm(w_val)
    
    # Recompute the final stress matrix using normalized weights
    final_stress = incidence_mat @ np.diag(w_norm) @ incidence_mat.T
    
    return final_stress