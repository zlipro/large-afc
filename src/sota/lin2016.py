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

def incidence_cauchy(N):
    """
    Constructs the oriented incidence matrix for a Cauchy Polygon framework.
    
    The framework consists of:
    - N boundary edges (cycle): (i, i+1).
    - N diagonals connecting (i, i+2).
    
    Note: For N=4, the diagonals (0,2) and (2,0) refer to the same edge. 
    This function handles that deduplication.
    
    Total Edges M:
    - If N = 4: M = 6 (4 boundary + 2 diagonals)
    - If N > 4: M = 2N (N boundary + N diagonals)
    
    Parameters:
    -----------
    N : int
        Number of nodes (must be >= 4).
        
    Returns:
    --------
    incidence_matrix : numpy.ndarray
        An (N x M) matrix where rows are nodes and columns are edges.
        Entries are 1 (start node) and -1 (end node).
    """
    
    if N < 4:
        raise ValueError("N must be at least 4 to define a Cauchy polygon structure.")

    edges = []
    
    # 1. Boundary Cycle: (0,1), (1,2), ..., (N-1,0)
    for i in range(N):
        u, v = i, (i + 1) % N
        edges.append((u, v))
        
    # 2. Cauchy Diagonals: Connect node i to node i+2
    # We use a set to track added edges to avoid duplicates (important for N=4)
    diag_edges_set = set()
    
    for i in range(N):
        u = i
        v = (i + 2) % N
        
        # Sort the tuple to ensure (0,2) and (2,0) are treated as the same edge
        edge_canonical = tuple(sorted((u, v)))
        
        if edge_canonical not in diag_edges_set:
            diag_edges_set.add(edge_canonical)
            # Append the directed version for the matrix (arbitrary orientation u->v)
            edges.append((u, v))
            
    M = len(edges)
    
    # 3. Construct the Matrix
    # Shape: (N nodes, M edges)
    inc_mat = np.zeros((N, M), dtype=int)
    
    for edge_index, (u, v) in enumerate(edges):
        inc_mat[u, edge_index] = 1
        inc_mat[v, edge_index] = -1
        
    return inc_mat

def optimize_with_incidence(incidence_mat, nominal_config):
    """
    Computes the stress matrix for a given graph structure and nominal configuration
    using Semidefinite Programming (SDP).

    Parameters:
    -----------
    incidence_mat : np.ndarray
        NxM incidence matrix encoding the graph structure.
    nominal_config : np.ndarray
        DxN matrix representing the nominal configuration (coordinates).

    Returns:
    --------
    np.ndarray
        NxN stress matrix.
    """
    # Ensure inputs are numpy arrays
    incidence_mat = np.array(incidence_mat)
    nominal_config = np.array(nominal_config)

    N, M = incidence_mat.shape
    D, _ = nominal_config.shape

    # --- 1. SVD on Augmented Configuration ---
    # MATLAB: augmented_config = [nominal_config', ones(N,1)];
    ones_col = np.ones((N, 1))
    augmented_config = np.hstack((nominal_config.T, ones_col))

    # Compute SVD
    # MATLAB: [U,~,~] = svd(augmented_config);
    # np.linalg.svd returns U, S, Vh. full_matrices=True guarantees U is NxN.
    U, _, _ = np.linalg.svd(augmented_config, full_matrices=True)

    # Extract Null Space Basis Q
    # MATLAB: Q = U(:, D+2:end); (D+2 is 1-based index)
    # Python: This corresponds to slicing from index D+1 to the end.
    # Columns 0 to D (total D+1 columns) are the range, D+1 onwards is null space.
    Q = U[:, D+1:]

    # --- 2. Convex Optimization (SDP) ---
    weights = cp.Variable(M)
    stress = cp.Variable((N, N), symmetric=True)

    constraints = []

    # Constraint 1: Stress definition
    # stress == incidence_mat * diag(weights) * incidence_mat'
    constraints.append(stress == incidence_mat @ cp.diag(weights) @ incidence_mat.T)

    # Constraint 2: PSD on the null space
    # Q'*stress*Q >= 1e-3*eye(N-D-1)
    # In CVXPY, '>>' denotes the Positive Semidefinite inequality.
    target_dim = N - D - 1
    constraints.append(Q.T @ stress @ Q >> 1e-3 * np.eye(target_dim))

    # Constraint 3: Equilibrium
    # stress * nominal_config' == 0
    constraints.append(stress @ nominal_config.T == 0)

    # Define problem (Feasibility problem, minimizing 0)
    prob = cp.Problem(cp.Minimize(0), constraints)
    
    # Solve
    # Using 'SCS' or 'CVXOPT' is common for SDPs, but default usually works.
    prob.solve()

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        raise ValueError(f"Optimization failed with status: {prob.status}")

    # --- 3. Post-processing ---
    # Extract values
    w_val = weights.value

    # Normalize weights
    # MATLAB: weights = weights./norm(weights);
    w_norm = w_val / np.linalg.norm(w_val)

    # Recompute stress to ensure consistency with normalized weights
    # MATLAB: stress = incidence_mat*diag(weights)*incidence_mat';
    stress_final = incidence_mat @ np.diag(w_norm) @ incidence_mat.T

    return stress_final

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