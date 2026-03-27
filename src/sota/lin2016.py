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

    # Define the Edges
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
    
    # Construct the Matrix
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
    
    # Boundary Cycle: (0,1), (1,2), ..., (N-1,0)
    for i in range(N):
        u, v = i, (i + 1) % N
        edges.append((u, v))
        
    # Cauchy Diagonals: Connect node i to node i+2
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
    
    # Construct the Matrix
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

    # --- SVD on Augmented Configuration ---
    ones_col = np.ones((N, 1))
    augmented_config = np.hstack((nominal_config.T, ones_col))

    # Compute SVD
    U, _, _ = np.linalg.svd(augmented_config, full_matrices=True)

    # Extract Null Space Basis Q
    Q = U[:, D+1:]

    # --- Convex Optimization (SDP) ---
    weights = cp.Variable(M)
    stress = cp.Variable((N, N), symmetric=True)

    constraints = []

    # Constraint 1: Stress definition
    # stress == incidence_mat * diag(weights) * incidence_mat'
    constraints.append(stress == incidence_mat @ cp.diag(weights) @ incidence_mat.T)

    target_dim = N - D - 1
    constraints.append(Q.T @ stress @ Q >> 1e-3 * np.eye(target_dim))

    # Constraint 3: Equilibrium
    # stress * nominal_config' == 0
    constraints.append(stress @ nominal_config.T == 0)

    # Define problem (Feasibility problem, minimizing 0)
    prob = cp.Problem(cp.Minimize(0), constraints)
    
    # Solve
    prob.solve()

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        raise ValueError(f"Optimization failed with status: {prob.status}")

    # --- Post-processing ---
    # Extract values
    w_val = weights.value

    # Normalize weights
    w_norm = w_val / np.linalg.norm(w_val)

    # Recompute stress to ensure consistency with normalized weights
    stress_final = incidence_mat @ np.diag(w_norm) @ incidence_mat.T

    return stress_final