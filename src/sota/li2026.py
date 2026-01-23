import numpy as np
from scipy.linalg import null_space

def vertex_additions(nominal_config):
    """
    Constructs the stress matrix for an affine formation using the Hierarchical 
    Affine Framework (HAF) vertex addition strategy.

    This reproduces the method described in "Distributed Framework Construction 
    for Affine Formation Control", specifically Algorithm 1 and Corollary 2.

    Args:
        nominal_config (np.ndarray): A (D x N) numpy array representing the 
                                     nominal configuration of N agents in D dimensions.

    Returns:
        np.ndarray: The (N x N) positive semi-definite stress matrix Omega.
    """
    nominal_config = np.array(nominal_config)
    
    # Validate input dimensions
    if nominal_config.ndim != 2:
        raise ValueError("Input configuration must be a 2D array (shape DxN).")
    
    D, N = nominal_config.shape
    
    # Initialize the global stress matrix [cite: 219]
    Omega = np.zeros((N, N))
    
    # A non-trivial stress matrix requires at least D+2 agents.
    # (The first D+1 agents form a simplex, which is stress-free in general position).
    if N < D + 2:
        return Omega

    # We treat the first D+1 agents as the initial basis (Leaders).
    # We iterate through the remaining agents (Followers) to add them to the framework.
    # New vertex index `u` ranges from D+1 to N-1.
    for u in range(D + 1, N):
        
        # --- Step 1: Select Parents ---
        # The paper requires selecting D+1 neighbors (parents)[cite: 159, 517].
        # To ensure the framework is well-defined without external topology data, 
        # we connect each new node to the first D+1 nodes (indices 0 to D).
        # This forms a (D+1)-lateration graph.
        parents_indices = list(range(D + 1))
        
        # --- Step 2: Construct Augmented Position Matrix (P_u) ---
        # We collect the positions of the parents and the new vertex u.
        # P_u = [[ p_i ... p_k  p_u ],
        #        [  1  ...  1    1  ]] 
        # This corresponds to Eq. (7) and Eq. (30) in the paper[cite: 222, 550].
        
        indices = parents_indices + [u]
        selected_positions = nominal_config[:, indices] # Shape (D, D+2)
        
        # Append a row of ones to handle affine dependence
        ones_row = np.ones((1, D + 2))
        P_u = np.vstack([selected_positions, ones_row]) # Shape (D+1, D+2)
        
        # --- Step 3: Compute Equilibrium Stress (Null Space) ---
        # We find vector phi such that P_u * phi = 0[cite: 225].
        # Assuming general position, the null space dimension (nullity) is 1.
        ns = null_space(P_u) # Shape (D+2, k)
        
        if ns.shape[1] == 0:
            # This occurs if points are not in general position (e.g., collinear in 2D).
            # The paper assumes generic configurations (Assumption 1).
            continue 
            
        phi = ns[:, 0] # Extract the vector
        
        # --- Step 4: Form Local Stress Update ---
        # Calculate Omega_u = s * phi * phi^T[cite: 230].
        # The paper states s must be positive (s > 0) for vertex addition.
        # We use s = 1.0 for simplicity.
        s = 1.0
        Omega_local = s * np.outer(phi, phi)
        
        # --- Step 5: Update Global Stress Matrix ---
        # Add the local contributions to the global matrix[cite: 234].
        for i_local, idx_i_global in enumerate(indices):
            for j_local, idx_j_global in enumerate(indices):
                Omega[idx_i_global, idx_j_global] += Omega_local[i_local, j_local]
                
    return Omega