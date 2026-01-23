import numpy as np
from scipy.linalg import svd

def sparse_nullspace_recon(nominal_config):
    """
    Constructs the universally rigid stress matrix for a given configuration
    using the five-step algorithm proposed by Yang et al.

    The algorithm assumes the configuration is in 'general position', meaning
    no d+1 points are affinely dependent.

    Args:
        nominal_config (np.ndarray): A (D x N) numpy array representing the 
                                     configuration, where D is the dimension 
                                     (e.g., 2 or 3) and N is the number of nodes.

    Returns:
        np.ndarray: The (N x N) symmetric stress matrix Omega.
    """
    # Validate input shape
    if nominal_config.ndim != 2:
        raise ValueError("nominal_config must be a 2D array of shape (D, N).")
    
    D_dim, N = nominal_config.shape
    
    # Check condition for general position processing: N >= D + 2
    # A generic framework needs at least D+2 nodes to have a non-trivial self-stress
    # capable of universal rigidity using this construction method.
    if N < D_dim + 2:
        raise ValueError(f"Number of nodes (N={N}) must be at least D+2 ({D_dim+2}) "
                         "for the construction of a universally rigid framework.")

    # --- Step 1 & Definition: Construct Extended Configuration Matrix Q* ---
    # According to Eq. (1), Q* is formed by stacking the configuration q* # with a row of ones[cite: 87, 139].
    # Q* has shape (D+1, N)
    ones_row = np.ones((1, N))
    Q_star = np.vstack([nominal_config, ones_row])

    # print(Q_star)

    # --- Step 2 & 3: Construct Matrix D ---
    # We construct a sparse matrix D such that Q* @ D = 0.
    # D corresponds to the 'Gale matrix' components.
    # The matrix D has shape N x (N - D - 1)[cite: 106].
    # The algorithm uses a sliding window of columns from Q* to create a band structure.
    
    num_cols_D = N - D_dim - 1
    D_matrix = np.zeros((N, num_cols_D))
    
    # In a general position, any D+2 columns of Q* are linearly dependent.
    # The window size is therefore D + 2.
    window_size = D_dim + 2
    
    for j in range(num_cols_D):
        # Identify the indices for the sliding window
        start_idx = j
        end_idx = j + window_size
        
        # Extract the submatrix of Q* consisting of D+2 columns
        # Shape: (D+1) x (D+2)
        sub_Q = Q_star[:, start_idx:end_idx]
        
        # Solve for the null space of this submatrix (Eq. 7)[cite: 168].
        # Since rows are D+1 and columns are D+2, the null space is 1-dimensional 
        # (assuming general position).
        # We use SVD: sub_Q = U * S * Vh
        # The null vector corresponds to the last row of Vh (or last column of V).
        u, s, vh = svd(sub_Q)
        null_vector = vh[-1, :]
        
        # Assign the computed non-zero elements to the appropriate position in D
        # The non-zeros are located at indices d+2 through d+2 in the window,
        # which maps to rows start_idx through end_idx in D[cite: 141, 144].
        D_matrix[start_idx:end_idx, j] = null_vector
    # print(D_matrix)
    # --- Step 4: Compute Stress Matrix Omega ---
    # The stress matrix is calculated as Omega = D * D.T[cite: 172].
    # Note: The paper notes that Omega can be D * Psi * D.T where Psi is a scaling matrix,
    # but sets Psi = Identity for simplicity[cite: 187, 188].
    Omega = D_matrix @ D_matrix.T

    return Omega