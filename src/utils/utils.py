import numpy as np

def incidence_matrix_fully_connected(n):
        """
        Generates the incidence matrix for a fully connected graph with n nodes.
        
        Returns:
            np.ndarray: Incidence matrix of shape (num_nodes, num_edges)
        """
        # Calculate the number of edges
        num_edges = n * (n- 1) // 2
        
        # Initialize the incidence matrix with zeros
        inc_mat = np.zeros((n, num_edges), dtype=int)
        
        # Initialize edge counter
        edge_counter = 0
        
        # Fill in the incidence matrix
        for i in range(n - 1):
            for j in range(i + 1, n):
                inc_mat[i, edge_counter] = 1
                inc_mat[j, edge_counter] = -1
                edge_counter += 1
        
        return inc_mat

def detect_num_edges(stress_mat, sparse_tol = 1e-5):
    """
    Detects the number of edges in a stress matrix based on non-zero entries.

    Args:
        stress_mat (np.ndarray): The stress matrix of shape (N, N).
        sparse_tol (float): Tolerance below which weights are considered zero.

    Returns:
        int: The number of edges detected in the stress matrix.
    """
    N = stress_mat.shape[0]
    edge_count = 0

    for i in range(N):
        for j in range(i + 1, N):
            if abs(stress_mat[i, j]) > sparse_tol:
                edge_count += 1

    return edge_count

def stress_normalizer(stress_mat):
    """
    Normalizes the stress matrix so that largest eigenvalue is 1.

    Args:
        stress_mat (np.ndarray): The stress matrix to be normalized.

    Returns:
        np.ndarray: The normalized stress matrix.
    """
    return stress_mat / np.linalg.norm(stress_mat, 2)

def generate_points_in_rect(corners, num_points=60):
    """
    Generates random points inside a rectangular box.
    corners: 3x8 numpy array
    """
    # Find min and max for each axis
    min_coords = np.min(corners, axis=1) # [min_x, min_y, min_z]
    max_coords = np.max(corners, axis=1) # [max_x, max_y, max_z]
    
    # Generate random points
    # Each row is a dimension (X, Y, Z)
    points = np.random.uniform(
        low=min_coords[:, np.newaxis], 
        high=max_coords[:, np.newaxis], 
        size=(3, num_points)
    )
    
    return points