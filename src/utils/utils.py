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

def detect_num_edges(stress_mat, sparse_tol=1e-5):
    """
    Detects the number of edges based on non-zero entries.
    Handles both single matrices and lists of matrices (Multi-Cluster).

    Args:
        stress_mat (np.ndarray or list): A single (N, N) matrix OR a list of matrices.
        sparse_tol (float): Tolerance below which weights are considered zero.

    Returns:
        int: The total number of edges detected.
    """
    # 1. Handle Multi-Cluster Input (List of Matrices)
    if isinstance(stress_mat, list):
        total_edges = 0
        for mat in stress_mat:
            if mat is not None:
                total_edges += detect_num_edges(mat, sparse_tol)
        return total_edges

    # Fast Vectorized Counting (replaces slow nested loops)
    # We take the Upper Triangle (k=1) to avoid double counting and diagonal
    upper_tri = np.triu(stress_mat, k=1)
    edge_count = np.sum(np.abs(upper_tri) > sparse_tol)

    return int(edge_count)

def stress_normalizer(stress_mat):
    """
    Normalizes the stress matrix (or list of matrices) so that the largest eigenvalue is 1.

    Args:
        stress_mat (np.ndarray or list[np.ndarray]): A single stress matrix or a list of 
                                                     stress matrices.

    Returns:
        np.ndarray or list[np.ndarray]: The normalized matrix or list of normalized matrices.
    """
    # Check if input is a list
    if isinstance(stress_mat, list):
        # Apply normalization to every matrix in the list
        return [mat / np.linalg.norm(mat, 2) for mat in stress_mat]
    
    # Otherwise, assume it is a single numpy array and normalize it
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

def find_global_stress_mat(stress_mats: list, subgraph_glb_idx: list, N: int):
    # combine stress matrices from subgraphs into a global stress matrix
    global_stress_mat = np.zeros((N, N))
    for stress_mat, subgraph in zip(stress_mats, subgraph_glb_idx):
        # fill rows and columns with the same indices

        global_stress_mat[np.ix_(subgraph,subgraph)] += stress_mat
    return global_stress_mat

def stitch_clusters(clusters_list: list, num_interfacing_pts: int):
    ########################### merging ##############################
    # This will hold the final clean matrix
    merged_config = None 
    
    # This will hold the list of lists of indices: [[0,1...], [..., 100], ...]
    cluster_indices = []
    current_global_idx = 0

    # *** FIX: Iterate over the LIST, not the hstacked matrix ***
    for i, cluster in enumerate(clusters_list):
        num_points_in_cluster = cluster.shape[1]
        
        if i == 0:
            # First cluster: take everything
            merged_config = cluster
            
            # Indices are simply 0 to N-1
            ids = list(range(current_global_idx, current_global_idx + num_points_in_cluster))
            cluster_indices.append(ids)
            
            # Advance counter
            current_global_idx += num_points_in_cluster
            
        else:
            # Subsequent clusters: Drop the first 'num_interfacing_pts' 
            # because they exist at the end of the previous cluster.
            
            # 1. Identify unique points (skip the hinge)
            unique_points = cluster[:, num_interfacing_pts:]
            
            # 2. Append to global matrix
            merged_config = np.hstack([merged_config, unique_points])
            
            # 3. Calculate Indices
            # The "start" of this cluster uses the "end" indices of the previous cluster (overlap)
            # The previous cluster ended at `current_global_idx - 1`.
            # The overlap indices are the last `num_interfacing_pts` before `current_global_idx`.
            
            overlap_ids = list(range(current_global_idx - num_interfacing_pts, current_global_idx))
            
            # The new indices for the unique points
            num_unique = unique_points.shape[1]
            new_ids = list(range(current_global_idx, current_global_idx + num_unique))
            
            # Combine
            cluster_indices.append(overlap_ids + new_ids)
            
            # Advance counter by the number of UNIQUE points added
            current_global_idx += num_unique
    return merged_config, cluster_indices

def get_symmetry_mapping_matrix(config, tol=1e-4):
    """
    Constructs the selection matrix S based on unique edge lengths (EDM shortcut).
    
    Args:
        config (np.ndarray): Shape (D, N)
        tol (float): Tolerance for grouping distances
        
    Returns:
        S (np.ndarray): Shape (M, K) mapping reduced vars to full edges
        unique_dists (list): List of unique distances found
    """
    D, N = config.shape
    num_edges = N * (N - 1) // 2
    
    # Calculate all pairwise edge lengths for the complete graph
    edge_lengths = []
    pair_indices = []
    
    k = 0
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.linalg.norm(config[:, i] - config[:, j])
            edge_lengths.append(dist)
            pair_indices.append(k)
            k += 1
            
    edge_lengths = np.array(edge_lengths)
    
    # Group edges by length
    # We round to avoid float precision issues during unique grouping
    rounded_lengths = np.round(edge_lengths / tol) * tol
    unique_vals = np.unique(rounded_lengths)
    K = len(unique_vals)
    
    S = np.zeros((num_edges, K))
    
    # Fill S: S[e, k] = 1 if edge e has length unique_vals[k]
    for k_idx, val in enumerate(unique_vals):
        # Find all edges with this length
        indices = np.where(np.abs(edge_lengths - val) < tol)[0]
        S[indices, k_idx] = 1.0
        
    return S, unique_vals

def random_multicluster_indices(num_nodes, num_clusters, node_per_cluster):
    """
    Generates cluster indices that are random but GUARANTEE every node 
    is included in at least one cluster.
    """
    if node_per_cluster * num_clusters < num_nodes:
        raise ValueError("Total capacity (clusters * size) must be >= num_nodes to guarantee coverage.")

    # 1. Create a random permutation of all nodes
    # This ensures we have a random order, but we possess every ID from 0 to N-1
    shuffled_nodes = np.random.permutation(num_nodes)
    
    # 2. Distribute the 'must-have' nodes first
    # We split the shuffled list into roughly equal 'core' chunks for each cluster
    core_chunks = np.array_split(shuffled_nodes, num_clusters)
    
    subgraph_glb_idx = []
    
    for i in range(num_clusters):
        core = core_chunks[i]
        current_len = len(core)
        needed = node_per_cluster - current_len
        
        # 3. Fill the rest of the cluster with random neighbors
        # We sample from ALL nodes excluding the ones we already have in the core
        # (Or just sample from all nodes and use np.unique, which is easier)
        
        remaining_pool = np.setdiff1d(np.arange(num_nodes), core)
        
        # Randomly pick extra nodes to meet the size requirement
        if needed > 0:
            extras = np.random.choice(remaining_pool, size=needed, replace=False)
            full_cluster = np.concatenate([core, extras])
        else:
            full_cluster = core
            
        # Sort for consistency
        full_cluster.sort()
        subgraph_glb_idx.append(full_cluster.tolist())

    return subgraph_glb_idx