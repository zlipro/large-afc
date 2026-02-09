import numpy as np
from itertools import product
from scipy.spatial.transform import Rotation as R
from scipy.spatial import KDTree
from .utils import generate_points_in_rect, stitch_clusters

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def circular_config(num_nodes, radius = 1, outlining_graph = False):
    """
    Generate positions of nodes placed evenly on a circle (regular polygon).
    
    Parameters:
        r (float): Radius of the circle.
        num_nodes (int): Number of nocdes to be placed.
    
    Returns:
        np.ndarray: A 2xN array where the first row is x-coordinates and the second row is y-coordinates.
    """
    # Create an array of angles excluding 2π
    angles = np.linspace(0, 2 * np.pi, num_nodes + 1)[:-1]
    
    # Compute x and y coordinates
    x_positions = radius * np.cos(angles)
    y_positions = radius * np.sin(angles)
    
    # Stack into a 2xN array
    positions = np.vstack((x_positions, y_positions))

    if outlining_graph:
        adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
        for i in range(num_nodes):
            adj_matrix[i, (i + 1) % num_nodes] = 1
            adj_matrix[(i + 1) % num_nodes, i] = 1
        return positions, adj_matrix
    
    return positions

def xiao_generic():
    nominal_config = np.array([[2,1,1.5,0,0.5,-1],[0,1,-1,1.5,-1,0.5]])
    return nominal_config

def random_config(N, D = 2, playground_size = 4):
    """
    Generate random configuration of N nodes in D dimensions within a square/cube of given size.
    
    Parameters:
        N (int): Number of nodes.
        D (int): Dimension of the space (default is 2).
        playground_size (float): Size of the square/cube in which nodes are placed (default is 4).
    
    Returns:
        np.ndarray: A DxN array where each column represents the coordinates of a node.
    """
    return np.random.rand(D, N) * playground_size

def cuboctahedron(outlining_graph = False):
    """
    Generate positions of nodes placed at the vertices of a cuboctahedron.
    
    Returns:
        np.ndarray: A 3x12 array where each column represents the coordinates of a vertex.
    """
    vertices = np.array([
        [1, 1, 0],
        [1, -1, 0],
        [-1, 1, 0],
        [-1, -1, 0],
        [1, 0, 1],
        [1, 0, -1],
        [-1, 0, 1],
        [-1, 0, -1],
        [0, 1, 1],
        [0, 1, -1],
        [0, -1, 1],
        [0, -1, -1]
    ]).T  # Transpose to get shape (3, 12)
    if outlining_graph:
        # a fake graph just for visualization purposes
        # Initialize the adjacency matrix
        num_vertices = vertices.shape[1]
        adj_matrix = np.zeros((num_vertices, num_vertices), dtype=int)

        # Calculate Euclidean distances to find edges
        # In this coordinate set, edges exist where distance is exactly sqrt(2)
        for i in range(num_vertices):
            for j in range(i + 1, num_vertices):
                dist = np.linalg.norm(vertices[:, i] - vertices[:, j])
                if np.isclose(dist, np.sqrt(2)):
                    adj_matrix[i, j] = 1
                    adj_matrix[j, i] = 1
    
        return vertices, adj_matrix
    return vertices



def truncated_icosahedron(outlining_graph = False):
    """
    Generate positions and the adjacency matrix for a truncated icosahedron.
    
    Returns:
        vertices (np.ndarray): A 3x60 array of vertex coordinates.
        adj_matrix (np.ndarray): A 60x60 symmetric matrix encoding the edges.
    """
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio

    # The 3 base vectors for cyclic permutations
    base_seeds = [
        [0, 1, 3 * phi],
        [2, 1 + 2 * phi, phi],
        [1, 2 + phi, 2 * phi]
    ]

    unique_vertices = set()

    for base in base_seeds:
        # 1. Generate Cyclic Permutations: (x,y,z), (y,z,x), (z,x,y)
        cyclic_perms = [
            (base[0], base[1], base[2]),
            (base[1], base[2], base[0]),
            (base[2], base[0], base[1])
        ]
        
        for p in cyclic_perms:
            # 2. Generate all Sign Combinations (2^3 = 8 per permutation)
            for signs in product([-1, 1], repeat=3):
                coords = (
                    p[0] * signs[0],
                    p[1] * signs[1],
                    p[2] * signs[2]
                )
                # This set handles duplicates (e.g., -0.0 vs 0.0) automatically
                unique_vertices.add(coords)

    # Convert to sorted array for consistent indexing
    vertices = np.array(sorted(list(unique_vertices)))

    # Verification
    if vertices.shape[0] != 60:
        raise ValueError(f"Generated {vertices.shape[0]} vertices instead of 60.")

    # Transpose to 3x60 as requested
    vertices = vertices.T

    if outlining_graph:
        # --- Generate Adjacency Matrix ---
        # With these coordinates, edge length is exactly 2.0
        num_vertices = 60
        adj_matrix = np.zeros((num_vertices, num_vertices), dtype=int)
        
        target_edge_length = 2.0
        tolerance = 1e-4

        # Calculate upper triangle of distance matrix
        for i in range(num_vertices):
            for j in range(i + 1, num_vertices):
                # dist = sqrt((x1-x2)^2 + ...)
                dist = np.linalg.norm(vertices[:, i] - vertices[:, j])
                
                if abs(dist - target_edge_length) < tolerance:
                    adj_matrix[i, j] = 1
                    adj_matrix[j, i] = 1
        
        # Validation: Every node must have exactly degree 3
        degrees = np.sum(adj_matrix, axis=0)
        if not np.all(degrees == 3):
            print(f"Warning: Not all vertices have degree 3. Degrees found: {np.unique(degrees)}")

        return vertices, adj_matrix
    return vertices

def nested_football(outlining_graph = False):
    """
    Generates 3 fused Truncated Icosahedra with WIDER separation (approx 116 degrees).
    Returns:
        vertices (3xN), adjacency (NxN), cluster_indices (list of arrays)
    """
    
    # --- 1. Generate Base Unit (Head) ---
    phi = (1 + np.sqrt(5)) / 2
    base_seeds = [[0, 1, 3 * phi], [2, 1 + 2 * phi, phi], [1, 2 + phi, 2 * phi]]
    unique_verts = set()
    for base in base_seeds:
        perms = [(base[0], base[1], base[2]), (base[1], base[2], base[0]), (base[2], base[0], base[1])]
        for p in perms:
            for signs in product([-1, 1], repeat=3):
                unique_verts.add((p[0]*signs[0], p[1]*signs[1], p[2]*signs[2]))
    verts_head = np.array(list(unique_verts))
    num_nodes_per_unit = len(verts_head) # Should be 60

    # --- 2. Identify Connectivity & Pentagons ---
    # Build Adjacency
    tree = KDTree(verts_head)
    pairs = tree.query_pairs(r=2.1)
    adj_list = {i: set() for i in range(len(verts_head))}
    for i, j in pairs:
        if abs(np.linalg.norm(verts_head[i] - verts_head[j]) - 2.0) < 0.1:
            adj_list[i].add(j)
            adj_list[j].add(i)

    # Find Pentagons (5-cycles)
    pentagon_centroids = []
    visited_paths = set()
    for i in range(len(verts_head)):
        for n in adj_list[i]:
            for a in adj_list[n]:
                if a == i: continue
                for b in adj_list[a]:
                    if b == n: continue
                    for c in adj_list[b]:
                        if c == a: continue
                        if i in adj_list[c]:
                            cycle = tuple(sorted([i, n, a, b, c]))
                            if cycle not in visited_paths:
                                visited_paths.add(cycle)
                                centroid = np.mean(verts_head[list(cycle)], axis=0)
                                pentagon_centroids.append(centroid)

    # --- 3. Select WIDER Attachment Sites ---
    pL = pentagon_centroids[0]
    pR = None
    target_angle = 116.57
    
    for cent in pentagon_centroids[1:]:
        dot = np.dot(pL, cent)
        norm_prod = np.linalg.norm(pL) * np.linalg.norm(cent)
        angle = np.degrees(np.arccos(np.clip(dot / norm_prod, -1, 1)))
        
        if abs(angle - target_angle) < 2.0:
            pR = cent
            break
            
    if pR is None: 
        raise ValueError(f"Could not find pentagons separated by {target_angle} degrees.")

    # --- 4. Robust Attachment Logic (Translate + Twist Check) ---
    def attach_ear(base_verts, face_centroid):
        trans_vec = 2.0 * face_centroid
        
        # Candidate A: Direct Translation
        candidate_A = base_verts + trans_vec
        
        # Candidate B: Twist 36 deg then Translate
        axis = face_centroid / np.linalg.norm(face_centroid)
        r_twist = R.from_rotvec(np.radians(36.0) * axis)
        candidate_B = r_twist.apply(base_verts) + trans_vec
        
        # Check Interface Error
        def get_err(cand):
            dists = np.linalg.norm(cand - face_centroid, axis=1)
            closest_idxs = np.where(dists < 0.5)[0] 
            if len(closest_idxs) == 0: return float('inf')
            err = 0
            for idx in closest_idxs:
                err += np.min(np.linalg.norm(verts_head - cand[idx], axis=1))
            return err

        return candidate_A if get_err(candidate_A) < get_err(candidate_B) else candidate_B

    verts_L = attach_ear(verts_head, pL)
    verts_R = attach_ear(verts_head, pR)

    # --- 5. Weld Vertices & Track Indices ---
    all_verts = np.vstack([verts_head, verts_L, verts_R])
    
    # Array to map every original vertex to its new final index
    # Initialize with -1
    mapping = np.full(len(all_verts), -1, dtype=int)
    
    final_verts_list = []
    tree_all = KDTree(all_verts)
    processed = np.zeros(len(all_verts), dtype=bool)
    
    current_final_idx = 0
    
    for i in range(len(all_verts)):
        if processed[i]: continue
        
        final_verts_list.append(all_verts[i])
        
        # Find duplicates (including self)
        neighbors = tree_all.query_ball_point(all_verts[i], r=0.1)
        
        # Mark all duplicates as processed and map them to the current final index
        processed[neighbors] = True
        mapping[neighbors] = current_final_idx
        
        current_final_idx += 1
        
    final_verts = np.array(final_verts_list)
    
    # Retrieve indices for each cluster based on original ranges
    # Head: 0 to 60, Left: 60 to 120, Right: 120 to 180
    idx_head = np.unique(mapping[0 : num_nodes_per_unit])
    idx_L    = np.unique(mapping[num_nodes_per_unit : 2*num_nodes_per_unit])
    idx_R    = np.unique(mapping[2*num_nodes_per_unit : 3*num_nodes_per_unit])
    
    cluster_indices = [idx_head, idx_L, idx_R]

    # --- 6. Orientation (Upright V-shape) ---
    c_L = np.mean(verts_L, axis=0)
    c_R = np.mean(verts_R, axis=0)
    
    # Bisector Up
    v_up = (c_L + c_R) / 2.0
    v_up /= np.linalg.norm(v_up)
    
    # Rotate Up to Z
    z_axis = np.array([0,0,1])
    rot_axis = np.cross(v_up, z_axis)
    if np.linalg.norm(rot_axis) > 1e-5:
        rot_angle = np.arccos(np.clip(np.dot(v_up, z_axis), -1, 1))
        r1 = R.from_rotvec(rot_angle * (rot_axis / np.linalg.norm(rot_axis)))
        final_verts = r1.apply(final_verts)
        c_R = r1.apply(c_R)

    # Rotate so Ears are symmetric across YZ plane
    angle_z = np.arctan2(c_R[1], c_R[0])
    r2 = R.from_rotvec(-angle_z * z_axis)
    final_verts = r2.apply(final_verts)
    
    final_vertices = final_verts.T

    # # --- 7. Adjacency Matrix ---
    # num_v = final_vertices.shape[1]
    # adj = np.zeros((num_v, num_v), dtype=int)
    
    # for i in range(num_v):
    #     diffs = final_vertices.T[i+1:] - final_vertices.T[i]
    #     dists_sq = np.sum(diffs**2, axis=1)
    #     neighbors_idx = np.where(np.abs(dists_sq - 4.0) < 0.2)[0]
        
    #     true_j = neighbors_idx + (i + 1)
    #     adj[i, true_j] = 1
    #     adj[true_j, i] = 1
    # if outlining_graph:
    #     return final_vertices, cluster_indices, adj
    # return final_vertices, cluster_indices
    # --- Index Mapping (Without np.unique) ---
    idx_head = mapping[0 : num_nodes_per_unit]
    idx_L    = mapping[num_nodes_per_unit : 2*num_nodes_per_unit]
    idx_R    = mapping[2*num_nodes_per_unit : 3*num_nodes_per_unit]
    
    cluster_indices = [idx_head.tolist(), idx_L.tolist(), idx_R.tolist()]

    if outlining_graph:
        # If you need adjacency, ensure 'adj' is calculated before this
        return final_vertices, cluster_indices, adj
        
    return final_vertices, cluster_indices
    


def letter_w(filling_points_per_seg = 50, outlining_graph = False):
    # Base square unit (the 4 corners of a hinge/tip)
    unit_square = 2*np.array([[1.0, 0.0, -1.0],
                            [1.0, 0.0, 1.0],
                            [-1.0, 0.0, 1.0],
                            [-1.0, 0.0, -1.0]]).T
    
    num_interfacing_pts = unit_square.shape[1]  # 4 corners (later could be more pts on a plane)
    
    translation_vector = np.array([[0.0, 5.0, 0.0]]).T

    # --- the base "bar" shape ---

    # left_tip ----- left_hinge ----- middle_hinge ----- right_hinge ----- right_tip
    #            ^                ^                  ^                 ^
    #            |                |                  |                 |
    #          seg 1            seg 2              seg 3             seg 4
    
    # Define positions by shifting the unit square
    middle_hinge = unit_square
    left_hinge   = middle_hinge - translation_vector
    right_hinge  = middle_hinge + translation_vector
    left_tip     = left_hinge   - translation_vector
    right_tip    = right_hinge  + translation_vector

    # random filling points between hinges, duplicate for all segments
    seg_1 = generate_points_in_rect(np.hstack([left_tip, left_hinge]), num_points=filling_points_per_seg)
    seg_2 = seg_1 + translation_vector
    seg_3 = seg_2 + translation_vector
    seg_4 = seg_3 + translation_vector
    

    clusters_1 = np.hstack([left_tip, seg_1, left_hinge])
    clusters_2 = np.hstack([left_hinge, seg_2, middle_hinge])
    clusters_3 = np.hstack([middle_hinge, seg_3, right_hinge])
    clusters_4 = np.hstack([right_hinge, seg_4, right_tip])

    # check if all clusters are identical upto a translation
    try:
        # Check Cluster 2 vs Cluster 1
        assert np.allclose(clusters_2, clusters_1 + translation_vector), "Cluster 2 is not a translation of Cluster 1"
        
        # Check Cluster 3 vs Cluster 2
        assert np.allclose(clusters_3, clusters_2 + translation_vector), "Cluster 3 is not a translation of Cluster 2"
        
        # Check Cluster 4 vs Cluster 3
        assert np.allclose(clusters_4, clusters_3 + translation_vector), "Cluster 4 is not a translation of Cluster 3"
        
        # print("Assertion Passed: All clusters are identical up to a translation of [0, 5, 0].")

    except AssertionError as e:
        print(f"Assertion Failed: {e}")

    config_bar = [clusters_1, clusters_2, clusters_3, clusters_4]


    # --- The V shape by shearing the middle hinges ---
    shearing_matrix_left = np.array([[1.0, 0, 0.0],
                                    [0.0, 1.0, 0.0],
                                    [0.0, 2.0, 1.0]])
    clusters_1 = shearing_matrix_left @ clusters_1
    clusters_2 = shearing_matrix_left @ clusters_2

    shearing_matrix_right = np.array([[1.0, 0, 0.0],
                                    [0.0, 1.0, 0.0],
                                    [0.0, -2.0, 1.0]])

    clusters_3 = shearing_matrix_right @ clusters_3
    clusters_4 = shearing_matrix_right @ clusters_4

    config_V = [clusters_1, clusters_2, clusters_3, clusters_4]

    # --- The W shape by applying another shearing ---
    # 1. Define the connection points (Hinges)
    # In the V-shape, the hinges have moved due to the first shear.
    # Left hinge was at [0, -5, 0]. After shear (slope 2), Z becomes 2*(-5) = -10.
    left_hinge_pt = np.array([[0.0, -5.0, -10.0]]).T
    
    # Right hinge was at [0, 5, 0]. After shear (slope -2), Z becomes -2*(5) = -10.
    right_hinge_pt = np.array([[0.0, 5.0, -10.0]]).T

    # 2. Define the "Correction" Shears
    # Cluster 1 currently has slope +2 (from the V setup). We want slope -2.
    # We need a net change of -4.
    shear_fix_left = np.array([[1.0, 0.0, 0.0],
                               [0.0, 1.0, 0.0],
                               [0.0, -4.0, 1.0]])

    # Cluster 4 currently has slope -2 (from the V setup). We want slope +2.
    # We need a net change of +4.
    shear_fix_right = np.array([[1.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0],
                                [0.0, 4.0, 1.0]])

    # 3. Apply transformation Relative to the Hinges
    # Logic: Move to origin -> Shear -> Move back to hinge
    
    # Shift to center the hinge at (0,0,0)
    c1_centered = clusters_1 - left_hinge_pt
    # Apply shear
    c1_sheared = shear_fix_left @ c1_centered
    # Shift back
    clusters_1_W = c1_sheared + left_hinge_pt

    # --- Fix Right Wing (Cluster 4) ---
    # Shift to center the hinge at (0,0,0)
    c4_centered = clusters_4 - right_hinge_pt
    # Apply shear
    c4_sheared = shear_fix_right @ c4_centered
    # Shift back
    clusters_4_W = c4_sheared + right_hinge_pt

    # 4. Combine into final W Configuration
    # (Clusters 2 and 3 are untouched as they form the inner V)
    config_W = [clusters_1_W, clusters_2, clusters_3, clusters_4_W]

    # --- Merge Clusters into Single Configuration & Get Indices ---
    merged_config_bar, cluster_indices = stitch_clusters(config_bar, num_interfacing_pts)
    merged_config_V, _ = stitch_clusters(config_V, num_interfacing_pts)
    merged_config_W, _ = stitch_clusters(config_W, num_interfacing_pts)


    # ---------------fine tuning the geometry---------------------
    # apply slight scaling on the y axis to strech a bit
    scaling_matrix = np.array([[2, 0, 0],
                               [0.0, 2.0, 0],
                               [0.0, 0, 1.5]])
    merged_config_bar = scaling_matrix @ merged_config_bar

    # flip V upside down
    flip_matrix = np.array([[2.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, -1.0]])
    
    merged_config_V = flip_matrix @ merged_config_V

    scaling_matrix = np.array([[2, 0, 0],
                               [0.0, 1.5, 0],
                               [0.0, 0, 1.5]])
    merged_config_W = scaling_matrix @ merged_config_W
   
    # ---------------------------------------------------------------


    nominal_configs = {"nominal_config_bar": merged_config_bar,
                       "nominal_config_V": merged_config_V,
                       "nominal_config_W": merged_config_W}
    
    # if outlining_graph is true, then also return adjacency matrix of the tips and hinges only
    if outlining_graph:
        num_total_nodes = merged_config_W.shape[1]
        adj_matrix = np.zeros((num_total_nodes, num_total_nodes), dtype=int)

        # Iterate through each segment (cluster) to outline it
        for idxs in cluster_indices:
            # In every cluster:
            # The first 'num_interfacing_pts' (4) are the start hinge/tip nodes
            # The last 'num_interfacing_pts' (4) are the end hinge/tip nodes
            start_nodes = idxs[:num_interfacing_pts]
            end_nodes = idxs[-num_interfacing_pts:]

            # 1. Create Longitudinal edges (Start Corner -> End Corner)
            for k in range(num_interfacing_pts):
                u, v = start_nodes[k], end_nodes[k]
                adj_matrix[u, v] = 1
                adj_matrix[v, u] = 1

            # 2. Create Cross-sectional edges (Outline the square faces)
            # Based on unit_square order: 0->1->2->3->0 forms the perimeter
            for corners in [start_nodes, end_nodes]:
                for k in range(num_interfacing_pts):
                    u = corners[k]
                    v = corners[(k + 1) % num_interfacing_pts] # Wrap around 0-1, 1-2, 2-3, 3-0
                    adj_matrix[u, v] = 1
                    adj_matrix[v, u] = 1

        return nominal_configs, cluster_indices, adj_matrix

    
    return nominal_configs, cluster_indices

if __name__ == "__main__":
    w_points, cluster_indices = letter_w(filling_points_per_seg = 70)
    # print(w_points.shape)
    # plot_letter_w_colored(w_points, filling_points_per_seg=50)

    print("Cluster Indices:", cluster_indices)
    # plot w_points
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(w_points[0, :], w_points[1, :], w_points[2, :], c='b', s=10)
    ax.set_title("3D Letter W Shape")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")      
    ax.set_aspect('equal')
    plt.show()
