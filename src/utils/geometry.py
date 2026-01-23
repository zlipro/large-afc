import numpy as np

def circular_config(num_nodes, radius = 1):
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

def cuboctahedron():
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