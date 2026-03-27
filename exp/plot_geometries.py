import numpy as np
import matplotlib.pyplot as plt

import src.utils.geometry as geometry
from src.graph import Graph

def plot_geometry(geometry_name, 
                  nominal_config, 
                  adj_mat, 
                  node_size=50, 
                  cluster_indices=None, 
                  view_angle=None):
    """
    Plot the geometries (2D or 3D) with optional cluster coloring and viewing angle customization.
    
    Parameters:
        geometry_name (str): Name used for the output filename.
        nominal_config (np.ndarray): DxN array of node positions (D=2 or D=3).
        adj_mat (np.ndarray): Adjacency matrix.
        node_size (int): Size of the nodes in the plot.
        cluster_indices (list of arrays, optional): List of index arrays for cluster coloring.
        view_angle (tuple, optional): (elev, azim) or (elev, azim, roll) - only for 3D plots.
    """
    
    # Check dimensionality: 2D or 3D
    is_3d = (nominal_config.shape[0] == 3)

    fig = plt.figure(figsize=(6, 6))
    
    if is_3d:
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.add_subplot(111)

    # Determine Node Colors
    if cluster_indices is not None:
        num_nodes = nominal_config.shape[1]
        node_colors = np.zeros((num_nodes, 4))
        node_colors[:] = [1.0, 0.0, 0.0, 1.0] 
        
        cmap = plt.get_cmap('tab10')
        
        for i, indices in enumerate(cluster_indices):
            node_colors[indices] = cmap(i % 10)
        color_arg = node_colors
    else:
        color_arg = 'red'

    # Visualize Graph
    outlining_graph = Graph(adj_mat)    
    outlining_graph.visualize(
        node_pos=nominal_config, 
        ax=ax, 
        node_color=color_arg,
        edge_color='black', 
        node_size=node_size,    
        linewidth=0.5
    )

    # Set Axis Limits with Padding
    # Helper to calculate padded limits
    def get_padded_limits(data, pad_factor=0.1):
        d_min, d_max = data.min(), data.max()
        span = d_max - d_min
        if span == 0: span = 1.0 # Prevent zero-division/flat axis issues
        pad = span * pad_factor
        return d_min - pad, d_max + pad

    ax.set_xlim(get_padded_limits(nominal_config[0]))
    ax.set_ylim(get_padded_limits(nominal_config[1]))

    if is_3d:
        ax.set_zlim(get_padded_limits(nominal_config[2]))
        
        # Set Viewing Angle (only if 3D)
        if view_angle is not None:
            if len(view_angle) == 3:
                ax.view_init(elev=view_angle[0], azim=view_angle[1], roll=view_angle[2])
            else:
                ax.view_init(elev=view_angle[0], azim=view_angle[1])
    
    ax.set_aspect('equal')
    ax.set_axis_off()

    # Save the figure
    output_filename = f"./results/geometry-plots/{geometry_name}.pdf"
    plt.savefig(output_filename, format='pdf', bbox_inches='tight')

    print(f"Saved geometry plot to {output_filename}")



if __name__ == "__main__":
    N_circular = 10
    nominal_config = geometry.circular_config(N_circular)
    plot_geometry("circular_config", nominal_config, np.zeros((N_circular,N_circular)), node_size=150)

    nominal_config, adj_mat = geometry.cuboctahedron()
    plot_geometry("cuboctahedron", nominal_config, adj_mat, node_size=120)

    nominal_config, adj_mat = geometry.truncated_icosahedron()
    plot_geometry("truncated_icosahedron", nominal_config, adj_mat)

    nominal_config, cluster_idxs, adj_mat = geometry.nested_football(outlining_graph=True)
    plot_geometry("nested_football", 
                  nominal_config, 
                  adj_mat,
                  cluster_indices=cluster_idxs, 
                  node_size=20,
                  view_angle=(6, -105, 0))
      
    nominal_configs, cluster_idxs, adj_mat = geometry.letter_w(filling_points_per_seg=80, outlining_graph=True)
    nominal_config_w = nominal_configs["nominal_config_W"]
    
    # Now passing cluster_indices and a view_angle (e.g., 30 deg elevation, 45 deg azimuth)
    plot_geometry("letter_w", 
                  nominal_config_w, 
                  adj_mat, 
                  node_size=20, 
                  cluster_indices=cluster_idxs, 
                  view_angle=(7, -15, 0))
    # plt.show()

