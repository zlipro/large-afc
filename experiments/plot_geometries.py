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
    Plot the geometries with optional cluster coloring and viewing angle customization.
    
    Parameters:
        geometry_name (str): Name used for the output filename.
        nominal_config (np.ndarray): 3xN array of node positions.
        adj_mat (np.ndarray): Adjacency matrix.
        node_size (int): Size of the nodes in the plot.
        cluster_indices (list of arrays, optional): A list where each element represents 
                                                    indices of nodes belonging to a specific cluster. 
                                                    Used for coloring nodes by cluster.
        view_angle (tuple, optional): A tuple (elevation, azimuth) to set the 3D camera angle.
    """
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # 1. Determine Node Colors
    if cluster_indices is not None:
        num_nodes = nominal_config.shape[1]
        # Initialize an RGBA array (N, 4) defaulting to red
        node_colors = np.zeros((num_nodes, 4))
        node_colors[:] = [1.0, 0.0, 0.0, 1.0] 
        
        # Use 'tab10' colormap for distinct cluster colors
        cmap = plt.get_cmap('tab10')
        
        for i, indices in enumerate(cluster_indices):
            # Assign specific color to indices belonging to cluster i
            node_colors[indices] = cmap(i % 10)
            
        color_arg = node_colors
    else:
        color_arg = 'red'

    # 2. Visualize Graph
    outlining_graph = Graph(adj_mat)    
    outlining_graph.visualize(
        node_pos=nominal_config, 
        ax=ax, 
        node_color=color_arg,   # Passes either single string or array of colors
        edge_color='lightgrey', 
        node_size=node_size,    
        linewidth=1.5
    )

    # 3. Set Axis Limits
    ax.set_xlim(nominal_config[0].min(), nominal_config[0].max())
    ax.set_ylim(nominal_config[1].min(), nominal_config[1].max())
    ax.set_zlim(nominal_config[2].min(), nominal_config[2].max())

    # 4. Set Viewing Angle (if provided)
    if view_angle is not None:
        # view_angle expected as (elevation, azimuth, roll)
        ax.view_init(elev=view_angle[0], azim=view_angle[1], roll=view_angle[2])

    ax.set_aspect('equal')
    ax.set_axis_off()

    # 5. Save the figure
    output_filename = f"./results/geometry-plots/{geometry_name}.pdf"
    plt.savefig(output_filename, format='pdf', bbox_inches='tight')

    print(f"Saved geometry plot to {output_filename}")




if __name__ == "__main__":
    nominal_config, adj_mat = geometry.cuboctahedron()
    plot_geometry("cuboctahedron", nominal_config, adj_mat, node_size=120)

    nominal_config, adj_mat = geometry.truncated_icosahedron()
    plot_geometry("truncated_icosahedron", nominal_config, adj_mat)

    nominal_config, adj_mat = geometry.nested_football()
    plot_geometry("nested_football", nominal_config, adj_mat, node_size=20)
      
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

