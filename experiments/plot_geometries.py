import numpy as np
import matplotlib.pyplot as plt

import src.utils.geometry as geometry
from src.graph import Graph

def plot_geometry(geometry_name, 
                  nominal_config, 
                  adj_mat, 
                  node_size = 50):

    """
    Plot the following geometries:
        - "random-5"
        - "random-10"
        - "random-50"
        - "random-100"
        - "random-500"
        - "circular-10"
        - "cuboctahedron"
        - "truncated_icosahedron"
        - "nested_football"
        - "bird"
    """
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    outlining_graph = Graph(adj_mat)    
    outlining_graph.visualize(
        node_pos=nominal_config, 
        ax=ax, 
        node_color='red',    # Customizing color
        edge_color='grey',  # Customizing edge color
        node_size=node_size,      # Customizing node size
        linewidth=1.5
    )
    ax.set_xlim(nominal_config[0].min(), nominal_config[0].max())
    ax.set_ylim(nominal_config[1].min(), nominal_config[1].max())
    ax.set_zlim(nominal_config[2].min(), nominal_config[2].max())

    ax.set_aspect('equal')
    ax.set_axis_off()

    # 4. Save the figure
    output_filename = f"./results/geometry-plots/{geometry_name}.pdf"

    # Optimization: format='pdf' ensures the backend handles it correctly
    # and metadata is stripped/managed for vector output.
    plt.savefig(output_filename, format='pdf', bbox_inches='tight')

    print(f"Saved geometry plot to {output_filename}")




if __name__ == "__main__":
    nominal_config, adj_mat = geometry.cuboctahedron()
    plot_geometry("cuboctahedron", nominal_config, adj_mat, node_size=120)

    nominal_config, adj_mat = geometry.truncated_icosahedron()
    plot_geometry("truncated_icosahedron", nominal_config, adj_mat)

    nominal_config, adj_mat = geometry.nested_football()
    plot_geometry("nested_football", nominal_config, adj_mat, node_size=20)
      