import numpy as np
import matplotlib.pyplot as plt

import src.utils.geometry as geometry
from src.utils.utils import detect_num_edges, stress_normalizer
from src import stress
from src.graph import Graph
from src.utils.plots import visualize_graph
from src.control import affine_formation_control_global, trackig_error

from src.sota import yang2019, xiao2022, li2026, lin2016

if __name__ == "__main__":
    nominal_config, adj_mat = geometry.cuboctahedron()
    N = nominal_config.shape[1]
    # hyperparam_dict = { 'alpha': 0.6, 'beta': 1, 'gamma': 0.1 }
    # stress_mat = stress_normalizer(stress.optimize_stress(nominal_config, hyperparam_dict))
    # stress_mat = np.zeros((N, N))  # Placeholder for stress matrix
    resulting_graph = Graph(adj_mat)
    fig = plt.figure(figsize=(6, 6))

    # This is the crucial step: projection='3d'
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect('equal')
    ax.axis('off') # Optional: Clean up the axes

    # 3. Call the visualize method
    resulting_graph.visualize(
        node_pos=nominal_config, 
        ax=ax, 
        node_color='red',    # Customizing color
        edge_color='black',  # Customizing edge color
        node_size=150, 
        linewidth=1.5
    )

    # 4. Save the figure
    output_filename = "./results/cuboctahedron.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Figure saved as {output_filename}")
