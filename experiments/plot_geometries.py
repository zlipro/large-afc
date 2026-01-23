import numpy as np
import matplotlib.pyplot as plt

import src.utils.geometry as geometry
from src.graph import Graph

def plot_geometry(geometry_name, nominal_config, adj_mat, node_size = 50):

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
    ax.set_aspect('equal')
    ax.axis('off') # Optional: Clean up the axes

    # 3. Call the visualize method
    outlining_graph.visualize(
        node_pos=nominal_config, 
        ax=ax, 
        node_color='red',    # Customizing color
        edge_color='grey',  # Customizing edge color
        node_size=node_size,      # Customizing node size
        linewidth=1.5
    )

    # 4. Save the figure
    output_filename = "./results/geometry-plots/" + geometry_name + ".png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Saved geometry plot to {output_filename}")



# if __name__ == "__main__":
#     nominal_config, adj_mat = geometry.cuboctahedron()
#     outlining_graph = Graph(adj_mat)
#     plot_geometry("cuboctahedron", nominal_config, adj_mat, node_size=120)

#     nominal_config, adj_mat = geometry.truncated_icosahedron()
#     outlining_graph = Graph(adj_mat)
#     plot_geometry("truncated_icosahedron", nominal_config, adj_mat)

#     nominal_config, adj_mat = geometry.nested_football()
#     outlining_graph = Graph(adj_mat)
#     plot_geometry("nested_football", nominal_config, adj_mat)
      
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np

def set_axes_equal(ax):
    """Make axes of 3D plot have equal scale so spheres look spherical."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    ax.set_box_aspect((1, 1, 1))

def plot_mickey_3d_view(vertices, adj_matrix, filename="mickey_3d.pdf"):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 1. Plot Edges (using Collection for speed/stability)
    rows, cols = np.where(np.triu(adj_matrix) == 1)
    segments = []
    for i, j in zip(rows, cols):
        segments.append([vertices[:, i], vertices[:, j]])
    
    # Made lines slightly thicker (1.0) and darker (alpha=0.4) for better depth perception
    lc = Line3DCollection(segments, colors='gray', linewidths=1.0, alpha=0.4)
    ax.add_collection3d(lc)

    # 2. Plot Nodes
    ax.scatter(vertices[0, :], vertices[1, :], vertices[2, :], 
               s=25, c='#D93025', depthshade=True, edgecolors='black', linewidth=0.3)

    # 3. Set Limits manually
    ax.set_xlim(vertices[0].min(), vertices[0].max())
    ax.set_ylim(vertices[1].min(), vertices[1].max())
    ax.set_zlim(vertices[2].min(), vertices[2].max())
    
    set_axes_equal(ax)
    ax.set_axis_off()
    
    # --- THE CHANGE IS HERE ---
    # Elevation 15: Look slightly down from above
    # Azimuth -75: Rotate slightly right from the front view (-90)
    ax.view_init(elev=15, azim=-75)

    plt.savefig(filename, bbox_inches='tight', pad_inches=0.01)
    print(f"Saved {filename}")
    plt.close()
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # --- 1. Plot Edges (The Fix) ---
    # Instead of one long line with Nones, we create a collection of segments.
    rows, cols = np.where(np.triu(adj_matrix) == 1)
    
    # Create a list of segments: [(x1, y1, z1), (x2, y2, z2)]
    segments = []
    for i, j in zip(rows, cols):
        segments.append([vertices[:, i], vertices[:, j]])
    
    # Line3DCollection is much faster and doesn't crash like ax.plot(None)
    lc = Line3DCollection(segments, colors='gray', linewidths=0.8, alpha=0.6)
    ax.add_collection3d(lc)

    # --- 2. Plot Nodes ---
    ax.scatter(vertices[0, :], vertices[1, :], vertices[2, :], 
               s=15, c='#D93025', depthshade=False, edgecolors='black', linewidth=0.2)

    # --- 3. Final Formatting ---
    # We must manually set limits because add_collection3d doesn't auto-scale the view
    ax.set_xlim(vertices[0].min(), vertices[0].max())
    ax.set_ylim(vertices[1].min(), vertices[1].max())
    ax.set_zlim(vertices[2].min(), vertices[2].max())
    
    set_axes_equal(ax)
    ax.set_axis_off()
    ax.view_init(elev=0, azim=-90)

    plt.savefig(filename, bbox_inches='tight', pad_inches=0.01)
    print(f"Saved {filename}")
    plt.show()
    plt.close()
    

# Usage
v, adj = geometry.nested_football_wide()
print(f"the shape of the config is {v.shape}")
plot_mickey_3d_view(v, adj)