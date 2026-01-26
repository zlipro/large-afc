import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_folding_bar(state="W", link_length=2.0, width=0.5, height=0.2):
    """
    Generates a configuration for a 4-segment structure.
    
    Args:
        state (str): "W" for a standing zigzag shape, "Flat" for a straight bar.
        link_length (float): Length of each segment.
        width (float): Width (y-axis) of the segment.
        height (float): Height/Thickness (z-axis) of the segment.
        
    Returns:
        P (np.ndarray): 3xN Coordinate matrix.
        clusters (list): Indices for each of the 4 segments.
        hinges (list): Indices of the bridging nodes.
    """
    x_coords, y_coords, z_coords = [], [], []
    clusters = []
    hinges = []
    
    # Starting point
    current_pos = np.array([0.0, 0.0, 0.0])
    
    # Define absolute pitch angles (rotation around Y-axis) for the segments
    if state == "W":
        # Down, Up, Down, Up to make a "W" shape in X-Z plane
        angles = [-60, 60, -60, 60] 
    else:
        # All zero -> Flat on the ground
        angles = [0, 0, 0, 0]
        
    node_idx = 0
    prev_end_nodes = []

    for i, deg in enumerate(angles):
        rad = np.radians(deg)
        
        # Calculate vector for this segment (in X-Z plane)
        dx = link_length * np.cos(rad)
        dz = link_length * np.sin(rad)
        vector = np.array([dx, 0, dz])
        
        start_center = current_pos
        end_center = current_pos + vector
        
        # --- Generate Box Corners (Face-sharing logic) ---
        # We define a generic box and rotate it to match the vector
        # Normal vector along the segment
        v_norm = vector / np.linalg.norm(vector)
        v_up = np.array([0, 1, 0]) # Width is along Y
        v_side = np.cross(v_norm, v_up) # Thickness/Height direction
        
        # Offsets for width (Y) and thickness (Z-ish)
        w_vec = v_up * (width / 2.0)
        h_vec = v_side * (height / 2.0)
        
        # 4 corners at Start
        s_corners = [
            start_center + w_vec + h_vec,
            start_center - w_vec + h_vec,
            start_center - w_vec - h_vec,
            start_center + w_vec - h_vec
        ]
        
        # 4 corners at End
        e_corners = [
            end_center + w_vec + h_vec,
            end_center - w_vec + h_vec,
            end_center - w_vec - h_vec,
            end_center + w_vec - h_vec
        ]
        
        current_cluster = []
        
        # -- Add Start Nodes --
        if i == 0:
            # First segment creates new start nodes
            for p in s_corners:
                x_coords.append(p[0]); y_coords.append(p[1]); z_coords.append(p[2])
                current_cluster.append(node_idx)
                node_idx += 1
        else:
            # Subsequent segments reuse the previous end nodes (The Hinge)
            current_cluster.extend(prev_end_nodes)
            hinges.extend(prev_end_nodes)
            
        # -- Add End Nodes --
        new_end_nodes = []
        for p in e_corners:
            x_coords.append(p[0]); y_coords.append(p[1]); z_coords.append(p[2])
            current_cluster.append(node_idx)
            new_end_nodes.append(node_idx)
            node_idx += 1
            
        clusters.append(current_cluster)
        
        # Update state for next loop
        current_pos = end_center
        prev_end_nodes = new_end_nodes
        
    return np.array([x_coords, y_coords, z_coords]), clusters, list(set(hinges))

# --- Visualization ---
fig = plt.figure(figsize=(12, 6))

# Plot 1: The Standing "W"
P_w, C_w, H_w = generate_folding_bar("W")
ax1 = fig.add_subplot(121, projection='3d')
colors = ['#FF5733', '#33FF57', '#3357FF', '#F333FF'] # Distinct colors for 4 segments

for i, idxs in enumerate(C_w):
    xs, ys, zs = P_w[0, idxs], P_w[1, idxs], P_w[2, idxs]
    # Draw edges of the box
    edges = [
        (0,1), (1,2), (2,3), (3,0), # Start face
        (4,5), (5,6), (6,7), (7,4), # End face
        (0,4), (1,5), (2,6), (3,7)  # Connecting lines
    ]
    # Note: Logic assumes order [s1, s2, s3, s4, e1, e2, e3, e4]
    # Re-map indices 0-7 to the actual node indices in 'idxs'
    local_pts = np.array([[xs[k], ys[k], zs[k]] for k in range(8)])
    
    for e_start, e_end in edges:
        ax1.plot(
            [local_pts[e_start, 0], local_pts[e_end, 0]],
            [local_pts[e_start, 1], local_pts[e_end, 1]],
            [local_pts[e_start, 2], local_pts[e_end, 2]],
            color=colors[i], linewidth=2
        )
    
    # Label segment
    center = np.mean(local_pts, axis=0)
    ax1.text(center[0], center[1], center[2]+0.5, f"Seg {i+1}", color=colors[i], fontweight='bold')

# Highlight Hinges
hx, hy, hz = P_w[0, H_w], P_w[1, H_w], P_w[2, H_w]
ax1.scatter(hx, hy, hz, c='black', s=50, label='Hinges (Shared Face)')

ax1.set_title("State 1: Standing 'W' Shape")
ax1.set_xlim(-1, 8); ax1.set_ylim(-2, 2); ax1.set_zlim(-3, 3)
ax1.set_xlabel("X"); ax1.set_zlabel("Height (Z)")
ax1.view_init(elev=20, azim=-80)

# Plot 2: The Laid Flat Bar
P_flat, C_flat, H_flat = generate_folding_bar("Flat")
ax2 = fig.add_subplot(122, projection='3d')

for i, idxs in enumerate(C_flat):
    xs, ys, zs = P_flat[0, idxs], P_flat[1, idxs], P_flat[2, idxs]
    local_pts = np.array([[xs[k], ys[k], zs[k]] for k in range(8)])
    for e_start, e_end in edges:
        ax2.plot(
            [local_pts[e_start, 0], local_pts[e_end, 0]],
            [local_pts[e_start, 1], local_pts[e_end, 1]],
            [local_pts[e_start, 2], local_pts[e_end, 2]],
            color=colors[i], linewidth=2
        )

# Highlight Hinges
hx, hy, hz = P_flat[0, H_flat], P_flat[1, H_flat], P_flat[2, H_flat]
ax2.scatter(hx, hy, hz, c='black', s=50)

ax2.set_title("State 2: Laid Flat Bar (4 Segments)")
ax2.set_xlim(-1, 9); ax2.set_ylim(-2, 2); ax2.set_zlim(-3, 3)
ax2.set_xlabel("X"); ax2.set_zlabel("Height (Z)")
ax2.view_init(elev=20, azim=-80)

plt.tight_layout()
plt.show()
