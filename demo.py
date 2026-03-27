import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import yaml
import time
import os
import argparse

# --- Project Imports ---
import src.utils.geometry as geometry
import src.stress as stress
from src.utils.utils import find_global_stress_mat
from src.control import multicluster_afc_global

import matplotlib.animation as animation

def save_formation_video(all_data, cluster_indices=None, leaders_idx=[], 
                         filename='results/demo_formation.gif', fps=30, downsample=10):
    """
    Generates a clean animation of the 3D formation process.
    
    Args:
        all_data: (Total_Steps, 3, N) trajectory array.
        cluster_indices: List of lists containing node indices for each segment.
        leaders_idx: List/Array of leader indices (will be black).
        filename: Output path.
        fps: Frames per second.
        downsample: Skip frames for faster rendering.
    """
    print(f">>> Rendering video to {filename}...")
    
    # Downsample Data
    data_subset = all_data[::downsample]
    num_frames, _, num_nodes = data_subset.shape
    
    # Setup Figure (No Frame)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    cmap = plt.get_cmap('cool') 
    node_colors = np.zeros((num_nodes, 4)) # RGBA
    
    if cluster_indices is not None:
        num_clusters = len(cluster_indices)
        for k, indices in enumerate(cluster_indices):
            # Distribute colors evenly from Cyan (0.0) to Magenta (1.0)
            val = k / (num_clusters - 1) if num_clusters > 1 else 0.5
            color = cmap(val)
            node_colors[indices] = color
    else:
        # Fallback if no clusters provided
        node_colors[:] = [0, 1, 1, 1] # Cyan

    # Override Leaders to Black
    if len(leaders_idx) > 0:
        node_colors[leaders_idx] = [0, 0, 0, 1]

    # Node Sizes (Leaders larger)
    sizes = np.full(num_nodes, 10)
    sizes[leaders_idx] = 60

    # --- EQUAL AXIS LOGIC ---
    all_x = all_data[:, 0, :].flatten()
    all_y = all_data[:, 1, :].flatten()
    all_z = all_data[:, 2, :].flatten()
    
    mid_x = (all_x.max() + all_x.min()) * 0.5
    mid_y = (all_y.max() + all_y.min()) * 0.5
    mid_z = (all_z.max() + all_z.min()) * 0.5
    
    max_range = np.max([
        all_x.max() - all_x.min(), 
        all_y.max() - all_y.min(), 
        all_z.max() - all_z.min()
    ]) / 2.0
    
    pad = max_range * 1.1
    ax.set_xlim(mid_x - pad, mid_x + pad)
    ax.set_ylim(mid_y - pad, mid_y + pad)
    ax.set_zlim(mid_z - pad, mid_z + pad)
    
    # Force Cubic Aspect
    ax.set_box_aspect([1, 1, 1])
    
    # Remove Axes/Grid for "Floating" look
    ax.set_axis_off()

    # 3. Initial Plot Object
    scatter = ax.scatter(
        data_subset[0, 0, :], 
        data_subset[0, 1, :], 
        data_subset[0, 2, :], 
        c=node_colors, 
        s=sizes, 
        alpha=0.9,
        depthshade=True
    )
    
    # Minimalist Phase Text (Top Left)
    phase_text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes, fontsize=14, fontweight='bold', color='black')

    # 4. Update Function
    def update(frame_idx):
        # Update Positions
        current_pos = data_subset[frame_idx]
        scatter._offsets3d = (current_pos[0, :], current_pos[1, :], current_pos[2, :])
        
        # Update Text
        progress = (frame_idx * downsample) / all_data.shape[0]
        if progress < 0.33:
            phase = "Phase 1: BAR"
        elif progress < 0.66:
            phase = "Phase 2: V-Shape"
        else:
            phase = "Phase 3: W-Shape"
            
        phase_text.set_text(phase)
        
        # Rotate Camera
        ax.view_init(elev=30, azim=frame_idx * 0.5)
        
        return scatter, phase_text

    # 5. Render
    ani = animation.FuncAnimation(fig, update, frames=num_frames, blit=False)
    
    if filename.endswith('.mp4'):
        try:
            writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
            ani.save(filename, writer=writer)
        except Exception:
            print("FFmpeg failed. Saving as GIF.")
            filename = filename.replace('.mp4', '.gif')
            writer = animation.PillowWriter(fps=fps)
            ani.save(filename, writer=writer)
    else:
        writer = animation.PillowWriter(fps=fps)
        ani.save(filename, writer=writer)

    print(f">>> Video saved: {filename}")
    plt.close(fig)

def plot_trajectory_3d(all_data, phase_steps, cluster_indices=None, leaders_idx=[], 
                       view_angle=(30, 45), node_size=20, trail_length=300, filename=None):
    """
    Plots the 3D configuration at specific phase steps with a short trailing trajectory.
    
    Args:
        all_data: (Total_Steps, 3, N) numpy array.
        phase_steps: List of step indices where phases end (e.g., [0, 500, 1000]).
        cluster_indices: (Optional) Used for logical grouping, though color is now 'cool' spectrum.
        leaders_idx: Flat list/array of leader node indices.
        view_angle: Tuple (elev, azim) for camera.
        node_size: Base size for scatter points.
        trail_length: How many steps back to plot the trajectory for each phase.
        filename: Output filename.
    """
    print(f">>> Plotting 3D Trajectory (Trails Only)...")
    
    num_steps, _, num_nodes = all_data.shape
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # We generate a spectrum of colors for the nodes
    cmap = plt.get_cmap('cool') 
    node_colors = [cmap(i / num_nodes) for i in range(num_nodes)]

    
    for step_idx in phase_steps:
        # Clamp index
        if step_idx >= num_steps: step_idx = num_steps - 1
        
        # Only plot trails if we are past the start
        if step_idx > 0:
            start_trail = max(0, step_idx - trail_length)
            # Slice the data for this specific time window: shape (T_window, 3, N)
            trail_data = all_data[start_trail : step_idx + 1]
            
            for i in range(num_nodes):
                is_leader = i in leaders_idx
                
                # Style Logic
                if is_leader:
                    c = 'black'
                    alpha = 0.6
                    lw = 1.2
                    zorder = 10
                else:
                    c = node_colors[i]
                    alpha = 0.2  # Very transparent for the trails
                    lw = 0.8
                    zorder = 1
                
                ax.plot(trail_data[:, 0, i], trail_data[:, 1, i], trail_data[:, 2, i], 
                        color=c, alpha=alpha, linewidth=lw, zorder=zorder)

        # --- Plot Scatter Dots (The "Head") ---
        current_config = all_data[step_idx]
        
        # Prepare arrays for scatter
        sizes = np.full(num_nodes, node_size)
        sizes[leaders_idx] = node_size * 1.5  # Leaders bigger
        
        # Color array for this frame
        frame_colors = np.array(node_colors)
        if len(leaders_idx) > 0:
            frame_colors[leaders_idx] = [0, 0, 0, 1] # Leaders black
            
        ax.scatter(current_config[0, :], current_config[1, :], current_config[2, :], 
                   c=frame_colors, 
                   s=sizes, 
                   alpha=1.0, 
                   depthshade=True)

    # 3. Equal Axis / Box Aspect Logic
    # Calculate global bounds for the cubic box
    all_x = all_data[:, 0, :].flatten()
    all_y = all_data[:, 1, :].flatten()
    all_z = all_data[:, 2, :].flatten()
    
    mid_x = (all_x.max() + all_x.min()) * 0.5
    mid_y = (all_y.max() + all_y.min()) * 0.5
    mid_z = (all_z.max() + all_z.min()) * 0.5
    
    max_range = np.max([
        all_x.max() - all_x.min(), 
        all_y.max() - all_y.min(), 
        all_z.max() - all_z.min()
    ]) / 2.0
    
    # Add small padding
    buffer = max_range * 1.1
    
    ax.set_xlim(mid_x - buffer, mid_x + buffer)
    ax.set_ylim(mid_y - buffer, mid_y + buffer)
    ax.set_zlim(mid_z - buffer, mid_z + buffer)
    
    # Force cubic aspect
    ax.set_box_aspect([1, 1, 1])
    
    # Final Polish
    if view_angle is not None:
        ax.view_init(elev=view_angle[0], azim=view_angle[1], roll = view_angle[2])
        
    ax.set_axis_off()
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f">>> Plot saved to {filename}")
    
    # plt.show()


def load_config(path='configs/benchmarks/letter-w.yaml'):
    if not os.path.exists(path):
        return {
            'opt_hyperparam_set': {'alpha': 1.0, 'beta': 1, 'gamma': 0.1},
        }
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def project_to_target(nominal, target_center):
    """Helper to center the nominal configuration at a specific 3D point."""
    centroid = np.mean(nominal, axis=1, keepdims=True)
    centered = nominal - centroid
    target_center = np.array(target_center).reshape(3, 1)
    return centered + target_center


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Run Formation Control Demo")
    parser.add_argument('--config', type=str, default='configs/benchmarks/letter-w.yaml', 
                        help='Path to the configuration YAML file')                      
    args = parser.parse_args()

    cfg = load_config(args.config)
    hparams = cfg.get('opt_hyperparam_set', {'alpha': 1.0, 'beta': 1, 'gamma': 0.1})
    print(">>> Configuration Loaded.")

    # 2. Generate Letter-W Geometry
    print(">>> Generating Geometry...")
    filling_points = cfg['geometry'].get('filling_points_per_seg')
    nominal_config_dict, subgraph_glb_idx = geometry.letter_w(filling_points_per_seg = filling_points)
    

    nominal_config_bar = nominal_config_dict['nominal_config_bar'] 
    D, N = nominal_config_bar.shape
    
    print(f"    Nodes: {N} | Dimension: {D} | Clusters: {len(subgraph_glb_idx)}")

    # the stress matrix is invariant. We compute it once for the base unit.
    print(f"    Computing stress (Single Unit)...")

    # Extract geometry of first cluster (Cluster 0 of Bar)
    indices_seg_0 = subgraph_glb_idx[0]
    config_seg_0 = nominal_config_bar[:, indices_seg_0]
    
    # Optimize ONCE
    t0 = time.perf_counter()
    stress_unit = stress.optimize_stress(config_seg_0, hparams)
    t1 = time.perf_counter()
    print(f"    Done ({t1 - t0:.4f}s).")

    
    # Replicate for all 4 clusters
    stress_list_common = [stress_unit for _ in subgraph_glb_idx]

    # randomly choose 3 leaders in each segment
    leaders_idx = []
    for seg_indices in subgraph_glb_idx:
        leaders_idx.append(np.random.choice(seg_indices, size=3, replace=False)) 
    
    # Define Targets
    offset = 25
    targets = {
        'bar': project_to_target(nominal_config_dict['nominal_config_bar'], [offset, offset, 0]),
        'V':   project_to_target(nominal_config_dict['nominal_config_V'],   [offset, -offset, 0]),
        'W':   project_to_target(nominal_config_dict['nominal_config_W'],   [-offset, -offset, 0])
    }
    
    # Initial State: Random cloud near Start Location
    start_loc = [-offset, offset, 0]
    current_config = project_to_target(np.random.randn(D, N) * 4, start_loc)


    full_trajectory = []
    # replicate initial config a few times for a longer start
    for _ in range(2000):
        full_trajectory.append(current_config[np.newaxis, :, :])  # Shape (1, D, N) 
    
    print("\n>>> Starting Simulation...")
    
    phases = ['bar', 'V', 'W']
    steps_per_phase = 2000
    
    for phase in phases:
        print(f"    [Phase: {phase.upper()}] Moving to target...")
        
        target_cfg = targets[phase]
        
        # Call API with the COMMON stress list
        traj = multicluster_afc_global(
            target_config=target_cfg,
            subgraph_glb_idx=subgraph_glb_idx,
            leaders_glb_idx=leaders_idx,
            stress_mat_list=stress_list_common,
            init_config=current_config,
            max_itr=steps_per_phase,
            dt=0.01,
            gain=20.0
        )
        
        # traj is (D, N, Steps). Transpose to (Steps, D, N) for storage
        traj_T = np.transpose(traj, (2, 0, 1))
        full_trajectory.append(traj_T)
        
        # Update start state for next phase
        current_config = traj[:, :, -1]

    print(">>> Simulation Complete.")

    all_data = np.concatenate(full_trajectory, axis=0)
    # save_formation_video(all_data, filename='results/demo_formation.gif', fps=30, downsample=20)
    checkpoints = [0, 4000, 6000, 8000] 

    # Flatten your list of leaders into a single list of indices
    # (Assuming leaders_idx is a list of arrays from your previous code)
    flat_leaders = np.concatenate(leaders_idx).tolist() 

    plot_trajectory_3d(
        all_data, 
        phase_steps=checkpoints, 
        cluster_indices=subgraph_glb_idx,
        leaders_idx=[],  # You can set this to flat_leaders if you want to highlight them] 
        view_angle=(13, -147, 0),
        node_size=5,
        trail_length=1600,
        filename='results/trajectory_final.png'
    )

    save_formation_video(
    all_data, 
    cluster_indices=subgraph_glb_idx, 
    leaders_idx=[],
    filename='results/demo_formation.gif',
    downsample=20 # Adjust speed here
    ) 