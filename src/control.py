import numpy as np
from src.utils.utils import find_global_stress_mat

def tracking_error(config, target_config):

    max_itr = config.shape[2]
    target_config = np.tile(target_config[:, :, np.newaxis], (1, 1, max_itr))

    return np.linalg.norm(config - target_config, axis=(0,1))

def affine_formation_control_global(target_config, stress_mat, leader_id, gain = 1, max_itr = 10000, dt = 0.01):

    # leader_id: list of leader node indices

    D, N = target_config.shape

    if leader_id is None:
        leader_id = list(range(D + 1))

    assert len(leader_id) >= D + 1, "leader nodes must be at least D + 1"

    config = np.random.rand(D, N)
    config_track = np.zeros((D, N, max_itr))
    # tracking_error = np.zeros(max_itr)
    
    for i in range(max_itr):
        control_input = stress_mat @ config.T
        config -= gain * dt * control_input.T
        config[:, leader_id] = target_config[:, leader_id]
        # log current configuration
        config_track[:, :, i] = config
        # tracking_error[i] = np.linalg.norm(config - target_config)
    
    return config_track


def multicluster_afc_global(target_config: np.ndarray, 
                        subgraph_glb_idx: list, 
                        leaders_glb_idx: list, 
                        stress_mat_list: list,
                        max_itr: int = 10000, 
                        dt: float = 0.01, 
                        gain: float = 10.0):
    
    D, N = target_config.shape

    stress_mat_global = find_global_stress_mat(stress_mat_list, subgraph_glb_idx, target_config.shape[1])

    init_config = np.random.rand(D, N)
    # global control
    config_track_glb = np.zeros((D, N, max_itr+1))
    config_track_glb[:, :, 0] = init_config  # initial configuration
    for itr in range(max_itr):
        control_input_glb = - gain * config_track_glb[:, :, itr] @ stress_mat_global.T  # shape (D,N)
        config_track_glb[:, :, itr+1] = config_track_glb[:, :, itr] + dt * control_input_glb
        # leaders fixed at target positions
        for leader_idx in leaders_glb_idx:
            config_track_glb[:, leader_idx, itr+1] = target_config[:, leader_idx]

    return config_track_glb