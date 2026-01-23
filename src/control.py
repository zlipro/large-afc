import numpy as np

def trackig_error(config, target_config):

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