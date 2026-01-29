import numpy as np
import matplotlib.pyplot as plt
from src import stress
from src.utils.utils import stress_normalizer
from src.control import multicluster_afc_global, tracking_error, multicluster_afc_distributed
import src.utils.geometry as geometry


# test multicluster stress optimization on random config
if __name__ == "__main__":


    # nominal_config = np.random.rand(2, 20) * 10  # Random configuration
    # subgraph_glb_idx = [list(range(0, 15)), list(range(10, 20))]  # Two clusters
    nominal_configs, subgraph_glb_idx = geometry.letter_w(filling_points_per_seg=10)

    hyperparam_dict = { 'alpha': 0.5, 'beta': 1, 'gamma': 0.1 }

    # stress_mat_list = stress_normalizer(stress.compute_stress_multicluster(
    #     nominal_config,
    #     subgraph_glb_idx,
    #     hyperparam_dict,
    #     parallel=True
    # ))
    nominal_config_bar = nominal_configs["nominal_config_bar"]
    nominal_config_V = nominal_configs["nominal_config_V"]
    nominal_config_W = nominal_configs["nominal_config_W"]

    ### in this special W case, where each cluster has identical configuraions,
    ### we could just reuse the same stress matrix for each cluster
    stress_mat_cluster_1 = stress_normalizer(stress.optimize_stress(
        nominal_config_bar[:, subgraph_glb_idx[0]],
        hyperparam_dict,
        verbose = False
    ))
    stress_mat_list = [stress_mat_cluster_1 for _ in subgraph_glb_idx]

    # done with stress computation, now test multicluster AFC

    TE_multicluster = tracking_error(multicluster_afc_global(
        nominal_config_W,
        subgraph_glb_idx,
        # leaders_glb_idx = np.random.choice(nominal_config.shape[1], 10, replace=False).tolist(),
        leaders_glb_idx = [6,7,17,25,31,41,47,59],
        stress_mat_list = stress_mat_list,
        gain = 60,
        max_itr = 10000,
        dt = 0.01
        ), 
        nominal_config_W)
    
    TE_multicluster_distributed = tracking_error(multicluster_afc_distributed(
        nominal_config_W,
        subgraph_glb_idx,
        # leaders_glb_idx = np.random.choice(nominal_config.shape[1], 10, replace=False).tolist(),
        leaders_glb_idx = [6,7,17,25,31,41,47,59],
        stress_mat_list = stress_mat_list,
        gain = 60,
        max_itr = 10000,
        dt = 0.01
        ),
        nominal_config_W)
    

    print("Subgraph Global Indices:", subgraph_glb_idx)
    plt.figure()
    plt.plot(TE_multicluster)
    plt.plot(TE_multicluster_distributed, linestyle='--')
    plt.legend(["Multicluster AFC - Global", "Multicluster AFC - Distributed"])
    plt.title("Tracking Error - Multicluster AFC")
    plt.xlabel("Iteration")
    plt.ylabel("Tracking Error")
    plt.grid()
    plt.yscale('log')
    plt.show()
        
