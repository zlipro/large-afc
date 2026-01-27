import numpy as np
import matplotlib.pyplot as plt
from src import stress
from src.utils.utils import stress_normalizer
import src.utils.geometry as geometry
import time
# compare the symmetric stress optimization with normal stress optimization on football case

if __name__ == "__main__":
    
    hyperparam_dict = { 'alpha': 1, 'beta': 1, 'gamma': 0.1 }
    nominal_config, _= geometry.truncated_icosahedron()

    

    # time the symmetric method
    start_time = time.time()
    stress_mat_symmetric = stress_normalizer(stress.optimized_stress_symm(
        nominal_config,
        hyperparam_dict,
        verbose = False
    ))
    end_time = time.time()
    print(f"Symmetric stress optimization time: {end_time - start_time:.2f} seconds")

    # time the computation
    start_time = time.time()
    # the normal method
    stress_mat_normal = stress_normalizer(stress.optimize_stress(
        nominal_config,
        hyperparam_dict,
        verbose = False
    ))
    end_time = time.time()
    print(f"Normal stress optimization time: {end_time - start_time:.2f} seconds")

    # simulate both stress matrices with AFC
    # pick 6 leaders randomly
    leader_ids = np.random.choice(nominal_config.shape[1], 6, replace=False).tolist()
    from src.control import affine_formation_control_global, tracking_error
    TE_normal = tracking_error(
        affine_formation_control_global(
            nominal_config, stress_mat_normal, leader_ids, gain=70
        ),
        nominal_config
    )
    TE_symmetric = tracking_error(
        affine_formation_control_global(
            nominal_config, stress_mat_symmetric, leader_ids, gain=70
        ),
        nominal_config
    )
    # plot the tracking errors
    plt.figure()
    plt.plot(TE_normal, label='Normal Stress AFC')
    plt.plot(TE_symmetric, label='Symmetric Stress AFC')
    plt.title("Tracking Error Comparison")
    plt.xlabel("Iteration")
    plt.ylabel("Tracking Error")
    plt.yscale('log')
    plt.legend()
    plt.grid()
    plt.show()