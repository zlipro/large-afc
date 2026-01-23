import numpy as np
import matplotlib.pyplot as plt

import src.utils.geometry as geometry
from src.utils.utils import detect_num_edges, stress_normalizer
from src.stress import optimize_stress
from src.graph import Graph
from src.utils.plots import visualize_graph
from src.control import affine_formation_control_global, trackig_error

from src.sota import yang2019, xiao2022
if __name__ == "__main__":
    

    SPARSE_TOL = 1e-6
    nominal_config = geometry.xiao_generic()

    hyperparam_dict = { 'alpha': 0.6, 'beta': 1, 'gamma': 0.1 }
    stress_mat = stress_normalizer(optimize_stress(nominal_config, hyperparam_dict, sparse_tol = SPARSE_TOL, message=True))
    stress_mat_yang = stress_normalizer(yang2019.sparse_nullspace_recon(nominal_config))

    # xiao2022.check_solvers()
    stress_mat_xiao = stress_normalizer(xiao2022.misdp_cutting_plane(nominal_config, beta = 0.1, mu = 0.6))

    print(f"Number of edges in optimized stress: {detect_num_edges(stress_mat, SPARSE_TOL)}")
    print(f"Number of edges in Yang et al. 2019 stress: {detect_num_edges(stress_mat_yang, SPARSE_TOL)}")
    print(f"Number of edges in Xiao et al. 2022 stress: {detect_num_edges(stress_mat_xiao, SPARSE_TOL)}")

    TE = trackig_error(
        affine_formation_control_global(
            nominal_config, stress_mat, None, gain=50
        ),
        nominal_config
    )

    TE_yang2019 = trackig_error(
        affine_formation_control_global(
            nominal_config, 
            stress_mat_yang,
            None,
            gain=50
        ),
        nominal_config
    )

    TE_xiao2022 = trackig_error(
        affine_formation_control_global(
            nominal_config, 
            stress_mat_xiao,
            None,
            gain=50
        ),
        nominal_config
    )

    # sort and plot the eigenvalues of each stress matrix

    # Calculate eigenvalues
    eigenvalues_stress = np.linalg.eigvals(stress_mat)
    eigenvalues_stress_yang = np.linalg.eigvals(stress_mat_yang)
    eigenvalues_stress_xiao = np.linalg.eigvals(stress_mat_xiao)

    # Sort eigenvalues
    sorted_eigenvalues_stress = np.sort(eigenvalues_stress)
    sorted_eigenvalues_stress_yang = np.sort(eigenvalues_stress_yang)
    sorted_eigenvalues_stress_xiao = np.sort(eigenvalues_stress_xiao)

    # Plot eigenvalues
    plt.figure()
    plt.plot(sorted_eigenvalues_stress, 'o-' , label='Optimized Sparse Stress')
    plt.plot(sorted_eigenvalues_stress_yang, 's-', label='Yang et al. 2019')
    plt.plot(sorted_eigenvalues_stress_xiao, '^-', label='Xiao et al. 2022')
    plt.title("Sorted Eigenvalues of Stress Matrices")
    plt.xlabel("Index")
    plt.ylabel("Eigenvalue")
    plt.legend()
    plt.grid()
    plt.savefig("results/eigs_xiao_generic.png", dpi=300)