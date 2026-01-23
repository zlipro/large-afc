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


    NUM_NODES = 10
    SPARSE_TOL = 5e-5
    nominal_config = geometry.circular_config(NUM_NODES, radius=2)
    # random configuration
    # nominal_config = np.random.rand(2, NUM_NODES) * 4

    hyperparam_dict = { 'alpha': 0.6, 'beta': 1, 'gamma': 0.1 }
    stress_mat = stress_normalizer(stress.optimize_stress(nominal_config, hyperparam_dict, sparse_tol = SPARSE_TOL, message=True))
    stress_mat_yang = stress_normalizer(yang2019.sparse_nullspace_recon(nominal_config))

    # xiao2022.check_solvers()
    stress_mat_xiao = stress_normalizer(xiao2022.misdp_faithful(nominal_config))
    # stress_mat_xiao = np.zeros((NUM_NODES, NUM_NODES))  # Placeholder if xiao2022 solver is not available

    stress_mat_li2026 = stress_normalizer(li2026.vertex_additions(nominal_config))

    incidence_mat = lin2016.incidence_grunbaum(NUM_NODES)
    stress_mat_lin2016 = stress_normalizer(lin2016.optimize_as_feasibility(incidence_mat, nominal_config))


    # save all the stress matrices in one .txt file with appropriate headers
    # np.savetxt("results/stress_matrices_circular_config.txt",
    #             np.array([stress_mat.flatten(),
    #                          stress_mat_yang.flatten(),
    #                          stress_mat_xiao.flatten()]),
    #             header="Optimized Sparse Stress\nYang et al. 2019 Stress\nXiao et al. 2022 Stress"
    #             )

    print(f"Number of edges in optimized stress: {detect_num_edges(stress_mat, SPARSE_TOL)}")
    print(f"Number of edges in Yang et al. 2019 stress: {detect_num_edges(stress_mat_yang, SPARSE_TOL)}")
    print(f"Number of edges in Xiao et al. 2022 stress: {detect_num_edges(stress_mat_xiao, SPARSE_TOL)}")
    print(f"Number of edges in Li et al. 2026 stress: {detect_num_edges(stress_mat_li2026, SPARSE_TOL)}")
    print(f"Number of edges in Lin et al. 2016 stress: {detect_num_edges(stress_mat_lin2016, SPARSE_TOL)}")

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

    TE_li2026 = trackig_error(
        affine_formation_control_global(
            nominal_config,
            stress_mat_li2026,
            None,
            gain=50
        ),
        nominal_config
    )

    TE_lin2016 = trackig_error(
        affine_formation_control_global(
            nominal_config,
            stress_mat_lin2016,
            None,
            gain=50
        ),
        nominal_config
    )

    plt.figure()
    plt.plot(TE)
    plt.plot(TE_yang2019)
    plt.plot(TE_xiao2022)
    plt.plot(TE_li2026)
    plt.plot(TE_lin2016)

    plt.yscale('log')
    plt.legend(["My Sparse Stress", "Yang2019", "Xiao2022", "Li2026", "Lin2016"])
    plt.title("Tracking Error under Optimized Sparse Stress")
    plt.xlabel("Iteration")
    plt.ylabel("Tracking Error (log scale)")
    plt.savefig("results/circular_stress_configuration.png", dpi=300)

    # Calculate eigenvalues
    eigenvalues_stress = np.linalg.eigvals(stress_mat)
    eigenvalues_stress_yang = np.linalg.eigvals(stress_mat_yang)
    eigenvalues_stress_xiao = np.linalg.eigvals(stress_mat_xiao)
    eigenvalues_stress_li2026 = np.linalg.eigvals(stress_mat_li2026)
    eigenvalues_stress_lin2016 = np.linalg.eigvals(stress_mat_lin2016)

    # Sort eigenvalues
    sorted_eigenvalues_stress = np.sort(eigenvalues_stress)
    sorted_eigenvalues_stress_yang = np.sort(eigenvalues_stress_yang)
    sorted_eigenvalues_stress_xiao = np.sort(eigenvalues_stress_xiao)
    sorted_eigenvalues_stress_li2026 = np.sort(eigenvalues_stress_li2026)
    sorted_eigenvalues_stress_lin2016 = np.sort(eigenvalues_stress_lin2016)

    # Plot eigenvalues
    plt.figure()
    plt.plot(sorted_eigenvalues_stress, 'o-' , label='Optimized Sparse Stress')
    plt.plot(sorted_eigenvalues_stress_yang, 's-', label='Yang et al. 2019')
    plt.plot(sorted_eigenvalues_stress_xiao, '^-', label='Xiao et al. 2022')
    plt.plot(sorted_eigenvalues_stress_li2026, 'd-', label='Li et al. 2026')
    plt.plot(sorted_eigenvalues_stress_lin2016, 'x-', label='Lin et al. 2016')
    plt.title("Sorted Eigenvalues of Stress Matrices")
    plt.xlabel("Index")
    plt.ylabel("Eigenvalue")
    plt.legend()
    plt.grid()
    plt.savefig("results/eigs_circular_config.png", dpi=300)