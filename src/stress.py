import numpy as np
import cvxpy as cp

from src.utils.utils import incidence_matrix_fully_connected

def optimize_stress(nominal_config: np.ndarray, hyperparam_dict: dict, sparse_tol=5e-5, message=False):

        """
        Our proposed framework to calculate the sparse stress matrix.
        cvxpy is needed for the convex optimization.
        
        nominal_config: np.ndarray
            The nominal configuration of shape (D, N) where D is the dimension and N is the number of nodes.
        hyperparam_dict: dict
            Dictionary containing the hyperparameters for the optimization
            alpha: sparsity-convergence trade-off; beta: largest eigenvalue; gamma: smallest eigenvalue 
        """

        D, N = nominal_config.shape
        incidence_full = incidence_matrix_fully_connected(N)
        
        M = incidence_full.shape[1]
        
        augmented_config = np.hstack((nominal_config.T, np.ones((N, 1))))
        U, _, _ = np.linalg.svd(augmented_config)
        Q = U[:, D+1:]
        
        E = np.vstack([augmented_config.T @ incidence_full @ np.diag(incidence_full.T[:, n]) for n in range(N)])
        Psi = Q.T @ incidence_full
        
        phi = np.diag(Psi.T @ Psi)

        ########## Convex Optimization ##########
        weights = cp.Variable(M)
        
        obj = cp.Minimize(cp.norm(weights, 1) - hyperparam_dict['alpha'] * phi.T @ weights)
        constraints = [
            cp.norm(incidence_full @ cp.diag(weights) @ incidence_full.T, 2) <= hyperparam_dict['beta'],
            Psi @ cp.diag(weights) @ Psi.T - hyperparam_dict['gamma'] * np.eye(N - D - 1) >> 0,
            E @ weights == 0
        ]
        
        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.SCS)

        weights_optimal = weights.value

        if message:
            print(f"There are: {np.count_nonzero(np.abs(weights_optimal) > sparse_tol)} edges")
            print(f"suggested alpha: {1/np.max(phi)}")

        # weights_optimal[np.abs(weights_optimal) < sparse_tol] = 0   # small weights to zero for plotting purpose

        stress_mat = incidence_full @ np.diag(weights_optimal) @ incidence_full.T
        return stress_mat
    


def optimize_stress_legacy(nominal_config, hyperparam_dict, sparse_tol=5e-5, message=False):
    """
    Our proposed framework to calculate the sparse stress matrix.
    cvxpy is needed for the convex optimization.
    
    hyperparam_dict: dict
        Dictionary containing the hyperparameters for the optimization.
    """

    D, N = nominal_config.shape
    incidence_full = incidence_matrix_fully_connected(N)
    M = incidence_full.shape[1]
    
    augmented_config = np.hstack((nominal_config.T, np.ones((N, 1))))
    U, _, _ = np.linalg.svd(augmented_config)
    Q = U[:, D+1:]
    
    E = np.vstack([augmented_config.T @ incidence_full @ np.diag(incidence_full.T[:, n]) for n in range(N)])
    Psi = Q.T @ incidence_full
    
    phi = np.diag(Psi.T @ Psi)

    
    weights = cp.Variable(M)
    
    obj = cp.Minimize(cp.norm(weights, 1) - hyperparam_dict['alpha'] * phi.T @ weights)
    constraints = [
        cp.norm(incidence_full @ cp.diag(weights) @ incidence_full.T, 2) <= hyperparam_dict['beta'],
        Psi @ cp.diag(weights) @ Psi.T - hyperparam_dict['gamma'] * np.eye(N - D - 1) >> 0,
        E @ weights == 0
    ]
    
    prob = cp.Problem(obj, constraints)
    # prob.solve()
    prob.solve(solver=cp.SCS)

    weights_optimal = weights.value
    # print(weights_optimal)
    # weights_optimal[weights_optimal < sparse_tol] = 0
    # print(weights_optimal)

    
    
    # if message:
        # print(f"There are: {np.count_nonzero(np.abs(weights_optimal) > sparse_tol)} edges")
        # print(f"suggested alpha upper bound: {1/np.max(phi)}")

    weights_optimal[np.abs(weights_optimal) < sparse_tol] = 0   # small weights to zero for plotting purpose
    # print(weights_optimal)
    stress = incidence_full @ np.diag(weights_optimal) @ incidence_full.T
    return stress