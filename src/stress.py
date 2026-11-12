import numpy as np
import scipy.sparse as sp
import scipy.linalg as sla
import cvxpy as cp

from src.utils.utils import get_symmetry_mapping_matrix

# ---------------------------------------------------------------------------
# Memory-efficient SDP construction (shared by P2 and P3/USI)
#
# The original formulation built  Psi @ diag(w) @ Psi.T  and  B @ diag(w) @ B.T
# with cp.diag, which makes CVXPY materialise dense coefficient tensors and runs
# out of memory around N = 60-100.  The constraints below are exactly equivalent
# but sparse in the edge weights:
#
#   Omega = B diag(w) B^T            ->  vec(Omega) = K w   (4 nonzeros / column)
#   Q^T Omega Q >= gamma I           ->  Omega + gamma U U^T >= gamma I
#                                        (U orthonormal basis of range(Pbar^T);
#                                         exact because Omega U = 0)
#   ||Omega||_2 <= beta              ->  Omega <= beta I    (Omega is PSD)
#   - alpha psi^T w                  ->  - alpha * 2 * 1^T w   (= tr(Omega); equal
#                                         to psi^T w on the feasible set)
#   E w = 0                          ->  (Pbar kron I) K w = 0, dependent rows removed
# ---------------------------------------------------------------------------

def _stress_map(N):
    """Sparse K (N^2 x M) with vec_rowmajor(B diag(w) B^T) = K @ w.

    Edge order matches incidence_matrix_fully_connected: (0,1), (0,2), ..., (N-2,N-1).
    """
    iu, ju = np.triu_indices(N, k=1)
    M = iu.size
    rows = np.concatenate([iu * N + iu, ju * N + ju, iu * N + ju, ju * N + iu])
    cols = np.tile(np.arange(M), 4)
    vals = np.concatenate([np.ones(2 * M), -np.ones(2 * M)])
    return sp.csr_matrix((vals, (rows, cols)), shape=(N * N, M))


def _equilibrium_rows(augmented_config_T, KG):
    """Independent rows of the equilibrium constraint Pbar Omega = 0, as a sparse matrix in w."""
    N = augmented_config_T.shape[1]
    Ed = (sp.kron(sp.csr_matrix(augmented_config_T), sp.eye(N)) @ KG).toarray()
    rank = np.linalg.matrix_rank(Ed)
    _, _, piv = sla.qr(Ed.T, pivoting=True, mode="economic")
    return sp.csr_matrix(Ed[np.sort(piv[:rank])])


def _solve_stress_sdp(nominal_config, hyperparam_dict, group_map=None, solve_kwargs=None):
    """
    Solve P2 (group_map=None) or P3/USI (group_map = M x S edge-to-class matrix)
    with the sparse formulation. Returns (stress_mat, prob, full_weights).
    """
    D, N = nominal_config.shape
    alpha = hyperparam_dict['alpha']
    beta = hyperparam_dict['beta']
    gamma = hyperparam_dict['gamma']

    Pbar = np.vstack([nominal_config, np.ones((1, N))])      # (D+1) x N
    K = _stress_map(N)
    G = sp.eye(K.shape[1], format="csr") if group_map is None else sp.csr_matrix(group_map)
    KG = (K @ G).tocsr()
    E_eq = _equilibrium_rows(Pbar, KG)
    U = np.linalg.qr(Pbar.T)[0]                              # N x (D+1), orthonormal

    theta = cp.Variable(G.shape[1])
    weights = theta if group_map is None else G @ theta
    Omega = cp.reshape(KG @ theta, (N, N), order="C")

    obj = cp.Minimize(cp.norm(weights, 1) - alpha * 2.0 * cp.sum(weights))
    constraints = [
        Omega + gamma * (U @ U.T) >> gamma * np.eye(N),      # <=> Q^T Omega Q >= gamma I
        Omega << beta * np.eye(N),                           # <=> ||Omega||_2 <= beta
        E_eq @ theta == 0,                                   # <=> Omega Pbar^T = 0
    ]
    prob = cp.Problem(obj, constraints)
    prob.solve(**(solve_kwargs or {}))

    if theta.value is None:
        return None, prob, None
    w = np.asarray(G @ theta.value).ravel()
    stress_mat = (K @ w).reshape(N, N)
    return stress_mat, prob, w


def _suggested_alpha(nominal_config):
    """1 / max(psi), psi_e = ||Q^T b_e||^2, computed without forming Q^T B."""
    D, N = nominal_config.shape
    augmented_config = np.hstack((nominal_config.T, np.ones((N, 1))))
    Q = np.linalg.svd(augmented_config)[0][:, D + 1:]
    QQ = Q @ Q.T
    i, j = np.triu_indices(N, k=1)
    psi = QQ[i, i] + QQ[j, j] - 2 * QQ[i, j]
    return 1.0 / np.max(psi)


def optimize_stress(nominal_config: np.ndarray, hyperparam_dict: dict, verbose=False):

        """
        Our proposed framework to calculate the sparse stress matrix (P2).
        cvxpy is needed for the convex optimization.

        nominal_config: np.ndarray
            The nominal configuration of shape (D, N) where D is the dimension and N is the number of nodes.
        hyperparam_dict: dict
            Dictionary containing the hyperparameters for the optimization
            alpha: sparsity-convergence trade-off; beta: largest eigenvalue; gamma: smallest eigenvalue
        """

        stress_mat, prob, _ = _solve_stress_sdp(nominal_config, hyperparam_dict,
                                                solve_kwargs=dict(solver=cp.MOSEK))

        if verbose:
            print(f"suggested alpha: {_suggested_alpha(nominal_config)}")

        return stress_mat

def optimized_stress_symm(nominal_config: np.ndarray, hyperparam_dict: dict, verbose=False):
    """
    The Unique Stress Identifier (USI) method (Section IV, Algorithm P3).
    Optimizes stress using symmetry groups identified via EDM.

    Args:
        nominal_config: np.ndarray (D, N)
        hyperparam_dict: dict with 'alpha', 'beta', 'gamma'

    Returns:
        stress_mat: np.ndarray (N, N)
    """
    # Symmetry Reduction (EDM Shortcut): omega_bar = S @ omega_underbar
    S_map, unique_dists = get_symmetry_mapping_matrix(nominal_config)

    if verbose:
        print(f"USI Method: Reduced variables from {S_map.shape[0]} edges to {S_map.shape[1]} unique groups.")

    try:
        stress_mat, prob, _ = _solve_stress_sdp(nominal_config, hyperparam_dict, group_map=S_map,
                                                solve_kwargs=dict(solver=cp.SCS, verbose=verbose))
    except Exception:
        stress_mat, prob, _ = _solve_stress_sdp(nominal_config, hyperparam_dict, group_map=S_map,
                                                solve_kwargs=dict(solver=cp.CVXOPT, verbose=verbose))

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        print(f"Warning: Problem status is {prob.status}")
        return None

    return stress_mat

def optimized_stress_symm_acc(nominal_config: np.ndarray, hyperparam_dict: dict, verbose=False):
    """
    Optimized Symmetric Stress (Fast Version).
    Same problem as optimized_stress_symm, with only K (groups) variables,
    solved with MOSEK when available.
    """
    S_map, unique_dists = get_symmetry_mapping_matrix(nominal_config)

    if verbose:
        print(f"USI Method: Collapsing {S_map.shape[0]} edges -> {S_map.shape[1]} groups.")

    # MOSEK is vastly superior for these SDPs
    solver = cp.MOSEK if 'MOSEK' in cp.installed_solvers() else cp.SCS
    stress_mat, prob, _ = _solve_stress_sdp(nominal_config, hyperparam_dict, group_map=S_map,
                                            solve_kwargs=dict(solver=solver, verbose=verbose))

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        if verbose: print(f"Warning: Problem status {prob.status}")
        return None

    return stress_mat

def compute_stress_multicluster(target_config, subgraph_glb_idx, hyperparam_dict, parallel=True):
    """
    loop over subgraphs and compute stress matrices and use parallel when specified.

    """
    
    if not parallel:
        stress_mat_list = []
        for subgraph in subgraph_glb_idx:
            nominal_subconfig = target_config[:, subgraph]
            stress_mat = optimize_stress(nominal_subconfig, hyperparam_dict)
            stress_mat_list.append(stress_mat)
        return stress_mat_list
    
    else:
        import psutil, multiprocessing
        from threadpoolctl import threadpool_limits
        
        N_CORES = psutil.cpu_count(logical=False)
        processes_needed = len(subgraph_glb_idx)
        
        # print(f"Starting parallel optimization on {processes_needed} tasks...")

        # --- PREPARE THE DATA ---
        tasks_args = [
            (target_config[:, subgraph], hyperparam_dict) 
            for subgraph in subgraph_glb_idx
        ]
        
        # --- EXECUTE ---
        # threadpool_limits here ensures that if the system forks (Linux), 
        with threadpool_limits(limits=6, user_api='blas'):
            
            # We use min() to avoid creating more processes than necessary
            with multiprocessing.Pool(processes=min(processes_needed, N_CORES)) as pool:
                stress_mat_list = pool.starmap(optimize_stress, tasks_args)

        return stress_mat_list
    

def sparsity_truncation(sparse_tol: float, stress_mat: np.ndarray):
    """

    # truncate small stress weights to zero

    """

    stress_mat_trunc = stress_mat.copy()
    stress_mat_trunc[np.abs(stress_mat_trunc) < sparse_tol] = 0

    return stress_mat_trunc

    
