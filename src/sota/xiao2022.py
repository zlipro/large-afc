import numpy as np
import cvxpy as cp
import scipy.linalg as la
import matplotlib.pyplot as plt
import sys

def check_solvers():
    """
    Checks for available solvers and returns the best one found.
    Priority: MOSEK > GUROBI > CVXPY Default
    """
    print("--- Checking Solver Installation ---\n")
    best_solver = None
    
    # 1. Check CVXPY availability
    try:
        import cvxpy as cp
        print(f"[OK] CVXPY is installed (Version: {cp.__version__})")
        
        # Check which solvers CVXPY detects
        installed_solvers = cp.installed_solvers()
        print(f"     Detected solvers: {', '.join(installed_solvers)}")
        
        # Check MOSEK specifically
        if 'MOSEK' in installed_solvers:
            print("     [OK] CVXPY detects MOSEK (Selected)")
            return cp.MOSEK
            
        # Check GUROBI specifically
        elif 'GUROBI' in installed_solvers:
            print("     [OK] CVXPY detects GUROBI (Selected)")
            return cp.GUROBI
            
        else:
            print("     [WARN] No commercial MISDP solver (MOSEK/GUROBI) detected.")
            print("     [WARN] Fallback to default solver (This may fail for Mixed-Integer problems).")
            return None # Let CVXPY choose
            
    except ImportError:
        print("[FAIL] CVXPY is not installed. Run: pip install cvxpy")
        return None


def misdp_cutting_plane(config_matrix, solver=None, mu=0.1, beta=0.5, q_min=0.01, q_max=1.0, max_iter=30, verbose=False):
    """
    Faithful reproduction of Xiao et al. (2022) with Multi-Cut Strategy.
    Accepts a 'solver' argument to avoid hardcoding MOSEK.
    """
    # Geometry & Setup
    if config_matrix.shape[0] > config_matrix.shape[1]:
        config_matrix = config_matrix.T
    d, n = config_matrix.shape
    
    # Gale Matrix & Null Space Basis
    P_bar = np.vstack((config_matrix, np.ones((1, n))))
    Z = la.null_space(P_bar)
    if Z.shape[1] == 0: raise ValueError("n must be >= d+2")
    U, _, _ = la.svd(Z)
    U1 = U[:, :n - d - 1]

    # Variables (Standard MISDP)
    w, x, y = {}, {}, {}
    lam_var = cp.Variable(nonneg=True, name="lambda")
    
    def get_idx(i, j): return tuple(sorted((i, j)))

    for i in range(n):
        for j in range(i + 1, n):
            idx = get_idx(i, j)
            w[idx] = cp.Variable()
            x[idx] = cp.Variable(integer=True)
            y[idx] = cp.Variable(integer=True)

    # Static Constraints
    constraints = []
    Omega_expr = [[0]*n for _ in range(n)]
    
    for i in range(n):
        row_sum_expr = 0
        for j in range(i + 1, n):
            idx = get_idx(i, j)
            # Binary Logic
            constraints += [x[idx] + y[idx] <= 1, x[idx] >= 0, y[idx] >= 0]
            
            # Weight Bounds
            constraints += [w[idx] >= -q_max * x[idx]]
            constraints += [w[idx] <= q_max * (1 - x[idx]) - q_min * x[idx]]
            constraints += [w[idx] >= -q_max * (1 - y[idx]) + q_min * y[idx]]
            constraints += [w[idx] <= q_max * y[idx]]
            
            Omega_expr[i][j] = -w[idx]
            Omega_expr[j][i] = -w[idx]
            row_sum_expr += w[idx]
        
        diag_sum = 0
        for k in range(n):
            if i != k: diag_sum += w[get_idx(i, k)]
        Omega_expr[i][i] = diag_sum
        constraints.append(Omega_expr[i][i] == diag_sum)

    Omega_cp = cp.bmat(Omega_expr)
    constraints.append(Omega_cp @ config_matrix.T == 0)

    # Objective Function
    comm_cost = 0
    total_dist = 0
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(config_matrix[:, i] - config_matrix[:, j])
            total_dist += dist
            idx = get_idx(i, j)
            comm_cost += dist * (x[idx] + y[idx])
            
    alpha1 = 1.0 / n
    alpha2 = 1.0 / total_dist if total_dist > 0 else 1.0
    objective = cp.Minimize(alpha1 * beta * lam_var + alpha2 * (1 - beta) * comm_cost)

    # Cutting Plane Loop
    dynamic_cuts = []
    
    # --- Solver Specific Parameters ---
    solve_kwargs = {"verbose": False}
    if solver == cp.MOSEK:
        solve_kwargs["mosek_params"] = {
            "MSK_DPAR_MIO_TOL_REL_GAP": 0.05,
            "MSK_DPAR_OPTIMIZER_MAX_TIME": 15.0,
        }
    elif solver == cp.GUROBI:
        # Gurobi uses simple kwargs or 'Param' dict depending on CVXPY version interface
        # but typically just passing them as kwargs works in recent versions
        # or via 'solver_specific_param'
        pass 

    if verbose: print(f"--- Faithful MISDP (N={n}, mu={mu}, solver={solver}) ---")

    for iter_idx in range(max_iter):
        prob = cp.Problem(objective, constraints + dynamic_cuts)
        
        try:
            # Use the detected solver here
            prob.solve(solver=solver, **solve_kwargs)
        except cp.error.SolverError as e:
            print(f"Solver failed step: {e}. Returning current best.")
            return Omega_cp.value

        # Check Spectral Properties
        if Omega_cp.value is None:
             print("Solver returned None (infeasible or failed).")
             return np.zeros((n,n))

        omega_val = Omega_cp.value
        omega_r = U1.T @ omega_val @ U1
        eigvals, eigvecs = la.eigh(omega_r)
        
        cuts_added = 0
        
        # 1. Stability Cuts (Lower Bound)
        for k in range(len(eigvals)):
            val = eigvals[k]
            if val < mu - 1e-4:
                v = eigvecs[:, k]
                cut = (v.T @ U1.T) @ Omega_cp @ (U1 @ v) >= mu
                dynamic_cuts.append(cut)
                cuts_added += 1
            else:
                break
        
        # 2. Condition Number Cuts (Upper Bound)
        current_lambda = lam_var.value
        if current_lambda is None: current_lambda = 100.0
        
        for k in range(len(eigvals)-1, -1, -1):
            val = eigvals[k]
            if val > (current_lambda * mu) + 1e-4:
                v = eigvecs[:, k]
                cut = (v.T @ U1.T) @ Omega_cp @ (U1 @ v) <= lam_var * mu
                dynamic_cuts.append(cut)
                cuts_added += 1
            else:
                break
                
        if verbose:
            min_eig, max_eig = eigvals[0], eigvals[-1]
            print(f"Iter {iter_idx}: MinEig={min_eig:.3f}, MaxEig={max_eig:.3f}, Cuts={cuts_added}")
            
        if cuts_added == 0:
            if verbose: print("Converged!")
            return omega_val

    print("Warning: Max iterations reached.")
    return Omega_cp.value