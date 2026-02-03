# import numpy as np
# import cvxpy as cp
# import scipy.linalg as la
# import matplotlib.pyplot as plt
# import sys

# def check_solvers():
#     print("--- Checking Solver Installation ---\n")
    
#     # 1. Check CVXPY availability
#     try:
#         import cvxpy as cp
#         print(f"[OK] CVXPY is installed (Version: {cp.__version__})")
        
#         # Check which solvers CVXPY detects
#         installed_solvers = cp.installed_solvers()
#         print(f"     Detected solvers: {', '.join(installed_solvers)}")
        
#         # Check MOSEK specifically in CVXPY
#         if 'MOSEK' in installed_solvers:
#             print("     [OK] CVXPY detects MOSEK")
#         else:
#             print("     [FAIL] CVXPY does NOT detect MOSEK")

#         # Check GUROBI specifically in CVXPY
#         if 'GUROBI' in installed_solvers:
#             print("     [OK] CVXPY detects GUROBI")
#         else:
#             print("     [FAIL] CVXPY does NOT detect GUROBI")
            
#     except ImportError:
#         print("[FAIL] CVXPY is not installed. Run: pip install cvxpy")
#         return

#     print("\n--- Checking Direct Library Imports ---\n")

#     # 2. Check MOSEK direct import
#     try:
#         import mosek
#         print(f"[OK] Mosek Python API is installed (Version: {mosek.Env.getversion()})")
#     except ImportError:
#         print("[FAIL] Mosek Python API not found. Run: pip install mosek")
#     except Exception as e:
#         print(f"[FAIL] Mosek error: {e}")

#     # 3. Check GUROBI direct import
#     try:
#         import gurobipy
#         print(f"[OK] Gurobi Python API is installed (Version: {gurobipy.gurobi.version()})")
#     except ImportError:
#         print("[FAIL] Gurobi Python API not found. Run: pip install gurobipy")
#     except Exception as e:
#         print(f"[FAIL] Gurobi error: {e}")



# def misdp_cutting_plane(config_matrix, mu=0.1, beta=0.5, q_min=0.01, q_max=1.0, max_iter=30, verbose=False):
#     """
#     Faithful reproduction of Xiao et al. (2022) with Multi-Cut Strategy.
    
#     This strictly follows Eq. (10) but adds cuts for ALL violating eigenvalues
#     at each step to prevent the solver from oscillating.
#     """
#     # 1. Geometry & Setup
#     # Ensure (d, n) shape
#     if config_matrix.shape[0] > config_matrix.shape[1]:
#         config_matrix = config_matrix.T
#     d, n = config_matrix.shape
    
#     # We do NOT normalize the configuration here to stay faithful to the input.
#     # Users should ensure config_matrix is not huge (e.g. coordinates ~ 1.0).
    
#     # Gale Matrix & Null Space Basis
#     P_bar = np.vstack((config_matrix, np.ones((1, n))))
#     Z = la.null_space(P_bar)
#     if Z.shape[1] == 0: raise ValueError("n must be >= d+2")
#     U, _, _ = la.svd(Z)
#     U1 = U[:, :n - d - 1]

#     # 2. Variables (Standard MISDP)
#     w, x, y = {}, {}, {}
#     lam_var = cp.Variable(nonneg=True, name="lambda")
    
#     def get_idx(i, j): return tuple(sorted((i, j)))

#     for i in range(n):
#         for j in range(i + 1, n):
#             idx = get_idx(i, j)
#             w[idx] = cp.Variable()
#             x[idx] = cp.Variable(integer=True)
#             y[idx] = cp.Variable(integer=True)

#     # 3. Static Constraints (Eq. 10 & 8)
#     constraints = []
#     Omega_expr = [[0]*n for _ in range(n)]
    
#     for i in range(n):
#         row_sum_expr = 0
#         for j in range(i + 1, n):
#             idx = get_idx(i, j)
#             # Binary Logic
#             constraints += [x[idx] + y[idx] <= 1, x[idx] >= 0, y[idx] >= 0]
            
#             # Weight Bounds [cite: 198-201]
#             constraints += [w[idx] >= -q_max * x[idx]]
#             constraints += [w[idx] <= q_max * (1 - x[idx]) - q_min * x[idx]]
#             constraints += [w[idx] >= -q_max * (1 - y[idx]) + q_min * y[idx]]
#             constraints += [w[idx] <= q_max * y[idx]]
            
#             # Omega Construction
#             Omega_expr[i][j] = -w[idx]
#             Omega_expr[j][i] = -w[idx]
#             row_sum_expr += w[idx]
        
#         # Diagonal Constraints
#         diag_sum = 0
#         for k in range(n):
#             if i != k: diag_sum += w[get_idx(i, k)]
#         Omega_expr[i][i] = diag_sum
#         constraints.append(Omega_expr[i][i] == diag_sum)

#     # Equilibrium Constraint
#     Omega_cp = cp.bmat(Omega_expr)
#     constraints.append(Omega_cp @ config_matrix.T == 0)

#     # 4. Objective Function (Eq. 10)
#     comm_cost = 0
#     total_dist = 0
#     for i in range(n):
#         for j in range(i + 1, n):
#             dist = np.linalg.norm(config_matrix[:, i] - config_matrix[:, j])
#             total_dist += dist
#             idx = get_idx(i, j)
#             comm_cost += dist * (x[idx] + y[idx])
            
#     # Normalization factors from paper
#     alpha1 = 1.0 / n
#     alpha2 = 1.0 / total_dist if total_dist > 0 else 1.0
    
#     # Strictly Eq. (10): Minimize alpha1 * beta * lambda + alpha2 * (1-beta) * cost
#     objective = cp.Minimize(alpha1 * beta * lam_var + alpha2 * (1 - beta) * comm_cost)

#     # 5. Cutting Plane Loop
#     dynamic_cuts = []
    
#     # Slightly relaxed gap (5%) for speed, but strict feasibility
#     mosek_params = {
#         "MSK_DPAR_MIO_TOL_REL_GAP": 0.05,
#         "MSK_DPAR_OPTIMIZER_MAX_TIME": 15.0,
#     }

#     if verbose: print(f"--- Faithful MISDP (N={n}, mu={mu}) ---")

#     for iter_idx in range(max_iter):
#         prob = cp.Problem(objective, constraints + dynamic_cuts)
        
#         try:
#             prob.solve(solver=cp.MOSEK, mosek_params=mosek_params, verbose=False)
#         except cp.error.SolverError:
#             print("Solver failed step. Returning current best.")
#             return Omega_cp.value

#         # Check Spectral Properties
#         omega_val = Omega_cp.value
#         omega_r = U1.T @ omega_val @ U1
#         eigvals, eigvecs = la.eigh(omega_r) # Ascending order
        
#         # --- MULTI-CUT GENERATION ---
#         # Instead of picking just the worst violation, we cut ALL violations.
#         # This prevents the solver from just "hopping" to the next unstable mode.
        
#         cuts_added = 0
        
#         # 1. Stability Cuts (Lower Bound)
#         # Cut every eigenvalue < mu
#         for k in range(len(eigvals)):
#             val = eigvals[k]
#             if val < mu - 1e-4: # Tolerance 1e-4
#                 v = eigvecs[:, k]
#                 # v^T U1^T Omega U1 v >= mu
#                 cut = (v.T @ U1.T) @ Omega_cp @ (U1 @ v) >= mu
#                 dynamic_cuts.append(cut)
#                 cuts_added += 1
#             else:
#                 # Since sorted, all subsequent eigenvalues are valid
#                 break
        
#         # 2. Condition Number Cuts (Upper Bound)
#         # Cut every eigenvalue > lambda * mu
#         # We check from the largest backwards
#         current_lambda = lam_var.value
#         if current_lambda is None: current_lambda = 100.0 # Safety fallback
        
#         for k in range(len(eigvals)-1, -1, -1):
#             val = eigvals[k]
#             # Use current_lambda for the check, but variable lam_var for the cut
#             if val > (current_lambda * mu) + 1e-4:
#                 v = eigvecs[:, k]
#                 # v^T U1^T Omega U1 v <= lam_var * mu
#                 cut = (v.T @ U1.T) @ Omega_cp @ (U1 @ v) <= lam_var * mu
#                 dynamic_cuts.append(cut)
#                 cuts_added += 1
#             else:
#                 break
                
#         if verbose:
#             min_eig, max_eig = eigvals[0], eigvals[-1]
#             print(f"Iter {iter_idx}: MinEig={min_eig:.3f}, MaxEig={max_eig:.3f}, Cuts={cuts_added}")
            
#         if cuts_added == 0:
#             print("Converged!")
#             return omega_val

#     print("Warning: Max iterations reached.")
#     return Omega_cp.value

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
    # 1. Geometry & Setup
    if config_matrix.shape[0] > config_matrix.shape[1]:
        config_matrix = config_matrix.T
    d, n = config_matrix.shape
    
    # Gale Matrix & Null Space Basis
    P_bar = np.vstack((config_matrix, np.ones((1, n))))
    Z = la.null_space(P_bar)
    if Z.shape[1] == 0: raise ValueError("n must be >= d+2")
    U, _, _ = la.svd(Z)
    U1 = U[:, :n - d - 1]

    # 2. Variables (Standard MISDP)
    w, x, y = {}, {}, {}
    lam_var = cp.Variable(nonneg=True, name="lambda")
    
    def get_idx(i, j): return tuple(sorted((i, j)))

    for i in range(n):
        for j in range(i + 1, n):
            idx = get_idx(i, j)
            w[idx] = cp.Variable()
            x[idx] = cp.Variable(integer=True)
            y[idx] = cp.Variable(integer=True)

    # 3. Static Constraints
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

    # 4. Objective Function
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

    # 5. Cutting Plane Loop
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