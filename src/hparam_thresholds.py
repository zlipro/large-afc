"""
Critical values of the hyperparameter alpha for the stress design problem P2 (7a):

    min_w  ||w||_1 - alpha * tr(Omega(w))
    s.t.   Q^T Omega(w) Q >= gamma I,   Omega(w) <= beta I,   E w = 0,

with Omega(w) = B diag(w) B^T (b_e = e_i - e_j, so Omega_ij = -w_e) and tr(Omega(w)) = psi^T w on the
equilibrium subspace (psi_e = ||Q^T b_e||^2). For 0 < gamma < beta:

    alpha < alpha_c   ->  lambda_{D+2}(Omega*) = gamma        (lower bound active, scale set by gamma)
    alpha > alpha_c   ->  lambda_max(Omega*)   = beta         (upper bound active)
    1/2 <= alpha_c <= alpha_bar = ||w(QQ^T)||_1 / (N - D - 1)  (calculation-free bracket)
    alpha >= alpha_sat ->  Omega* = beta QQ^T                 (complete graph, condition number 1)

    alpha_c   = 1 / tau(P),  tau(P) = max { tr Omega(w) : E w = 0, Omega(w) >= 0, ||w||_1 <= 1 }
    alpha_sat = min alpha  s.t. KKT of (7a) hold at Omega = beta QQ^T:
                A*(Z) + E^T y = alpha psi - s            on supp(w(QQ^T))
                |alpha psi - A*(Z) - E^T y| <= 1         off supp(w(QQ^T))
                Z >= 0,  Z U = 0                         (A*(Z)_e = b_e^T Z b_e)

All thresholds depend only on the configuration P (not on beta or gamma).

Numerical notes
  * Omega(w) >= 0 with E w = 0 is imposed as Omega(w) + U U^T >= 0 (U orthonormal basis of
    range(Pbar^T)): equivalent, but strictly feasible (no 3-dim face), which keeps the IPM well-posed.
  * Z >= 0, Z U = 0 is imposed through Zt = Z + U U^T >= 0, Zt U = U, for the same reason.
  * psi is replaced by 2*1 in the alpha_c SDP and kept exact in the alpha_sat KKT system; the two differ
    by an element of range(E^T), which y absorbs.
"""
import time
import numpy as np
import cvxpy as cp

from src.stress import _stress_map, _equilibrium_rows


def _geometry(nominal_config):
    """Pbar, Q (N x N-D-1, orthonormal basis of null(Pbar)), U (N x D+1), K, E, psi, w(QQ^T)."""
    D, N = nominal_config.shape
    Pbar = np.vstack([nominal_config, np.ones((1, N))])
    Uf = np.linalg.svd(Pbar.T)[0]
    U, Q = Uf[:, :D + 1], Uf[:, D + 1:]
    K = _stress_map(N)
    E = _equilibrium_rows(Pbar, K)
    QQ = Q @ Q.T
    i, j = np.triu_indices(N, k=1)
    psi = QQ[i, i] + QQ[j, j] - 2 * QQ[i, j]               # = A*(QQ^T)
    w_QQ = -QQ[i, j]                                        # stress vector of QQ^T
    return dict(D=D, N=N, Pbar=Pbar, Q=Q, U=U, K=K, E=E, psi=psi, w_QQ=w_QQ)


def alpha_bar(nominal_config, geo=None):
    """Calculation-free upper bound on alpha_c: ||w(QQ^T)||_1 / (N - D - 1)."""
    g = geo or _geometry(nominal_config)
    return float(np.abs(g['w_QQ']).sum() / (g['N'] - g['D'] - 1))


def alpha_c(nominal_config, geo=None, solver=cp.MOSEK, return_info=False, **solve_kwargs):
    """alpha_c = 1 / tau(P), tau(P) = max trace per unit l1-norm over PSD equilibrium stresses."""
    g = geo or _geometry(nominal_config)
    N, K, E, U = g['N'], g['K'], g['E'], g['U']
    t0 = time.perf_counter()
    w = cp.Variable(K.shape[1])
    Omega = cp.reshape(K @ w, (N, N), order="C")
    prob = cp.Problem(cp.Maximize(2.0 * cp.sum(w)),
                      [E @ w == 0, Omega + U @ U.T >> 0, cp.norm(w, 1) <= 1])
    prob.solve(solver=solver, **solve_kwargs)
    tau = float(prob.value)
    if not return_info:
        return 1.0 / tau
    h = np.asarray(w.value).ravel()
    return 1.0 / tau, dict(tau=tau, h=h, status=prob.status, time=time.perf_counter() - t0)


def alpha_sat(nominal_config, geo=None, solver=cp.MOSEK, support_tol=1e-10, return_info=False,
              **solve_kwargs):
    """Smallest alpha for which Omega = beta QQ^T satisfies the KKT conditions of (7a)."""
    g = geo or _geometry(nominal_config)
    N, K, E, U, psi, w_QQ = g['N'], g['K'], g['E'], g['U'], g['psi'], g['w_QQ']
    t0 = time.perf_counter()
    on = np.abs(w_QQ) > support_tol * np.abs(w_QQ).max()
    s = np.sign(w_QQ)
    KT = K.T.tocsr()
    a_UU = KT @ (U @ U.T).ravel()                           # A*(U U^T), removed from A*(Zt)

    a = cp.Variable()
    Zt = cp.Variable((N, N), PSD=True)                      # Zt = Z + U U^T
    y = cp.Variable(E.shape[0])
    r = a * psi - (KT @ cp.vec(Zt, order="C") - a_UU) - E.T @ y   # = alpha psi - A*(Z) - E^T y
    cons = [Zt @ U == U]
    if on.any():
        cons.append(r[np.flatnonzero(on)] == s[on])
    if (~on).any():
        cons.append(cp.abs(r[np.flatnonzero(~on)]) <= 1)
    prob = cp.Problem(cp.Minimize(a), cons)
    prob.solve(solver=solver, **solve_kwargs)
    if not return_info:
        return float(a.value)
    Z = Zt.value - U @ U.T
    return float(a.value), dict(Z=Z, y=np.asarray(y.value).ravel(), status=prob.status,
                                n_zero_edges=int((~on).sum()), time=time.perf_counter() - t0)


def compute_thresholds(nominal_config, solver=cp.MOSEK, verbose=False):
    """alpha_c (exact), alpha_bar (bound), alpha_sat (exact) and the 1/2 lower bound."""
    g = _geometry(nominal_config)
    ab = alpha_bar(nominal_config, g)
    ac, ic = alpha_c(nominal_config, g, solver=solver, return_info=True)
    asat, isat = alpha_sat(nominal_config, g, solver=solver, return_info=True)
    out = dict(alpha_lower=0.5, alpha_c=ac, alpha_bar=ab, alpha_sat=asat, tau=ic['tau'],
               status_c=ic['status'], status_sat=isat['status'],
               time_c=ic['time'], time_sat=isat['time'])
    if verbose:
        print(f"N={g['N']} D={g['D']}:  1/2 <= alpha_c = {ac:.6f} <= alpha_bar = {ab:.6f};"
              f"  alpha_sat = {asat:.6f}   ({ic['status']}/{isat['status']}, "
              f"{ic['time']:.1f}s + {isat['time']:.1f}s)")
    return out


if __name__ == "__main__":
    import sys
    import src.utils.geometry as geometry
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    np.random.seed(seed)
    P = geometry.random_config(N, playground_size=5.0)
    compute_thresholds(P, verbose=True)
