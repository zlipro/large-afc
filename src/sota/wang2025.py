"""
Wang, Yang, Xiao, Fang, Chen, "Accelerated Optimized Topology Design in Affine
Formation Control Using ADMM", IEEE TNSE 12(4), 2025.

Implements the paper's improved ADMM with closed-form sub-problem solutions:

  * Algorithm 2 (problem P1 / eq. (17)): prescribed minimum non-zero eigenvalue mu,
        min_{w, lmax}  lmax
        s.t.  Omega <= lmax I,   Omega + alpha Pbar Pbar^T / n >= mu I,
              lmax <= pi / (2 tau*),   ||C_w w||_1 <= r,   C_E w = 0,   1^T w = w_t
  * Algorithm 3 (problem P2 / eq. (18)): unprescribed minimum eigenvalue,
        w = w+ - w-  (w+, w- >= 0), variables (w+, w-, lmin, lmax),
        min  lmax - c_lambda * lmin
        s.t.  Omega <= lmax I,   Omega + alpha Pbar Pbar^T / n >= lmin I,   lmin >= 0, ...

Following the paper, the l1 constraint is handled through the auxiliary variable
z = C_w w, updated by soft thresholding (Alg. 2) or the non-negative proximity
operator (Alg. 3) with a fixed sparsity factor kappa (the paper uses kappa = 0.5).
The PSD constraints are split with slacks S1, S2 and updated by eigenvalue
projection (Proposition 2). The x-update (SP1) is an equality-constrained QP whose
Hessian is diagonal in the edge weights once the diagonal of the slack matrices is
kept as separate variables (s1, s2); it is solved in closed form through a small
reduced KKT system that is factorised once before the iterations (Proposition 1).

Parameters not reported in the paper (w_t, mu, c_min/c_max, tau*) are exposed as
arguments; defaults are documented on wang2025_admm.
"""
import time
import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp


def _equilibrium_rows(Pbar_T, N, M, iu, ju):
    """Independent rows of C_E (full row rank) with C_E w = 0 <=> Omega Pbar = 0."""
    # (Omega Pbar)_{i,:} = sum_{k=(i,j)} w_k (pbar_i - pbar_j)
    D1 = Pbar_T.shape[0]
    rows, cols, vals = [], [], []
    for d in range(D1):
        diff = Pbar_T[d, iu] - Pbar_T[d, ju]
        rows += [iu * D1 + d, ju * D1 + d]
        cols += [np.arange(M), np.arange(M)]
        vals += [diff, -diff]
    E = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(N * D1, M)).toarray()
    rank = np.linalg.matrix_rank(E)
    _, _, piv = sla.qr(E.T, pivoting=True, mode="economic")
    return sp.csr_matrix(E[np.sort(piv[:rank])])


def _proj_psd(X):
    ev, V = np.linalg.eigh((X + X.T) / 2)
    return (V * np.maximum(ev, 0)) @ V.T


def _proj_nsd(X):
    ev, V = np.linalg.eigh((X + X.T) / 2)
    return (V * np.minimum(ev, 0)) @ V.T


def wang2025_admm(nominal_config, prescribed=True, mu=1.0, w_t=None, alpha=10.0, kappa=0.5,
                  rho=1.0, eps_pri=1e-3, eps_dual=1e-3, max_iter=20000, c_min=1.0, c_max=2.0,
                  tau_star=None, c_lambda=1.0, sparse_output=True, return_info=False, verbose=False,
                  max_time=None, callback=None):
    """
    Stress matrix design of Wang et al. (2025).

    Args:
        nominal_config: (D, N) target configuration.
        prescribed:   True -> Algorithm 2 (P1, minimum non-zero eigenvalue >= mu);
                      False -> Algorithm 3 (P2, minimum eigenvalue is optimised).
        mu:           prescribed lower bound on the minimum non-zero eigenvalue (Alg. 2 only).
                      Not reported in the paper; default 1.0.
        w_t:          total edge weight 1^T w. Not reported in the paper; default N, which
                      reproduces trace(Omega) = 24 of the paper's 12-agent example.
        alpha:        scaling of Pbar Pbar^T / n in the shifted LMI (paper: 10).
        kappa:        sparsity factor of the z-update (paper: 0.5).
        rho:          ADMM penalty (paper: 1).
        eps_pri, eps_dual: stopping tolerances on the primal / dual gaps (paper: 1e-3).
        c_min, c_max: range of the distance-based communication costs c_k (not reported;
                      default 1 and 2).
        tau_star:     time-delay bound giving lmax <= pi / (2 tau*). Default None, which uses
                      the redundant bound lmax <= trace(Omega) = 2 w_t.
        c_lambda:     weight of lmin in the Alg. 3 objective lmax - c_lambda * lmin (paper: 1).
        sparse_output: zero the weights whose auxiliary variable z is exactly zero (the
                      topology given by the l1 step); otherwise return the raw x iterate.
        return_info:  also return a dict with iterations, gaps and timing.
        max_time:     optional wall-clock limit in seconds for the iterations.
        callback:     optional f(it, primal_gap, dual_gap, w, z, l) called every iteration
                      (e.g. to record accuracy checkpoints); return True to stop.

    Returns:
        stress_mat (N, N) [, info]
    """
    t_start = time.perf_counter()
    P = np.asarray(nominal_config, dtype=float)
    D, N = P.shape
    iu, ju = np.triu_indices(N, k=1)
    M = iu.size
    w_t = float(N) if w_t is None else float(w_t)
    # lmax <= pi/(2 tau*). Without a delay bound use trace(Omega) = 2 w_t, which is redundant
    # for a PSD Omega but keeps the slack block well scaled (a huge cap stalls the ADMM).
    lmax_cap = 2.0 * w_t if tau_star is None else np.pi / (2.0 * tau_star)

    # --- problem data ---
    Pbar = np.hstack([P.T, np.ones((N, 1))])                    # n x (d+1), paper notation
    C_E = _equilibrium_rows(Pbar.T, N, M, iu, ju)
    dist = np.linalg.norm(P[:, iu] - P[:, ju], axis=0)
    span = dist.max() - dist.min()
    c = (c_max - c_min) * (dist - dist.min()) / (span if span > 0 else 1.0) + c_min

    shift = alpha * Pbar @ Pbar.T / N                           # alpha Pbar Pbar^T / n
    # B01: constant of the "PSD" constraint B1(x) + B01 >= 0
    if prescribed:
        B01 = shift - mu * np.eye(N)                            # n x n
    else:
        B01 = np.zeros((N + 1, N + 1)); B01[:N, :N] = shift     # blkdiag(shift, 0)
    n1 = B01.shape[0]
    n2 = N + 1
    b02 = np.zeros(n2); b02[N] = -lmax_cap                      # B02 = blkdiag(0, -lmax_cap)

    # unsigned incidence (diagonal of Omega = Bu @ w)
    Bu = sp.csr_matrix((np.ones(2 * M), (np.concatenate([iu, ju]), np.tile(np.arange(M), 2))),
                       shape=(N, M))

    # --- variables: edge block v (M x p), w = v @ g; scalars l; slacks s1 (n1), s2 (n2) ---
    p = 1 if prescribed else 2
    g = np.array([1.0]) if prescribed else np.array([1.0, -1.0])
    nl = 1 if prescribed else 2                                 # l = [lmax] or [lmin, lmax]
    i_lmax = nl - 1

    # Hessian of the edge block per edge: rho (c_k^2 I_p + 4 g g^T)
    Hk = rho * (c[:, None, None] ** 2 * np.eye(p)[None] + 4.0 * np.outer(g, g)[None])
    Hk_inv = np.linalg.inv(Hk)                                  # M x p x p
    h = np.einsum('i,kij,j->k', g, Hk_inv, g)                   # g^T Hk^-1 g

    # Linear constraints  F_w w + F_u u = rhs,  u = (l, s1, s2)
    r_E = C_E.shape[0]
    F_w = sp.vstack([C_E, sp.csr_matrix(np.ones((1, M))),
                     sp.vstack([Bu, sp.csr_matrix((n1 - N, M))]),
                     sp.vstack([Bu, sp.csr_matrix((1, M))])]).tocsr()
    n_rows = F_w.shape[0]
    nu_dim = nl + n1 + n2
    F_u = np.zeros((n_rows, nu_dim))
    o1, o2 = r_E + 1, r_E + 1 + n1                              # row offsets of diag1, diag2
    F_u[o1:o1 + n1, nl:nl + n1] = np.eye(n1)
    F_u[o2:o2 + n2, nl + n1:] = np.eye(n2)
    F_u[o2:o2 + N, i_lmax] = -1.0                               # Omega - lmax I
    F_u[o2 + N, i_lmax] = 1.0                                   # lmax
    if not prescribed:
        F_u[o1:o1 + N, 0] = -1.0                                # Omega - lmin I
        F_u[o1 + N, 0] = 1.0                                    # lmin
    rhs = np.zeros(n_rows)
    rhs[r_E] = w_t
    rhs[o1:o1 + n1] = -np.diag(B01)
    rhs[o2:o2 + n2] = -b02

    H_u = np.zeros((nu_dim, nu_dim))
    H_u[nl:, nl:] = rho * np.eye(n1 + n2)
    c_l = np.zeros(nl); c_l[i_lmax] = 1.0
    if not prescribed:
        c_l[0] = -c_lambda

    S_mat = (F_w.multiply(h[None, :]) @ F_w.T).toarray()        # F_w diag(h) F_w^T
    KKT = np.block([[H_u, F_u.T], [F_u, -S_mat]])
    lu = sla.lu_factor(KKT)
    t_setup = time.perf_counter() - t_start

    # --- ADMM state ---
    z = np.zeros((M, p)); mu_d = np.zeros((M, p))
    S1b = np.zeros((n1, n1)); L1 = np.zeros((n1, n1))
    S2b = np.zeros((n2, n2)); L2 = np.zeros((n2, n2))
    nz = np.zeros(M, dtype=bool)
    it, r_gap, s_gap = 0, np.inf, np.inf

    for it in range(1, max_iter + 1):
        # ---------- SP1: x-update in closed form ----------
        T1 = S1b - L1 / rho
        T2 = S2b - L2 / rho
        a0 = z - mu_d / rho                                     # target of C_w v
        a1 = B01[iu, ju] + 0.5 * (T1[iu, ju] + T1[ju, iu])      # target of w from S1 off-diag
        a2 = 0.5 * (T2[iu, ju] + T2[ju, iu])                    # target of w from S2 off-diag
        g_v = -rho * (c[:, None] * a0 + 2.0 * (a1 + a2)[:, None] * g[None, :])
        g_u = np.concatenate([c_l, -rho * np.diag(T1), -rho * np.diag(T2)])

        q = np.einsum('i,kij,kj->k', g, Hk_inv, g_v)            # g^T Hk^-1 g_v
        sol = sla.lu_solve(lu, np.concatenate([-g_u, rhs + F_w @ q]))
        u, nu = sol[:nu_dim], sol[nu_dim:]
        v = -np.einsum('kij,kj->ki', Hk_inv, g_v + (F_w.T @ nu)[:, None] * g[None, :])
        w = v @ g
        lvar = u[:nl]

        Om = np.zeros((N, N)); Om[iu, ju] = -w; Om[ju, iu] = -w
        Om[np.arange(N), np.arange(N)] = Bu @ w
        if prescribed:
            B1x = Om
        else:
            B1x = np.zeros((n1, n1)); B1x[:N, :N] = Om - lvar[0] * np.eye(N); B1x[N, N] = lvar[0]
        B2x = np.zeros((n2, n2)); B2x[:N, :N] = Om - lvar[i_lmax] * np.eye(N); B2x[N, N] = lvar[i_lmax]
        S1 = -(B1x + B01)                                       # S1 <= 0  <=> B1 + B01 >= 0
        S2 = -(B2x + np.diag(b02))                              # S2 >= 0  <=> B2 + B02 <= 0

        # ---------- SP2: z, S1bar, S2bar ----------
        y = c[:, None] * v + mu_d / rho
        z_old, S1b_old, S2b_old = z, S1b, S2b
        if prescribed:
            z = np.sign(y) * np.maximum(np.abs(y) - kappa, 0.0)     # soft thresholding (29)
        else:
            z = np.maximum(y - kappa, 0.0)                          # prox on z >= 0 (34)
        S1b = _proj_nsd(S1 + L1 / rho)
        S2b = _proj_psd(S2 + L2 / rho)

        # ---------- dual update (22) ----------
        res_z = c[:, None] * v - z
        mu_d = mu_d + rho * res_z
        L1 = L1 + rho * (S1 - S1b)
        L2 = L2 + rho * (S2 - S2b)

        r_gap = max(np.linalg.norm(res_z), np.linalg.norm(S1 - S1b), np.linalg.norm(S2 - S2b))
        s_gap = rho * max(np.linalg.norm(z - z_old), np.linalg.norm(S1b - S1b_old),
                          np.linalg.norm(S2b - S2b_old))
        if verbose and it % 500 == 0:
            print(f"[wang2025] it {it}: primal gap {r_gap:.2e}, dual gap {s_gap:.2e}")
        if r_gap <= eps_pri and s_gap <= eps_dual:
            break
        if callback is not None and callback(it, r_gap, s_gap, w, z, lvar):
            break
        if max_time is not None and time.perf_counter() - t_start > max_time:
            break

    nz = np.any(z != 0, axis=1)
    w_out = np.where(nz, w, 0.0) if sparse_output else w
    stress_mat = np.zeros((N, N))
    stress_mat[iu, ju] = -w_out; stress_mat[ju, iu] = -w_out
    stress_mat[np.arange(N), np.arange(N)] = Bu @ w_out

    if not return_info:
        return stress_mat
    return stress_mat, dict(iterations=it, converged=bool(r_gap <= eps_pri and s_gap <= eps_dual),
                            primal_gap=r_gap, dual_gap=s_gap, lambdas=lvar,
                            support=int(nz.sum()), t_setup=t_setup,
                            t_total=time.perf_counter() - t_start, w_raw=w)
