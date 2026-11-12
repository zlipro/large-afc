# Memory-efficient implementation of P2 / P3: theory, code and tests

**Audience:** a coding assistant (or developer) updating the stress-matrix design code of the paper
*Multicluster design and control of large-scale affine formations* (Li, Leus, Rajan).
**Goal:** make the generic formulation P2 solvable at N = 100 (it currently fails out of memory with 64 GB)
without changing its solution, and apply the same structure to P3 (USI) and to each cluster problem.

Read the whole document before editing code. Section 6 lists the tests that must pass.

---

## 1. Summary

The optimization variables are tiny (N = 100 gives M = 4,950 edge weights, about 40 KB). Memory runs out
because of how the problem is *built* for the solver:

1. Expressions such as `Psi @ cp.diag(w) @ Psi.T` or `B @ cp.diag(w) @ B.T` make CVXPY build huge
   intermediate coefficient tensors. Measured peak memory: 0.14 GB (N = 20), 0.34 GB (N = 30),
   1.06 GB (N = 40), out of memory on a 7 GB machine at N = 60.
2. The projected constraint QᵀΩQ ⪰ γI is dense: every weight affects all (N−D−1)² entries.
3. `cp.sigma_max(Omega) <= beta` is turned into a 2N × 2N LMI.

The fix replaces each constraint by an **exactly equivalent** one whose coefficients are sparse in the
weights. With the fix, N = 100 solves with the open-source interior-point solver Clarabel in 3.7 GB.

| Original (paper) | Replacement | Equivalent because |
|---|---|---|
| Ω = B̄ diag(ω̄) B̄ᵀ via `cp.diag` | vec(Ω) = K ω̄, K sparse (4 nonzeros per column) | same linear map |
| QᵀΩQ − γI ⪰ 0 (7d)/(10b) | Ω + γUUᵀ ⪰ γI, U orthonormal basis of range(P̄ᵀ) | Section 3.2 |
| ‖Ω‖₂ ≤ β (7c)/(10c) | Ω ⪯ βI | Ω ⪰ 0 already (Section 3.3) |
| −αψᵀω̄ | −2α·1ᵀω̄ | ψᵀω̄ = 2·1ᵀω̄ = tr(Ω) on the feasible set (Section 3.4) |
| E ω̄ = 0 (10d) | (P̄ ⊗ I_N) K ω̄ = 0, redundant rows removed | same constraint (Section 3.5) |

---

## 2. Notation (as in the paper)

- P ∈ ℝ^{D×N}: target configuration. P̄ = [P; 1ᵀ] ∈ ℝ^{(D+1)×N}, assumed full row rank.
- Complete graph with M = N(N−1)/2 edges e = (i, j), i < j. B̄ ∈ ℝ^{N×M}: incidence matrix,
  column e = e_i − e_j.
- ω̄ ∈ ℝ^M: edge weights (stresses). Ω = B̄ diag(ω̄) B̄ᵀ.
- Q ∈ ℝ^{N×(N−D−1)}: orthonormal basis of the kernel of P̄ (columns ⟂ rows of P̄).
- U ∈ ℝ^{N×(D+1)}: orthonormal basis of range(P̄ᵀ). Note QQᵀ + UUᵀ = I_N.
- Ψ = QᵀB̄, ψ = diag(ΨᵀΨ), i.e. ψ_e = ‖Qᵀb_e‖².
- P2: minimize ‖ω̄‖₁ − αψᵀω̄ subject to Ψ diag(ω̄)Ψᵀ ⪰ γI, ‖B̄ diag(ω̄)B̄ᵀ‖₂ ≤ β, Eω̄ = 0.

---

## 3. Theory: why each replacement is exact

### 3.1 Sparse linear map for Ω

For edge e = (i, j), B̄ diag(ω̄) B̄ᵀ adds ω_e at (i,i) and (j,j) and −ω_e at (i,j) and (j,i).
So vec(Ω) = K ω̄, where K ∈ ℝ^{N²×M} has exactly four nonzeros per column. Building Ω through K
avoids every `cp.diag(w)` product.

### 3.2 Projected LMI → sparse shifted LMI (no tuning)

**Claim.** If ΩP̄ᵀ = 0, then for any γ:

  QᵀΩQ ⪰ γI_{N−D−1}  ⟺  Ω + γUUᵀ ⪰ γI_N.

**Proof.** Write Ω + γUUᵀ − γI in the orthonormal basis [Q U]. Since ΩU = 0 (U spans range(P̄ᵀ)
and ΩP̄ᵀ = 0), the matrix is block diagonal:

- the Q-block is QᵀΩQ − γI;
- the U-block is UᵀΩU + γI − γI = 0.

So Ω + γUUᵀ − γI ⪰ 0 exactly when QᵀΩQ − γI ⪰ 0. ∎

**Important:** U must be **orthonormal** and the shift must be **exactly γ**. Using the non-orthonormal
P̄ with an arbitrary scale α (as in Wang et al., TNSE 2025) is only a sufficient condition and can cap
the eigenvalue being optimized. With orthonormal U and shift γ there is nothing to tune.

The new LMI is N × N, but each weight touches only four entries of Ω (UUᵀ is a constant), so its
coefficients are sparse. The old (N−D−1)² × M coefficient block was dense.

### 3.3 Spectral norm → one-sided LMI

The LMI in 3.2 with γ > 0 implies Ω ⪰ 0. For a PSD matrix, ‖Ω‖₂ = λ_max(Ω), so

  ‖Ω‖₂ ≤ β  ⟺  Ω ⪯ βI_N.

This is an N × N LMI instead of the 2N × 2N LMI that `cp.sigma_max` creates.

### 3.4 Trace term: ψᵀω̄ = 2·1ᵀω̄ on the feasible set

- tr(Ω) = Σ_e ω_e ‖b_e‖² = 2 Σ_e ω_e = 2·1ᵀω̄, for any ω̄.
- ψᵀω̄ = Σ_e ω_e ‖Qᵀb_e‖² = tr(QᵀΩQ) = tr(Ω) − tr(UᵀΩU).
- On the feasible set ΩU = 0, so tr(UᵀΩU) = 0 and ψᵀω̄ = tr(Ω) = 2·1ᵀω̄.

The two objectives take identical values on the feasible set, so they have the same optimal
solutions. Replacing ψ by the constant vector 2·1 removes the dense precomputation of ψ.

*(Side note for the paper, not the code: because ψ can be replaced by 2·1 without changing the
problem, the critical value α* = 1/‖ψ‖∞ is not intrinsic. Treat it as a heuristic.)*

### 3.5 Equilibrium constraint

ΩP̄ᵀ = 0 ⟺ P̄Ω = 0 (Ω is symmetric). With row-major vectorization, vec(P̄ Ω) = (P̄ ⊗ I_N) vec(Ω),
so the constraint is (P̄ ⊗ I_N) K ω̄ = 0, which is sparse. It has (D+1)N rows but rank DN − D(D+1)/2:
the 1ᵀ row is automatic (rows of Ω always sum to zero) and the remaining rows have symmetric
duplicates. Removing dependent rows is optional; in our tests it did not change the solution or the
solver status.

### 3.6 Feasibility and solver guarantees (useful for the paper)

For any 0 < γ < β, Ω = cQQᵀ with γ < c < β is a complete-graph equilibrium stress (rows sum to zero,
ΩP̄ᵀ = 0) that satisfies both LMIs strictly. So P2 is feasible if and only if γ ≤ β, Slater's
condition holds for γ < β, and strong duality holds. ‖Ω‖₂ ≤ β bounds Ω, and on the complete graph ω̄ is
read off the off-diagonal entries of Ω, so the feasible set is compact and the optimum is attained.
Standard primal-dual interior-point guarantees therefore apply.

---

## 4. Extension to P3 (USI) and to clusters

### 4.1 P3: one variable per symmetry class

Let the M edges be partitioned into S classes (the **exact symmetry orbits**, not equal-distance
classes; see Section 6, test T6). Let G ∈ {0,1}^{M×S} be the class indicator matrix (G_{es} = 1 if
edge e is in class s) and c = Gᵀ1 the class sizes. Substitute ω̄ = Gθ, θ ∈ ℝ^S:

- vec(Ω) = (KG) θ;
- ‖ω̄‖₁ = Σ_s c_s |θ_s| (weighted ℓ1);
- tr(Ω) = 2·1ᵀGθ = 2 cᵀθ;
- the constraints of Section 3 are unchanged, with K replaced by KG.

Setting G = I recovers P2, so one function covers both (Section 5).

For P3 specifically, a second form is also fine because S is small: precompute
A_s = QᵀΩ_sQ (dense (N−D−1)² matrices, one per class, where Ω_s is the stress of class s with unit
weight) and impose Σ_s θ_s A_s ⪰ γI. Both forms gave identical optima in testing.

### 4.2 Multicluster

Apply the same function to each cluster with its own local configuration P_c (D × N_c). Nothing else
changes.

---

## 5. Reference implementation

Tested with Python 3, NumPy, SciPy and CVXPY 1.9 with Clarabel. With MOSEK, pass `solver="MOSEK"`.

```python
import itertools
import numpy as np
import scipy.sparse as sp
import scipy.linalg as sla
import cvxpy as cp


def edges_complete(N):
    """Edge list of the complete graph, order (0,1),(0,2),...,(N-2,N-1)."""
    return list(itertools.combinations(range(N), 2))


def build_K(N, edges):
    """Sparse K (N^2 x M) with vec_rowmajor(Omega) = K @ w, Omega = B diag(w) B^T."""
    rows, cols, vals = [], [], []
    for k, (i, j) in enumerate(edges):
        for r, c, v in ((i, i, 1.0), (j, j, 1.0), (i, j, -1.0), (j, i, -1.0)):
            rows.append(r * N + c); cols.append(k); vals.append(v)
    return sp.csr_matrix((vals, (rows, cols)), shape=(N * N, len(edges)))


def build_equilibrium(Pbar, K, reduce_rows=True):
    """Sparse E_eq with E_eq @ w = 0  <=>  Pbar @ Omega = 0  (i.e. Omega Pbar^T = 0)."""
    N = Pbar.shape[1]
    E = (sp.kron(sp.csr_matrix(Pbar), sp.eye(N)) @ K).tocsr()   # vec_r(Pbar Omega)
    if not reduce_rows:
        return E
    Ed = E.toarray()
    r = np.linalg.matrix_rank(Ed)
    _, _, piv = sla.qr(Ed.T, pivoting=True, mode="economic")
    return sp.csr_matrix(Ed[np.sort(piv[:r])])


def solve_p2_sparse(P, alpha, gamma, beta, classes=None, solver="CLARABEL", **solver_kw):
    """
    P       : D x N target configuration.
    classes : optional length-M integer array, class label per edge (P3 / USI),
              edges ordered as edges_complete(N). None -> generic P2.
    """
    D, N = P.shape
    Pbar = np.vstack([P, np.ones((1, N))])                 # (D+1) x N
    edges = edges_complete(N)
    M = len(edges)
    K = build_K(N, edges)
    U = np.linalg.qr(Pbar.T)[0]                            # N x (D+1), orthonormal

    if classes is None:
        G = sp.eye(M, format="csr")
    else:
        classes = np.asarray(classes)
        G = sp.csr_matrix((np.ones(M), (np.arange(M), classes)),
                          shape=(M, classes.max() + 1))
    KG = (K @ G).tocsr()
    Eeq = build_equilibrium(Pbar, KG)
    csize = np.asarray(G.sum(axis=0)).ravel()              # edges per variable

    theta = cp.Variable(G.shape[1])
    Om = cp.reshape(KG @ theta, (N, N), order="C")
    objective = cp.Minimize(csize @ cp.abs(theta)          # ||w||_1
                            - alpha * 2.0 * (csize @ theta))  # - alpha * tr(Omega)
    constraints = [
        Om + gamma * (U @ U.T) >> gamma * np.eye(N),       # <=> Q^T Omega Q >= gamma I
        Om << beta * np.eye(N),                            # <=> ||Omega||_2 <= beta
        Eeq @ theta == 0,                                  # <=> Omega Pbar^T = 0
    ]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=solver, **solver_kw)

    w = np.asarray(G @ theta.value).ravel()
    Omega = (K @ w).reshape(N, N)
    ev = np.sort(np.linalg.eigvalsh(Omega))
    return dict(status=prob.status, value=prob.value, w=w, Omega=Omega, eig=ev,
                lam_D2=ev[D + 1], lam_max=ev[-1],
                residual=np.abs(Omega @ Pbar.T).max(),
                edges=int((np.abs(w) > 1e-4 * np.abs(w).max()).sum()),
                solve_time=prob.solver_stats.solve_time)
```

### Implementation notes and pitfalls

- **Never** build Ω or QᵀΩQ with `cp.diag(w)` inside a matrix product. That is the memory problem.
- `U` must be orthonormal (`np.linalg.qr(Pbar.T)[0]`). Do not use P̄ᵀ directly.
- Keep `order="C"` in `cp.reshape` consistent with the row-major K (index r·N + c).
- λ_{D+2} is `ev[D+1]` with zero-based indexing of the ascending eigenvalues.
- For 3D configurations D = 3; P̄ always gets one extra row of ones.
- Count edges with a relative threshold (here 10⁻⁴ of the largest weight), since solvers return tiny
  nonzeros instead of exact zeros.
- Clarabel often reports `optimal_inaccurate` on these problems at default tolerances. In every test
  the objective still matched the original formulation to 5 digits. Prefer MOSEK when available; it is
  also much faster on SDPs.
- Edge order: if the existing code uses a different edge ordering, either reorder `classes` and the
  output weights, or replace `edges_complete` with the existing ordering everywhere.

---

## 6. Tests (all must pass)

Run with γ = 0.1, β = 1 unless stated. Reference values below were obtained with Clarabel.

**T1. K is correct.** For random N = 7 and random w: `(K @ w).reshape(N, N)` equals
`B @ np.diag(w) @ B.T`.

**T2. Same optimum as the original P2.** For a random 2D configuration
(`np.random.default_rng(0).normal(size=(2, 30))`), α = 0.6: the new function and the original
formulation (with ψ, `sigma_max` and the projected LMI) give the same optimal value.
Reference: **−1.25197** for both, with Clarabel and SCS. Repeat for N = 8 and N = 12; values must agree
to relative tolerance 10⁻⁴.

**T3. Wang et al. 12-agent benchmark.** Coordinates (x, y pairs):
`[2,0, 1.6,1.2, 1,1.6, 0,1.8, -1,0.9, -1.1,-0.2, -0.6,-1.3, 0.5,-1.4, 1.5,-1.2, -0.3,0.7, 0.2,-0.5, 1,0.7]`
reshaped to 12 × 2 and transposed. Reference:

| α | Edges | Condition number λ_max/λ_{D+2} | max \|ΩP̄ᵀ\| |
|---|---|---|---|
| 0.5 | 32 | 2.45 | ≤ 1e-9 |
| 0.6 | 29 | 3.21 | ≤ 1e-9 |

**T4. Post-solve checks on every solution.** Ω symmetric; row sums ≈ 0; max |ΩP̄ᵀ| ≤ 1e-8 (MOSEK) or
1e-6 (Clarabel); λ_{D+2} ≥ γ − 1e-6; λ_max ≤ β + 1e-6; exactly D+1 eigenvalues ≈ 0.

**T5. Scaling (random 2D configurations, α = 0.6).** The old formulation must not be needed at all.
Reference on a 7 GB cloud VM with Clarabel (MOSEK should be faster):

| N | Weights | Solve time | Peak memory | max \|ΩP̄ᵀ\| |
|---|---|---|---|---|
| 30 | 435 | 0.5 s | 0.15 GB | 2e-12 |
| 60 | 1,770 | 17.5 s | 0.71 GB | 2e-13 |
| 100 | 4,950 | 219 s | 3.7 GB | 1e-9 |

Record solve time and peak memory (e.g. `resource.getrusage(resource.RUSAGE_SELF).ru_maxrss`) and
update Table I of the paper, where P2 is currently listed as N/A at N = 100.

**T6. P3 equals P2 with exact orbits (checks Theorem 2), and EDM classes lose optimality.**
Truncated icosahedron (60 vertices: even permutations of (0, ±1, ±3φ), (±1, ±(2+φ), ±2φ),
(±φ, ±2, ±φ³)), coordinates scaled so the largest absolute coordinate is 1, D = 3, α = 0.6.
Compute the exact orbits of vertex pairs under its full symmetry group (order 120), as in
`compare_edm_vs_automorphism.m`. Reference:

| Formulation | Classes | Optimal value | λ_{D+2} | Edges |
|---|---|---|---|---|
| P2 (all edges) | 1,770 | −1.77918 | 0.1238 | 150 |
| P3, exact orbits | 23 | −1.77918 | 0.1238 | 150 |
| P3, equal-distance (EDM) classes | 21 | −1.69191 | 0.1477 | 330 |

P3 with exact orbits must match P2 to 5 digits. The EDM partition gives a strictly worse optimum and a
much denser stress (330 edges instead of 150). This is a concrete numerical answer to the reviewer's
question about equal-length edges and can go in the response letter.

**T7. Trace identity.** For any feasible solution, `psi @ w`, `2 * w.sum()` and `np.trace(Omega)` agree
to 1e-8.

---

## 7. What to change in the paper afterwards

- Table I: replace "N/A" for P2 at N = 100 with the new runtime. Keep the multicluster method for
  N ≥ a few hundred: interior-point methods still form a dense M × M Schur complement (about 124 GB at
  N = 500), so monolithic P2 stops scaling somewhere around N = 200–300.
- Implementation section: one sentence that QᵀΩQ ⪰ γI is implemented through the equivalent sparse
  LMI Ω + γUUᵀ ⪰ γI, and ‖Ω‖₂ ≤ β through Ω ⪯ βI.
- Optionally add the feasibility proposition of Section 3.6 (P2 feasible iff γ ≤ β; Slater for γ < β).
- Use test T6 in the response to the reviewer's question on equal-distance classes.
