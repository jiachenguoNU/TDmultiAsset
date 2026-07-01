"""
solver1d.py
---------------------------
Subspace-iteration (alternating least squares / PGD) assembly and solve for the
1D Black-Scholes equation cast as a Space (x = ln S) - Parameter (sigma) -
Time (tau = T - t) tensor-decomposition problem.

This is the BS analogue of the heat-equation `subIterSPT1D.py`. It implements
exactly the per-direction systems derived in ``TD_subspace_derivation.md``:

  x-solve  ( ACTIVE = x , 3 Kronecker terms ):
      ( M_x (x) coef_dtau_r  +  K_x (x) coef_diff  +  NB_x (x) coef_adv ) U_x = Q_x
  sigma-solve ( ACTIVE = sigma , 2 terms ):
      ( M_s (x) coef1_s  +  W_s (x) coef2_s ) U_s = Q_s
  tau-solve ( ACTIVE = tau , 2 terms ):
      ( PB_t (x) coef_dtau  +  M_t (x) coef_sp ) U_t = Q_t

1D matrices (all scipy CSR):
    M_x  = K_Nx_Nx          (space mass,        symmetric)
    K_x  = K_Bx_Bx          (space stiffness,   symmetric)
    NB_x = K_Nx_Bx          (space advection,   NON-symmetric, test N / trial B)
    M_s  = K_Ns_Ns          (parameter L2 mass)
    W_s  = K_Ns_s2_Ns       (parameter sigma^2-weighted mass)
    M_t  = K_Nt_Nt          (time L2 mass)
    PB_t = K_Nt_Bt          (time Petrov-Galerkin, NON-symmetric, test N / trial B)

Gram convention (matches the example): with U_d of shape (num_mode, n_d),
    gram(U_d, A)[m, n] = sum_{i,j} U_d[m,i] A[i,j] U_d[n,j]  ==  U_d @ A @ U_d.T
(test mode m = row, trial mode n = column; non-symmetric A keep orientation).

The per-direction linear systems are solved by explicitly forming the
Kronecker-sum operator and a sparse direct solve (`kron_sum_solve`). Unlike the
heat case, the BS x-solve has THREE Kronecker terms, so the two-term
eigen/Schur Sylvester solvers (`td_solve` / `td_solve_shur`) do not apply
directly; the explicit-Kronecker solve handles any number of terms and the
systems here are small (n_d * num_mode). The 2-term sigma/tau systems could
alternatively reuse `td_solve` / `td_solve_shur` for speed.
"""
import numpy as onp
import scipy.sparse as sp
import scipy.sparse.linalg as spla


class bcolors:
    HEADER = '\033[95m'; OKGREEN = '\033[92m'; WARNING = '\033[93m'
    FAIL = '\033[91m'; ENDC = '\033[0m'; BOLD = '\033[1m'


# --------------------------------------------------------------------------- #
#  Building blocks
# --------------------------------------------------------------------------- #
def gram(U, K):
    """Modal Gram U @ K @ U.T  ->  dense (num_mode, num_mode).

    U: (num_mode, n) ; K: (n, n) scipy sparse or dense.
    Orientation preserved for non-symmetric K (test mode = row).
    """
    return onp.asarray(U @ (K @ U.T))


def assemble_rhs(Q_a, Q_p1, U_p1, Q_p2, U_p2, num_mode):
    """Separable-forcing RHS for the ACTIVE direction ``a`` (passives p1, p2).

    Implements  Q^[a]_{i,m} = sum_r Q_a[r,i] * (sum_i' U_p1[m,i'] Q_p1[r,i'])
                                            * (sum_i'' U_p2[m,i''] Q_p2[r,i'']),
    i.e. TD_subspace_derivation.md eq. in B.10 (one J-projection per passive dim).
    Matches `AFI_*_STP_syvl` Q_term5 in the heat example.

    Q_a:  (num_source, n_a)   active-direction 1D loads  Q_a^{(r)}
    Q_p1: (num_source, n_p1)  passive-1 1D loads,  U_p1: (num_mode, n_p1)
    Q_p2: (num_source, n_p2)  passive-2 1D loads,  U_p2: (num_mode, n_p2)
    Returns Q: (n_a, num_mode)
    """
    proj1 = onp.sum(U_p1[None, :, :] * Q_p1[:, None, :], axis=2)   # (num_source, num_mode)
    proj2 = onp.sum(U_p2[None, :, :] * Q_p2[:, None, :], axis=2)   # (num_source, num_mode)
    Qt = Q_a[:, None, :] * proj1[:, :, None] * proj2[:, :, None]   # (num_source, num_mode, n_a)
    Qt = onp.sum(Qt, axis=0)                                       # (num_mode, n_a)
    return Qt.reshape(num_mode, -1).T                              # (n_a, num_mode)


def kron_sum_solve(terms, Q, bc_idx, num_mode):
    """Solve  sum_i ( C_i (x) K_i ) vec_F(X) = vec_F(Q)  for X (n, num_mode).

    The matrix (Sylvester) form is  sum_i  K_i X C_i^T = Q, vectorised
    column-major so that vec_F(K X C^T) = (C (x) K) vec_F(X). Dirichlet rows
    (node indices ``bc_idx``, every mode) are set homogeneous.

    terms : list of (C_i  dense (num_mode, num_mode), K_i  sparse (n, n))
    Q     : (n, num_mode) RHS
    bc_idx: 1D array of constrained node indices (homogeneous); [] for none.
    Returns X: (n, num_mode)
    """
    n = terms[0][1].shape[0]
    A = None
    for C, K in terms:
        block = sp.kron(sp.csr_matrix(onp.asarray(C)), sp.csr_matrix(K), format='csr')
        A = block if A is None else (A + block)
    rhs = onp.asarray(Q, dtype=float).reshape(-1, order='F').copy()

    bc_idx = onp.asarray(bc_idx, dtype=int)
    if bc_idx.size > 0:
        # global dof for (node i, mode m) under column-major vec_F is m*n + i
        gidx = (onp.arange(num_mode)[:, None] * n + bc_idx[None, :]).reshape(-1)
        A = A.tolil()
        A[gidx, :] = 0.0
        A[:, gidx] = 0.0
        A[gidx, gidx] = 1.0
        A = A.tocsr()
        rhs[gidx] = 0.0

    sol = spla.spsolve(A.tocsc(), rhs)
    return onp.asarray(sol).reshape((n, num_mode), order='F')


# --------------------------------------------------------------------------- #
#  Per-direction coefficient assembly  (returns list of (coef, K) + RHS)
# --------------------------------------------------------------------------- #
def AFI_x_BS(mats, Qx, Qs, Qt, U_x, U_s, U_t, r, q):
    """ACTIVE = x.  Three Kronecker terms (mass, stiffness, advection)."""
    M_x, K_x, NB_x, M_s, W_s, M_t, PB_t = mats
    num_mode = U_x.shape[0]
    # passive Grams (sigma, tau)
    Gs_NN = gram(U_s, M_s); Gs_W = gram(U_s, W_s)
    Gt_NN = gram(U_t, M_t); Gt_NB = gram(U_t, PB_t)
    Gs_adv = -(r - q) * Gs_NN + 0.5 * Gs_W            # U_s^T[-(r-q)M_s + 1/2 W_s]U_s

    coef_dtau_r = (Gs_NN * Gt_NB) + r * (Gs_NN * Gt_NN)   # -> M_x   (time deriv + reaction)
    coef_diff   = 0.5 * (Gs_W * Gt_NN)                    # -> K_x   (diffusion)
    coef_adv    = Gs_adv * Gt_NN                          # -> NB_x  (advection)

    terms = [(coef_dtau_r, M_x), (coef_diff, K_x), (coef_adv, NB_x)]
    Q = assemble_rhs(Qx, Qs, U_s, Qt, U_t, num_mode)
    return terms, Q


def AFI_sigma_BS(mats, Qx, Qs, Qt, U_x, U_s, U_t, r, q):
    """ACTIVE = sigma.  Two Kronecker terms (L2 mass, sigma^2-weighted mass)."""
    M_x, K_x, NB_x, M_s, W_s, M_t, PB_t = mats
    num_mode = U_x.shape[0]
    Gx_M = gram(U_x, M_x); Gx_K = gram(U_x, K_x); Gx_NB = gram(U_x, NB_x)
    Gt_NN = gram(U_t, M_t); Gt_NB = gram(U_t, PB_t)

    coef1 = (Gx_M * Gt_NB) - (r - q) * (Gx_NB * Gt_NN) + r * (Gx_M * Gt_NN)   # -> M_s
    coef2 = 0.5 * (Gx_K * Gt_NN) + 0.5 * (Gx_NB * Gt_NN)                      # -> W_s

    terms = [(coef1, M_s), (coef2, W_s)]
    Q = assemble_rhs(Qs, Qx, U_x, Qt, U_t, num_mode)
    return terms, Q


def AFI_tau_BS(mats, Qx, Qs, Qt, U_x, U_s, U_t, r, q):
    """ACTIVE = tau.  Two Kronecker terms (Petrov-Galerkin, L2 mass)."""
    M_x, K_x, NB_x, M_s, W_s, M_t, PB_t = mats
    num_mode = U_x.shape[0]
    Gx_M = gram(U_x, M_x); Gx_K = gram(U_x, K_x); Gx_NB = gram(U_x, NB_x)
    Gs_NN = gram(U_s, M_s); Gs_W = gram(U_s, W_s)
    Gs_adv = -(r - q) * Gs_NN + 0.5 * Gs_W

    coef_dtau = Gx_M * Gs_NN                                                   # -> PB_t
    coef_sp = 0.5 * (Gx_K * Gs_W) + (Gx_NB * Gs_adv) + r * (Gx_M * Gs_NN)      # -> M_t

    terms = [(coef_dtau, PB_t), (coef_sp, M_t)]
    Q = assemble_rhs(Qt, Qx, U_x, Qs, U_s, num_mode)
    return terms, Q


# --------------------------------------------------------------------------- #
#  One full alternating sweep  x -> sigma -> tau
# --------------------------------------------------------------------------- #
def TD_solver_BS_sweep(mats, Qx, Qs, Qt, U_x, U_s, U_t, r, q,
                       dirichlet_idx, ic_idx):
    """One alternating (Gauss-Seidel) sweep over the three directions.

    mats   : (M_x, K_x, NB_x, M_s, W_s, M_t, PB_t)  scipy CSR
    Qx/Qs/Qt : (num_source, n_d) separable 1D loads
    U_x/U_s/U_t : (num_mode, n_d) current modal coefficients
    dirichlet_idx : x-direction Dirichlet node indices (both ends)
    ic_idx        : tau-direction initial-condition node index ([0])
    Returns updated (U_x, U_s, U_t, variation, norm).
    """
    num_mode = U_x.shape[0]
    U_x0, U_s0, U_t0 = U_x, U_s, U_t

    terms, Q = AFI_x_BS(mats, Qx, Qs, Qt, U_x, U_s, U_t, r, q)
    U_x = kron_sum_solve(terms, Q, dirichlet_idx, num_mode).T          # (num_mode, n_x)

    terms, Q = AFI_sigma_BS(mats, Qx, Qs, Qt, U_x, U_s, U_t, r, q)
    U_s = kron_sum_solve(terms, Q, [], num_mode).T                     # no BC on sigma

    terms, Q = AFI_tau_BS(mats, Qx, Qs, Qt, U_x, U_s, U_t, r, q)
    U_t = kron_sum_solve(terms, Q, ic_idx, num_mode).T                 # IC at tau node 0

    variation = (1. / 3.) * (onp.linalg.norm(U_x0 - U_x) / onp.linalg.norm(U_x0)
                             + onp.linalg.norm(U_s0 - U_s) / onp.linalg.norm(U_s0)
                             + onp.linalg.norm(U_t0 - U_t) / onp.linalg.norm(U_t0))
    norm = onp.hstack((onp.linalg.norm(U_x), onp.linalg.norm(U_s), onp.linalg.norm(U_t)))
    return U_x, U_s, U_t, variation, norm
