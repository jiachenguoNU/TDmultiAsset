"""
nd_bs.py
===========================
General-D multi-asset Black-Scholes tensor-decomposition (PGD / subspace-iteration,
C-HiDeNN finite-element) solver, treated as Space (x_a = ln S_a, a = 0..D-1) - Time
(tau = T - t).  Volatilities sigma_a and correlations rho_ab are FIXED scalars here
(not parameter axes), which keeps the cross-check monolithic solve and the Monte-Carlo
validation tractable while still exercising the genuinely new structure of the nD
problem -- the correlation CROSS-DERIVATIVE term.

Governing equation (forward in tau, constant-coefficient in x; see
``../TD_subspace_derivation.md`` and its B.11 fixed-sigma reduction):

    du/dtau  -  1/2 sum_{a,b} rho_ab sigma_a sigma_b d2u/dx_a dx_b
             -  sum_a b_a du/dx_a  +  r u  =  f ,      b_a = r - q_a - 1/2 sigma_a^2 .

Per-direction Kronecker structure (fixed sigma => the sigma-axes of the full nD
derivation collapse to the scalars below):

  active x_g : M_x[g] (x) coef_M  +  K_x[g] (x) coef_K
             + NB_x[g] (x) coef_NB  +  BN_x[g] (x) coef_BN          (4 LEFT factors)
  active tau : PB_t (x) coef_PBt  +  M_t (x) coef_Mt                (2 LEFT factors)

with (G^M,G^K,G^NB,G^BN = modal Grams of mass/stiffness/advection/advection^T;
G_t^NN,G_t^NB = tau L2 / Petrov Grams; (.) = Hadamard product; the cross term is the
ONLY genuinely new piece vs 1D and uses the advection matrix and its transpose):

  coef_K[g]  = 1/2 sigma_g^2  (prod_{l!=g} G^M_l) o G_t^NN
  coef_NB[g] = (-b_g) (prod_{l!=g} G^M_l) o G_t^NN
             + sum_{a!=g} 1/2 rho_ag sigma_a sigma_g  G^BN_a o (prod_{l!=g,a} G^M_l) o G_t^NN
  coef_BN[g] = sum_{b!=g} 1/2 rho_gb sigma_g sigma_b G^NB_b o (prod_{l!=g,b} G^M_l) o G_t^NN
  coef_M[g]  = (prod_{l!=g} G^M_l) o (G_t^NB + r G_t^NN)
             + sum_{d!=g} 1/2 sigma_d^2          G^K_d  o (prod_{l!=g,d} G^M_l) o G_t^NN
             + sum_{d!=g} (-b_d)                 G^NB_d o (prod_{l!=g,d} G^M_l) o G_t^NN
             + sum_{a!=b, a,b!=g} 1/2 rho_ab s_a s_b G^BN_a o G^NB_b o (prod_{l!=g,a,b}G^M_l) o G_t^NN

  coef_PBt   = prod_l G^M_l
  coef_Mt    = sum_g 1/2 sigma_g^2 G^K_g o (prod_{l!=g}G^M_l)
             + sum_{a!=b} 1/2 rho_ab s_a s_b G^BN_a o G^NB_b o (prod_{l!=a,b}G^M_l)
             + sum_g (-b_g) G^NB_g o (prod_{l!=g}G^M_l)
             + r prod_l G^M_l

The same operator is also provided as (a) a list of (coeff, [1D matrix per axis]) for
cheap tensor mode-product application  -L g  (no full Kronecker), and (b) an explicit
sparse Kronecker sum for the monolithic cross-check.
"""
import time
import numpy as onp
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import jax
jax.config.update("jax_enable_x64", True)

class bcolors:
    HEADER = '\033[95m'; OKGREEN = '\033[92m'; OKCYAN = '\033[96m'
    WARNING = '\033[93m'; FAIL = '\033[91m'; ENDC = '\033[0m'; BOLD = '\033[1m'


# --------------------------------------------------------------------------- #
#  Linear algebra helpers
# --------------------------------------------------------------------------- #
def gram(U, K):
    """Modal Gram U @ K @ U.T -> dense (num_mode, num_mode); orientation preserved."""
    return onp.asarray(U @ (K @ U.T))


def _had(mats):
    out = mats[0].copy()
    for m in mats[1:]:
        out = out * m
    return out


def kron_sum_solve(terms, Q, bc_idx, num_mode):
    """Solve  sum_i (C_i (x) K_i) vec_F(X) = vec_F(Q)  for X (n, num_mode).

    Sylvester form sum_i K_i X C_i^T = Q, column-major vectorised. Dirichlet rows
    (node indices bc_idx, every mode) set homogeneous. (Same as the 1D BS solver.)
    """
    n = terms[0][1].shape[0]
    A = None
    for C, K in terms:
        block = sp.kron(sp.csr_matrix(onp.asarray(C)), sp.csr_matrix(K), format='csr')
        A = block if A is None else (A + block)
    rhs = onp.asarray(Q, dtype=float).reshape(-1, order='F').copy()
    bc_idx = onp.asarray(bc_idx, dtype=int)
    if bc_idx.size > 0:
        gidx = (onp.arange(num_mode)[:, None] * n + bc_idx[None, :]).reshape(-1)
        A = A.tolil(); A[gidx, :] = 0.0; A[:, gidx] = 0.0; A[gidx, gidx] = 1.0; A = A.tocsr()
        rhs[gidx] = 0.0
    sol = spla.spsolve(A.tocsc(), rhs)
    return onp.asarray(sol).reshape((n, num_mode), order='F')


# --------------------------------------------------------------------------- #
#  Per-direction coefficient assembly (interior operator, fixed sigma)
# --------------------------------------------------------------------------- #
def interior_terms(active, A, U, M, sigma, rho, r, q):
    """Kronecker-term list  [(coef MxM, K sparse)]  for the active direction.

    active : 0..k-1 -> spatial asset (local index); k -> tau.
    A      : tuple of GLOBAL asset ids alive in this sub-problem (len k).
    U      : list length k+1 of modal coeffs (num_mode, n_d); U[k] is tau.
    M      : matrices dict from build_matrices.
    """
    k = len(A)
    num_mode = U[0].shape[0]
    sl = onp.array([sigma[a] for a in A])
    ql = onp.array([q[a] for a in A])
    adv = -(r - ql) + 0.5 * sl ** 2                 # = -b_a  (advection scalar)
    rl = onp.array([[rho[A[a], A[b]] for b in range(k)] for a in range(k)])

    GM = [gram(U[l], M['Mx'][A[l]]) for l in range(k)]
    GK = [gram(U[l], M['Kx'][A[l]]) for l in range(k)]
    GNB = [gram(U[l], M['NBx'][A[l]]) for l in range(k)]
    GBN = [g.T for g in GNB]
    GtNN = gram(U[k], M['Mt']); GtNB = gram(U[k], M['PBt'])
    ones = onp.ones((num_mode, num_mode))

    def gmp(exclude):
        facs = [GM[l] for l in range(k) if l not in exclude]
        return _had(facs) if facs else ones

    if active < k:                                   # ---- spatial direction g ----
        g = active
        coef_K = 0.5 * sl[g] ** 2 * (gmp({g}) * GtNN)
        coef_NB = adv[g] * (gmp({g}) * GtNN)
        for a in range(k):
            if a == g:
                continue
            coef_NB = coef_NB + 0.5 * rl[a, g] * sl[a] * sl[g] * (GBN[a] * gmp({g, a}) * GtNN)
        coef_BN = onp.zeros((num_mode, num_mode))
        for b in range(k):
            if b == g:
                continue
            coef_BN = coef_BN + 0.5 * rl[g, b] * sl[g] * sl[b] * (GNB[b] * gmp({g, b}) * GtNN)
        coef_M = gmp({g}) * (GtNB + r * GtNN)
        for d in range(k):
            if d == g:
                continue
            coef_M = coef_M + 0.5 * sl[d] ** 2 * (GK[d] * gmp({g, d}) * GtNN)
            coef_M = coef_M + adv[d] * (GNB[d] * gmp({g, d}) * GtNN)
        for a in range(k):
            for b in range(k):
                if a == b or a == g or b == g:
                    continue
                coef_M = coef_M + 0.5 * rl[a, b] * sl[a] * sl[b] * (GBN[a] * GNB[b] * gmp({g, a, b}) * GtNN)
        terms = [(coef_M, M['Mx'][A[g]]), (coef_K, M['Kx'][A[g]]),
                 (coef_NB, M['NBx'][A[g]])]
        if k > 1:
            terms.append((coef_BN, M['BNx'][A[g]]))
        return terms

    # ---- time direction tau ----
    coef_PBt = gmp(set())
    coef_Mt = r * gmp(set())
    for g in range(k):
        coef_Mt = coef_Mt + 0.5 * sl[g] ** 2 * (GK[g] * gmp({g}))
        coef_Mt = coef_Mt + adv[g] * (GNB[g] * gmp({g}))
    for a in range(k):
        for b in range(k):
            if a == b:
                continue
            coef_Mt = coef_Mt + 0.5 * rl[a, b] * sl[a] * sl[b] * (GBN[a] * GNB[b] * gmp({a, b}))
    return [(coef_PBt, M['PBt']), (coef_Mt, M['Mt'])]


# --------------------------------------------------------------------------- #
#  Operator as (coeff, [1D matrix per axis]) -- mode-product apply & monolithic
# --------------------------------------------------------------------------- #
def interior_terms_generic(active, op, U):
    """Per-direction subspace Kronecker terms derived generically from an operator
    term-list `op` = [(coef, [A_0..A_{nax-1}])].  For active dimension a:
        sum_i ( coef_i * prod_{d!=a} U_d^T A_i[d] U_d ) (x) A_i[a].
    Works for ANY operator (fixed-sigma or sigma-parametric) -- no hand-coded grouping.
    kron_sum_solve sums the (unmerged) blocks, so correctness does not need merging.
    """
    num_mode = U[0].shape[0]
    terms = []
    for coef, mats in op:
        C = coef * onp.ones((num_mode, num_mode))
        for d in range(len(mats)):
            if d == active:
                continue
            C = C * gram(U[d], mats[d])
        terms.append((C, mats[active]))
    return terms


def operator_terms(A, M, sigma, rho, r, q):
    """List of (coeff, [matrix per axis]) over axes (x for a in A) + tau.

    Same operator as interior_terms, in plain (un-Gram'd) Kronecker form.
    """
    k = len(A)
    sl = onp.array([sigma[a] for a in A]); ql = onp.array([q[a] for a in A])
    adv = -(r - ql) + 0.5 * sl ** 2
    base_x = [M['Mx'][A[l]] for l in range(k)]
    out = []
    out.append((1.0, base_x + [M['PBt']]))                              # time
    out.append((r, base_x + [M['Mt']]))                                 # reaction
    for g in range(k):                                                  # self-diffusion
        mats = list(base_x); mats[g] = M['Kx'][A[g]]; out.append((0.5 * sl[g] ** 2, mats + [M['Mt']]))
    for g in range(k):                                                  # advection
        mats = list(base_x); mats[g] = M['NBx'][A[g]]; out.append((adv[g], mats + [M['Mt']]))
    for a in range(k):                                                  # cross / correlation
        for b in range(k):
            if a == b:
                continue
            mats = list(base_x); mats[a] = M['BNx'][A[a]]; mats[b] = M['NBx'][A[b]]
            out.append((0.5 * rho[A[a], A[b]] * sl[a] * sl[b], mats + [M['Mt']]))
    return out


def _mode_product(T, Asp, axis):
    """Contract sparse matrix Asp (m,n) along `axis` of dense tensor T (n=T.shape[axis])."""
    T2 = onp.moveaxis(T, axis, 0)
    shp = T2.shape
    R = Asp @ T2.reshape(shp[0], -1)
    R = onp.asarray(R).reshape((Asp.shape[0],) + shp[1:])
    return onp.moveaxis(R, 0, axis)


def apply_operator(T, op_terms):
    """Apply the Kronecker-sum operator to full tensor T (mode products, no full kron)."""
    out = onp.zeros_like(T)
    for coef, mats in op_terms:
        g = T
        for ax, Asp in enumerate(mats):
            g = _mode_product(g, Asp, ax)
        out = out + coef * g
    return out


def monolithic_operator(op_terms):
    """Explicit sparse Kronecker-sum operator (for cross-check / small solves)."""
    A = None
    for coef, mats in op_terms:
        blk = mats[0]
        for Asp in mats[1:]:
            blk = sp.kron(blk, Asp, format='csr')
        blk = coef * blk
        A = blk if A is None else (A + blk)
    return A.tocsr()


# --------------------------------------------------------------------------- #
#  RHS builders
# --------------------------------------------------------------------------- #
def rhs_from_full(b, U, active):
    """Subspace RHS by contracting full load tensor b against the OTHER modes.

    Q[i_a, m] = sum_{i != a} b[i] * prod_{d!=a} U[d][m, i_d].   (no separability of b needed)
    """
    num_mode = U[0].shape[0]
    Ta = onp.moveaxis(b, active, 0)
    other = [d for d in range(b.ndim) if d != active]
    Q = onp.zeros((Ta.shape[0], num_mode))
    for m in range(num_mode):
        Tm = Ta
        for d in other:
            Tm = onp.tensordot(Tm, U[d][m], axes=([1], [0]))   # consumes axes in order
        Q[:, m] = Tm
    return Q


def rhs_separable(Q_list, U, active):
    """Subspace RHS for a separable load: Q_list[d] = (num_source, n_d).

    Q[i_a, m] = sum_r Q_list[a][r,i_a] * prod_{d!=a} (sum_i U[d][m,i] Q_list[d][r,i]).
    """
    num_mode = U[0].shape[0]
    nd = len(Q_list)
    proj = onp.ones((Q_list[active].shape[0], num_mode))
    for d in range(nd):
        if d == active:
            continue
        proj = proj * onp.einsum('ri,mi->rm', Q_list[d], U[d])
    return onp.einsum('ra,rm->am', Q_list[active], proj)


# --------------------------------------------------------------------------- #
#  PGD sweep / solve for one sub-problem
# --------------------------------------------------------------------------- #
def reconstruct(U):
    """Full nodal tensor from modal factors: sum_m outer_d U[d][m]."""
    num_mode = U[0].shape[0]
    out = None
    for m in range(num_mode):
        term = U[0][m]
        for d in range(1, len(U)):
            term = onp.multiply.outer(term, U[d][m])
        out = term if out is None else out + term
    return out


def pgd_solve(A, M, sigma, rho, r, q, rhs_fn, bc_list, num_mode,
              max_iter=200, tol=1e-9, seed=0, verbose=False):
    """Alternating subspace iteration for asset-set A (+tau), homogeneous BC.

    rhs_fn(active, U) -> (n_active, num_mode) RHS.
    bc_list[d]        -> constrained node indices for direction d.
    Returns list U (k+1 modal factors) and the reconstructed nodal tensor.
    """
    k = len(A)
    dims = [M['n_x']] * k + [M['n_t']]
    rng = onp.random.default_rng(seed)
    U = [rng.standard_normal((num_mode, nd)) for nd in dims]

    u_prev = reconstruct(U)
    for it in range(max_iter):
        for active in range(k + 1):
            terms = interior_terms(active, A, U, M, sigma, rho, r, q)
            Q = rhs_fn(active, U)
            U[active] = kron_sum_solve(terms, Q, bc_list[active], num_mode).T
        nc = onp.prod([onp.linalg.norm(Ud) for Ud in U]) ** (1.0 / len(U))
        for d in range(len(U)):
            nrm = onp.linalg.norm(U[d])
            if nrm > 0:
                U[d] = nc * U[d] / nrm
        u_now = reconstruct(U)
        delta = onp.linalg.norm(u_now - u_prev) / (onp.linalg.norm(u_now) + 1e-30)
        u_prev = u_now
        if verbose and ((it + 1) % 25 == 0 or delta < tol):
            print(f"    sweep {it+1:3d}: reconstruction change = {delta:.2e}")
        if delta < tol:
            break
    return U, u_now


def pgd_greedy(A, M, sigma, rho, r, q, b_load, bc_list, max_modes=40, als_iters=25,
              tol=1e-9, patience=3, seed=0, verbose=False):
    """Greedy (progressive rank-enrichment) PGD for  L u = b_load  with homogeneous BC.

    Adds one rank-1 mode at a time, each fitted to the current residual by a few ALS
    sweeps (1-mode subspace solves, never rank-degenerate). Stops on relative residual
    `tol`, on the mode cap, or after `patience` enrichments with no residual improvement.
    Robust where fixed-rank ALS goes rank-singular once #modes exceeds the true rank.
    """
    k = len(A); dims = [M['n_x']] * k + [M['n_t']]
    op = operator_terms(A, M, sigma, rho, r, q)
    # interior mask (0 on constrained Dirichlet/IC nodes) -- the residual there is
    # un-reducible (those dofs are pinned) and must be excluded from the metric/fit.
    grids = onp.indices(dims)
    interior = onp.ones(dims, dtype=bool)
    for d in range(k + 1):
        if len(bc_list[d]):
            interior &= ~onp.isin(grids[d], onp.asarray(bc_list[d]))
    u_full = onp.zeros(dims)
    bnorm = onp.linalg.norm((b_load * interior)) + 1e-30
    rng = onp.random.default_rng(seed)
    best = onp.inf; stale = 0
    for enr in range(max_modes):
        res = (b_load - apply_operator(u_full, op)) * interior
        rnorm = onp.linalg.norm(res) / bnorm
        if verbose and (enr % 5 == 0 or rnorm < tol):
            print(f"      enrich {enr:2d}: rel residual = {rnorm:.3e}")
        if rnorm < tol:
            break
        stale = 0 if rnorm < best * (1.0 - 1e-4) else stale + 1
        best = min(best, rnorm)
        if stale >= patience:
            if verbose:
                print(f"      stop at {enr} modes (rel res {rnorm:.3e}, no improvement)")
            break
        v = [rng.standard_normal(nd) for nd in dims]
        for _ in range(als_iters):
            for active in range(k + 1):
                U1 = [v[d][None, :] for d in range(k + 1)]
                terms = interior_terms(active, A, U1, M, sigma, rho, r, q)
                Q = rhs_from_full(res, U1, active)
                v[active] = kron_sum_solve(terms, Q, bc_list[active], 1)[:, 0]
            nrm = [onp.linalg.norm(vd) for vd in v]
            gm = onp.prod(nrm) ** (1.0 / len(v))
            for d in range(len(v)):
                if nrm[d] > 0:
                    v[d] = gm * v[d] / nrm[d]
        u_full = u_full + reconstruct([vd[None, :] for vd in v])
    return u_full


def bc_homogeneous(A, M):
    """Homogeneous Dirichlet on both ends of every x-axis, IC at tau node 0."""
    k = len(A)
    bc = [onp.array([0, M['n_x'] - 1], dtype=int) for _ in range(k)]
    bc.append(onp.array([0], dtype=int))             # tau initial node
    return bc


# --------------------------------------------------------------------------- #
#  Monolithic solve (cross-check)  for a separable OR full-tensor load
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
#  Nested lower-boundary hierarchy  (TD_subspace_derivation.md B.12)
# --------------------------------------------------------------------------- #
def _embed_full(T, present, A):
    """Embed tensor T (axes = sorted `present` assets + tau) into the full level-A
    layout (axes = sorted A + tau) by inserting size-1 axes for the missing assets."""
    out = onp.asarray(T)
    for pos, a in enumerate(A):
        if a not in present:
            out = onp.expand_dims(out, axis=pos)
    return out


def build_lift(A, M, Vsub, payoff_tensor, asymp_tensor, check=True):
    """Transfinite (Boolean-sum) separable lift g matching all 2|A| spatial faces and
    the IC, from the known trace data (lower levels Vsub[.] + analytic upper asymptote).
    Single (|A|+1)-way construction: g = g_sp ; g[...,tau=0] := payoff  (B.12.4).

    Precondition (B.12.3): the supplied lower-level traces Vsub[.] must be mutually
    consistent at shared lower corners (Vsub[A\\b]|_{Sc=0} == Vsub[A\\c]|_{Sb=0}); the
    nested_price hierarchy guarantees this by construction. `errs` (returned) reports the
    residual face/IC reproduction error, which is 0 except for the documented O(S_min)
    lower-upper edge mismatch -- check it stays small.

    Returns (g  full nodal tensor (n_x^k, n_t),  errs dict of face/IC residuals).
    """
    from itertools import combinations, product as iproduct
    k = len(A); n_x = M['n_x']; n_t = M['n_t']
    xg = onp.asarray(M['xg']); xmin, xmax = xg[0], xg[-1]
    phi = {'lo': (xmax - xg) / (xmax - xmin), 'hi': (xg - xmin) / (xmax - xmin)}
    full = (n_x,) * k + (n_t,)
    g_sp = onp.zeros(full)
    for rr in range(1, k + 1):
        for B in combinations(range(k), rr):
            sign = (-1.0) ** (rr + 1)
            for s in iproduct(('lo', 'hi'), repeat=rr):
                sm = dict(zip(B, s))
                B_lo = [j for j in B if sm[j] == 'lo']
                B_hi = [j for j in B if sm[j] == 'hi']
                if not B_lo:
                    T = onp.asarray(asymp_tensor); present = list(A)
                else:
                    src = tuple(sorted(set(A) - {A[j] for j in B_lo}))
                    T = onp.asarray(Vsub[src]); present = list(src)
                for j in B_hi:                                   # slice upper node
                    ax = present.index(A[j]); T = onp.take(T, T.shape[ax] - 1, axis=ax); present.remove(A[j])
                term = _embed_full(T, present, A)
                for j in B:
                    shp = [1] * (k + 1); shp[j] = n_x
                    term = term * phi[sm[j]].reshape(shp)
                g_sp = g_sp + sign * term
    g = g_sp.copy()
    g[..., 0] = payoff_tensor                                    # inject IC at tau-node 0

    errs = {}
    if check:
        for j in range(k):
            lo = onp.take(g_sp, 0, axis=j)
            errs[f'face_lo_{A[j]}'] = float(onp.max(onp.abs(lo - onp.asarray(Vsub[tuple(a for a in A if a != A[j])]))))
            hi = onp.take(g_sp, n_x - 1, axis=j)
            errs[f'face_hi_{A[j]}'] = float(onp.max(onp.abs(hi - onp.take(onp.asarray(asymp_tensor), n_x - 1, axis=j))))
        errs['ic'] = float(onp.max(onp.abs(onp.take(g, 0, axis=k) - onp.asarray(payoff_tensor))))
    return g, errs


def solve_level(A, M, sigma, rho, r, q, payoff_fn, asymp_fn, H0, Vsub,
                num_mode=40, tol=1e-9, patience=3, seed=0, verbose=False):
    """Solve one lattice level (asset-set A) by greedy PGD with the transfinite lift.
    `num_mode` is the greedy mode cap; raise `patience` (and `num_mode`) for tighter
    convergence on slowly-separable (diagonal-kink) loads."""
    k = len(A)
    if k == 0:                                                   # corner ODE
        return H0 * onp.exp(-r * onp.asarray(M['tg']))
    g, errs = build_lift(A, M, Vsub, payoff_fn(A), asymp_fn(A))
    bc_list = bc_homogeneous(A, M)
    b_load = -apply_operator(g, operator_terms(A, M, sigma, rho, r, q))
    if verbose:
        print(f"    level {A}: lift face/IC residual max = {max(errs.values()):.2e}")
    w = pgd_greedy(A, M, sigma, rho, r, q, b_load, bc_list,
                   max_modes=num_mode, tol=tol, patience=patience, seed=seed, verbose=verbose)
    return g + w


def nested_price(D, M, sigma, rho, r, q, payoff_fn, asymp_fn, H0,
                 num_mode=40, tol=1e-9, patience=3, verbose=True):
    """Bottom-up sweep over the subset lattice (corner -> singletons -> ... -> full).
    Returns dict A(tuple) -> nodal solution tensor; the full price is V[(0..D-1)]."""
    from itertools import combinations
    V = {(): H0 * onp.exp(-r * onp.asarray(M['tg']))}
    for k in range(1, D + 1):
        for A in combinations(range(D), k):
            t0 = time.time()
            V[A] = solve_level(A, M, sigma, rho, r, q, payoff_fn, asymp_fn, H0, V,
                               num_mode=num_mode, tol=tol, patience=patience, verbose=verbose)
            if verbose:
                print(f"  level |A|={k} A={A}: {time.time()-t0:.2f}s")
    return V


def monolithic_solve(A, M, op_terms, b_full, bc_list):
    """Solve  L u = b_full  (full nodal tensor RHS) with Dirichlet/IC from bc_list."""
    k = len(A)
    dims = [M['n_x']] * k + [M['n_t']]
    Afull = monolithic_operator(op_terms).tolil()
    rhs = onp.asarray(b_full, dtype=float).reshape(-1).copy()
    # global Dirichlet indices
    grids = onp.indices(dims)
    mask = onp.zeros(dims, dtype=bool)
    for d in range(k + 1):
        if len(bc_list[d]) == 0:
            continue
        mask |= onp.isin(grids[d], onp.asarray(bc_list[d]))
    gidx = onp.where(mask.reshape(-1))[0]
    Afull[gidx, :] = 0.0; Afull[gidx, gidx] = 1.0
    rhs[gidx] = 0.0
    u = spla.spsolve(Afull.tocsr().tocsc(), rhs)
    return u.reshape(dims)
