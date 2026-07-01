"""
nd_bs_cp.py
===========================
Fully-SEPARATED (CP / canonical-polyadic) version of the nD Black-Scholes TD solver.
Removes the O(n_x^D) full-tensor memory of nd_bs.py: every field (solution, lift, load,
residual) is stored as a CP factor list and every operation is done on factors, so the
memory is O(D * n_x * rank) and the solver scales to D = 5, 6, ... .

A CP tensor over axes (x_a for a in A, then tau) is a list F = [F_0, ..., F_k] with
F_d of shape (R, n_d); the field is sum_{rho<R} prod_d F_d[rho, :]  (rank R).

Reuses from nd_bs.py the assembly primitives that already work on factor lists:
  interior_terms (1-mode Grams), operator_terms (Kronecker list), rhs_separable
  (contracts a separated RHS against modes), kron_sum_solve, bc_homogeneous,
  build_matrices.  Only the full-tensor pieces are re-implemented here in CP form:
  apply_operator, the residual/greedy loop, the transfinite lift, and the hierarchy.

The one remaining O(n_x^|A|) touch is compressing the PAYOFF (the diagonal-kink IC is
genuinely high-rank): it is formed once per level on the spatial grid and greedily
compressed (cp_from_full). The forward ASYMPTOTE is built analytically in CP form
(rank |A|+1) and never materialised. For production-scale meshes the payoff would use a
cross-approximation that never forms the full grid; here the one-time spatial form is
cheap at the demo meshes.
"""
import numpy as onp

from .nd_bs import (interior_terms, interior_terms_generic,
                       operator_terms, kron_sum_solve, bc_homogeneous, rhs_separable, bcolors)


# --------------------------------------------------------------------------- #
#  CP utilities
# --------------------------------------------------------------------------- #
def cp_rank(F):
    return F[0].shape[0]

def cp_dims(F):
    return [f.shape[1] for f in F]

def cp_zero(dims):
    return [onp.zeros((0, n)) for n in dims]

def cp_neg(F):
    return [-F[0]] + [f for f in F[1:]]

def cp_scale(F, c):
    return [c * F[0]] + [f for f in F[1:]]

def cp_add(*Fs):
    Fs = [F for F in Fs if cp_rank(F) > 0]
    if not Fs:
        return None
    k1 = len(Fs[0])
    return [onp.concatenate([F[d] for F in Fs], axis=0) for d in range(k1)]

def cp_cross_grams(F, G):
    return [F[d] @ G[d].T for d in range(len(F))]      # list of (R_F, R_G)

def cp_inner(F, G):
    if cp_rank(F) == 0 or cp_rank(G) == 0:
        return 0.0
    H = None
    for g in cp_cross_grams(F, G):
        H = g if H is None else H * g
    return float(H.sum())

def cp_norm(F, masks=None):
    if cp_rank(F) == 0:
        return 0.0
    Fm = F if masks is None else [F[d] * masks[d] for d in range(len(F))]
    return onp.sqrt(max(cp_inner(Fm, Fm), 0.0))

def cp_eval(F, idx):
    """Field value at a single multi-index idx (length k+1)."""
    if cp_rank(F) == 0:
        return 0.0
    p = F[0][:, idx[0]].copy()
    for d in range(1, len(F)):
        p = p * F[d][:, idx[d]]
    return float(p.sum())

def cp_to_full(F):
    """Materialise the full tensor (VALIDATION / small-D extraction only)."""
    dims = cp_dims(F)
    out = onp.zeros(dims)
    for rho in range(cp_rank(F)):
        t = F[0][rho]
        for d in range(1, len(F)):
            t = onp.multiply.outer(t, F[d][rho])
        out += t
    return out


def cp_apply_operator(F, op):
    """Apply the Kronecker-sum operator (op = list of (coef, [A_0..A_k] sparse 1D)) to a
    CP field. Returns a CP field of rank (#terms * rank(F)); the coef is folded into axis 0."""
    if cp_rank(F) == 0:
        return cp_zero(cp_dims(F))
    k1 = len(F)
    blocks = [[] for _ in range(k1)]
    for coef, mats in op:
        for d in range(k1):
            fac = (mats[d] @ F[d].T).T            # (R, n_d)
            blocks[d].append(coef * fac if d == 0 else fac)
    return [onp.concatenate(blocks[d], axis=0) for d in range(k1)]


def _rank1_als_fit(target_contract, denom_fun, dims, als_iters, rng):
    """Generic rank-1 ALS used by both cp_round and cp_from_full.
    target_contract(d, v) -> length-n_d numerator vector; denom_fun(d, v) -> scalar."""
    v = [rng.standard_normal(n) for n in dims]
    for _ in range(als_iters):
        for d in range(len(dims)):
            v[d] = target_contract(d, v) / denom_fun(d, v)
    return v


def cp_round(F, tol=1e-8, max_rank=200, als_iters=20, seed=0):
    """Greedy L2 CP recompression: low-rank CP S with ||F - S|| <= tol*||F||."""
    dims = cp_dims(F)
    Fn = cp_norm(F)
    if Fn == 0.0:
        return cp_zero(dims)
    S = cp_zero(dims); rng = onp.random.default_rng(seed)
    for _ in range(max_rank):
        Res = cp_add(F, cp_neg(S)) if cp_rank(S) > 0 else F
        if cp_norm(Res) / Fn < tol:
            break
        def contract(d, v, Res=Res):
            coefs = onp.ones(cp_rank(Res))
            for e in range(len(dims)):
                if e != d:
                    coefs = coefs * (Res[e] @ v[e])
            return Res[d].T @ coefs
        def denom(d, v):
            p = 1.0
            for e in range(len(dims)):
                if e != d:
                    p *= v[e] @ v[e]
            return p
        v = _rank1_als_fit(contract, denom, dims, als_iters, rng)
        blk = [v[d][None, :] for d in range(len(dims))]
        S = cp_add(S, blk) if cp_rank(S) > 0 else blk
    return S


def cp_from_full(T, tol=1e-9, max_rank=200, als_iters=25, seed=0):
    """Greedy L2 CP fit of a FULL tensor T (used once for the payoff IC)."""
    dims = list(T.shape); k1 = len(dims)
    Tn = onp.linalg.norm(T)
    if Tn == 0.0:
        return cp_zero(dims)
    S = cp_zero(dims); R = T.copy(); rng = onp.random.default_rng(seed)
    for _ in range(max_rank):
        if onp.linalg.norm(R) / Tn < tol:
            break
        def contract(d, v, R=R):
            tmp = onp.moveaxis(R, d, 0)
            for e in [e for e in range(k1) if e != d]:
                tmp = onp.tensordot(tmp, v[e], axes=([1], [0]))
            return tmp
        def denom(d, v):
            p = 1.0
            for e in range(k1):
                if e != d:
                    p *= v[e] @ v[e]
            return p
        v = _rank1_als_fit(contract, denom, dims, als_iters, rng)
        outer = v[0]
        for d in range(1, k1):
            outer = onp.multiply.outer(outer, v[d])
        R = R - outer
        blk = [v[d][None, :] for d in range(k1)]
        S = cp_add(S, blk) if cp_rank(S) > 0 else blk
    return S


# --------------------------------------------------------------------------- #
#  Greedy PGD in CP form
# --------------------------------------------------------------------------- #
def pgd_greedy_op(dims, op, b_cp, bc_list, max_modes=60, als_iters=25,
                  tol=1e-9, patience=3, res_round_rank=None, res_round_tol=1e-7,
                  seed=0, verbose=False):
    """Operator-driven greedy CP PGD: solve (sum_i coef_i prod A_i) u = b_cp, homogeneous BC.
    Generic over the axis layout -- `op` is an operator term-list, `dims`/`bc_list` describe
    the axes. `res_round_rank` (if set) compresses the residual each enrichment -- essential
    at high D where the operator has O(D^2) terms so the raw residual rank (#terms*rank(u))
    explodes. Used by both the fixed-sigma and the sigma-parametric front-ends."""
    nax = len(dims)
    masks = []
    for d in range(nax):
        m = onp.ones(dims[d])
        if len(bc_list[d]):
            m[onp.asarray(bc_list[d], dtype=int)] = 0.0
        masks.append(m)
    if res_round_rank is None:
        res_round_rank = 60                                   # always bound the residual (fast)
    u = cp_zero(dims)
    res = [f.copy() for f in b_cp]                            # residual, maintained INCREMENTALLY
    bnorm = cp_norm(b_cp, masks) + 1e-30
    rng = onp.random.default_rng(seed); best = onp.inf; stale = 0
    for enr in range(max_modes):
        rnorm = cp_norm(res, masks) / bnorm
        if verbose and (enr % 10 == 0 or rnorm < tol):
            print(f"      enrich {enr:2d}: rel residual = {rnorm:.3e}  (res rank {cp_rank(res)})")
        if rnorm < tol:
            break
        stale = 0 if rnorm < best * (1 - 1e-4) else stale + 1
        best = min(best, rnorm)
        if stale >= patience:
            if verbose:
                print(f"      stop at {enr} modes (rel res {rnorm:.3e})")
            break
        v = [rng.standard_normal(n) for n in dims]
        for _ in range(als_iters):
            for active in range(nax):
                U1 = [v[d][None, :] for d in range(nax)]
                terms = interior_terms_generic(active, op, U1)
                Q = rhs_separable(res, U1, active)
                v[active] = kron_sum_solve(terms, Q, bc_list[active], 1)[:, 0]
            nrm = [onp.linalg.norm(vd) for vd in v]
            gm = onp.prod(nrm) ** (1.0 / len(v))
            for d in range(len(v)):
                if nrm[d] > 0:
                    v[d] = gm * v[d] / nrm[d]
        blk = [v[d][None, :] for d in range(nax)]
        u = cp_add(u, blk) if cp_rank(u) > 0 else blk
        # incremental residual update: res <- res - L v  (L v has rank n_terms only), then bound
        res = cp_add(res, cp_neg(cp_apply_operator(blk, op)))
        if cp_rank(res) > res_round_rank:
            res = cp_round(res, tol=res_round_tol, max_rank=res_round_rank)
    return u if cp_rank(u) > 0 else cp_zero(dims)


def pgd_greedy_cp(A, M, sigma, rho, r, q, b_cp, bc_list, max_modes=60, als_iters=25,
                  tol=1e-9, patience=3, seed=0, verbose=False):
    """Greedy rank-enrichment PGD for L u = b (all in CP form), homogeneous BC.
    Returns the solution u as a CP field of rank = #modes added."""
    k = len(A); dims = [M['n_x']] * k + [M['n_t']]
    op = operator_terms(A, M, sigma, rho, r, q)
    masks = []
    for d in range(k + 1):
        m = onp.ones(dims[d])
        if len(bc_list[d]):
            m[onp.asarray(bc_list[d], dtype=int)] = 0.0
        masks.append(m)
    u = cp_zero(dims)
    bnorm = cp_norm(b_cp, masks) + 1e-30
    rng = onp.random.default_rng(seed); best = onp.inf; stale = 0
    for enr in range(max_modes):
        res = cp_add(b_cp, cp_neg(cp_apply_operator(u, op))) if cp_rank(u) > 0 else b_cp
        rnorm = cp_norm(res, masks) / bnorm
        if verbose and (enr % 5 == 0 or rnorm < tol):
            print(f"      enrich {enr:2d}: rel residual = {rnorm:.3e}  (res rank {cp_rank(res)})")
        if rnorm < tol:
            break
        stale = 0 if rnorm < best * (1 - 1e-4) else stale + 1
        best = min(best, rnorm)
        if stale >= patience:
            if verbose:
                print(f"      stop at {enr} modes (rel res {rnorm:.3e})")
            break
        v = [rng.standard_normal(n) for n in dims]
        for _ in range(als_iters):
            for active in range(k + 1):
                U1 = [v[d][None, :] for d in range(k + 1)]
                terms = interior_terms(active, A, U1, M, sigma, rho, r, q)
                Q = rhs_separable(res, U1, active)           # res factors as separated RHS
                v[active] = kron_sum_solve(terms, Q, bc_list[active], 1)[:, 0]
            nrm = [onp.linalg.norm(vd) for vd in v]
            gm = onp.prod(nrm) ** (1.0 / len(v))
            for d in range(len(v)):
                if nrm[d] > 0:
                    v[d] = gm * v[d] / nrm[d]
        blk = [v[d][None, :] for d in range(k + 1)]
        u = cp_add(u, blk) if cp_rank(u) > 0 else blk
    return u if cp_rank(u) > 0 else cp_zero(dims)


# --------------------------------------------------------------------------- #
#  Transfinite (Boolean-sum) lift in CP form
# --------------------------------------------------------------------------- #
def _cp_slice(F, axis, node):
    """Slice CP field at (axis, node): fold the column into the first remaining factor."""
    col = F[axis][:, node]                                    # (R,)
    rest = [F[d] for d in range(len(F)) if d != axis]
    rest = [rest[0] * col[:, None]] + rest[1:]
    return rest

def _cp_insert(F, pos, vec):
    """Insert a constant-along-axis factor (each rank row = vec) at position pos."""
    R = cp_rank(F)
    fac = onp.broadcast_to(vec, (R, vec.shape[0])).copy()
    return F[:pos] + [fac] + F[pos:]


def build_lift_cp(A, M, Vsub, payoff_cp, asymp_cp, round_tol=1e-7, round_rank=200):
    """CP transfinite lift (derivation B.12.4) built from CP trace data:
    Vsub[A\\{..}] (lower levels), asymp_cp (analytic upper face), payoff_cp (IC).
    Returns (g_cp, errs)."""
    from itertools import combinations, product as iproduct
    k = len(A); nx = M['n_x']; nt = M['n_t']
    xg = onp.asarray(M['xg']); xmin, xmax = xg[0], xg[-1]
    phi = {'lo': (xmax - xg) / (xmax - xmin), 'hi': (xg - xmin) / (xmax - xmin)}
    e0 = onp.zeros(nt); e0[0] = 1.0
    dims = [nx] * k + [nt]

    # accumulate the 3^|A|-1 inclusion-exclusion blocks with INCREMENTAL rounding so the
    # working rank stays O(round_rank) instead of O(3^|A| * sub-rank).
    g_sp = cp_zero(dims)
    for rr in range(1, k + 1):
        for B in combinations(range(k), rr):
            sign = (-1.0) ** (rr + 1)
            for s in iproduct(('lo', 'hi'), repeat=rr):
                sm = dict(zip(B, s))
                B_lo = [j for j in B if sm[j] == 'lo']
                B_hi = [j for j in B if sm[j] == 'hi']
                if not B_lo:
                    F = [f.copy() for f in asymp_cp]; present = list(A)
                else:
                    src = tuple(sorted(set(A) - {A[j] for j in B_lo}))
                    F = [f.copy() for f in Vsub[src]]; present = list(src)
                for j in B_hi:                                # slice upper node
                    ax = present.index(A[j]); F = _cp_slice(F, ax, nx - 1); present.remove(A[j])
                for pos, a in enumerate(A):                   # embed missing axes + blends
                    if a not in present:
                        loc = A.index(a)
                        vec = phi[sm[loc]] if loc in B else onp.ones(nx)
                        F = _cp_insert(F, pos, vec)
                blk = cp_scale(F, sign)
                g_sp = cp_add(g_sp, blk) if cp_rank(g_sp) > 0 else blk
                if cp_rank(g_sp) > 3 * round_rank:
                    g_sp = cp_round(g_sp, tol=round_tol, max_rank=round_rank)
    g_sp = cp_round(g_sp, tol=round_tol, max_rank=round_rank)

    # IC injection: g = g_sp + (payoff (x) e0) - (g_sp|_{tau=0} (x) e0)
    payoff_t = payoff_cp + [onp.broadcast_to(e0, (cp_rank(payoff_cp), nt)).copy()]   # add tau factor
    gsp_tau0 = _cp_slice(g_sp, k, 0)                          # spatial CP over A
    gsp_tau0_t = gsp_tau0 + [onp.broadcast_to(e0, (cp_rank(gsp_tau0), nt)).copy()]
    g = cp_add(g_sp, payoff_t, cp_neg(gsp_tau0_t))
    g = cp_round(g, tol=round_tol, max_rank=round_rank)

    # residual diagnostics (cheap, CP inner products)
    errs = {}
    for j in range(k):
        Vlo = Vsub[tuple(a for a in A if a != A[j])]
        face = _cp_slice(g_sp, j, 0)
        diff = cp_add(face, cp_neg(Vlo))
        errs[f'face_lo_{A[j]}'] = cp_norm(diff) / (cp_norm(Vlo) + 1e-30)
    return g, errs


# --------------------------------------------------------------------------- #
#  One level + the nested hierarchy, all in CP
# --------------------------------------------------------------------------- #
def solve_level_cp(A, M, sigma, rho, r, q, payoff_full_fn, asymp_cp_fn, H0, Vsub,
                   num_mode=60, tol=1e-9, patience=3, round_tol=1e-7, round_rank=200,
                   payoff_tol=1e-8, seed=0, verbose=False):
    k = len(A)
    if k == 0:                                                # corner ODE, rank-1 in tau
        return [(H0 * onp.exp(-r * onp.asarray(M['tg'])))[None, :]]
    payoff_cp = cp_from_full(payoff_full_fn(A), tol=payoff_tol, max_rank=round_rank)
    asymp_cp = asymp_cp_fn(A)
    g, errs = build_lift_cp(A, M, Vsub, payoff_cp, asymp_cp, round_tol, round_rank)
    op = operator_terms(A, M, sigma, rho, r, q)
    b = cp_neg(cp_apply_operator(g, op))
    bc = bc_homogeneous(A, M)
    if verbose:
        print(f"    level {A}: lift rank {cp_rank(g)}, payoff rank {cp_rank(payoff_cp)}, "
              f"face residual max {max(errs.values()):.2e}")
    w = pgd_greedy_cp(A, M, sigma, rho, r, q, b, bc, max_modes=num_mode, tol=tol,
                      patience=patience, seed=seed, verbose=verbose)
    V = cp_add(g, w) if cp_rank(w) > 0 else g
    V = cp_round(V, tol=round_tol, max_rank=round_rank)       # bound rank up the lattice
    return V


def nested_price_cp(D, M, sigma, rho, r, q, payoff_full_fn, asymp_cp_fn, H0,
                    num_mode=60, tol=1e-9, patience=3, round_tol=1e-7, round_rank=200,
                    verbose=True):
    """Bottom-up subset-lattice sweep, all fields CP. Returns dict A->CP; price = V[(0..D-1)]."""
    import time
    from itertools import combinations
    V = {(): [(H0 * onp.exp(-r * onp.asarray(M['tg'])))[None, :]]}
    for k in range(1, D + 1):
        for A in combinations(range(D), k):
            t0 = time.time()
            V[A] = solve_level_cp(A, M, sigma, rho, r, q, payoff_full_fn, asymp_cp_fn, H0, V,
                                  num_mode=num_mode, tol=tol, patience=patience,
                                  round_tol=round_tol, round_rank=round_rank, verbose=verbose)
            if verbose:
                print(f"  level |A|={k} A={A}: rank {cp_rank(V[A])}, {time.time()-t0:.1f}s")
    return V


# --------------------------------------------------------------------------- #
#  Basket helpers (payoff full-fn, analytic asymptote in CP)
# --------------------------------------------------------------------------- #
def basket_payoff_full(M, w, K):
    S = onp.exp(onp.asarray(M['xg']))
    def fn(A):
        k = len(A); basket = onp.zeros((M['n_x'],) * k)
        for j, a in enumerate(A):
            shp = [1] * k; shp[j] = M['n_x']; basket = basket + (w[a] * S).reshape(shp)
        return onp.maximum(basket - K, 0.0)
    return fn

def basket_asymp_cp(M, w, K, r, q):
    S = onp.exp(onp.asarray(M['xg'])); tn = onp.asarray(M['tg']); nx = M['n_x']; nt = M['n_t']
    def fn(A):
        k = len(A)
        rows_x = [[] for _ in range(k)]; rows_t = []
        for j, a in enumerate(A):                              # term w_a S_a e^{-q_a tau}
            for d in range(k):
                rows_x[d].append(w[a] * S if d == j else onp.ones(nx))
            rows_t.append(onp.exp(-q[a] * tn))
        for d in range(k):                                     # term -K e^{-r tau}
            rows_x[d].append(onp.ones(nx))
        rows_t.append(-K * onp.exp(-r * tn))
        return [onp.array(rows_x[d]) for d in range(k)] + [onp.array(rows_t)]
    return fn
