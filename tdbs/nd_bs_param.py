"""
nd_bs_param.py
===========================
sigma-PARAMETRIC multi-asset Black-Scholes TD solver: the volatilities sigma_1..sigma_D
are now PARAMETER AXES (not fixed scalars). One CP solve over the whole sigma-hypercube
yields a stored separated solution V(x_1..x_D, sigma_1..sigma_D, tau); pricing at any
sigma-vector is then an O(rank) CP evaluation -- no re-solve. This is the parametric
payoff of TD: solve once, infer for arbitrarily many sigma-samples almost for free, vs
Monte-Carlo which re-simulates per sample.

Axes for a level with alive asset-set A (|A|=k), canonical order:
    [ x_a : a in A ]   (positions 0..k-1, Dirichlet BC)
    [ s_a : a in A ]   (positions k..2k-1, NO BC -- parameters)
    [ tau ]            (position 2k, IC at node 0)

The operator is the sigma-parametric form of derivation B.6-B.8: every coefficient that
was a fixed scalar sigma_a^p becomes a sigma^p-WEIGHTED parameter mass on the s_a axis
(p=2 for diffusion / advection-sigma^2, p=1 for the cross term on BOTH its s-axes, p=0
plain mass elsewhere). The lower-boundary faces are sigma_a-INDEPENDENT (B.12.1), so the
lift embeds the s-axes as constant-1 / keeps the sub-solution's s-factors.

Built on the CP core in nd_bs_cp.py (operator-driven pgd_greedy_op + cp utilities) and the
generic interior_terms_generic in nd_bs.py.
"""
import numpy as onp

from .generate_mesh import uniform_mesh_new
from .fem import get_FEM_shape_fun_dict, get_matrix_x, get_matrix_t, bcoo_2_csr
from .bs_assembly import get_matrix_weighted_mass
from . import nd_bs_cp as cpm
from .nd_bs_cp import (cp_rank, cp_zero, cp_neg, cp_add, cp_scale, cp_norm, cp_eval,
                          cp_apply_operator, cp_round, cp_from_full, cp_to_full, pgd_greedy_op)


# --------------------------------------------------------------------------- #
#  Matrices: x (mass/stiff/advection), sigma (plain / sigma^1 / sigma^2 mass), tau
# --------------------------------------------------------------------------- #


_ELEM_TYPE = {1: 'D1LN2N', 2: 'D1LQ3N'}      # polynomial order -> Lagrange element type


def _axis_orders(order):
    """Normalise `order` to a per-axis dict {'x':., 's':., 't':.} (int -> same on all axes)."""
    if isinstance(order, dict):
        return {k: int(order.get(k, 1)) for k in ('x', 's', 't')}
    return {k: int(order) for k in ('x', 's', 't')}


def build_matrices_param(D, x_dom, sig_dom, t_dom, nelem_x, nelem_s, nelem_t,
                         Gauss=8, order=1):
    """`order` selects the Lagrange element order per axis: 1 = P1 (linear), 2 = P2
    (quadratic).  Pass an int (same on every axis) or a dict {'x':.,'s':.,'t':.} -- e.g.
    P2 on the price axes for O(h^2) basket Deltas.  The per-axis mesh (node grid,
    connectivity, element type) is returned so `price_at`/`greeks_at` can reconstruct the
    FE field and its shape-gradient B off-grid."""
    orders = _axis_orders(order); et = {k: _ELEM_TYPE[o] for k, o in orders.items()}
    Lx = x_dom[1] - x_dom[0]; Ls = sig_dom[1] - sig_dom[0]; Tt = t_dom[1] - t_dom[0]
    x, En_x = uniform_mesh_new(Lx, nelem_x, orders['x']); x = onp.array(x) + x_dom[0]
    s, En_s = uniform_mesh_new(Ls, nelem_s, orders['s']); s = onp.array(s) + sig_dom[0]
    t, En_t = uniform_mesh_new(Tt, nelem_t, orders['t']); t = onp.array(t) + t_dom[0]
    n_x, n_s, n_t = x.shape[0], s.shape[0], t.shape[0]      # = order*nelem + 1 per axis
    # Lagrange finite-element shape functions (per-axis element type via the dict `et`)
    inp = {'coor': {'x': x, 's': s, 't': t}, 'Elem_nodes': {'x': En_x, 's': En_s, 't': En_t}}
    sfd = get_FEM_shape_fun_dict(inp, Gauss, et)
    gd = lambda k: (sfd[k]['N_fe'], sfd[k]['Grad_N'], sfd[k]['JxW'], sfd[k]['Elem_nodes'])
    Nx, Gx, Jx, Ex = gd('x'); Ns, Gs, Js, Es = gd('s'); Nt, Gt, Jt, Et = gd('t')
    (Kbb, Knn) = get_matrix_x(x, En_x, Nx, Gx, Jx, Ex, Gauss, et['x'])
    (Knb, _) = get_matrix_t(x, En_x, Nx, Gx, Jx, Ex, Gauss, et['x'])
    Mx = bcoo_2_csr(Knn); Kx = bcoo_2_csr(Kbb); NBx = bcoo_2_csr(Knb); BNx = NBx.T.tocsr()
    (_, Mssn) = get_matrix_x(s, En_s, Ns, Gs, Js, Es, Gauss, et['s'])
    Ms = bcoo_2_csr(Mssn)
    Ws1 = bcoo_2_csr(get_matrix_weighted_mass(s, En_s, Ns, Js, Es, Gauss, et['s'], power=1))
    Ws2 = bcoo_2_csr(get_matrix_weighted_mass(s, En_s, Ns, Js, Es, Gauss, et['s'], power=2))
    (Ptb, Mtt) = get_matrix_t(t, En_t, Nt, Gt, Jt, Et, Gauss, et['t'])
    Mt = bcoo_2_csr(Mtt); PBt = bcoo_2_csr(Ptb)
    return dict(xg=x.reshape(-1), sg=s.reshape(-1), tg=t.reshape(-1), n_x=n_x, n_s=n_s, n_t=n_t,
                Mx=[Mx] * D, Kx=[Kx] * D, NBx=[NBx] * D, BNx=[BNx] * D,
                Ms=Ms, Ws1=Ws1, Ws2=Ws2, Mt=Mt, PBt=PBt,
                Ex=onp.asarray(En_x), Es=onp.asarray(En_s), Et=onp.asarray(En_t), et=et)


# --------------------------------------------------------------------------- #
#  Parametric operator (sigma as axes; coefficients are pure scalars)
# --------------------------------------------------------------------------- #
def operator_terms_param(A, M, rho, r, q):
    """Term-list over axes [x_0..x_{k-1}, s_0..s_{k-1}, tau]; sigma^p enters via the
    sigma^p-weighted parameter mass on the s-axes (not as scalars)."""
    k = len(A)
    Mx, Kx, NBx, BNx = M['Mx'], M['Kx'], M['NBx'], M['BNx']
    Ms, Ws1, Ws2, Mt, PBt = M['Ms'], M['Ws1'], M['Ws2'], M['Mt'], M['PBt']
    def base(tau_mat):
        return [Mx[A[l]] for l in range(k)] + [Ms for _ in range(k)] + [tau_mat]
    out = [(1.0, base(PBt)), (r, base(Mt))]                       # time, reaction
    for g in range(k):                                            # self-diffusion: x_g K, s_g sigma^2
        m = base(Mt); m[g] = Kx[A[g]]; m[k + g] = Ws2; out.append((0.5, m))
    for g in range(k):                                            # advection: (r-q) piece + sigma^2 piece
        m = base(Mt); m[g] = NBx[A[g]]; out.append((-(r - q[A[g]]), m))
        m2 = base(Mt); m2[g] = NBx[A[g]]; m2[k + g] = Ws2; out.append((0.5, m2))
    for a in range(k):                                            # cross: x_a BN, x_b NB, s_a/s_b sigma^1
        for b in range(k):
            if a == b:
                continue
            m = base(Mt); m[a] = BNx[A[a]]; m[b] = NBx[A[b]]; m[k + a] = Ws1; m[k + b] = Ws1
            out.append((0.5 * rho[A[a], A[b]], m))
    return out


def dims_param(A, M):
    k = len(A); return [M['n_x']] * k + [M['n_s']] * k + [M['n_t']]

def bc_param(A, M):
    k = len(A)
    bc = [onp.array([0, M['n_x'] - 1], dtype=int) for _ in range(k)]    # x Dirichlet
    bc += [onp.array([], dtype=int) for _ in range(k)]                  # sigma: none
    bc += [onp.array([0], dtype=int)]                                   # tau IC
    return bc


# --------------------------------------------------------------------------- #
#  Label-based CP embedding for the sigma-aware transfinite lift
# --------------------------------------------------------------------------- #
def _labels(A):
    return [('x', a) for a in A] + [('s', a) for a in A] + [('t',)]

def _nsize(M):
    return {'x': M['n_x'], 's': M['n_s'], 't': M['n_t']}

def _slice_label(F, labels, lab, node):
    ax = labels.index(lab); col = F[ax][:, node]
    newF = [F[d] for d in range(len(F)) if d != ax]
    newlab = [l for l in labels if l != lab]
    newF[0] = newF[0] * col[:, None]                                   # fold scalar into first remaining factor
    return newF, newlab

def _embed(F, labels, tgt, M):
    R = cp_rank(F); ns = _nsize(M); by = dict(zip(labels, F))
    out = []
    for lab in tgt:
        if lab in by:
            out.append(by[lab])
        else:
            out.append(onp.ones((R, ns[lab[0]])))
    return out


def build_lift_param(A, M, Vsub, payoff_cp, asymp_cp, round_tol=1e-6, round_rank=120):
    """sigma-parametric transfinite lift. payoff_cp/asymp_cp are sigma-INDEPENDENT (over
    x-axes [+ tau for asymp]); lower faces (Vsub) carry their own sigma-factors."""
    from itertools import combinations, product as iproduct
    k = len(A); nx = M['n_x']; nt = M['n_t']
    xg = onp.asarray(M['xg']); xmin, xmax = xg[0], xg[-1]
    phi = {'lo': (xmax - xg) / (xmax - xmin), 'hi': (xg - xmin) / (xmax - xmin)}
    e0 = onp.zeros(nt); e0[0] = 1.0
    tgt = _labels(A); dims = dims_param(A, M)
    asymp_lab = [('x', a) for a in A] + [('t',)]
    payoff_lab = [('x', a) for a in A]

    g_sp = cp_zero(dims)
    for rr in range(1, k + 1):
        for B in combinations(range(k), rr):
            sign = (-1.0) ** (rr + 1)
            for s in iproduct(('lo', 'hi'), repeat=rr):
                sm = dict(zip(B, s))
                lo = {A[j] for j in B if sm[j] == 'lo'}
                hi = {A[j] for j in B if sm[j] == 'hi'}
                if not lo:
                    F = [f.copy() for f in asymp_cp]; lab = list(asymp_lab)
                else:
                    src = tuple(sorted(set(A) - lo)); F = [f.copy() for f in Vsub[src]]; lab = _labels(src)
                for a in sorted(hi):
                    F, lab = _slice_label(F, lab, ('x', a), nx - 1)
                F = _embed(F, lab, tgt, M)
                for j in B:                                          # blends on the B x-axes
                    ax = tgt.index(('x', A[j])); F[ax] = F[ax] * phi[sm[j]]
                blk = cp_scale(F, sign)
                g_sp = cp_add(g_sp, blk) if cp_rank(g_sp) > 0 else blk
                if cp_rank(g_sp) > 3 * round_rank:
                    g_sp = cp_round(g_sp, tol=round_tol, max_rank=round_rank)
    g_sp = cp_round(g_sp, tol=round_tol, max_rank=round_rank)

    # IC injection at tau=0: g = g_sp + payoff (x) 1_sigma (x) e0  -  g_sp|_{tau0} (x) e0
    Rp = cp_rank(payoff_cp)
    pay_t = _embed(payoff_cp, payoff_lab, tgt[:-1], M) + [onp.broadcast_to(e0, (Rp, nt)).copy()]
    gsp0, gsp0_lab = _slice_label(g_sp, tgt, ('t',), 0)
    gsp0_t = _embed(gsp0, gsp0_lab, tgt[:-1], M) + [onp.broadcast_to(e0, (cp_rank(gsp0), nt)).copy()]
    g_pre = cp_add(g_sp, pay_t, cp_neg(gsp0_t))               # full-rank lift before final compression
    g = cp_round(g_pre, tol=round_tol, max_rank=round_rank)

    errs = {}
    # relative L2 error introduced by compressing the lift g_A to its stored CP rank
    g_pre_norm = cp_norm(g_pre)
    errs['lift_round'] = cp_norm(cp_add(g_pre, cp_neg(g))) / (g_pre_norm + 1e-30)
    for j in range(k):
        Vlo = Vsub[tuple(a for a in A if a != A[j])]
        face, flab = _slice_label(g_sp, tgt, ('x', A[j]), 0)
        # reorder face to Vlo's label order for comparison
        face = _embed(face, flab, _labels(tuple(a for a in A if a != A[j])), M)
        diff = cp_add(face, cp_neg(Vlo))
        errs[f'face_lo_{A[j]}'] = cp_norm(diff) / (cp_norm(Vlo) + 1e-30)
    return g, errs


# --------------------------------------------------------------------------- #
#  Level solve + nested hierarchy (parametric)
# --------------------------------------------------------------------------- #
def solve_level_param(A, M, rho, r, q, payoff_full_fn, asymp_cp_fn, H0, Vsub,
                      num_mode=60, tol=1e-8, patience=3, round_tol=1e-6, round_rank=120,
                      payoff_tol=1e-7, res_round_rank=None, seed=0, verbose=False, solver_fn=None):
    k = len(A)
    if k == 0:
        return [(H0 * onp.exp(-r * onp.asarray(M['tg'])))[None, :]]
    payoff_dense = payoff_full_fn(A)                          # full IC tensor over the |A| price axes
    payoff_cp = cp_from_full(payoff_dense, tol=payoff_tol, max_rank=round_rank)
    asymp_cp = asymp_cp_fn(A)
    g, errs = build_lift_param(A, M, Vsub, payoff_cp, asymp_cp, round_tol, round_rank)
    op = operator_terms_param(A, M, rho, r, q)
    b = cp_neg(cp_apply_operator(g, op))
    bc = bc_param(A, M); dims = dims_param(A, M)
    if verbose:
        print(f"    level {A}: axes={len(dims)}, lift rank {cp_rank(g)}, "
              f"payoff rank {cp_rank(payoff_cp)}, face res max {max(errs.values()):.2e}")
    if solver_fn is not None:                                 # e.g. joint ALS (cuDSS)
        w = solver_fn(dims, op, b, bc, num_mode, seed, verbose)
    else:                                                     # default greedy PGD
        w = pgd_greedy_op(dims, op, b, bc, max_modes=num_mode, tol=tol, patience=patience,
                          res_round_rank=res_round_rank, seed=seed, verbose=verbose)
    V = cp_add(g, w) if cp_rank(w) > 0 else g
    return cp_round(V, tol=round_tol, max_rank=round_rank)


def nested_price_param(D, M, rho, r, q, payoff_full_fn, asymp_cp_fn, H0,
                       num_mode=60, tol=1e-8, patience=3, round_tol=1e-6, round_rank=120,
                       res_round_rank=None, verbose=True, solver_fn=None):
    import time
    from itertools import combinations
    V = {(): [(H0 * onp.exp(-r * onp.asarray(M['tg'])))[None, :]]}
    for k in range(1, D + 1):
        for A in combinations(range(D), k):
            t0 = time.time()
            V[A] = solve_level_param(A, M, rho, r, q, payoff_full_fn, asymp_cp_fn, H0, V, solver_fn=solver_fn,
                                     num_mode=num_mode, tol=tol, patience=patience,
                                     round_tol=round_tol, round_rank=round_rank,
                                     res_round_rank=res_round_rank, verbose=verbose)
            if verbose:
                print(f"  level |A|={k} A={A}: rank {cp_rank(V[A])}, {time.time()-t0:.1f}s")
    return V


# --------------------------------------------------------------------------- #
#  Basket payoff (sigma-independent): spatial full-fn + analytic CP asymptote
# --------------------------------------------------------------------------- #
def basket_payoff_full(M, w, K):
    S = onp.exp(onp.asarray(M['xg']))
    def fn(A):
        k = len(A); bk = onp.zeros((M['n_x'],) * k)
        for j, a in enumerate(A):
            shp = [1] * k; shp[j] = M['n_x']; bk = bk + (w[a] * S).reshape(shp)
        return onp.maximum(bk - K, 0.0)
    return fn

def basket_asymp_cp(M, w, K, r, q):
    """CP asymptote over [x_a : a in A] + [tau] (sigma-independent), rank |A|+1."""
    S = onp.exp(onp.asarray(M['xg'])); tn = onp.asarray(M['tg']); nx = M['n_x']; nt = M['n_t']
    def fn(A):
        k = len(A); rows_x = [[] for _ in range(k)]; rows_t = []
        for j, a in enumerate(A):
            for d in range(k):
                rows_x[d].append(w[a] * S if d == j else onp.ones(nx))
            rows_t.append(onp.exp(-q[a] * tn))
        for d in range(k):
            rows_x[d].append(onp.ones(nx))
        rows_t.append(-K * onp.exp(-r * tn))
        return [onp.array(rows_x[d]) for d in range(k)] + [onp.array(rows_t)]
    return fn


# --------------------------------------------------------------------------- #
#  High-D single-level path (the 2^D hierarchy & 3^D Boolean lift do NOT scale):
#  minimal rank-(|A|+1) lift (asymptote + IC), one solve, residual rounding.
# --------------------------------------------------------------------------- #
def minimal_lift_param(A, M, payoff_cp, asymp_cp):
    """Cheap lift = asymptote (sigma-independent) with the payoff IC injected at tau=0.
    No 3^|A| Boolean sum -- matches the upper asymptote & IC but not the lower faces
    exactly (boundary-approximate; for high-D speed/storage demos)."""
    k = len(A); nt = M['n_t']; e0 = onp.zeros(nt); e0[0] = 1.0
    tgt = _labels(A); asymp_lab = [('x', a) for a in A] + [('t',)]; payoff_lab = [('x', a) for a in A]
    g_sp = _embed([f.copy() for f in asymp_cp], asymp_lab, tgt, M)
    Rp = cp_rank(payoff_cp)
    pay_t = _embed(payoff_cp, payoff_lab, tgt[:-1], M) + [onp.broadcast_to(e0, (Rp, nt)).copy()]
    gsp0, gsp0_lab = _slice_label(g_sp, tgt, ('t',), 0)
    gsp0_t = _embed(gsp0, gsp0_lab, tgt[:-1], M) + [onp.broadcast_to(e0, (cp_rank(gsp0), nt)).copy()]
    return cp_add(g_sp, pay_t, cp_neg(gsp0_t))


def solve_single_param(D, M, rho, r, q, payoff_cp, asymp_cp, num_mode=30, tol=1e-6,
                       patience=2, round_tol=1e-5, round_rank=60, res_round_rank=80,
                       seed=0, verbose=True):
    """ONE top-level sigma-parametric solve (no nested hierarchy) for the full asset set,
    with the minimal lift and residual rounding. Returns the stored CP solution."""
    A = tuple(range(D))
    g = minimal_lift_param(A, M, payoff_cp, asymp_cp)
    g = cp_round(g, tol=round_tol, max_rank=round_rank)
    op = operator_terms_param(A, M, rho, r, q)
    b = cp_neg(cp_apply_operator(g, op))
    bc = bc_param(A, M); dims = dims_param(A, M)
    if verbose:
        print(f"  single-level D={D}: axes={len(dims)}, op terms={len(op)}, lift rank {cp_rank(g)}")
    w = pgd_greedy_op(dims, op, b, bc, max_modes=num_mode, tol=tol, patience=patience,
                      res_round_rank=res_round_rank, seed=seed, verbose=verbose)
    V = cp_add(g, w) if cp_rank(w) > 0 else g
    return cp_round(V, tol=round_tol, max_rank=round_rank)


# --------------------------------------------------------------------------- #
#  Portfolio-of-calls payoff (SEPARABLE: rank D, no n_x^D tensor; sigma-dependent;
#  analytic reference = sum_a w_a BS(S_a, sigma_a)). The basket diagonal kink is
#  non-separable and does not scale to high D -- this is the validatable high-D case.
# --------------------------------------------------------------------------- #
def portfolio_payoff_cp(M, w, Kvec):
    """sum_a w_a max(S_a - K_a, 0) as a rank-D CP over [x_a : a in A] (no tau)."""
    S = onp.exp(onp.asarray(M['xg'])); nx = M['n_x']
    def fn(A):
        k = len(A); rows = [[] for _ in range(k)]
        for j, a in enumerate(A):
            for d in range(k):
                rows[d].append(w[a] * onp.maximum(S - Kvec[a], 0.0) if d == j else onp.ones(nx))
        return [onp.array(rows[d]) for d in range(k)]
    return fn

def portfolio_asymp_cp(M, w, Kvec, r, q):
    """Upper asymptote sum_a w_a (S_a e^{-q_a tau} - K_a e^{-r tau}) over [x_a]+[tau]."""
    S = onp.exp(onp.asarray(M['xg'])); tn = onp.asarray(M['tg']); nx = M['n_x']; nt = M['n_t']
    def fn(A):
        k = len(A); rows_x = [[] for _ in range(k)]; rows_t = []
        for j, a in enumerate(A):                                  # w_a S_a e^{-q_a tau}
            for d in range(k):
                rows_x[d].append(w[a] * S if d == j else onp.ones(nx))
            rows_t.append(onp.exp(-q[a] * tn))
            for d in range(k):                                     # -w_a K_a e^{-r tau}
                rows_x[d].append(-w[a] * Kvec[a] * onp.ones(nx) if d == j else onp.ones(nx))
            rows_t.append(onp.exp(-r * tn))
        return [onp.array(rows_x[d]) for d in range(k)] + [onp.array(rows_t)]
    return fn


def price_at(V_top, M, A, x_idx, s_idx, t_idx):
    """CP evaluation of the parametric solution at given (x-node, sigma-node, tau-node) indices."""
    return cp_eval(V_top, tuple(x_idx) + tuple(s_idx) + (t_idx,))
