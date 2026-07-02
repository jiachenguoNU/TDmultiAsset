"""High-level D=2 parametric basket-call pricer: build the C-HiDeNN operators and solve
the basket PDE as a 5-axis (x0, x1, sigma0, sigma1, tau) separated representation via the
nested-hierarchy joint fixed-rank ALS.

`solve_basket_2d` returns the CP solution plus the mesh and the C-HiDeNN evaluation dict.

Sigma padding
-------------
sigma is a free PARAMETER axis with no boundary condition, so the Galerkin representation
degrades at the sigma-mesh edges. `sig_pad` widens the MESH by (about) `sig_pad` on each
side of the region of interest `sig_dom` -- by a whole number of elements, preserving the
spacing h_s -- so the `sig_dom` endpoints become interior nodes and that boundary
degradation is pushed outside the region you actually price in.
"""
import numpy as np

from .nd_bs_param import (build_matrices_param, nested_price_param,
                          basket_payoff_full, basket_asymp_cp)
from .joint_als import als_joint_op


def solve_basket_2d(K=1.0, r=0.05, T=1.0, w=None, q=None, rho=None,
                    x_dom=(-5.0, 2.0), sig_dom=(0.15, 0.40),
                    Nx=29, Ns=29, Nt=29, rank1=16, rank2=32, niter=60,
                    sig_pad=0.0, round_tol=1e-7, seed=0, order=1):
    """Solve the D=2 basket call. Returns a dict with:
        V        : the 2-asset solution in CP form (axis order x0,x1,s0,s1,tau)
        M        : mesh-matrix dict (node grids 'xg','sg','tg', connectivity + element type
                   for tdbs.price_at / tdbs.greeks_at)
        sig_dom  : the region of interest;  sig_mesh : the padded mesh domain actually solved
        w, q, rho, K, r, T, D : the market/option parameters used

    `order` selects the Lagrange element order (1 = P1 linear, 2 = P2 quadratic); pass an int
    for all axes or a dict {'x':.,'s':.,'t':.} for per-axis control.  P2 on the price/vol axes
    makes the shape-gradient B element-linear, so `greeks_at` recovers Delta/Vega at O(h^2).
    Note each P2 axis roughly doubles its node count (and the joint-ALS solve cost).
    """
    D = 2
    w   = np.ones(D) / D if w is None else np.asarray(w, float)
    q   = np.array([0.00, 0.01]) if q is None else np.asarray(q, float)
    rho = np.array([[1.0, 0.3], [0.3, 1.0]]) if rho is None else np.asarray(rho, float)

    # whole-element sigma-mesh padding (keeps the sig_dom endpoints on interior nodes)
    nelem_s_roi = Ns - 1
    h_s = (sig_dom[1] - sig_dom[0]) / nelem_s_roi
    pad_elems = int(round(sig_pad / h_s)) if sig_pad > 0 else 0
    sig_mesh = (sig_dom[0] - pad_elems * h_s, sig_dom[1] + pad_elems * h_s)
    nelem_s = nelem_s_roi + 2 * pad_elems
    t_dom = (0.0, T)

    M = build_matrices_param(D, x_dom, sig_mesh, t_dom, Nx - 1, nelem_s, Nt - 1, order=order)
    pf = basket_payoff_full(M, w, K)
    af = basket_asymp_cp(M, w, K, r, q)

    def solver_fn(dims, op, b, bc, num_mode, sd, verbose):
        # a k-asset subproblem has 2k+1 axes; rank1 for k=1, rank2 for k=2.
        k = max(1, (len(dims) - 1) // 2)
        return als_joint_op(dims, op, b, bc, rank=rank1 if k == 1 else rank2, n_iter=niter,
                            reg=1e-10, patience=niter, use_gpu=False, seed=seed, verbose=False)

    maxrank = max(rank1, rank2)                  # also the lift / stored-solution CP-compression rank
    V = nested_price_param(D, M, rho, r, q, pf, af, 0.0, num_mode=maxrank,
                           round_tol=round_tol, round_rank=maxrank, solver_fn=solver_fn, verbose=False)
    return {'V': V[tuple(range(D))], 'M': M, 'sig_dom': sig_dom, 'sig_mesh': sig_mesh,
            'w': w, 'q': q, 'rho': rho, 'K': K, 'r': r, 'T': T, 'D': D, 'order': order}
