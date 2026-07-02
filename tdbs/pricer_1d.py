"""High-level 1D European-call pricer: build the finite-element operators and solve the
Black-Scholes PDE as a Space(x=lnS) x Parameter(sigma) x Time(tau) separated
representation by greedy PGD subspace iteration.

Uses Lagrange finite elements of selectable order (`order` in `solve_call_1d`): P1
(linear, 2-node) or P2 (quadratic, 3-node), chosen independently per axis.  The basis is
interpolatory, so the solved nodal values ARE the price; off-grid prices and Greeks are the
exact FE reconstruction -- shape functions N for the value, their derivatives B for the
Greeks -- evaluated on the containing element (`interp_price`, `greeks_1d`).

`solve_call_1d` returns the price surface u(x, sigma, tau) on the mesh nodes, the solution
in separated (CP) form, and the per-axis mesh needed to reconstruct off-grid.
"""
import numpy as np
import jax
import jax.numpy as jnp

from .generate_mesh import uniform_mesh_new
from .fem import (get_FEM_shape_fun_dict, get_matrix_x, get_matrix_t, bcoo_2_csr,
                  get_shape_val_functions, get_shape_grad_functions)
from .bs_assembly import get_matrix_weighted_mass
from .solver1d import TD_solver_BS_sweep

_ELEM_TYPE = {1: 'D1LN2N', 2: 'D1LQ3N'}      # polynomial order -> Lagrange element type


def _axis_orders(order):
    """Normalise the `order` argument to a per-axis dict {'x':., 's':., 't':.}.
    `order` may be a single int (same order on every axis) or a partial/full dict."""
    if isinstance(order, dict):
        return {k: int(order.get(k, 1)) for k in ('x', 's', 't')}
    return {k: int(order) for k in ('x', 's', 't')}


def _sine_init(num_mode, dof):
    """Deterministic orthogonal sine-mode initialisation: row m = sin(m pi n / dof).
    Far more robust than a random start -- it avoids the local-minimum 'lottery' the
    alternating least-squares iteration otherwise suffers from."""
    m = np.arange(1, num_mode + 1); n = np.arange(1, dof + 1)
    return np.sin(m[:, None] * np.pi * n[None, :] / dof)


def solve_call_1d(K=1.0, r=0.05, q=0.0, T=1.0, x_dom=(-3.0, 2.0), s_dom=(0.10, 0.50),
                  nelem_x=100, nelem_s=16, nelem_t=24, num_mode=10, num_max_iter=60,
                  gauss=8, tol=1e-9, init='sine', seed=0, order=1):
    """Price a European call across the (S, sigma, tau) grid in one offline solve.

    `num_mode` is a FIXED separated rank: this subspace iteration updates all modes
    together, so it is NOT monotone in rank -- the 1D BS solution is genuinely
    low-rank, and over-ranking (e.g. 24+) injects noise and makes the price *worse*,
    not better. ~10 modes is near-optimal here; raise it only with care.

    `order` selects the Lagrange element order: 1 = P1 (linear), 2 = P2 (quadratic).
    Pass an int for the same order on every axis, or a dict for per-axis control, e.g.
    ``order={'x': 2, 's': 1, 't': 1}`` -- P2 on the log-price axis (where Delta lives and
    the payoff/exp nonlinearity is) sharpens the Greeks from O(h) to O(h^2) at the least
    cost.  P2 gives a piecewise-quadratic price (O(h^3)) and an element-linear gradient B,
    so `greeks_1d` need not sit at an element midpoint to be accurate.

    Returns a dict with:
        u  : (n_x, n_s, n_t) price surface on the nodes (u[:, j, k] = price vs S at sigma_j, tau_k)
        S  : (n_x,) spot grid = exp(x);  x, sigma, tau : 1D node coordinates
        sweeps : number of PGD sweeps taken
        factors : (Vx, Vs, Vt) the full solution in SEPARATED (CP) form -- the rank-2 lift g
                  stacked on top of the PGD modes, each of shape (2 + num_mode, n_axis), so
                  V(x, sigma, tau) = sum_m Vx[m] (x) * Vs[m] (sigma) * Vt[m] (tau) == u exactly.
        mesh : {'x': (nodes, Elem_nodes, elem_type), 's': ..., 't': ...} per-axis mesh used by
               `interp_price` / `greeks_1d` to reconstruct the FE field (and B) off-grid.
    """
    orders = _axis_orders(order)
    et = {k: _ELEM_TYPE[o] for k, o in orders.items()}   # per-axis Lagrange element type
    x_min, x_max = x_dom; s_min, s_max = s_dom
    S_max = np.exp(x_max)

    x, En_x = uniform_mesh_new(x_max - x_min, nelem_x, orders['x']); x = np.array(x) + x_min
    s, En_s = uniform_mesh_new(s_max - s_min, nelem_s, orders['s']); s = np.array(s) + s_min
    t, En_t = uniform_mesh_new(T, nelem_t, orders['t'])
    n_x, n_s, n_t = x.shape[0], s.shape[0], t.shape[0]   # = order*nelem + 1 per axis

    # --- finite-element shape functions (no patch / adjacency / dilation needed) ---
    inp = {'coor': {'x': x, 's': s, 't': t},
           'Elem_nodes': {'x': En_x, 's': En_s, 't': En_t}}
    sfd = get_FEM_shape_fun_dict(inp, gauss, et)     # et: per-axis element type
    #   N_fe: (nelem, quad, nodes_per_elem)   Grad_N: (nelem, quad, nodes_per_elem, dim)
    #   JxW:  (nelem, quad)                   Elem_nodes: (nelem, nodes_per_elem)  -- connectivity
    gd = lambda k: (sfd[k]['N_fe'], sfd[k]['Grad_N'], sfd[k]['JxW'], sfd[k]['Elem_nodes'])
    Nx, Gx, Jx, Ex = gd('x'); Ns, Gs, Js, Es = gd('s'); Nt, Gt, Jt, Et = gd('t')

    # --- assemble the 1D operator matrices (same builders, FE shape functions) ---
    (K_Bx_Bx, K_Nx_Nx) = get_matrix_x(x, En_x, Nx, Gx, Jx, Ex, gauss, et['x'])   # stiffness, mass (x)
    (K_Nx_Bx, _)       = get_matrix_t(x, En_x, Nx, Gx, Jx, Ex, gauss, et['x'])   # advection int N B (x)
    (_, K_Ns_Ns)       = get_matrix_x(s, En_s, Ns, Gs, Js, Es, gauss, et['s'])   # mass (sigma)
    K_Ns_s2_Ns         = get_matrix_weighted_mass(s, En_s, Ns, Js, Es, gauss, et['s'], power=2)  # sigma^2 mass
    (K_Nt_Bt, K_Nt_Nt) = get_matrix_t(t, En_t, Nt, Gt, Jt, Et, gauss, et['t'])   # time int N B, mass (tau)
    M_x, K_x, NB_x = bcoo_2_csr(K_Nx_Nx), bcoo_2_csr(K_Bx_Bx), bcoo_2_csr(K_Nx_Bx)
    M_s, W_s       = bcoo_2_csr(K_Ns_Ns), bcoo_2_csr(K_Ns_s2_Ns)
    M_t, PB_t      = bcoo_2_csr(K_Nt_Nt), bcoo_2_csr(K_Nt_Bt)
    mats = (M_x, K_x, NB_x, M_s, W_s, M_t, PB_t)
    A_s_adv = (-(r - q)) * M_s + 0.5 * W_s
    xn, sn, tn = x.reshape(-1), s.reshape(-1), t.reshape(-1)

    # separable lift g (payoff IC + x-boundaries) and load b = -L g
    px   = np.maximum(np.exp(xn) - K, 0.0)
    phix = (xn - x_min) / (x_max - x_min)
    ones_s, ones_t = np.ones(n_s), np.ones(n_t)
    psit = (S_max * np.exp(-q * tn) - K * np.exp(-r * tn)) - (S_max - K)
    lift = [(px, ones_s, ones_t), (phix, ones_s, psit)]
    ops = [(M_x, M_s, PB_t, 1.0), (K_x, W_s, M_t, 0.5),
           (NB_x, A_s_adv, M_t, 1.0), (M_x, M_s, M_t, r)]
    Qx, Qs, Qt = [], [], []
    for (Ax, As, At, c) in ops:
        for (gx, gs, gt) in lift:
            Qx.append(-c * (Ax @ gx)); Qs.append(As @ gs); Qt.append(At @ gt)
    Qx, Qs, Qt = np.array(Qx), np.array(Qs), np.array(Qt)
    dirichlet_idx = np.array([0, n_x - 1], dtype=int)
    ic_idx = np.array([0], dtype=int)

    # PGD subspace iteration: L w = -L g (homogeneous data) -> u = g + w
    if init == 'sine':
        U_x, U_s, U_t = _sine_init(num_mode, n_x), _sine_init(num_mode, n_s), _sine_init(num_mode, n_t)
    else:
        rng = np.random.default_rng(seed)
        U_x = rng.random((num_mode, n_x)); U_s = rng.random((num_mode, n_s)); U_t = rng.random((num_mode, n_t))
    u_prev = np.zeros((n_x, n_s, n_t)); sweeps = 0
    for j in range(num_max_iter):
        U_x, U_s, U_t, _, _ = TD_solver_BS_sweep(mats, Qx, Qs, Qt, U_x, U_s, U_t, r, q, dirichlet_idx, ic_idx)
        nc = (np.linalg.norm(U_x) * np.linalg.norm(U_s) * np.linalg.norm(U_t)) ** (1.0 / 3)
        U_x = nc * U_x / np.linalg.norm(U_x); U_s = nc * U_s / np.linalg.norm(U_s); U_t = nc * U_t / np.linalg.norm(U_t)
        w = np.einsum('mx,ms,mt->xst', U_x, U_s, U_t)
        delta = np.linalg.norm(w - u_prev) / (np.linalg.norm(w) + 1e-30); u_prev = w; sweeps = j + 1
        if delta < tol:
            break
    g = sum(np.einsum('x,s,t->xst', gx, gs, gt) for gx, gs, gt in lift)

    # Full solution in separated (CP) form: stack the rank-2 lift g onto the PGD modes so
    # V = g + w = sum_m Vx[m] (x) Vs[m] (sigma) Vt[m] (tau).  `greeks_1d` reads Greeks off
    # these factors by swapping the shape-function value N for its derivative B per direction.
    Vx = np.vstack([np.asarray(gx) for (gx, _, _) in lift] + [np.asarray(U_x)])
    Vs = np.vstack([np.asarray(gs) for (_, gs, _) in lift] + [np.asarray(U_s)])
    Vt = np.vstack([np.asarray(gt) for (_, _, gt) in lift] + [np.asarray(U_t)])
    mesh = {'x': (xn, np.asarray(En_x), et['x']),
            's': (sn, np.asarray(En_s), et['s']),
            't': (tn, np.asarray(En_t), et['t'])}
    return {'u': np.asarray(g + w), 'S': np.exp(xn), 'x': xn, 'sigma': sn, 'tau': tn,
            'sweeps': sweeps, 'factors': (Vx, Vs, Vt), 'mesh': mesh}


def _cp_contract(q, axis_mesh, factor, deriv=False):
    """Reconstruct one CP factor (and its shape functions) at a scalar query `q`.

    `axis_mesh` = (nodes, Elem_nodes, elem_type) for this axis.  Locate the element that
    contains q, map q to the parent coordinate xi in [-1, 1], and contract the element's
    nodal factor values with either the shape-function VALUES N_a(xi)  (deriv=False) or
    their PHYSICAL derivatives B_a = dN_a/dq = N_a'(xi) * dxi/dq  (deriv=True):

        deriv=False -> sum_a N_a(xi) U_a      (the interpolated value,  == jnp.interp for P1)
        deriv=True  -> sum_a B_a     U_a      (the first derivative d/dq)

    This is exact for any Lagrange order (P1, P2, ...).  Returns (rank,) over the modes."""
    nodes, conn, elem_type = axis_mesh
    nodes = np.asarray(nodes).reshape(-1); conn = np.asarray(conn)
    nelem = conn.shape[0]
    v0 = nodes[conn[:, 0]]                      # left  end-vertex of each element
    vL = nodes[conn[:, -1]]                     # right end-vertex of each element
    qc = float(np.clip(q, nodes.min(), nodes.max()))            # clamp: no extrapolation
    e = int(np.clip(np.searchsorted(v0, qc, side='right') - 1, 0, nelem - 1))
    a, b = float(v0[e]), float(vL[e])
    xi = 2.0 * (qc - a) / (b - a) - 1.0         # physical -> parent coordinate
    xi_arr = jnp.array([xi])
    if deriv:
        fns = get_shape_grad_functions(elem_type)               # [dN_a/dxi]
        w = jnp.array([f(xi_arr)[0] for f in fns]) * (2.0 / (b - a))   # dN_a/dq (chain rule)
    else:
        fns = get_shape_val_functions(elem_type)                # [N_a]
        w = jnp.array([f(xi_arr) for f in fns])
    return jnp.asarray(factor)[:, conn[e]] @ w                  # (rank, npe) @ (npe,)


def interp_price(res, S, sigma, tau=None):
    """Online price at an off-grid (S, sigma, tau) as the exact FE reconstruction of the
    separated solution (default tau = T):  price = sum_m [N_x.Vx][N_s.Vs][N_t.Vt].

    For P1 elements this is the usual linear interpolation of the interpolatory nodal
    solution; for P2 it is the piecewise-quadratic reconstruction (both handled by
    `_cp_contract` via the element's shape functions N)."""
    Vx, Vs, Vt = res['factors']; mesh = res['mesh']
    tau = float(res['tau'][-1]) if tau is None else tau
    Nx = _cp_contract(jnp.log(S), mesh['x'], Vx)
    Ns = _cp_contract(sigma,      mesh['s'], Vs)
    Nt = _cp_contract(tau,        mesh['t'], Vt)
    return float(jnp.sum(Nx * Ns * Nt))


def greeks_1d(res, S, sigma, tau=None):
    """First-order Greeks at an off-grid (S, sigma, tau) straight from the separated
    solution, by replacing the shape-function value N with its derivative B in the
    differentiated direction (default tau = T).

    With V = sum_m Vx[m] (x) Vs[m] (sigma) Vt[m] (tau), x = ln S, tau = T - t:

        price = sum_m [N_x.Vx] [N_s.Vs] [N_t.Vt]
        dV/dx = sum_m [B_x.Vx] [N_s.Vs] [N_t.Vt]   ->  Delta = dV/dS = (1/S) dV/dx
        Vega  = sum_m [N_x.Vx] [B_s.Vs] [N_t.Vt]   =   dV/dsigma
        dV/dtau = sum_m [N_x.Vx] [N_s.Vs] [B_t.Vt] ->  Theta = dV/dt = -dV/dtau

    B is the shape-function derivative of whatever element order was solved with, so P2 gives
    an element-linear (O(h^2)) gradient vs P1's piecewise-constant (O(h)) one.  Returns
    {'price', 'delta', 'vega', 'theta'} (theta per unit CALENDAR time; vega per unit vol)."""
    Vx, Vs, Vt = res['factors']; mesh = res['mesh']
    tau = float(res['tau'][-1]) if tau is None else tau
    x = jnp.log(S)

    Nx, Ns, Nt = (_cp_contract(x, mesh['x'], Vx),
                  _cp_contract(sigma, mesh['s'], Vs),
                  _cp_contract(tau, mesh['t'], Vt))
    Bx = _cp_contract(x,     mesh['x'], Vx, deriv=True)
    Bs = _cp_contract(sigma, mesh['s'], Vs, deriv=True)
    Bt = _cp_contract(tau,   mesh['t'], Vt, deriv=True)

    dV_dx   = float(jnp.sum(Bx * Ns * Nt))       # d/d(ln S)
    vega    = float(jnp.sum(Nx * Bs * Nt))       # d/dsigma
    dV_dtau = float(jnp.sum(Nx * Ns * Bt))       # d/dtau  (tau = T - t)
    return {'price': float(jnp.sum(Nx * Ns * Nt)),
            'delta': dV_dx / float(S),           # dV/dS = (1/S) dV/d(ln S)
            'vega':  vega,
            'theta': -dV_dtau}                   # dV/dt = -dV/dtau
