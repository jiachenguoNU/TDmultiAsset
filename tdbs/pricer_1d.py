"""High-level 1D European-call pricer: build the finite-element operators and solve the
Black-Scholes PDE as a Space(x=lnS) x Parameter(sigma) x Time(tau) separated
representation by greedy PGD subspace iteration.

Uses plain Lagrange (linear, 2-node) finite-element shape functions -- no C-HiDeNN /
reproducing-kernel machinery -- so the basis is interpolatory: the solved nodal values
ARE the price, and off-grid prices follow by linear interpolation (`interp_price`).

`solve_call_1d` returns the full price surface u(x, sigma, tau) on the mesh nodes.
"""
import numpy as np
import jax
import jax.numpy as jnp

from .generate_mesh import uniform_mesh_new
from .fem import get_FEM_shape_fun_dict, get_matrix_x, get_matrix_t, bcoo_2_csr
from .bs_assembly import get_matrix_weighted_mass
from .solver1d import TD_solver_BS_sweep


def _sine_init(num_mode, dof):
    """Deterministic orthogonal sine-mode initialisation: row m = sin(m pi n / dof).
    Far more robust than a random start -- it avoids the local-minimum 'lottery' the
    alternating least-squares iteration otherwise suffers from."""
    m = np.arange(1, num_mode + 1); n = np.arange(1, dof + 1)
    return np.sin(m[:, None] * np.pi * n[None, :] / dof)


def solve_call_1d(K=1.0, r=0.05, q=0.0, T=1.0, x_dom=(-3.0, 2.0), s_dom=(0.10, 0.50),
                  nelem_x=100, nelem_s=16, nelem_t=24, num_mode=10, num_max_iter=60,
                  gauss=8, tol=1e-9, init='sine', seed=0):
    """Price a European call across the (S, sigma, tau) grid in one offline solve.

    `num_mode` is a FIXED separated rank: this subspace iteration updates all modes
    together, so it is NOT monotone in rank -- the 1D BS solution is genuinely
    low-rank, and over-ranking (e.g. 24+) injects noise and makes the price *worse*,
    not better. ~10 modes is near-optimal here; raise it only with care.

    Returns a dict with:
        u  : (n_x, n_s, n_t) price surface on the nodes (u[:, j, k] = price vs S at sigma_j, tau_k)
        S  : (n_x,) spot grid = exp(x);  x, sigma, tau : 1D node coordinates
        sweeps : number of PGD sweeps taken
    """
    elem_type = 'D1LN2N'                              # 1D linear, 2-node Lagrange element
    x_min, x_max = x_dom; s_min, s_max = s_dom
    S_max = np.exp(x_max)

    x, En_x = uniform_mesh_new(x_max - x_min, nelem_x); x = np.array(x) + x_min
    s, En_s = uniform_mesh_new(s_max - s_min, nelem_s); s = np.array(s) + s_min
    t, En_t = uniform_mesh_new(T, nelem_t)
    n_x, n_s, n_t = nelem_x + 1, nelem_s + 1, nelem_t + 1

    # --- finite-element shape functions (no patch / adjacency / dilation needed) ---
    inp = {'coor': {'x': x, 's': s, 't': t},
           'Elem_nodes': {'x': En_x, 's': En_s, 't': En_t}}
    sfd = get_FEM_shape_fun_dict(inp, gauss, elem_type)
    #   N_fe: (nelem, quad, nodes_per_elem=2)   Grad_N: (nelem, quad, 2, dim)
    #   JxW:  (nelem, quad)                      Elem_nodes: (nelem, 2)  -- the element connectivity
    gd = lambda k: (sfd[k]['N_fe'], sfd[k]['Grad_N'], sfd[k]['JxW'], sfd[k]['Elem_nodes'])
    Nx, Gx, Jx, Ex = gd('x'); Ns, Gs, Js, Es = gd('s'); Nt, Gt, Jt, Et = gd('t')

    # --- assemble the 1D operator matrices (same builders, FE shape functions) ---
    (K_Bx_Bx, K_Nx_Nx) = get_matrix_x(x, En_x, Nx, Gx, Jx, Ex, gauss, elem_type)   # stiffness, mass (x)
    (K_Nx_Bx, _)       = get_matrix_t(x, En_x, Nx, Gx, Jx, Ex, gauss, elem_type)   # advection int N B (x)
    (_, K_Ns_Ns)       = get_matrix_x(s, En_s, Ns, Gs, Js, Es, gauss, elem_type)   # mass (sigma)
    K_Ns_s2_Ns         = get_matrix_weighted_mass(s, En_s, Ns, Js, Es, gauss, elem_type, power=2)  # sigma^2 mass
    (K_Nt_Bt, K_Nt_Nt) = get_matrix_t(t, En_t, Nt, Gt, Jt, Et, gauss, elem_type)   # time int N B, mass (tau)
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
    return {'u': np.asarray(g + w), 'S': np.exp(xn), 'x': xn, 'sigma': sn, 'tau': tn, 'sweeps': sweeps}


def _interp_axis(q, grid, values, axis):
    """Linear-interpolate `values` along `axis` at scalar `q` (grid = that axis' nodes)."""
    v = jnp.moveaxis(jnp.asarray(values), axis, -1)
    flat = v.reshape(-1, v.shape[-1])
    out = jax.vmap(lambda row: jnp.interp(q, jnp.asarray(grid), row))(flat)
    return out.reshape(v.shape[:-1])


def interp_price(res, S, sigma, tau=None):
    """Online price at an off-grid (S, sigma, tau) by LINEAR interpolation of the solved
    nodal surface (default tau = T).

    Because the Lagrange finite-element basis is interpolatory, the nodal values are the
    function values, so plain linear interpolation inside an element is exactly the FE
    reconstruction -- this is `jnp.interp` applied along tau, then sigma, then x = ln S.
    """
    u, xg, sg, tg = res['u'], res['x'], res['sigma'], res['tau']
    tau = float(tg[-1]) if tau is None else tau
    a = _interp_axis(tau, tg, u, 2)            # (n_x, n_s)  at tau
    b = _interp_axis(sigma, sg, a, 1)          # (n_x,)      at sigma
    return float(_interp_axis(jnp.log(S), xg, b, 0))   # scalar  at S
