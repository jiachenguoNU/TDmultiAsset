"""Evaluate a stored CP (separated-representation) solution at arbitrary physical points.

The parametric solution V is a CP factor list with axis order
    [x_0, ..., x_{D-1}, sigma_0, ..., sigma_{D-1}, tau]
(log-price axes, then volatility axes, then time-to-maturity); factor V[d] has shape
(rank, n_nodes_on_axis_d) and stores each mode's NODAL values on that axis.

An off-grid value is the exact finite-element reconstruction: on the element containing the
query, contract each factor with the Lagrange shape functions N (`price_at`), or -- for the
first-order Greeks -- swap N for its derivative B on the differentiated axis (`greeks_at`),
exactly the method used in 1D (`tdbs.greeks_1d`).  This is order-agnostic: P1 gives the usual
linear interpolation, P2 the piecewise-quadratic reconstruction with an O(h^2) gradient B.
"""
import numpy as np
import jax
import jax.numpy as jnp

from .pricer_1d import _cp_contract     # (q, (nodes, conn, elem_type), factor, deriv) -> (rank,)

_GRID_KEY = {'x': 'xg', 's': 'sg', 't': 'tg'}
_CONN_KEY = {'x': 'Ex', 's': 'Es', 't': 'Et'}


def _axis_mesh(M, axis):
    """(nodes, connectivity, elem_type) for axis 'x' | 's' | 't' from the mesh dict M.
    Falls back to a linear (P1) connectivity for legacy meshes that predate the element-type
    metadata, so old stored M dicts still evaluate correctly."""
    grid = np.asarray(M[_GRID_KEY[axis]]).reshape(-1)
    if 'et' in M and _CONN_KEY[axis] in M:
        return (grid, np.asarray(M[_CONN_KEY[axis]]), M['et'][axis])
    n = grid.shape[0]                                              # legacy P1 mesh
    conn = np.stack([np.arange(n - 1), np.arange(1, n)], axis=1)
    return (grid, conn, 'D1LN2N')


def _interp_modes(q, grid, factor):
    """Linear-interpolate every mode (row) of `factor` (rank, n) at scalar `q`. Returns (rank,).
    Kept for backward compatibility; `price_at` now uses the FE reconstruction `_cp_contract`."""
    grid = jnp.asarray(grid); factor = jnp.asarray(factor)
    return jax.vmap(lambda row: jnp.interp(q, grid, row))(factor)


def price_at(V, D, S_vec, sigma_vec, tau, M):
    """Price at a physical point (S_vec, sigma_vec, tau) by FE reconstruction + contraction.

    V         : CP factor list, axis order [x_0..x_{D-1}, s_0..s_{D-1}, tau].
    S_vec     : length-D spot per asset;  sigma_vec : length-D volatility per asset.
    tau       : time-to-maturity;  M : mesh dict with node grids + connectivity + element type.
    """
    mx, ms, mt = _axis_mesh(M, 'x'), _axis_mesh(M, 's'), _axis_mesh(M, 't')
    prod = jnp.ones(V[0].shape[0])
    for d in range(D):                                   # log-price axes
        prod = prod * _cp_contract(jnp.log(S_vec[d]), mx, V[d])
    for j in range(D):                                   # volatility axes
        prod = prod * _cp_contract(sigma_vec[j], ms, V[D + j])
    prod = prod * _cp_contract(tau, mt, V[2 * D])        # tau axis
    return float(prod.sum())


def greeks_at(V, D, S_vec, sigma_vec, tau, M):
    """First-order basket Greeks at (S_vec, sigma_vec, tau) via the shape-gradient B -- the
    same method as `tdbs.greeks_1d`, on every CP axis.  With
    V = sum_m prod_a Vx_a[m](x_a) * prod_j Vs_j[m](sigma_j) * Vtau[m](tau):

        delta_a = dV/dS_a     = (1/S_a) * (swap N->B on axis x_a)   [per asset, length D]
        vega_j  = dV/dsigma_j =           (swap N->B on axis s_j)   [per asset, length D]
        theta   = dV/dt       = -dV/dtau = -(swap N->B on axis tau)  [scalar; calendar time]

    Returns {'price', 'delta' (D,), 'vega' (D,), 'theta'}.  Uses whatever element order the
    solve used; P2 on the price/vol axes gives O(h^2) delta/vega."""
    mx, ms, mt = _axis_mesh(M, 'x'), _axis_mesh(M, 's'), _axis_mesh(M, 't')
    # per-axis shape-function (N) and shape-gradient (B) contractions, canonical axis order
    N, B = [], []
    for d in range(D):                                   # x_a = log-price axes
        N.append(_cp_contract(jnp.log(S_vec[d]), mx, V[d]))
        B.append(_cp_contract(jnp.log(S_vec[d]), mx, V[d], deriv=True))
    for j in range(D):                                   # sigma_j = volatility axes
        N.append(_cp_contract(sigma_vec[j], ms, V[D + j]))
        B.append(_cp_contract(sigma_vec[j], ms, V[D + j], deriv=True))
    N.append(_cp_contract(tau, mt, V[2 * D]))            # tau axis
    B.append(_cp_contract(tau, mt, V[2 * D], deriv=True))

    n_ax = 2 * D + 1
    def contract(swap):                                  # product over axes, N except B at `swap`
        prod = jnp.ones(V[0].shape[0])
        for a in range(n_ax):
            prod = prod * (B[a] if a == swap else N[a])
        return float(jnp.sum(prod))

    price = contract(-1)                                 # no axis swapped -> the value
    delta = np.array([contract(a) / float(S_vec[a]) for a in range(D)])   # dV/dS_a
    vega  = np.array([contract(D + j) for j in range(D)])                 # dV/dsigma_j
    theta = -contract(2 * D)                                              # dV/dt = -dV/dtau
    return {'price': price, 'delta': delta, 'vega': vega, 'theta': theta}
