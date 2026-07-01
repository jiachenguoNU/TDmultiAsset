"""Evaluate a stored CP (separated-representation) solution at arbitrary physical points.

The parametric solution V is a CP factor list with axis order
    [x_0, ..., x_{D-1}, sigma_0, ..., sigma_{D-1}, tau]
(log-price axes, then volatility axes, then time-to-maturity); factor V[d] has shape
(rank, n_nodes_on_axis_d) and stores each mode's NODAL values on that axis.

With plain Lagrange finite elements the basis is interpolatory (nodal values are the
function values), so a price at an off-grid point is just a LINEAR interpolation of each
factor along its axis (`jnp.interp`), then a contraction over the modes.
"""
import jax
import jax.numpy as jnp


def _interp_modes(q, grid, factor):
    """Linear-interpolate every mode (row) of `factor` (rank, n) at scalar `q`. Returns (rank,)."""
    grid = jnp.asarray(grid); factor = jnp.asarray(factor)
    return jax.vmap(lambda row: jnp.interp(q, grid, row))(factor)


def price_at(V, D, S_vec, sigma_vec, tau, M):
    """Price at a physical point (S_vec, sigma_vec, tau) by linear interpolation + contraction.

    V         : CP factor list, axis order [x_0..x_{D-1}, s_0..s_{D-1}, tau].
    S_vec     : length-D spot per asset;  sigma_vec : length-D volatility per asset.
    tau       : time-to-maturity;  M : mesh dict with node grids 'xg', 'sg', 'tg'.
    """
    xg, sg, tg = M['xg'], M['sg'], M['tg']
    prod = jnp.ones(V[0].shape[0])
    for d in range(D):                                   # log-price axes
        prod = prod * _interp_modes(jnp.log(S_vec[d]), xg, V[d])
    for j in range(D):                                   # volatility axes
        prod = prod * _interp_modes(sigma_vec[j], sg, V[D + j])
    prod = prod * _interp_modes(tau, tg, V[2 * D])       # tau axis
    return float(prod.sum())
