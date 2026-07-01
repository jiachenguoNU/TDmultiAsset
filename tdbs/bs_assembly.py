"""
bs_assembly.py
---------------------------
Black-Scholes-specific 1D finite-element matrix assembly, built on the shape-function /
assembly machinery in `fem.py`.

The plain mass / stiffness / advection 1D matrices come from `fem.get_matrix_x/t`; this
module adds the sigma^p-weighted parameter mass that BS needs

    K_Ns_s2_Ns[i,j] = integral_{Omega_sigma}  N_i(sigma) * sigma^2 * N_j(sigma) d sigma,

the parametric factor of both the diffusion term ( (1/2) sigma^2 ) and the sigma^2-part of
the advection coefficient ( b = r - q - (1/2) sigma^2 ).
"""
import jax.numpy as np
from .fem import get_shape_vals, assembly


def get_matrix_weighted_mass(coor, Elem_nodes, N_til, JxW, Elemental_patch_nodes_st,
                             Gauss_Num, elem_type, power=2):
    """Assemble the coordinate-power-weighted mass matrix

        K_w[i,j] = integral  N_i(xi) * xi**power * N_j(xi)  d xi.

    With ``power=2`` and ``coor`` the volatility grid this is K_Ns_s2_Ns.

    Args:
        coor:   (nnode, 1) nodal coordinates of this direction
        N_til:  (nelem, quad_num, nodes_per_elem)
        JxW:    (nelem, quad_num)
    Returns:
        K_w (BCOO)  -> convert with bcoo_2_csr before use.
    """
    dim = 1
    shape_vals = get_shape_vals(Gauss_Num, dim, elem_type)  # (quad_num, nodes_per_elem)

    # N_i N_j at every (elem, quad): (nelem, quad_num, edex_max, edex_max)
    NxT_Nx = np.matmul(N_til[:, :, :, None],
                       np.transpose(N_til[:, :, :, None], (0, 1, 3, 2)))

    # coordinate value at every quad point: (nelem, quad_num)
    x_coor = np.take(coor, Elem_nodes, axis=0)                      # (nelem, nodes_per_elem, dim)
    xs = np.sum(shape_vals[None, :, :, None] * x_coor[:, None, :, :], axis=2)[:, :, 0]

    weight = xs ** power                                           # (nelem, quad_num)
    NxT_w_Nx = weight[:, :, None, None] * NxT_Nx

    K_w = assembly(NxT_w_Nx, JxW, Elemental_patch_nodes_st)
    return K_w
