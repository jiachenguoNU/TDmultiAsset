"""
generate_mesh.py  (trimmed)
---------------------------
1D uniform mesh generator, copied from the example_code but with the optional
gmsh / meshio dependencies removed (only `uniform_mesh_new` is used by the
tensor-decomposition solver).
"""
import numpy as onp


def uniform_mesh_new(L, nelem_x, order=1):
    """ 1D uniform Lagrange mesh generator.
    --- Inputs ---
    L      : length of the domain
    nelem_x: number of elements
    order  : element (polynomial) order -- 1 = linear P1 (2 nodes/elem),
             2 = quadratic P2 (3 nodes/elem, with a centered mid-edge node)
    --- Outputs ---
    XY        : nodal coordinates, shape (nnode, 1), nnode = order*nelem + 1,
                uniformly spaced (so P2 mid-edge nodes sit at element centers)
    Elem_nodes: element connectivity, shape (nelem, order+1), local order
                [left, right] (P1) or [left, mid, right] (P2); consecutive
                elements share their end vertex, and global nodes 0 and nnode-1
                are the two domain endpoints.
    """
    dim = 1
    nelem = nelem_x
    npe = order + 1                       # nodes per element (2 for P1, 3 for P2)
    nnode = order * nelem + 1
    dx = L / (order * nelem)              # node spacing (h for P1, h/2 for P2)

    XY = (onp.arange(nnode, dtype=onp.double) * dx).reshape(nnode, dim)
    # element e owns nodes [order*e, order*e + 1, ..., order*e + order]
    Elem_nodes = (order * onp.arange(nelem)[:, None]
                  + onp.arange(npe)[None, :]).astype(onp.int32)
    return XY, Elem_nodes
