"""
generate_mesh.py  (trimmed)
---------------------------
1D uniform mesh generator, copied from the example_code but with the optional
gmsh / meshio dependencies removed (only `uniform_mesh_new` is used by the
tensor-decomposition solver).
"""
import numpy as onp


def uniform_mesh_new(L, nelem_x):
    """ 1D uniform mesh generator.
    --- Inputs ---
    L      : length of the domain
    nelem_x: number of elements
    --- Outputs ---
    XY        : nodal coordinates, shape (nnode, 1)
    Elem_nodes: element connectivity, shape (nelem, 2)
    """
    dim = 1
    nelem = nelem_x
    nnode = nelem + 1

    XY = onp.zeros([nnode, dim], dtype=onp.double)
    dx = L / nelem
    for i in range(1, nelem + 2):
        XY[i - 1, 0] = (i - 1) * dx

    nodes_per_elem = 2
    Elem_nodes = onp.zeros([nelem, nodes_per_elem], dtype=onp.int32)
    for j in range(1, nelem + 1):
        Elem_nodes[j - 1, 0] = j - 1
        Elem_nodes[j - 1, 1] = j
    return XY, Elem_nodes
