"""Lagrange finite-element shape functions, 1D operator matrices and sparse
assembly for the tensor-decomposition Black-Scholes solver (no C-HiDeNN).
"""
import numpy as onp
from scipy.sparse import csc_matrix, csr_matrix
import jax
import jax.numpy as np
from jax.experimental.sparse import BCOO
from itertools import combinations
from jax.experimental.sparse import BCSR
from functools import partial
from scipy.special import roots_legendre
import os,sys
# petsc4py.init()


os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # add this for memory pre-allocation
jax.config.update("jax_enable_x64", True)

jit_repeat = jax.jit(np.repeat, static_argnames=['axis', 'total_repeat_length'])   # used by assembly()



def gauss_legendre(n):
    # Get the nodes and weights
    nodes, weights = roots_legendre(n)
    
    # Convert to Python's list format without rounding
    Gauss_Weight1D = list(weights)
    Gauss_Point1D = list(nodes)
    
    return Gauss_Weight1D, Gauss_Point1D

def GaussSet(Gauss_Num = 2, cuda=False):
    if Gauss_Num == 2:
        Gauss_Weight1D = [1, 1]
        Gauss_Point1D = [-1/np.sqrt(3), 1/np.sqrt(3)]
    
    elif Gauss_Num == 0:
        Gauss_Weight1D = [1.]
        Gauss_Point1D = [-1.]
       
    elif Gauss_Num == 3:
        Gauss_Weight1D, Gauss_Point1D = gauss_legendre(3)
       
        
    elif Gauss_Num == 4:
        Gauss_Weight1D, Gauss_Point1D = gauss_legendre(4)

    elif Gauss_Num == 6: # double checked, 16 digits
        Gauss_Weight1D, Gauss_Point1D = gauss_legendre(6)


       
    elif Gauss_Num == 8: # double checked, 20 digits
        Gauss_Weight1D, Gauss_Point1D = gauss_legendre(8)
        
    elif Gauss_Num == 10:
        Gauss_Weight1D, Gauss_Point1D = gauss_legendre(10)
    elif Gauss_Num == 12:
        Gauss_Weight1D, Gauss_Point1D = gauss_legendre(12)
    elif Gauss_Num == 14:
        Gauss_Weight1D, Gauss_Point1D = gauss_legendre(14)        
                
    elif Gauss_Num == 20:
        Gauss_Weight1D, Gauss_Point1D = gauss_legendre(20)
    
    return Gauss_Weight1D, Gauss_Point1D

def get_quad_points(Gauss_Num, dim):
    """ Quadrature point and weight generator
    --- Inputs ---
    --- Outputs ---
    """
    Gauss_Weight1D, Gauss_Point1D = GaussSet(Gauss_Num)
    quad_points, quad_weights = [], []
    
    for ipoint, iweight in zip(Gauss_Point1D, Gauss_Weight1D):
        if dim == 1:
            quad_points.append([ipoint])
            quad_weights.append(iweight)
        else:
            for jpoint, jweight in zip(Gauss_Point1D, Gauss_Weight1D):
                if dim == 2:
                    quad_points.append([ipoint, jpoint])
                    quad_weights.append(iweight * jweight)
                else: # dim == 3
                    for kpoint, kweight in zip(Gauss_Point1D, Gauss_Weight1D):
                        quad_points.append([ipoint, jpoint, kpoint])
                        quad_weights.append(iweight * jweight * kweight)
    
    quad_points = np.array(quad_points) # (quad_degree*dim, dim)
    quad_weights = np.array(quad_weights) # (quad_degree,)
    return quad_points, quad_weights

def get_shape_val_functions(elem_type):
    """ Shape function generator (parent domain xi in [-1, 1])
    """
    ############ 1D ##################
    if elem_type == 'D1LN2N': # 1D linear (P1) element, 2 nodes at xi = -1, +1
        f1 = lambda x: 1./2.*(1 - x[0])
        f2 = lambda x: 1./2.*(1 + x[0])
        shape_fun = [f1, f2] # a list of functions

    elif elem_type == 'D1LQ3N': # 1D quadratic (P2) element, 3 nodes at xi = -1, 0, +1
        f1 = lambda x: 1./2.*x[0]*(x[0] - 1.)   # left  vertex  (xi = -1)
        f2 = lambda x: 1. - x[0]**2             # mid   node    (xi =  0)
        f3 = lambda x: 1./2.*x[0]*(x[0] + 1.)   # right vertex  (xi = +1)
        shape_fun = [f1, f2, f3]                # local order [left, mid, right]

    return shape_fun

def get_shape_grad_functions(elem_type):
    """ Shape function gradient in the parent domain
    """
    shape_fns = get_shape_val_functions(elem_type)
    return [jax.grad(f) for f in shape_fns]

def get_shape_vals(Gauss_Num, dim, elem_type):
    """ Measure shape function values at quadrature points
    """
    shape_val_fns = get_shape_val_functions(elem_type)
    quad_points, quad_weights = get_quad_points(Gauss_Num, dim)
    shape_vals = []
    for quad_point in quad_points:
        physical_shape_vals = []
        for shape_val_fn in shape_val_fns:
            physical_shape_val = shape_val_fn(quad_point) 
            physical_shape_vals.append(physical_shape_val)
 
        shape_vals.append(physical_shape_vals)

    shape_vals = np.array(shape_vals) # (quad_num, nodes_per_elem)
    return shape_vals #N_I at different quads

@partial(jax.jit, static_argnames=['Gauss_Num', 'dim', 'elem_type']) # necessary
def get_shape_grads(Gauss_Num, dim, elem_type, XY, Elem_nodes):
    """ Meature shape function gradient values at quadrature points
    --- Outputs
    shape_grads_physical: shape function gradient in physcial coordinate (nelem, quad_num, nodes_per_elem, dim)
    JxW: Jacobian determinant times Gauss quadrature weights (nelem, quad_num)
    """
    shape_grad_fns = get_shape_grad_functions(elem_type)
    quad_points, quad_weights = get_quad_points(Gauss_Num, dim)
    shape_grads = []
    for quad_point in quad_points:
        physical_shape_grads = []
        for shape_grad_fn in shape_grad_fns:
            physical_shape_grad = shape_grad_fn(quad_point)
            physical_shape_grads.append(physical_shape_grad)
        shape_grads.append(physical_shape_grads)

    shape_grads = np.array(shape_grads) # (quad_num, nodes_per_elem, dim)
    physical_coos = np.take(XY, Elem_nodes, axis=0) # (nelem, nodes_per_elem, dim)
    jacobian_dx_deta = np.sum(physical_coos[:, None, :, :, None] * shape_grads[None, :, :, None, :], axis=2, keepdims=True) # dx/deta
    # (nelem, quad_num, nodes_per_elem, dim, dim) -> (nelem, quad_num, 1, dim, dim)
    
    jacbian_det = np.squeeze(np.linalg.det(jacobian_dx_deta)) # det(J) (nelem, quad_num)
    jacobian_deta_dx = np.linalg.inv(jacobian_dx_deta) # deta/dx (nelem, quad_num, 1, dim, dim)
    shape_grads_physical = (shape_grads[None, :, :, None, :] @ jacobian_deta_dx)[:, :, :, 0, :]
    JxW = jacbian_det * quad_weights[None, :] #(nelem, quad_num)
    return shape_grads_physical, JxW

def get_FEM_shape_fun_dict(input_dict, Gauss_Num_FEM, elem_type):
    """`elem_type` may be a single string (same element on every coordinate) or a dict
    mapping each coordinate key to its own element type, e.g. {'x': 'D1LQ3N', 's': 'D1LN2N',
    't': 'D1LN2N'} -- this is what lets the axes independently be P1 or P2."""
    # Extract coordinate dictionary
    coor = input_dict['coor']  # {'x': ..., 't': ..., 'ksi': ...}

    # Extract element nodes dictionary
    elem_nodes = input_dict['Elem_nodes']  # {'x': ..., 't': ..., 'ksi': ...}
    dim = 1

    results = {}

    # Loop over coordinate types ('x', 't', 'ksi') to perform computations
    for coord in coor:

        # per-coordinate element type (P1 'D1LN2N' or P2 'D1LQ3N')
        et = elem_type[coord] if isinstance(elem_type, dict) else elem_type
        shape_vals = get_shape_vals(Gauss_Num_FEM, dim, et) # (quad_num, nodes_per_elem) @ quads

        nelem_coords = len(elem_nodes[coord])

        # Fetch parameters from the dictionaries
        value = coor[coord]
        elem_node = elem_nodes[coord]

        physical_coor = np.take(value, elem_node, axis=0) # (nelem, nodes_per_elem, dim)
        gauss_pts_coor = np.sum(shape_vals[None, :, :, None] * physical_coor[:, None, :, :], axis=2) # (nelem, quad_num,)#gauss points coor


        Grad_N, JxW = get_shape_grads(Gauss_Num_FEM, dim, et, value, elem_node) # (nelem, quad_num, nodes_per_elem, dim)
        # Perform shape function computation
        N_fe = np.repeat(shape_vals[np.newaxis, :, :], nelem_coords, axis=0)
        (K_Bx_Bx, K_Nx_Nx) = get_matrix_x(value,elem_node,N_fe, Grad_N,
                JxW, elem_node,
                Gauss_Num_FEM, et)
        
        results[coord] = {
            'N_fe': N_fe, # (nelem, quad_num, edex_max)
            'Grad_N': Grad_N, # (nelem, quad_num, edex_max, dim)
            'JxW': JxW, # (nelem, quad_num)
            'Elem_nodes': elem_node, # (nelem, nodes_per_elem)
            'gauss_pts_coor': gauss_pts_coor, # (nelem, quad_num, dim)
            'K_Bx_Bx': K_Bx_Bx, # (nelem, quad_num, nodes_per_elem, nodes_per_elem)
            'K_Nx_Nx': K_Nx_Nx # (nelem, quad_num, nodes_per_elem, nodes_per_elem)
        }


    return results

def get_matrix_x(x, 
                  Elem_nodes_x, 
                  N_til_x, 
                  Grad_N_til_x, 
                  JxW_x, 
                  Elemental_patch_nodes_st_x, 
                  Gauss_Num, elem_type):
    """ 
     
        x: (nelem, quad_num, dim)
        Elem_nodes_x: (nelem, nodes_per_elem)
        N_til_x: (nelem, quad_num, edex_max)
        Grad_N_til_x: (nelem, quad_num, edex_max, dim)
        JxW_x: (nelem, quad_num)
        Elemental_patch_nodes_st_x: (nelem, nodes_per_elem)
        Gauss_Num: int
        elem_type: str

    Returns:
        K_Bx_Bx: (nelem, quad_num, nodes_per_elem, nodes_per_elem)
        K_Nx_Nx: (nelem, quad_num, nodes_per_elem, nodes_per_elem)
    """
    radial_basis = 'cubicSpline'
    nelem_x = len(Elem_nodes_x); dof_global_x = Elem_nodes_x.shape[0] + 1
    
    
    dim = 1
    quad_num = Gauss_Num
    shape_vals = get_shape_vals(Gauss_Num, dim, elem_type) # (quad_num, nodes_per_elem)  --shape fun values @ quads: same for differnt parameters

    #Need this Jacobian info for different parameters for integration
    # N_til_X: (nelem, quad_num, edex_max)

    BxT_Bx= np.matmul(Grad_N_til_x, np.transpose(Grad_N_til_x, (0,1,3,2))) # (nelem, quad_num, nodes_per_elem, nodes_per_elem) #element stiffness matrix in space
    
    NxT_Nx = np.matmul(N_til_x[:, :, :, None], np.transpose(N_til_x[:, :, :, None], (0,1,3,2)))

    
    
    K_Bx_Bx = assembly(BxT_Bx, JxW_x, Elemental_patch_nodes_st_x)

    
    K_Nx_Nx = assembly(NxT_Nx, JxW_x, Elemental_patch_nodes_st_x)
                    
    return (K_Bx_Bx, K_Nx_Nx)

def get_matrix_t(x, 
                  Elem_nodes_x, 
                  N_til_x, 
                  Grad_N_til_x, 
                  JxW_x, 
                  Elemental_patch_nodes_st_x, 
                  Gauss_Num, elem_type):
    """ N^T B (and mass) matrices
        return the matrices required to compute the linear system of equations in alternating fixed iteration
        K_NN = int N^T@N
        K_NxN= int N^T @ x @ N -> for material parameter terms that has a coefficient in the fun 
        K_BB = int B^T@B 
        K_NB = int N^T@B 
        K_N_petrov B = int N_p^T@B  for time derivative term
        Q_N = int N^T
    """
    radial_basis = 'cubicSpline'
    nelem_x = len(Elem_nodes_x); dof_global_x = Elem_nodes_x.shape[0] + 1
    
    
    dim = 1
    quad_num = Gauss_Num
    shape_vals = get_shape_vals(Gauss_Num, dim, elem_type) # (quad_num, nodes_per_elem)  --shape fun values @ quads: same for differnt parameters

    #Need this Jacobian info for different parameters for integration
    # N_til_X: (nelem, quad_num, edex_max)
    
    NxT_Nx = np.matmul(N_til_x[:, :, :, None], np.transpose(N_til_x[:, :, :, None], (0,1,3,2)))
    NxT_Bx = np.matmul(N_til_x[:, :, :, None], np.transpose(Grad_N_til_x, (0,1,3,2))) 
    
    K_Nx_Bx = assembly(NxT_Bx, JxW_x, Elemental_patch_nodes_st_x)
    

    
    K_Nx_Nx = assembly(NxT_Nx, JxW_x, Elemental_patch_nodes_st_x)
                    
    return (K_Nx_Bx, K_Nx_Nx)

@jax.jit
def assembly(BT_B, JxW, connectivity):
    """
    Args:
        BT_B: Input Tensor. 
              Shape: (nelem, quad_num, nodes_per_elem, nodes_per_elem) 
        JxW:  Jacobian * Weights. 
              Shape: (nelem, quad_num)
        connectivity: Connectivity for Mesh. Shape: (nelem, nodes_per_elem)
    """
    # nnode = nelem * (nodes_per_elem - 1) + 1 for a contiguous 1D mesh: P1 -> nelem+1,
    # P2 -> 2*nelem+1.  (Both shapes are static under jit, unlike connectivity.max().)
    dof_global = JxW.shape[0] * (connectivity.shape[1] - 1) + 1
    edex_max = connectivity.shape[1]
    V = np.sum(BT_B * JxW[:, :, None, None], axis=(1)).reshape(-1) # (nelem, edex, edex) -> (1 ,)
    # I = np.repeat(connectivity, edex_max, axis=1).reshape(-1)
    I = jit_repeat(connectivity, edex_max, axis=1, total_repeat_length=connectivity.shape[1] * edex_max).reshape(-1)
    # J = np.repeat(connectivity, edex_max, axis=0).reshape(-1)
    J = jit_repeat(connectivity, edex_max, axis=0, total_repeat_length=connectivity.shape[0] * edex_max).reshape(-1)
    # V_s, J_s, indptr, = compute_indptr(V, I, J, dof_global)
    # K = BCSR((V_s, J_s, indptr), shape=(dof_global, dof_global))
    bcoo_indices = np.hstack((I.reshape(-1, 1), J.reshape(-1, 1)))
    K = BCOO((V, bcoo_indices), shape=(dof_global, dof_global))
    return K

def bcoo_2_csr(A):
    """ Convert BCOO format to CSR format """
    return csr_matrix((A.data, (A.indices[:,0], A.indices[:,1])), shape = A.shape)
