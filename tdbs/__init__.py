"""tdbs — tensor-decomposition Black-Scholes pricer (Lagrange finite elements).

European options solved as a Space (log-price x = ln S) - Parameter (volatility sigma)
- Time (tau = T - t) separated representation:

  * a 1D vanilla call (greedy PGD subspace iteration), and
  * a D=2 parametric basket call (nested joint alternating-least-squares).

The PDE is discretised once with plain Lagrange (linear) finite-element 1D operators; the
solution is stored in CP (canonical-polyadic) form, giving O(N) storage and microsecond
pricing across the whole volatility surface after a single offline solve. Off-grid prices
are linear interpolations of the nodal solution (the FE basis is interpolatory).

Public API
----------
high-level pricers:
    solve_call_1d, interp_price         (1D call; off-grid price by linear interpolation)
    solve_basket_2d, price_at           (D=2 basket; off-grid price by linear interpolation)

building blocks:
    build_matrices_param, nested_price_param, basket_payoff_full, basket_asymp_cp,
    als_joint_op (joint fixed-rank ALS sub-solver), TD_solver_BS_sweep (1D PGD sweep)

references:
    bs_call, bs_call_weighted, geo_basket_price, basket_cv_mc
"""
from .nd_bs_param import (build_matrices_param, nested_price_param,
                          basket_payoff_full, basket_asymp_cp)
from .joint_als import als_joint_op, gpu_device_name
from .nd_bs_cp import cp_eval, cp_rank, cp_norm
from .solver1d import TD_solver_BS_sweep
from .pricer_1d import solve_call_1d, interp_price
from .pricer_2d import solve_basket_2d
from .evaluate import price_at
from .reference import bs_call, bs_call_weighted, geo_basket_price, basket_cv_mc

__all__ = [
    "build_matrices_param", "nested_price_param", "basket_payoff_full", "basket_asymp_cp",
    "als_joint_op", "gpu_device_name", "cp_eval", "cp_rank", "cp_norm",
    "TD_solver_BS_sweep", "solve_call_1d", "interp_price", "solve_basket_2d", "price_at",
    "bs_call", "bs_call_weighted", "geo_basket_price", "basket_cv_mc",
]
__version__ = "0.1.0"
