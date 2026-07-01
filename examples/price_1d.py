"""Price a 1D European call with the tensor-decomposition (greedy PGD) finite-element solver
and compare against the closed-form Black-Scholes price across several volatilities.

The PDE is solved ONCE as a Space (x = ln S) x Parameter (sigma) x Time (tau = T - t)
separated representation u(x, sigma, tau) ~ sum_m X_m(x) S_m(sigma) T_m(tau) on plain
Lagrange finite elements; the whole volatility surface comes from that single offline solve,
and any off-grid price is a linear interpolation of the nodal solution.

Run:  python examples/price_1d.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import jax; jax.config.update("jax_enable_x64", True)

from tdbs import solve_call_1d, interp_price, bs_call

K, r, q, T = 1.0, 0.05, 0.0, 1.0
# 100 linear finite elements on every axis (x, sigma, tau)
res = solve_call_1d(K=K, r=r, q=q, T=T, nelem_x=100, nelem_s=100, nelem_t=100)   # full surface u(x, sigma, tau)
u, S, sigma, tau = res['u'], res['S'], res['sigma'], res['tau']
kT = len(tau) - 1                                    # tau = T (largest time-to-maturity)
band = (S >= 0.5 * K) & (S <= 2.0 * K)               # near-the-money band (away from kink tail / far field)

print(f"1D European call  K={K} r={r} q={q} T={T} | mesh {u.shape}, {res['sweeps']} PGD sweeps (FEM, sine init)")
print(f"{'sigma (node)':>12}  {'rel-L2 (S in [0.5,2]K)':>22}  {'max abs err':>12}")
for sig in (0.15, 0.25, 0.35, 0.45):
    js = int(np.argmin(np.abs(sigma - sig)))         # nearest sigma node
    u_num = u[:, js, kT]
    u_ana = bs_call(S, K, r, q, sigma[js], T)
    rel = np.linalg.norm((u_num - u_ana)[band]) / np.linalg.norm(u_ana[band])
    print(f"{sigma[js]:>12.3f}  {rel:>22.3e}  {np.max(np.abs(u_num - u_ana)[band]):>12.3e}")

# --- online prediction at OFF-grid (S, sigma) by linear interpolation of the nodal surface ---
print("\noff-grid prediction (linear interp of the solved surface) vs analytic BS, tau=T:")
print(f"{'S':>6} {'sigma':>7}  {'TD interp':>10}  {'BS':>10}  {'rel.err':>9}")
for (Sq, sq) in [(1.00, 0.27), (1.10, 0.333), (0.90, 0.215), (1.25, 0.48)]:
    td = interp_price(res, Sq, sq, T)
    an = bs_call(Sq, K, r, q, sq, T)
    print(f"{Sq:>6.2f} {sq:>7.3f}  {td:>10.5f}  {an:>10.5f}  {abs(td - an) / abs(an):>9.2e}")
