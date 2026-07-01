"""Price a D=2 parametric BASKET call with the joint tensor-decomposition C-HiDeNN solver
and compare against control-variate Monte-Carlo across volatilities.

The 2-asset basket  max(w0 S0 + w1 S1 - K, 0)  is solved as a 5-axis separated
representation over (x0, x1, sigma0, sigma1, tau).  A nested hierarchy lifts the 1-asset
solutions into the 2-asset boundary data; each subproblem is solved by fixed-rank joint
alternating-least-squares.  One offline solve prices the whole 2D volatility surface.

Run:  python examples/price_2d.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import jax; jax.config.update("jax_enable_x64", True)

from tdbs import solve_basket_2d, price_at, basket_cv_mc

# High-accuracy config: 90 nodes on each log-price axis, 50 on sigma/tau, per-level CP
# ranks 19 / 80, 100 ALS sweeps, sigma-mesh padded by 0.05 so the compared sigmas are interior.
res = solve_basket_2d(Nx=90, Ns=50, Nt=50, rank1=19, rank2=80, niter=100, sig_pad=0.05)
V, M, D, K, r, T = res['V'], res['M'], res['D'], res['K'], res['r'], res['T']
w, q, rho = res['w'], res['q'], res['rho']

# compare the ATM basket (S=1 per asset) across interior sigma nodes, vs control-variate MC
sg = np.asarray(M['sg'])
lo, hi = sg[0], sg[-1]
sig_nodes = sg[(sg > res['sig_dom'][0] + 1e-9) & (sg < res['sig_dom'][1] - 1e-9)]   # interior of ROI
sig_nodes = sig_nodes[np.linspace(0, len(sig_nodes) - 1, 5).round().astype(int)]
S0 = np.ones(D)

print(f"D=2 basket call  K={K} r={r} T={T} rho=0.3 | ATM (S=1), tau=T")
print(f"{'sigma (both)':>13}  {'TD price':>10}  {'CV-MC':>10}  {'rel.err':>9}")
td_all, mc_all = [], []
for sv in sig_nodes:
    td = price_at(V, D, S0, [sv, sv], T, M)
    mc = basket_cv_mc(S0, np.array([sv, sv]), K, r, q, w, rho, T, n=400_000)
    td_all.append(td); mc_all.append(mc)
    print(f"{sv:>13.3f}  {td:>10.5f}  {mc:>10.5f}  {abs(td - mc) / abs(mc):>9.2e}")
td_all, mc_all = np.array(td_all), np.array(mc_all)
print(f"rel-L2 over {len(sig_nodes)} volatilities = {np.linalg.norm(td_all - mc_all) / np.linalg.norm(mc_all):.3e}")
