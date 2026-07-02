"""Generate the README result figures and the Monte-Carlo speed/accuracy benchmark.

Produces:
  docs/price_1d_vs_bs.png    1D call: TD solver vs closed-form Black-Scholes across sigma
  docs/greeks_1d_vs_bs.png   1D call Greeks (Delta/Vega/Theta via shape-gradient B) vs BS
  docs/basket_2d_vs_mc.png   D=2 basket: TD solver vs control-variate Monte-Carlo vs spot
and prints the offline solve time + per-price latency (TD nodal contraction vs Monte-Carlo).

Configs match the project's reference runs: 1D mesh 100/100/100; 2D Nx=90, Ns=Nt=50,
ranks 19/80, 100 ALS sweeps, sigma-mesh padded by 0.05.

Run:  pip install -e ".[figures]"  then  python examples/make_figures.py
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import jax; jax.config.update("jax_enable_x64", True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tdbs import (solve_call_1d, bs_call, greeks_1d, bs_greeks,
                  solve_basket_2d, price_at, basket_cv_mc, cp_eval)

DOCS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs")
os.makedirs(DOCS, exist_ok=True)

# =========================================================================== #
# Figure 1 -- 1D European call: TD solver vs analytic Black-Scholes
# =========================================================================== #
K, r, q, T = 1.0, 0.05, 0.0, 1.0
print("Solving 1D call (mesh 100/100/100; one offline solve gives the whole surface)...")
res1 = solve_call_1d(K=K, r=r, q=q, T=T, nelem_x=100, nelem_s=100, nelem_t=100)
u, S, sigma, tau = res1['u'], res1['S'], res1['sigma'], res1['tau']
kT = len(tau) - 1
band = (S >= 0.5 * K) & (S <= 2.0 * K)               # near-the-money band (away from kink tail / far field)
Sb = S[band]; order = np.argsort(Sb)

fig, ax = plt.subplots(figsize=(6.4, 4.6))
colors = plt.cm.viridis(np.linspace(0.1, 0.85, 4))
rels = []
for c, sig in zip(colors, (0.15, 0.25, 0.35, 0.45)):
    js = int(np.argmin(np.abs(sigma - sig))); sv = sigma[js]
    u_num = u[:, js, kT][band]; u_ana = bs_call(Sb, K, r, q, sv, T)
    rels.append(np.linalg.norm(u_num - u_ana) / np.linalg.norm(u_ana))
    ax.plot(Sb[order], u_ana[order], '-', color=c, lw=1.6, label=f"BS  $\\sigma$={sv:.2f}")
    ax.plot(Sb[order][::7], u_num[order][::7], 'o', color=c, ms=4, mfc='none')
ax.set_xlabel("spot  $S$"); ax.set_ylabel("call price  $C(S,\\sigma,T)$")
ax.set_title(f"1D European call: TD solver (markers) vs Black-Scholes (lines)\n"
             f"rel-L2 over $\\sigma$: {', '.join(f'{e:.1e}' for e in rels)}")
ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(DOCS, "price_1d_vs_bs.png"), dpi=130); plt.close(fig)
print(f"  saved docs/price_1d_vs_bs.png  (1D rel-L2 ~ {np.mean(rels):.2e})")

# =========================================================================== #
# Figure 2 -- 1D first-order Greeks from the shape-gradient B (P2 on log-price)
# =========================================================================== #
# The Greeks come for free from the SAME separated solve: differentiate the FE field by
# swapping the shape function N for its derivative B in one direction (greeks_1d). P2
# (quadratic) elements on x = ln S make B element-linear -> O(h^2) Delta.
print("Computing 1D Greeks (P2 on x = ln S; Delta/Vega/Theta from shape-gradient B)...")
resG = solve_call_1d(K=K, r=r, q=q, T=T, nelem_x=100, nelem_s=100, nelem_t=100,
                     order={'x': 2, 's': 1, 't': 1})
Sg = np.linspace(0.6, 1.6, 61)
GK = ('delta', 'vega', 'theta')
GLAB = {'delta': r'$\Delta=\partial V/\partial S$',
        'vega':  r'$\mathcal{V}=\partial V/\partial \sigma$',
        'theta': r'$\Theta=\partial V/\partial t$'}
fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))
g_rel = {k: [] for k in GK}
for c, sv in zip([plt.cm.viridis(0.15), plt.cm.viridis(0.7)], (0.20, 0.35)):
    num = {k: np.array([greeks_1d(resG, float(s), sv, T)[k] for s in Sg]) for k in GK}
    ana = bs_greeks(Sg, K, r, q, sv, T)
    band = (Sg >= 0.5 * K) & (Sg <= 2.0 * K)          # near-the-money band for the reported error
    for ax, k in zip(axes, GK):
        g_rel[k].append(np.linalg.norm((num[k] - ana[k])[band]) / np.linalg.norm(ana[k][band]))
        ax.plot(Sg, ana[k], '-', color=c, lw=1.6, label=f"BS  $\\sigma$={sv:.2f}")
        ax.plot(Sg[::4], num[k][::4], 'o', color=c, ms=4, mfc='none', label=f"TD via $B$  $\\sigma$={sv:.2f}")
for ax, k in zip(axes, GK):
    ax.set_xlabel("spot  $S$"); ax.set_title(GLAB[k]); ax.grid(alpha=0.3)
axes[0].legend(fontsize=8)
fig.suptitle("1D European-call first-order Greeks: TD solver via shape-gradient $B$ (markers) "
             "vs Black-Scholes (lines)   |   P2 elements on $x=\\ln S$", y=1.03)
fig.tight_layout()
fig.savefig(os.path.join(DOCS, "greeks_1d_vs_bs.png"), dpi=130, bbox_inches='tight'); plt.close(fig)
print("  saved docs/greeks_1d_vs_bs.png  (rel-L2  "
      + ", ".join(f"{k} {np.mean(g_rel[k]):.1e}" for k in GK) + ")")

# =========================================================================== #
# D=2 basket solve (one offline solve), then figure + benchmark
# =========================================================================== #
print("Solving D=2 basket (Nx=90, Ns=Nt=50, ranks 19/80, 100 sweeps, sig-pad 0.05)...")
t0 = time.time()
res2 = solve_basket_2d(Nx=90, Ns=50, Nt=50, rank1=19, rank2=80, niter=100, sig_pad=0.05)
t_solve = time.time() - t0
V, M, D = res2['V'], res2['M'], res2['D']
w, q2, rho = res2['w'], res2['q'], res2['rho']

# --- Figure 3: basket price vs common spot S0=S1, at two volatilities, TD vs CV-MC ---
spots = np.linspace(0.6, 1.6, 11)
fig, ax = plt.subplots(figsize=(6.4, 4.6))
b_rels = []
for c, sv in zip([plt.cm.plasma(0.2), plt.cm.plasma(0.65)], (0.20, 0.35)):
    td = np.array([price_at(V, D, [sp, sp], [sv, sv], T, M) for sp in spots])
    mc = np.array([basket_cv_mc(np.array([sp, sp]), np.array([sv, sv]), K, r, q2, w, rho, T, n=300_000)
                   for sp in spots])
    b_rels.append(np.linalg.norm(td - mc) / np.linalg.norm(mc))
    ax.plot(spots, mc, '-', color=c, lw=1.6, label=f"CV-MC  $\\sigma$={sv:.2f}")
    ax.plot(spots, td, 'o', color=c, ms=5, mfc='none', label=f"TD  $\\sigma$={sv:.2f}")
ax.set_xlabel("spot per asset  $S_0=S_1$"); ax.set_ylabel("basket call price")
ax.set_title(f"D=2 basket call: TD solver (markers) vs control-variate MC (lines)\n"
             f"rel-L2: {', '.join(f'{e:.1e}' for e in b_rels)}  |  K={K}, $\\rho$=0.3, ATM at S=1")
ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(DOCS, "basket_2d_vs_mc.png"), dpi=130); plt.close(fig)
print(f"  saved docs/basket_2d_vs_mc.png  (2D rel-L2 ~ {np.mean(b_rels):.2e})")

# =========================================================================== #
# Benchmark -- nodal price (integer-index CP contraction) vs Monte-Carlo
# =========================================================================== #
print("\nBenchmarking per-price latency (D=2 basket, ATM)...")
xg, sg, tg = np.asarray(M['xg']), np.asarray(M['sg']), np.asarray(M['tg'])
ix = int(np.argmin(np.abs(np.exp(xg) - 1.0))); js = int(np.argmin(np.abs(sg - 0.25))); kt = len(tg) - 1
idx = (ix, ix, js, js, kt)
cp_eval(V, idx)                                       # warm up
nrep = 20_000
t = time.time()
for _ in range(nrep):
    cp_eval(V, idx)
td_us = (time.time() - t) / nrep * 1e6

S0, sig = np.ones(D), np.array([0.25, 0.25]); mc_paths = 300_000
t = time.time(); basket_cv_mc(S0, sig, K, r, q2, w, rho, T, n=mc_paths); mc_ms = (time.time() - t) * 1e3
speedup = (mc_ms * 1e3) / td_us
breakeven = t_solve / max(mc_ms * 1e-3 - td_us * 1e-6, 1e-12)

print("=" * 66)
print(f"  D=2 basket | Nx=90, Ns=Nt=50 | ranks 19/80 | offline solve {t_solve:.1f}s")
print(f"  TD nodal price (CP contraction)   : {td_us:10.2f}  us / price")
print(f"  Monte-Carlo ({mc_paths/1e3:g}k paths + CV)  : {mc_ms:10.3f}  ms / price")
print(f"  -> TD is ~{speedup:,.0f}x faster per query after the one-time solve")
print(f"  -> breakeven: the {t_solve:.0f}s solve is amortized after ~{breakeven:,.0f} priced points")
print("=" * 66)
