"""Closed-form and Monte-Carlo reference prices used to validate the TD solver.

All functions are pure (no global state): pass the option/market parameters explicitly.
"""
import numpy as np
from scipy.special import ndtr            # standard normal CDF


def bs_call(S, K, r, q, sigma, tau):
    """Black-Scholes-Merton European call on a single asset."""
    S = np.asarray(S, float)
    if tau <= 0:
        return np.maximum(S - K, 0.0)
    sq = sigma * np.sqrt(tau)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * tau) / sq
    d2 = d1 - sq
    return S * np.exp(-q * tau) * ndtr(d1) - K * np.exp(-r * tau) * ndtr(d2)


def bs_call_weighted(S, K, r, q, sigma, tau, w):
    """Value of the payoff max(w*S - K, 0): a single asset with weight w.

    This is the per-asset 'level-1' payoff of a weighted basket; equals
    w * BS_call(S, K/w, ...). Used to validate single-asset subproblems.
    """
    if tau <= 0:
        return max(w * S - K, 0.0)
    kap = K / w
    sq = sigma * np.sqrt(tau)
    d1 = (np.log(S / kap) + (r - q + 0.5 * sigma ** 2) * tau) / sq
    d2 = d1 - sq
    return float(w * S * np.exp(-q * tau) * ndtr(d1) - K * np.exp(-r * tau) * ndtr(d2))


def geo_basket_price(S0, sigma, tau, K, r, q, w, rho):
    """Closed-form GEOMETRIC-basket call price (used as the Monte-Carlo control variate).

    S0, sigma, q, w are length-D arrays; rho is the DxD correlation matrix.
    """
    S0 = np.asarray(S0, float); sigma = np.asarray(sigma, float)
    w = np.asarray(w, float); q = np.asarray(q, float)
    y = float(np.sum(w * np.log(S0)))
    sigG2 = float(w @ (rho * np.outer(sigma, sigma)) @ w)
    muG = float(np.sum(w * (r - q - 0.5 * sigma ** 2)))
    if tau <= 0:
        return max(np.exp(y) - K, 0.0)
    sigG = np.sqrt(sigG2)
    FG = np.exp(y) * np.exp((muG + 0.5 * sigG2) * tau)
    d1 = (np.log(FG / K) + 0.5 * sigG2 * tau) / (sigG * np.sqrt(tau))
    d2 = d1 - sigG * np.sqrt(tau)
    return float(np.exp(-r * tau) * (FG * ndtr(d1) - K * ndtr(d2)))


def basket_cv_mc(S0, sigma, K, r, q, w, rho, tau, n=2_000_000, seed=1):
    """Arithmetic-basket European call price by Monte-Carlo with the geometric-basket
    control variate (+ antithetic variates). Returns a scalar price.

    S0, sigma, q, w are length-D arrays; rho is the DxD correlation matrix.
    """
    S0 = np.asarray(S0, float); sigma = np.asarray(sigma, float)
    w = np.asarray(w, float); q = np.asarray(q, float)
    D = len(S0)
    L = np.linalg.cholesky(rho)
    rng = np.random.default_rng(seed)
    Zh = rng.standard_normal((n // 2, D)) @ L.T
    Z = np.vstack([Zh, -Zh])                              # antithetic
    ST = S0 * np.exp((r - q - 0.5 * sigma ** 2) * tau + sigma * np.sqrt(tau) * Z)
    pa = np.maximum(ST @ w - K, 0.0)                      # arithmetic basket payoff
    pg = np.maximum(np.exp(np.log(ST) @ w) - K, 0.0)      # geometric basket payoff (control)
    Eg = geo_basket_price(S0, sigma, tau, K, r, q, w, rho) * np.exp(r * tau)
    beta = np.cov(pa, pg)[0, 1] / (np.var(pg) + 1e-300)
    return float(np.exp(-r * tau) * (pa.mean() - beta * (pg.mean() - Eg)))
