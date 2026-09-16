"""
Deterministic ODE mean-field companion to the unified CTMC.

Solves the same model using scipy solve_ivp, treating all state variables
as continuous. The ODE represents E[X(t)] of the stochastic process in the
large-N (N→∞) limit. At finite N the two diverge after the stochastic peak,
which is why both are included -- their comparison is informative.

The environmental compartment W is treated as fully continuous (first-order
ODE), which is consistent with the exact-exponential hybrid approximation
used in the CTMC for W.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from typing import Optional
from core.config_schema import DiseaseConfig


def ode_rhs(t: float, y: np.ndarray, cfg: DiseaseConfig) -> np.ndarray:
    """
    Right-hand side of the deterministic ODE system.

    State vector y: [S, E, I, Q, R, D, W]  (always 7 elements)
    - Q = 0 always if not quarantine_switch
    - W = 0 always if not environmental_switch (kappa very large otherwise)
    """
    S, E, I, Q, R, D, W = y
    N = cfg.N
    gamma_q = cfg.gamma_q if cfg.gamma_q is not None else cfg.gamma

    # Force of infection
    lambda_SE = cfg.beta * I / N
    if cfg.quarantine_switch and cfg.epsilon_q > 0:
        lambda_SE += cfg.beta * cfg.epsilon_q * Q / N
    if cfg.environmental_switch:
        lambda_SE += cfg.beta_W * W / N

    new_exposures = lambda_SE * S
    new_infectious = cfg.sigma * E

    dS = -new_exposures
    dE = new_exposures - new_infectious

    if cfg.quarantine_switch:
        dI = new_infectious - (cfg.gamma + cfg.delta) * I
        dQ = cfg.delta * I - gamma_q * Q
        dR = cfg.gamma * (1 - cfg.mu) * I + gamma_q * (1 - cfg.mu) * Q
        dD = cfg.gamma * cfg.mu * I + gamma_q * cfg.mu * Q
    else:
        dI = new_infectious - cfg.gamma * I
        dQ = 0.0
        dR = cfg.gamma * (1 - cfg.mu) * I
        dD = cfg.gamma * cfg.mu * I

    if cfg.environmental_switch:
        dW = cfg.epsilon_shed * I - cfg.kappa * W
    else:
        dW = 0.0

    return np.array([dS, dE, dI, dQ, dR, dD, dW])


def run_ode(cfg: DiseaseConfig, t_eval: Optional[np.ndarray] = None) -> pd.DataFrame:
    """
    Solve the deterministic ODE system and return a tidy daily DataFrame.
    """
    if t_eval is None:
        t_eval = np.linspace(0, cfg.T_end, int(cfg.T_end) + 1)

    y0 = np.array([
        cfg.N - cfg.I0 - cfg.E0 - cfg.Q0,  # S
        cfg.E0,
        cfg.I0,
        cfg.Q0,
        0.0,   # R
        0.0,   # D
        cfg.W0,  # W
    ], dtype=float)

    sol = solve_ivp(
        ode_rhs, t_span=[0, cfg.T_end], y0=y0, t_eval=t_eval,
        args=(cfg,), method="RK45", max_step=0.5,
        dense_output=False, rtol=1e-6, atol=1e-8,
    )

    df = pd.DataFrame(sol.y.T, columns=["S", "E", "I", "Q", "R", "D", "W"])
    df.insert(0, "day", sol.t)
    return df


def compute_r0_ngm(cfg: DiseaseConfig) -> float:
    """
    Next-generation matrix R0 -- identical to cfg.R0() but explicit.
    Provided as a stand-alone function for unit testing and documentation.

    See MATHEMATICAL_FRAMEWORK.md §3 for derivation.
    """
    return cfg.R0()


def final_attack_rate_analytical(cfg: DiseaseConfig) -> float:
    """
    Solve the implicit final-size equation for the SIR/SEIR limit.
    (Applies only when there is no vital dynamics, no stochastic fadeout,
    and no W or Q compartment for simplicity.)

    Equation (Kermack-McKendrick 1927):
        s_∞ = s_0 · exp(-R0 · (1 - s_∞))
    Solved numerically via Brentq on (0, 1).
    """
    R0 = cfg.R0()
    s0 = (cfg.N - cfg.I0 - cfg.E0) / cfg.N
    if R0 <= 1.0:
        return 0.0  # no major epidemic
    try:
        s_inf = brentq(lambda s: s - s0 * np.exp(-R0 * (1 - s)), 1e-8, s0 - 1e-8)
        return 1.0 - s_inf
    except ValueError:
        return float("nan")
