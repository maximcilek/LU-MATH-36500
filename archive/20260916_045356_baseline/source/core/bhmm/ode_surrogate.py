"""
core/bhmm/ode_surrogate.py
==========================
R0-indexed ODE surrogate: precomputed trajectory grid + differentiable
linear interpolation in PyTensor.

WHY THIS MODULE EXISTS  (v6 critical bug fix)
---------------------------------------------
In v5 the mechanistic trajectory was solved ONCE, outside the model, at the
prior-mean rate parameters:

    e_base = ode_prevalence(EBOLA_PRIOR.log_beta_mu, ...)      # fixed numpy array
    e_mu_obs = exp(log_intercept_e + log_rw_e[t]) * e_base[t]  # R0 absent

The sampled `e_R0` / `flu_R0` therefore never entered the likelihood. They were
pure `pm.Deterministic` transformations of the hyperprior and an un-updated
Normal(0,1) raw variable. Empirical confirmation from the v5 production trace
(4 chains x 2000 draws, seed=20260914):

    e_log_R0_raw    posterior mean = -0.0157,  sd = 1.0105   (prior: 0, 1)
    flu_log_R0_raw  posterior mean = -0.0005,  sd = 1.0124   (prior: 0, 1)
    n_log_R0_raw    posterior mean = -0.6206,  sd = 0.8949   (prior: 0, 1)  <- updated

Only Norovirus updated, because only Norovirus routed R0 into the likelihood
(via the Kermack-McKendrick final-size equation). The reported Ebola and
Influenza R0 values were the Norovirus-driven hyperprior rescaled by each
disease's literature R0_base -- not estimates from their own data.

THE FIX
-------
Precompute the mechanistic trajectory on a grid of R0 values, then interpolate
at the sampled R0 inside the model. The interpolation is differentiable, so
NUTS gradients flow from the likelihood back to R0 and the data identifies it.

    idx = (log R0 - log R0_min) / dlog
    i0  = floor(idx);  w = idx - i0
    traj(R0) = (1-w) * GRID[i0] + w * GRID[i0+1]

d traj / d log R0 = (GRID[i0+1] - GRID[i0]) / dlog, the central finite-difference
derivative. With a fine grid (default 160 nodes spanning R0 in [0.3, 15]) the
piecewise-linear surrogate error is far below Monte Carlo error.

PRECEDENT
---------
Emulating an expensive simulator on a parameter grid and interpolating is
standard practice in Bayesian calibration of computer models:
  Kennedy MC & O'Hagan A (2001). Bayesian calibration of computer models.
    J R Stat Soc B 63(3):425-464.
  Conti S & O'Hagan A (2010). Bayesian emulation of complex multi-output and
    dynamic computer models. J Stat Plan Inference 140(3):640-651.
We use exact ODE solutions on a dense grid rather than a GP emulator, so there
is no emulator uncertainty to propagate -- only bounded interpolation error,
which is reported by `grid_interpolation_error()`.

SCOPE / LIMITATION (documented, not hidden)
-------------------------------------------
The grid varies R0 only. The remaining rate parameters (sigma, gamma, delta)
are held at their literature prior means. R0 = beta / (removal rate), so
varying R0 at fixed removal rates is equivalent to varying beta. The generation
time is therefore fixed, not estimated. This is a deliberate restriction: with
14 (Influenza) and 139 (Ebola) observations, jointly identifying R0 and the
generation-interval distribution is not feasible, and attempting it produces the
same non-identifiability this module exists to fix. Sensitivity of R0 to the
assumed generation time is assessed in scripts/02_prefit_diagnostics.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pytensor.tensor as pt
from scipy.integrate import solve_ivp

# ---------------------------------------------------------------------------
# Compartmental right-hand sides
# ---------------------------------------------------------------------------


def _rhs_seird(t, y, beta, sigma, gamma, mu, N):
    """S -> E -> I -> {R, D}.  Used for Ebola (with delta folded into gamma)."""
    S, E, I, R, D = y
    foi = beta * I / N
    return [
        -foi * S,
        foi * S - sigma * E,
        sigma * E - gamma * I,
        gamma * (1.0 - mu) * I,
        gamma * mu * I,
    ]


def _rhs_seiqr_erlang(t, y, beta, sigma, delta, gamma_q, N, kE, kI, kQ):
    """
    S -> E(kE stages) -> I(kI stages) -> Q(kQ stages) -> R, via the linear
    chain trick: k sequential sub-compartments each at rate k*rate give an
    Erlang(k, k*rate) residence time with the same mean and variance 1/(k*rate^2).
    k = 1 recovers the exponential case.

    Non-exponential residence times are required to fit the 1978 boarding-school
    prevalence curve; see Avilov KK et al. (2024) J R Soc Interface 21:20240394,
    who show that exponential residence times produce a curve that is too broad
    and significantly overestimates the outbreak timespan.

    Reference for the linear chain trick:
      Lloyd AL (2001). Realistic distributions of infectious periods in epidemic
        models. Theor Popul Biol 60(1):59-71.
      Wearing HJ, Rohani P & Keeling MJ (2005). Appropriate models for the
        management of infectious diseases. PLoS Med 2(7):e174.
    """
    S = y[0]
    E = y[1:1 + kE]
    I = y[1 + kE:1 + kE + kI]
    Q = y[1 + kE + kI:1 + kE + kI + kQ]
    foi = beta * I.sum() / N          # Q excluded from FOI by construction

    dS = -foi * S
    dE = np.empty(kE)
    dE[0] = foi * S - kE * sigma * E[0]
    for j in range(1, kE):
        dE[j] = kE * sigma * (E[j - 1] - E[j])
    dI = np.empty(kI)
    dI[0] = kE * sigma * E[-1] - kI * delta * I[0]
    for j in range(1, kI):
        dI[j] = kI * delta * (I[j - 1] - I[j])
    dQ = np.empty(kQ)
    dQ[0] = kI * delta * I[-1] - kQ * gamma_q * Q[0]
    for j in range(1, kQ):
        dQ[j] = kQ * gamma_q * (Q[j - 1] - Q[j])
    dR = kQ * gamma_q * Q[-1]
    return np.concatenate([[dS], dE, dI, dQ, [dR]])


def _rhs_seiqr(t, y, beta, sigma, delta, gamma_q, N):
    """
    S -> E -> I -> Q -> R.

    Q is the *confined-to-bed / isolated* compartment. Individuals in Q are
    removed from transmission (they do not appear in the force of infection).
    This is the structure required when the observed quantity counts people who
    are withdrawn from circulation rather than people who are transmitting.

    1/delta   = mean infectious (circulating) period before confinement
    1/gamma_q = mean duration of confinement
    R0        = beta / delta   (removal from the infectious pool happens at delta)
    """
    S, E, I, Q, R = y
    foi = beta * I / N          # Q excluded from FOI by construction
    return [
        -foi * S,
        foi * S - sigma * E,
        sigma * E - delta * I,
        delta * I - gamma_q * Q,
        gamma_q * Q,
    ]


# ---------------------------------------------------------------------------
# Grid specification
# ---------------------------------------------------------------------------


@dataclass
class ODEGrid:
    """
    A precomputed, R0-indexed trajectory grid ready for PyTensor interpolation.

    Attributes
    ----------
    traj        : (G, T) trajectories of the observed quantity, in person-units
    log_r0      : (G,) log R0 nodes, uniformly spaced
    log_r0_lo   : first node
    dlog        : node spacing
    G, T        : grid size and window length
    observable  : which compartment/flux the rows represent
    meta        : all structural constants used to build the grid
    """

    traj: np.ndarray
    log_r0: np.ndarray
    log_r0_lo: float
    dlog: float
    G: int
    T: int
    observable: str
    meta: dict = field(default_factory=dict)

    # -- PyTensor side ----------------------------------------------------
    def as_tensor(self):
        return pt.as_tensor_variable(self.traj.astype(np.float64))

    def interpolate(self, log_R0):
        """
        Differentiable linear interpolation of the trajectory at a sampled log R0.

        Parameters
        ----------
        log_R0 : scalar PyTensor variable

        Returns
        -------
        (T,) PyTensor vector: the mechanistic trajectory at that R0.
        """
        grid = self.as_tensor()
        idx = (log_R0 - self.log_r0_lo) / self.dlog
        idx = pt.clip(idx, 0.0, float(self.G - 1) - 1e-6)
        i0 = pt.cast(pt.floor(idx), "int64")
        w = idx - pt.floor(idx)
        return (1.0 - w) * grid[i0] + w * grid[i0 + 1]

    # -- diagnostics ------------------------------------------------------
    def interpolation_error(self, n_test: int = 40, seed: int = 0) -> dict:
        """
        Max relative error of the piecewise-linear surrogate against exact ODE
        solves at random off-node R0 values. Reported in the pre-fit audit so a
        reviewer can see the surrogate error is negligible.
        """
        rng = np.random.default_rng(seed)
        lo, hi = self.log_r0[0], self.log_r0[-1]
        test = rng.uniform(lo, hi, n_test)
        rel = []
        for lr in test:
            exact = solve_window(np.exp(lr), **self.meta)
            idx = (lr - self.log_r0_lo) / self.dlog
            i0 = int(np.floor(idx))
            i0 = min(max(i0, 0), self.G - 2)
            w = idx - i0
            approx = (1 - w) * self.traj[i0] + w * self.traj[i0 + 1]
            denom = np.maximum(np.abs(exact), 1e-9)
            rel.append(float(np.max(np.abs(approx - exact) / denom)))
        rel = np.asarray(rel)
        return {
            "n_test": n_test,
            "max_rel_error": float(rel.max()),
            "mean_rel_error": float(rel.mean()),
            "p95_rel_error": float(np.quantile(rel, 0.95)),
        }


# ---------------------------------------------------------------------------
# Single-R0 solve
# ---------------------------------------------------------------------------


def solve_window(
    R0: float,
    *,
    structure: Literal["seird_incidence", "seiqr_prevalence"],
    sigma: float,
    removal: float,
    gamma_q: float,
    mu: float,
    N: int,
    I0: float,
    t_seed: float,
    T: int,
    dt_obs: float = 1.0,
    kE: int = 1,
    kI: int = 1,
    kQ: int = 1,
) -> np.ndarray:
    """
    Solve the mechanistic model at one R0 and return the observed quantity on
    the T-point observation window.

    `t_seed` shifts the observation window forward from the epidemic seed, so
    that observation day 0 corresponds to model time t_seed. Outbreak records
    almost never begin at the true seeding event; forcing them to do so is what
    produced the v5 timing failure (ODE peaked at day 36, data peaked at day 6).

    structure="seird_incidence"
        Returns the incidence flux sigma * E(t) -- new symptomatic cases per
        unit time. Correct observable for a daily *onset* series (Ebola).

    structure="seiqr_prevalence"
        Returns Q(t) -- the number currently confined. Correct observable for a
        daily *confined-to-bed* series (Influenza 1978).
    """
    t_eval = t_seed + np.arange(T) * dt_obs
    t_end = float(t_eval[-1])

    if structure == "seird_incidence":
        beta = R0 * removal
        y0 = [N - I0, 0.0, float(I0), 0.0, 0.0]
        sol = solve_ivp(
            _rhs_seird, [0.0, t_end], y0, t_eval=t_eval,
            args=(beta, sigma, removal, mu, N),
            method="RK45", rtol=1e-8, atol=1e-10,
        )
        E = np.clip(sol.y[1], 0.0, None)
        return sigma * E                      # incidence flux, persons/day

    if structure == "seiqr_prevalence":
        beta = R0 * removal                   # removal == delta (I -> Q)
        y0 = np.zeros(1 + kE + kI + kQ + 1)
        y0[0] = N - I0
        y0[1 + kE] = float(I0)                # seed the first I stage
        sol = solve_ivp(
            _rhs_seiqr_erlang, [0.0, t_end], y0, t_eval=t_eval,
            args=(beta, sigma, removal, gamma_q, N, kE, kI, kQ),
            method="RK45", rtol=1e-8, atol=1e-10,
        )
        Q = sol.y[1 + kE + kI:1 + kE + kI + kQ]
        return np.clip(Q.sum(axis=0), 0.0, None)   # total confined, persons

    raise ValueError(f"unknown structure: {structure!r}")


# ---------------------------------------------------------------------------
# Grid builder
# ---------------------------------------------------------------------------


def build_grid(
    *,
    structure: Literal["seird_incidence", "seiqr_prevalence"],
    sigma: float,
    removal: float,
    N: int,
    T: int,
    gamma_q: float = 0.0,
    mu: float = 0.0,
    I0: float = 1.0,
    t_seed: float = 0.0,
    dt_obs: float = 1.0,
    kE: int = 1,
    kI: int = 1,
    kQ: int = 1,
    R0_lo: float = 0.3,
    R0_hi: float = 15.0,
    G: int = 160,
) -> ODEGrid:
    """
    Precompute trajectories over a log-uniform R0 grid.

    The default span [0.3, 15] comfortably contains every posterior mass region
    the hyperprior can reach, so `interpolate` never clips in practice. Clipping
    would silently flatten the gradient, so `scripts/02_prefit_diagnostics.py`
    asserts that the posterior stays inside the grid.
    """
    log_r0 = np.linspace(np.log(R0_lo), np.log(R0_hi), G)
    meta = dict(
        structure=structure, sigma=sigma, removal=removal, gamma_q=gamma_q,
        mu=mu, N=N, I0=I0, t_seed=t_seed, T=T, dt_obs=dt_obs,
        kE=kE, kI=kI, kQ=kQ,
    )
    rows = [solve_window(float(np.exp(lr)), **meta) for lr in log_r0]
    traj = np.vstack(rows)

    # Guard against a degenerate grid: if the trajectory does not change with
    # R0, interpolation carries no gradient and R0 is unidentified again.
    spread = float(np.max(np.std(traj, axis=0) / (np.mean(np.abs(traj), axis=0) + 1e-12)))
    if spread < 1e-3:
        raise RuntimeError(
            f"ODE grid for structure={structure!r} is flat in R0 "
            f"(max relative spread {spread:.2e}). R0 would be unidentified. "
            "Check sigma/removal/t_seed/T."
        )

    return ODEGrid(
        traj=traj, log_r0=log_r0, log_r0_lo=float(log_r0[0]),
        dlog=float(log_r0[1] - log_r0[0]), G=G, T=T,
        observable=("incidence sigma*E(t)" if structure == "seird_incidence" else "prevalence Q(t)"),
        meta=meta,
    )


# ---------------------------------------------------------------------------
# Irregular-time GRW scaling
# ---------------------------------------------------------------------------


def grw_sigma_vector(day_index: np.ndarray, sigma_rw) -> "pt.TensorVariable":
    """
    Per-step innovation SD for a random walk observed at irregular times.

    A Gaussian random walk is the discretisation of a Wiener process. If
    consecutive observations are separated by dt days, the innovation variance
    over that interval is sigma^2 * dt, not sigma^2. Using a constant sigma on
    an irregularly spaced series understates the variance across gaps and
    overstates it across consecutive days.

    This matters for Ebola: 53 of 192 calendar days are excluded as structural
    zeros, so the 139 retained observations are irregularly spaced. Indexing the
    GRW by *observation* rather than by calendar day also reduces the latent
    dimension from T=193 to T=139, improving the latent-to-observation ratio
    from 1.39 to 1.00.

    Reference
    ---------
    Durbin J & Koopman SJ (2012). Time Series Analysis by State Space Methods,
      2e. Oxford University Press. Ch. 3 (continuous-time state space models and
      their discrete-time equivalents at irregular observation times).

    Returns
    -------
    (n_obs - 1,) PyTensor vector of per-step innovation SDs.
    """
    gaps = np.diff(np.asarray(day_index, dtype=np.float64))
    if np.any(gaps <= 0):
        raise ValueError("day_index must be strictly increasing")
    return sigma_rw * pt.as_tensor_variable(np.sqrt(gaps))
