"""
Zombie SEZR prior-predictive inference.

The zombie disease has no real data. It appears in the BHMM comparison as
a prior-predictive distribution: the range of epidemic behaviour the model
predicts given the SEZR simulation parameters AS PRIORS, without any
likelihood update from real observations.

This is methodologically distinct from:
  (a) synthetic data generation (we do not create fake zombie observations)
  (b) calibration (we do not fit to zombie data — there is none)

It is equivalent to asking: "If we only knew the original SEZR parameters
and had no real outbreak data, what would the model expect?"

This approach is documented in the epidemiology literature as
"expert-elicited prior predictive checks" (Gelman et al. 2020,
Bayesian Data Analysis 3e, Ch. 6).
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import json
import pandas as pd
from core.bhmm.bhmm_model import ode_prevalence


def zombie_prior_predictive(
    prior_params_path: Path,
    n_samples: int = 1000,
    seed: int = 20260914,
) -> dict:
    """
    Draw n_samples from the zombie prior and propagate each through the ODE.

    Returns a dict with:
      - 'I_t_samples' : shape (n_samples, T_end+1) — prevalence trajectories
      - 'R0_samples'  : shape (n_samples,) — implied R0 per draw
      - 'peak_I_samples': shape (n_samples,) — peak prevalence per draw
      - 'peak_day_samples': shape (n_samples,) — day of peak per draw
      - 'final_D_samples': shape (n_samples,) — final deaths/N per draw
      - 'params_df'   : DataFrame of sampled parameters

    No observations are generated. No likelihood is computed.
    This is purely a forward simulation from the prior.
    """
    rng = np.random.default_rng(seed)

    with open(prior_params_path) as f:
        spec = json.load(f)

    params = spec["parameters"]
    N   = spec["N_population"]
    T   = spec["T_end_days"]
    I0  = 1   # single index case

    # Sample from prior distributions
    # Beta is LogNormal: sample log_beta ~ Normal(log(mu), sd)
    log_beta_samples  = rng.normal(
        loc=np.log(params["beta"]["prior_mean"]),
        scale=params["beta"]["prior_sd"],
        size=n_samples,
    )
    log_sigma_samples = rng.normal(
        loc=np.log(params["sigma"]["prior_mean"]),
        scale=params["sigma"]["prior_sd"],
        size=n_samples,
    )
    log_gamma_samples = rng.normal(
        loc=np.log(params["gamma"]["prior_mean"]),
        scale=params["gamma"]["prior_sd"],
        size=n_samples,
    )
    # mu from Beta(2, 18) → mean ≈ 0.10
    mu_samples = rng.beta(
        a=params["mu"]["prior_mean"] * 20,
        b=(1 - params["mu"]["prior_mean"]) * 20,
        size=n_samples,
    )

    # Propagate each draw through the ODE
    I_t_list      = []
    R0_list       = []
    peak_I_list   = []
    peak_day_list = []
    final_D_list  = []

    for i in range(n_samples):
        lb, ls, lg, mu = (log_beta_samples[i], log_sigma_samples[i],
                          log_gamma_samples[i], mu_samples[i])
        I_t = ode_prevalence(lb, ls, lg, mu, N, I0, T, T + 1)
        R0  = np.exp(lb) / np.exp(lg)

        I_t_list.append(I_t / N)    # store as prevalence (0-1)
        R0_list.append(R0)
        peak_I_list.append(I_t.max() / N)
        peak_day_list.append(int(np.argmax(I_t)))
        # Approximate final deaths: mu * (N - S_inf)
        # S_inf ≈ N * exp(-R0) for large R0 (K-M approximation)
        S_inf = N * np.exp(-R0) if R0 > 1 else N * 0.99
        final_D_list.append(mu * max(0, N - S_inf) / N)

    params_df = pd.DataFrame({
        "beta":  np.exp(log_beta_samples),
        "sigma": np.exp(log_sigma_samples),
        "gamma": np.exp(log_gamma_samples),
        "mu":    mu_samples,
        "R0":    np.array(R0_list),
    })

    return {
        "I_t_samples":    np.array(I_t_list),   # (n_samples, T+1)
        "R0_samples":     np.array(R0_list),
        "peak_I_samples": np.array(peak_I_list),
        "peak_day_samples": np.array(peak_day_list),
        "final_D_samples": np.array(final_D_list),
        "params_df":      params_df,
        "T_end":          T,
        "N":              N,
        "n_samples":      n_samples,
        "source":         "PRIOR PREDICTIVE ONLY — no real zombie data exists",
    }


def summarise_prior_predictive(pp_result: dict) -> pd.DataFrame:
    """Return summary statistics of the prior predictive distribution."""
    R0 = pp_result["R0_samples"]
    pk = pp_result["peak_I_samples"]
    pd_ = pp_result["peak_day_samples"]
    fd  = pp_result["final_D_samples"]

    rows = []
    for name, arr in [("R0", R0), ("peak_prevalence", pk),
                       ("peak_day", pd_), ("final_dead_fraction", fd)]:
        rows.append({
            "quantity": name,
            "mean":  np.mean(arr),
            "sd":    np.std(arr),
            "q025":  np.quantile(arr, 0.025),
            "q250":  np.quantile(arr, 0.25),
            "q500":  np.quantile(arr, 0.50),
            "q750":  np.quantile(arr, 0.75),
            "q975":  np.quantile(arr, 0.975),
        })
    return pd.DataFrame(rows)
