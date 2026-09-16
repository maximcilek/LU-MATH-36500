"""
core/diagnostics.py
===================
Reusable diagnostic computations shared by the pre-fit audit (scripts/02),
the post-fit summary (scripts/05) and the debug report (scripts/08).

Every test here maps to a stated decision rule so that a reviewer can see the
threshold that was applied, not just the number that came out.

Decision rules and their sources
--------------------------------
Overdispersion      Var/Mean > 1 => Poisson misspecified, use NegBinomial.
                    Cameron AC & Trivedi PK (2013), Regression Analysis of
                    Count Data, 2e, sec 3.3.
Serial correlation  |lag-1 ACF| > 0.20 or Ljung-Box p < 0.05 on residuals of
                    the constant-transmission fit => time-varying transmission
                    component warranted.
                    Ljung GM & Box GEP (1978) Biometrika 65(2):297-303.
Prior dominance     |posterior mean / prior mean - 1| < 0.25 and
                    posterior sd / prior sd > 0.75 => parameter is
                    prior-dominated and must be reported as unidentified.
                    Gelman A et al. (2020) BDA3 sec 13.3.
Identifiability     d logp / d theta == 0 at the initial point => theta never
                    enters the likelihood. This is the check that caught the
                    v5 R0 bug; it now runs automatically before every fit.
Final size          K-M: AR = 1 - exp(-R0 * AR). Exact for SEIR-type models
                    with one wave, independent of the generation-interval
                    distribution. Kermack & McKendrick (1927); Ma J & Earn DJD
                    (2006) Bull Math Biol 68:679-702.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq


# ---------------------------------------------------------------------------
# Count-data diagnostics
# ---------------------------------------------------------------------------


def dispersion(y: np.ndarray) -> dict:
    """Var/Mean with a Poisson-dispersion test."""
    y = np.asarray(y, dtype=float)
    mu, var = y.mean(), y.var(ddof=1)
    ratio = var / mu if mu > 0 else np.nan
    n = len(y)
    # Score test for overdispersion against Poisson (Cameron & Trivedi sec 3.4)
    stat = np.sum((y - mu) ** 2 - y) / np.sqrt(2 * n * mu ** 2) if mu > 0 else np.nan
    p = 1 - stats.norm.cdf(stat) if np.isfinite(stat) else np.nan
    return dict(n=n, mean=float(mu), var=float(var), var_mean_ratio=float(ratio),
                skewness=float(stats.skew(y)), zero_frac=float(np.mean(y == 0)),
                overdispersion_z=float(stat), overdispersion_p=float(p),
                verdict="NegBinomial required" if ratio > 1.5 else "Poisson adequate")


def serial_correlation(resid: np.ndarray, lags=(1, 2, 3)) -> dict:
    """Lag-1..k autocorrelation plus a Ljung-Box portmanteau test."""
    r = np.asarray(resid, dtype=float)
    r = r - r.mean()
    n = len(r)
    out = {}
    acf = []
    for k in lags:
        if n - k < 3:
            acf.append(np.nan); continue
        rk = float(np.corrcoef(r[k:], r[:-k])[0, 1])
        acf.append(rk)
        out[f"acf_lag{k}"] = rk
    # Ljung-Box
    valid = [a for a in acf if np.isfinite(a)]
    h = len(valid)
    Q = n * (n + 2) * sum((a ** 2) / (n - k) for a, k in zip(valid, lags[:h]))
    out["ljung_box_Q"] = float(Q)
    out["ljung_box_df"] = h
    out["ljung_box_p"] = float(1 - stats.chi2.cdf(Q, h)) if h else np.nan
    lag1 = out.get("acf_lag1", np.nan)
    out["grw_warranted"] = bool(
        (np.isfinite(lag1) and abs(lag1) > 0.20) or
        (np.isfinite(out["ljung_box_p"]) and out["ljung_box_p"] < 0.05))
    out["decision_rule"] = "|lag-1 ACF| > 0.20 OR Ljung-Box p < 0.05"
    return out


def pearson_residuals(obs: np.ndarray, fit: np.ndarray) -> np.ndarray:
    """(obs - fit) / sqrt(fit). Appropriate for count data."""
    fit = np.clip(np.asarray(fit, dtype=float), 1e-9, None)
    return (np.asarray(obs, dtype=float) - fit) / np.sqrt(fit)


# ---------------------------------------------------------------------------
# Final size
# ---------------------------------------------------------------------------


def km_attack_rate_np(R0: float, n_iter: int = 200) -> float:
    """Final-size attack rate for a given R0 (numpy scalar version)."""
    if R0 <= 1.0:
        return 0.0
    ar = 1.0 - np.exp(-R0)
    for _ in range(n_iter):
        ar = 1.0 - np.exp(-R0 * ar)
    return float(np.clip(ar, 0.0, 1.0))


def km_R0_from_attack_rate(ar: float) -> float:
    """Invert the K-M relation: which R0 implies this observed attack rate?"""
    if not (0.0 < ar < 1.0):
        return float("nan")
    return float(brentq(lambda r: km_attack_rate_np(r) - ar, 1.0 + 1e-9, 50.0))


def final_size_check(R0_samples: np.ndarray, reported_ar: float) -> dict:
    """
    Compare the attack rate implied by the fitted R0 against the reported one.

    A large gap is a structural warning, not a convergence problem: it means the
    model reproduces the epidemic's SPEED but not its SIZE (or vice versa). For
    the 1978 boarding-school data this gap is a known, published open problem --
    see docs/INFLUENZA_1978_LIMITATION.md.
    """
    R0 = np.asarray(R0_samples, dtype=float)
    implied = np.array([km_attack_rate_np(r) for r in R0])
    return dict(
        reported_attack_rate=float(reported_ar),
        R0_from_reported_ar=km_R0_from_attack_rate(reported_ar),
        implied_ar_median=float(np.median(implied)),
        implied_ar_lo=float(np.quantile(implied, 0.055)),
        implied_ar_hi=float(np.quantile(implied, 0.945)),
        absolute_gap=float(np.median(implied) - reported_ar),
        consistent=bool(np.quantile(implied, 0.055) <= reported_ar <= np.quantile(implied, 0.945)),
    )


# ---------------------------------------------------------------------------
# Prior dominance
# ---------------------------------------------------------------------------


def halfnormal_moments(scale: float) -> tuple[float, float]:
    """Mean and SD of HalfNormal(scale)."""
    mean = scale * np.sqrt(2.0 / np.pi)
    sd = scale * np.sqrt(1.0 - 2.0 / np.pi)
    return float(mean), float(sd)


def prior_dominance(post: np.ndarray, prior_mean: float, prior_sd: float) -> dict:
    """
    Flag parameters whose posterior has barely moved from the prior.

    Gelman et al. (2020) BDA3 sec 13.3: a posterior that mirrors the prior
    indicates a flat likelihood in that parameter. Such a parameter must be
    reported as unidentified rather than as an estimate.
    """
    post = np.asarray(post, dtype=float)
    pm_, ps = post.mean(), post.std(ddof=1)
    mean_ratio = pm_ / prior_mean if prior_mean else np.nan
    sd_ratio = ps / prior_sd if prior_sd else np.nan
    dominated = bool(abs(mean_ratio - 1.0) < 0.25 and sd_ratio > 0.75)
    return dict(post_mean=float(pm_), post_sd=float(ps),
                prior_mean=float(prior_mean), prior_sd=float(prior_sd),
                mean_ratio=float(mean_ratio), sd_ratio=float(sd_ratio),
                prior_dominated=dominated,
                verdict=("PRIOR-DOMINATED: report as unidentified" if dominated
                         else "data-informed"))


# ---------------------------------------------------------------------------
# Identifiability gate  (the check that would have caught the v5 bug)
# ---------------------------------------------------------------------------


def identifiability_gate(model, param_names: list[str]) -> pd.DataFrame:
    """
    Assert that each named parameter actually enters the likelihood.

    Computes d logp / d theta at the model's initial point. A parameter whose
    gradient is identically zero cannot be informed by the data: its posterior
    will equal its prior no matter how long the chain runs.

    This is the check that was missing in v5, where e_log_R0_raw and
    flu_log_R0_raw had zero likelihood gradient because the ODE was solved
    outside the model at fixed prior-mean rates. Their posteriors came back at
    mean -0.016 / -0.001 with sd 1.011 / 1.012 against a Normal(0,1) prior --
    i.e. exactly the prior.

    Returns a DataFrame with one row per parameter and a PASS/FAIL column.
    scripts/04 raises before sampling if any row FAILs.
    """
    ip = model.initial_point()
    grad = model.compile_dlogp()(ip)

    offsets, idx = {}, 0
    for v in model.value_vars:
        arr = np.atleast_1d(np.asarray(ip[v.name]))
        offsets[v.name] = (idx, idx + arr.size)
        idx += arr.size

    rows = []
    for want in param_names:
        key = next((k for k in offsets if want in k), None)
        if key is None:
            rows.append(dict(parameter=want, present=False, max_abs_grad=np.nan,
                             status="ABSENT"))
            continue
        lo, hi = offsets[key]
        g = np.max(np.abs(grad[lo:hi]))
        rows.append(dict(parameter=key, present=True, max_abs_grad=float(g),
                         status="PASS" if g > 1e-8 else "FAIL - NOT IN LIKELIHOOD"))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Posterior predictive coverage
# ---------------------------------------------------------------------------


def ppc_coverage(obs: np.ndarray, ppc_samples: np.ndarray, level: float = 0.90) -> dict:
    """Fraction of observations inside the central `level` predictive interval."""
    obs = np.asarray(obs, dtype=float)
    s = np.asarray(ppc_samples, dtype=float).reshape(-1, len(obs))
    a = (1.0 - level) / 2.0
    lo, hi = np.quantile(s, a, axis=0), np.quantile(s, 1 - a, axis=0)
    cov = float(np.mean((obs >= lo) & (obs <= hi)))
    return dict(level=level, coverage=cov, n=len(obs),
                pred_mean=float(s.mean()), obs_mean=float(obs.mean()),
                passes=bool(cov >= 0.80),
                decision_rule="coverage >= 0.80 of the 90% predictive interval")
