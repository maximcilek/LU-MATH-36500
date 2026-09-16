"""
core/mcmc_audit.py
==================
Full audit of the fitting process itself, not just its output.

The gap this closes: the repository reported WHAT the sampler converged to but
almost nothing about HOW it got there. For a neural network you would expect a
loss curve and a ROC curve; the MCMC equivalents are the ones below. All of them
are written to CSV so the run is auditable without re-opening the trace.

    tables/mcmc_stepsize.csv            step size per chain per draw
    tables/mcmc_treedepth.csv           tree depth per chain per draw
    tables/mcmc_energy.csv              Hamiltonian energy + BFMI per chain
    tables/mcmc_accept.csv              acceptance probability per draw
    tables/mcmc_logp_trace.csv          log-posterior per draw (the "loss curve")
    tables/mcmc_rank_uniformity.csv     rank-plot uniformity test per parameter
    tables/mcmc_ess_evolution.csv       ESS as a function of chain length
    tables/mcmc_mass_matrix.csv         tuned diagonal mass matrix
    tables/mcmc_posterior_correlation.csv   parameter correlation matrix
    tables/mcmc_mcse_evolution.csv      Monte Carlo error vs draws used
    tables/mcmc_convergence_ledger.csv  one row per parameter, every diagnostic

Diagnostic references
---------------------
  Vehtari A, Gelman A, Simpson D, Carpenter B & Burkner P-C (2021).
    Rank-normalization, folding, and localization: an improved Rhat for
    assessing convergence of MCMC. Bayesian Analysis 16(2):667-718.
  Betancourt M (2018). A Conceptual Introduction to Hamiltonian Monte Carlo.
    arXiv:1701.02434.  (E-BFMI, energy diagnostics)
  Hoffman MD & Gelman A (2014). The No-U-Turn Sampler. JMLR 15:1351-1381.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import arviz as az
from scipy import stats


# ---------------------------------------------------------------------------
# Sampler-state extraction
# ---------------------------------------------------------------------------

def _stat(trace, name):
    ss = trace.sample_stats
    return np.asarray(ss[name].values) if name in ss else None


def step_size_table(trace) -> pd.DataFrame | None:
    """Step size per chain per draw. Flat after warmup means dual averaging settled."""
    v = _stat(trace, "step_size")
    if v is None:
        return None
    nc, nd = v.shape
    return pd.DataFrame({"chain": np.repeat(np.arange(nc), nd),
                         "draw": np.tile(np.arange(nd), nc),
                         "step_size": v.ravel()})


def treedepth_table(trace) -> pd.DataFrame | None:
    """
    Tree depth per draw. Saturation at max_treedepth means the sampler ran out
    of budget before the trajectory turned around: not wrong, but inefficient
    and usually a sign of a difficult geometry.
    """
    v = _stat(trace, "tree_depth")
    if v is None:
        v = _stat(trace, "treedepth")
    if v is None:
        return None
    nc, nd = v.shape
    return pd.DataFrame({"chain": np.repeat(np.arange(nc), nd),
                         "draw": np.tile(np.arange(nd), nc),
                         "tree_depth": v.ravel()})


def energy_table(trace) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """
    Hamiltonian energy per draw, plus E-BFMI per chain.

    E-BFMI below about 0.3 indicates the sampler is not exploring the energy
    distribution efficiently, typically because of heavy tails
    (Betancourt 2018, sec 6.1).
    """
    e = _stat(trace, "energy")
    if e is None:
        return None, None
    nc, nd = e.shape
    per_draw = pd.DataFrame({"chain": np.repeat(np.arange(nc), nd),
                             "draw": np.tile(np.arange(nd), nc),
                             "energy": e.ravel()})
    rows = []
    for c in range(nc):
        ec = e[c]
        de = np.diff(ec)
        bfmi = float(np.sum(de ** 2) / (len(ec) * np.var(ec))) if np.var(ec) > 0 else np.nan
        rows.append(dict(chain=c, mean_energy=float(ec.mean()),
                         sd_energy=float(ec.std()), bfmi=bfmi,
                         bfmi_ok=bool(np.isfinite(bfmi) and bfmi > 0.3)))
    return per_draw, pd.DataFrame(rows)


def accept_table(trace) -> pd.DataFrame | None:
    v = _stat(trace, "acceptance_rate")
    if v is None:
        v = _stat(trace, "mean_tree_accept")
    if v is None:
        return None
    nc, nd = v.shape
    return pd.DataFrame({"chain": np.repeat(np.arange(nc), nd),
                         "draw": np.tile(np.arange(nd), nc),
                         "accept_prob": v.ravel()})


def logp_trace(trace) -> pd.DataFrame | None:
    """
    Log-posterior per draw: the MCMC analogue of a training-loss curve.

    A healthy run shows all chains at the same level with no drift. A chain
    sitting systematically lower is stuck in a worse mode.
    """
    for nm in ("lp", "logp"):
        v = _stat(trace, nm)
        if v is not None:
            nc, nd = v.shape
            return pd.DataFrame({"chain": np.repeat(np.arange(nc), nd),
                                 "draw": np.tile(np.arange(nd), nc),
                                 "log_posterior": v.ravel()})
    return None


def mass_matrix_table(trace) -> pd.DataFrame | None:
    """Tuned diagonal mass matrix (inverse metric), if the sampler recorded it."""
    for nm in ("inverse_mass_matrix", "mass_matrix_inv"):
        v = _stat(trace, nm)
        if v is not None:
            arr = np.asarray(v)
            flat = arr.reshape(arr.shape[0], -1) if arr.ndim > 2 else arr
            return pd.DataFrame(flat).T.reset_index().rename(columns={"index": "param_index"})
    return None


# ---------------------------------------------------------------------------
# Convergence over the course of the run
# ---------------------------------------------------------------------------

def ess_evolution(trace, params: list[str], n_points: int = 12) -> pd.DataFrame:
    """
    ESS recomputed on increasing prefixes of the chain.

    A straight line through the origin means each additional draw buys the same
    amount of information. Flattening means the chain has stopped mixing and
    longer runs will not help.
    """
    post = trace.posterior
    rows = []
    for p in params:
        if p not in post:
            continue
        x = np.asarray(post[p].values)          # (chain, draw)
        nd = x.shape[1]
        for frac in np.linspace(1.0 / n_points, 1.0, n_points):
            k = max(int(nd * frac), 50)
            sub = az.convert_to_dataset({p: x[:, :k]})
            rows.append(dict(parameter=p, draws_used=k,
                             ess_bulk=float(az.ess(sub, var_names=[p])[p].values),
                             ess_tail=float(az.ess(sub, var_names=[p],
                                                   method="tail")[p].values)))
    return pd.DataFrame(rows)


def mcse_evolution(trace, params: list[str], n_points: int = 12) -> pd.DataFrame:
    """Monte Carlo standard error as a function of chain length. Should fall as 1/sqrt(n)."""
    post = trace.posterior
    rows = []
    for p in params:
        if p not in post:
            continue
        x = np.asarray(post[p].values)
        nd = x.shape[1]
        for frac in np.linspace(1.0 / n_points, 1.0, n_points):
            k = max(int(nd * frac), 50)
            sub = az.convert_to_dataset({p: x[:, :k]})
            rows.append(dict(parameter=p, draws_used=k,
                             mcse=float(az.mcse(sub, var_names=[p])[p].values),
                             running_mean=float(x[:, :k].mean())))
    return pd.DataFrame(rows)


def rank_uniformity(trace, params: list[str]) -> pd.DataFrame:
    """
    Rank-plot uniformity test (Vehtari et al. 2021).

    Pool all draws, rank them, and split by chain. If the chains are sampling
    the same distribution each chain's ranks are uniform. A chi-square test
    against uniform gives a single number per parameter; small p means the
    chains disagree about where the mass is, which Rhat can miss.
    """
    post = trace.posterior
    rows = []
    for p in params:
        if p not in post:
            continue
        x = np.asarray(post[p].values)
        nc, nd = x.shape
        r = stats.rankdata(x.ravel()).reshape(nc, nd)
        nbin = 20
        edges = np.linspace(0, nc * nd, nbin + 1)
        chi_tot, ok = 0.0, True
        for c in range(nc):
            obs, _ = np.histogram(r[c], bins=edges)
            exp = nd / nbin
            chi_tot += float(np.sum((obs - exp) ** 2 / exp))
        dof = nc * (nbin - 1)
        pval = float(1 - stats.chi2.cdf(chi_tot, dof))
        rows.append(dict(parameter=p, chi2=chi_tot, dof=dof, p_value=pval,
                         uniform=bool(pval > 0.01)))
    return pd.DataFrame(rows)


def posterior_correlation(trace, params: list[str]) -> pd.DataFrame:
    """
    Correlation matrix of the posterior draws.

    Strong pairwise correlation means the two parameters trade off against each
    other: the sampler must move them together, which slows mixing and is the
    signature of a ridge in the posterior.
    """
    post = trace.posterior
    cols = {p: np.asarray(post[p].values).ravel() for p in params if p in post}
    if not cols:
        return pd.DataFrame()
    df = pd.DataFrame(cols)
    df.columns = [c.replace("bhmm::", "") for c in df.columns]
    return df.corr()


def convergence_ledger(trace, params: list[str] | None = None) -> pd.DataFrame:
    """One row per parameter with every convergence diagnostic and a verdict."""
    s = az.summary(trace)
    if params:
        s = s.loc[[p for p in params if p in s.index]]
    led = s[["mean", "sd", "r_hat", "ess_bulk", "ess_tail", "mcse_mean"]].copy()
    led["mcse_over_sd_pct"] = led.mcse_mean / led.sd.replace(0, np.nan) * 100
    led["ess_per_1000"] = led.ess_bulk / max(int(trace.posterior.sizes["draw"]), 1) * 1000
    led["rhat_ok"] = led.r_hat <= 1.01
    led["ess_ok"] = led.ess_bulk >= 400
    led["mcse_ok"] = led.mcse_over_sd_pct <= 5
    led["verdict"] = np.where(led.rhat_ok & led.ess_ok & led.mcse_ok, "PASS", "REVIEW")
    return led.reset_index().rename(columns={"index": "parameter"})


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

def write_all(trace, tables_dir: Path, params: list[str]) -> dict:
    """Compute and persist every sampler-process table. Returns what was written."""
    td = Path(tables_dir); td.mkdir(parents=True, exist_ok=True)
    written = {}

    def put(name, df):
        if df is not None and len(df):
            df.to_csv(td / name, index=(name == "mcmc_posterior_correlation.csv"))
            written[name] = len(df)

    put("mcmc_stepsize.csv", step_size_table(trace))
    put("mcmc_treedepth.csv", treedepth_table(trace))
    e_draw, e_chain = energy_table(trace)
    put("mcmc_energy.csv", e_draw)
    put("mcmc_energy_bfmi.csv", e_chain)
    put("mcmc_accept.csv", accept_table(trace))
    put("mcmc_logp_trace.csv", logp_trace(trace))
    put("mcmc_mass_matrix.csv", mass_matrix_table(trace))
    put("mcmc_ess_evolution.csv", ess_evolution(trace, params))
    put("mcmc_mcse_evolution.csv", mcse_evolution(trace, params))
    put("mcmc_rank_uniformity.csv", rank_uniformity(trace, params))
    put("mcmc_posterior_correlation.csv", posterior_correlation(trace, params))
    put("mcmc_convergence_ledger.csv", convergence_ledger(trace, params))
    return written
