"""
core/bhmm/zombie_arm.py
=======================
Two routes for an undocumented pathogen, implemented side by side.

    Route A  ASSUMPTION-DRIVEN   (prior predictive; v5-v7 behaviour)
    Route B  POOLING-DRIVEN      (hierarchical borrowing; new in v8)

WHY BOTH
--------
The zombie SEZR arm has no observations and none may be invented. That leaves
exactly two defensible things to do, and they answer different questions.

Route A -- prior predictive
    Choose values for beta, sigma, gamma from the SEZR specification, push them
    through the ODE, and report the resulting distribution of outbreaks.

    The posterior equals the prior. Nothing is learned. The output is the
    assumption re-expressed in epidemic units. The only honest claim is
    conditional:

        "IF a pathogen had these transmission and recovery properties,
         THEN its outbreak would look like this."

    Saying "the zombie R0 is 3.03" is not a finding; it is the input restated.
    Reference: Gelman A et al. (2020) Bayesian Data Analysis 3e, sec 6.4
    (prior predictive checking).

Route B -- hierarchical borrowing
    Place the pathogen INSIDE the cross-disease hierarchy with no likelihood
    term. It contributes nothing to the fit, but its log R0 is drawn from the
    same hyperprior that Ebola, Measles and Norovirus inform. Its posterior is
    then the posterior predictive of the hyperprior: what a fourth, unobserved
    human pathogen would plausibly look like given three that were measured.

        "GIVEN three human outbreaks we measured, a fourth unmeasured one
         would plausibly fall in this range."

    This is a genuine statement about human pathogens, supported by data,
    without inventing a single zombie observation. It is the standard
    hierarchical device for a new group with no data of its own.
    Reference: Gelman A et al. (2020) BDA3 sec 5.5 (prediction for a new
    group); Gelman A & Hill J (2007) Data Analysis Using Regression and
    Multilevel/Hierarchical Models, sec 12.2.

WHAT ROUTE B IS NOT
-------------------
It is not a statement about zombies specifically. The model has no zombie
information and cannot acquire any. Route B describes the population of human
pathogens the three measured arms are drawn from. Presenting it as a
zombie-specific estimate would be the same error as Route A, one level up.

Both routes are computed every run and reported side by side so the difference
is visible rather than assumed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Route A -- assumption-driven
# ---------------------------------------------------------------------------

def route_a_prior_predictive(prior_params_path: Path, n: int = 4000,
                             seed: int = 20260914) -> dict:
    """
    Draw from the declared SEZR parameter priors and report implied R0.

    No observations are created and no likelihood is evaluated. The returned
    distribution is the assumption, propagated.
    """
    spec = json.loads(Path(prior_params_path).read_text())
    p = spec["parameters"]
    rng = np.random.default_rng(seed)

    beta = np.exp(rng.normal(np.log(p["beta"]["prior_mean"]), p["beta"]["prior_sd"], n))
    gamma = np.exp(rng.normal(np.log(p["gamma"]["prior_mean"]), p["gamma"]["prior_sd"], n))
    sigma = np.exp(rng.normal(np.log(p["sigma"]["prior_mean"]), p["sigma"]["prior_sd"], n))
    mu = rng.beta(p["mu"]["prior_mean"] * 20, (1 - p["mu"]["prior_mean"]) * 20, n)
    R0 = beta / gamma

    return dict(route="A", label="assumption-driven (prior predictive)",
                R0=R0, beta=beta, gamma=gamma, sigma=sigma, mu=mu,
                source=spec.get("source", "declared SEZR parameters"),
                claim=("conditional only: IF a pathogen had these properties, "
                       "THEN its outbreak would look like this"),
                learns_from_data=False)


# ---------------------------------------------------------------------------
# Route B -- pooling-driven
# ---------------------------------------------------------------------------

def route_b_from_trace(trace, sd_floor: float = 0.1, seed: int = 20260914) -> dict:
    """
    Posterior predictive for a NEW arm with no data of its own.

    For each posterior draw (mu_hyp, sd_hyp) from the fitted hierarchy, draw

        eta_new ~ Normal(0, 1)
        log R0_new = mu_hyp + (sd_hyp + sd_floor) * eta_new

    This is exactly the generative step the model applies to every disease arm,
    executed for an arm that contributes no likelihood term. It integrates over
    both the location of the shared prior and the spread between diseases, so
    the result is wider than any single fitted arm -- correctly so, because a
    new pathogen could be more or less transmissible than the ones measured.

    Reference: Gelman A et al. (2020) BDA3 sec 5.5.
    """
    post = trace.posterior
    mu_h = np.asarray(post["bhmm::hyper_log_R0_mu"].values).ravel()
    sd_h = np.asarray(post["bhmm::hyper_log_R0_sd"].values).ravel()
    rng = np.random.default_rng(seed)
    eta = rng.normal(0.0, 1.0, size=mu_h.size)
    log_R0 = mu_h + (sd_h + sd_floor) * eta

    return dict(route="B", label="pooling-driven (posterior predictive for a new arm)",
                R0=np.exp(log_R0), log_R0=log_R0, hyper_mu=mu_h, hyper_sd=sd_h,
                n_draws=int(mu_h.size),
                claim=("GIVEN the measured human outbreaks, a fourth unmeasured "
                       "one would plausibly fall in this range"),
                learns_from_data=True,
                caveat=("this describes the population of human pathogens the "
                        "measured arms are drawn from, NOT this specific "
                        "hypothetical pathogen"))


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare_routes(a: dict, b: dict, measured: dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Side-by-side summary of both routes against the measured arms.

    `measured` maps arm name -> posterior R0 draws.
    """
    rows = []

    def summarise(name, x, note):
        x = np.asarray(x, dtype=float)
        rows.append(dict(
            arm=name,
            median=float(np.median(x)),
            eti89_lo=float(np.quantile(x, 0.055)),
            eti89_hi=float(np.quantile(x, 0.945)),
            eti89_width=float(np.quantile(x, 0.945) - np.quantile(x, 0.055)),
            cv_pct=float(np.std(x) / max(abs(np.mean(x)), 1e-12) * 100),
            note=note))

    for k, v in measured.items():
        summarise(k, v, "measured arm: informed by its own likelihood")
    summarise("Zombie — Route A", a["R0"],
              "assumption restated; posterior equals prior; learns nothing from data")
    summarise("Zombie — Route B", b["R0"],
              "new-arm prediction from the fitted hierarchy; informed by the measured arms")
    return pd.DataFrame(rows)


def write_outputs(a: dict, b: dict, measured: dict[str, np.ndarray],
                  tables_dir: Path, generated_dir: Path) -> pd.DataFrame:
    """Persist both routes and the comparison for audit."""
    tables_dir = Path(tables_dir); generated_dir = Path(generated_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    generated_dir.mkdir(parents=True, exist_ok=True)

    cmp_df = compare_routes(a, b, measured)
    cmp_df.to_csv(tables_dir / "zombie_route_comparison.csv", index=False)

    np.savez(generated_dir / "zombie_routes.npz",
             route_a_R0=a["R0"], route_b_R0=b["R0"],
             hyper_mu=b["hyper_mu"], hyper_sd=b["hyper_sd"])

    (tables_dir / "zombie_routes_meta.json").write_text(json.dumps(dict(
        route_a=dict(label=a["label"], claim=a["claim"],
                     learns_from_data=a["learns_from_data"],
                     median=float(np.median(a["R0"]))),
        route_b=dict(label=b["label"], claim=b["claim"],
                     learns_from_data=b["learns_from_data"],
                     caveat=b["caveat"], n_draws=b["n_draws"],
                     median=float(np.median(b["R0"]))),
    ), indent=2))
    return cmp_df
