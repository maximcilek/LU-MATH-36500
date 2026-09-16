"""
scripts/03_prior_predictive_calibration.py
==========================================
Calibrate the GRW innovation prior by PRIOR PREDICTIVE CHECKING.

WHY THIS REPLACES THE v5 FORMULA
--------------------------------
v5 justified sigma_rw with

    sigma_max = log(R0_hi / R0_lo) / sqrt(T)

presented as "Lemma 1 (Durbin & Koopman 2012, sec 2.3)". That citation is
overstated. Durbin & Koopman sec 2.3 discusses the signal-to-noise ratio of the
local level model; it does not contain that lemma. A reviewer checking the
reference would not find it. The formula is a reasonable heuristic but it is the
author's own construction and must be labelled as such.

This script uses prior predictive checking instead, which IS a standard,
citable procedure with an explicit place in the Bayesian workflow:

    Gelman A et al. (2020). Bayesian Data Analysis, 3e. CRC Press, sec 6.4.
    Gabry J, Simpson D, Vehtari A, Betancourt M & Gelman A (2019).
      Visualization in Bayesian workflow. J R Stat Soc A 182(2):389-402.
    Schad DJ, Betancourt M & Vasishth S (2021). Toward a principled Bayesian
      workflow in cognitive science. Psychological Methods 26(1):103-126.

PROCEDURE
---------
1. Draw sigma_rw from the candidate HalfNormal prior.
2. Simulate the GRW forward over the arm's actual T.
3. Record the implied multiplicative range of beta(t): max(exp z) / min(exp z).
4. A prior is acceptable if that range stays epidemiologically plausible --
   transmission in a closed setting over a two-week outbreak should not swing
   by an order of magnitude absent an intervention.
5. Report the range across candidate scales; the selected scale is the largest
   whose 95th percentile stays under the plausibility ceiling.

The heuristic sigma_max is still computed and reported ALONGSIDE, clearly
labelled as an author-derived rule of thumb rather than a cited result, so the
two lines of reasoning can be compared.

Run:  python scripts/03_prior_predictive_calibration.py
"""
import sys, json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.bhmm.priors import EBOLA, INFLUENZA

ROOT = Path(__file__).resolve().parents[1]
TAB, FIG = ROOT/"tables", ROOT/"figures"/"prefit"
TAB.mkdir(exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)

SEED = 20260914
N_SIM = 4000
# Plausibility ceiling: beta(t) may vary by at most this factor across the
# observation window in the absence of a modelled intervention. 3x over two
# weeks in a closed boarding school is already generous; 10x is not credible.
CEILING = 3.0
CANDIDATES = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]


def simulate(scale, T, n=N_SIM, seed=SEED, gaps=None):
    rng = np.random.default_rng(seed)
    sig = np.abs(rng.normal(0.0, scale, n))               # HalfNormal(scale)
    if gaps is None:
        gaps = np.ones(T)
    sd = np.sqrt(gaps)
    eps = rng.normal(0.0, 1.0, size=(n, T))
    z = np.cumsum(eps * sd[None, :], axis=1) * sig[:, None]
    b = np.exp(z)
    return sig, b.max(axis=1) / b.min(axis=1)


def sigma_max_heuristic(T, lo, hi):
    """AUTHOR-DERIVED heuristic, not a cited theorem. Reported for comparison."""
    return float(np.log(hi / lo) / np.sqrt(T))


def calibrate(name, T, gaps, lit_lo, lit_hi):
    rows = []
    for sc in CANDIDATES:
        _, rng_ = simulate(sc, T, gaps=gaps)
        rows.append(dict(disease=name, scale=sc,
                         range_median=float(np.median(rng_)),
                         range_p95=float(np.quantile(rng_, 0.95)),
                         frac_over_ceiling=float(np.mean(rng_ > CEILING)),
                         acceptable=bool(np.quantile(rng_, 0.95) <= CEILING)))
    df = pd.DataFrame(rows)
    ok = df[df.acceptable]
    chosen = float(ok.scale.max()) if len(ok) else float(df.scale.min())
    heur = (sigma_max_heuristic(T, lit_lo, lit_hi) if (lit_lo and lit_hi) else np.nan)
    return df, chosen, heur


def main():
    print("Prior predictive calibration of the GRW innovation prior")
    print(f"  plausibility ceiling: beta(t) range <= {CEILING}x across the window")
    print(f"  acceptance: 95th percentile of the simulated range must not exceed it\n")

    e_days = pd.read_csv(ROOT/"data"/"canonical"/"ebola_kikwit_1995_analysis_ready.csv")
    e_days = e_days[e_days.analysis_eligible].day.values
    e_gaps = np.concatenate([[1.0], np.diff(e_days).astype(float)])
    T_flu = len(pd.read_csv(ROOT/"data"/"canonical"/"influenza_england_1978_analysis_ready.csv"))

    out, chosen = [], {}
    for name, T, gaps, lo, hi, cfg in [
        ("Ebola", len(e_days), e_gaps, EBOLA.R0_lit_lo, EBOLA.R0_lit_hi, EBOLA),
        ("Influenza", T_flu, None, INFLUENZA.R0_lit_lo, INFLUENZA.R0_lit_hi, INFLUENZA),
    ]:
        df, ch, heur = calibrate(name, T, gaps, lo, hi)
        out.append(df); chosen[name] = ch
        print(f"  {name} (T={T})")
        for _, r in df.iterrows():
            print(f"    HalfNormal({r.scale:<4}) median range {r.range_median:6.2f}x  "
                  f"p95 {r.range_p95:7.2f}x  {'OK' if r.acceptable else 'too wide'}")
        print(f"    -> prior predictive selects HalfNormal({ch})")
        print(f"    -> author heuristic log(R0_hi/R0_lo)/sqrt(T) = {heur:.4f}  "
              f"(NOT a cited theorem; reported for comparison)")
        print(f"    -> configured in priors.py: HalfNormal({cfg.sigma_rw_scale})\n")

    df = pd.concat(out, ignore_index=True)
    df.to_csv(TAB/"prior_predictive_calibration.csv", index=False)
    (TAB/"prior_predictive_calibration.json").write_text(json.dumps(dict(
        ceiling=CEILING, selected=chosen,
        configured=dict(Ebola=EBOLA.sigma_rw_scale, Influenza=INFLUENZA.sigma_rw_scale),
        method="prior predictive checking (Gelman et al. 2020 BDA3 sec 6.4; "
               "Gabry et al. 2019 JRSSA 182:389)"), indent=2))

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    fig.suptitle("Prior predictive calibration of the GRW innovation SD",
                 fontsize=12, fontweight="bold", color="#1F4E79")
    for a, nm, col in zip(ax, ["Ebola", "Influenza"], ["#C0392B", "#1A7B6B"]):
        d = df[df.disease == nm]
        a.plot(d.scale, d.range_median, "o-", color=col, lw=2, label="median range")
        a.plot(d.scale, d.range_p95, "s--", color=col, lw=1.4, alpha=.7, label="95th pct")
        a.axhline(CEILING, color="k", ls=":", lw=1.5, label=f"ceiling {CEILING}x")
        a.axvline(chosen[nm], color="green", ls="--", lw=1.5, label=f"selected {chosen[nm]}")
        a.set(xlabel="HalfNormal scale for sigma_rw", ylabel="beta(t) range across window",
              yscale="log", title=nm)
        a.legend(frameon=False, fontsize=7.5)
        a.spines.top.set_visible(False); a.spines.right.set_visible(False)
    fig.tight_layout(); fig.savefig(FIG/"prior_predictive_calibration.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  figure -> {FIG/'prior_predictive_calibration.png'}")
    print(f"  tables -> {TAB/'prior_predictive_calibration.csv'}")


if __name__ == "__main__":
    main()
