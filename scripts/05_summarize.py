"""
scripts/05_summarize.py
=======================
Load the saved trace and print/write the headline posterior summary.
Does not resample. Safe to run any number of times after a fit.

Run:  python scripts/05_summarize.py [--quick]
"""
import sys, argparse, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, arviz as az

ROOT = Path(__file__).resolve().parents[1]
GEN, TAB = ROOT/"data"/"generated", ROOT/"tables"

KEY = ["bhmm::e_R0","bhmm::mea_R0","bhmm::flu_R0","bhmm::n_R0","bhmm::n_attack_rate",
       "bhmm::log_intercept_e","bhmm::log_intercept_flu",
       "bhmm::sigma_rw_e","bhmm::sigma_rw_flu",
       "bhmm::e_alpha_nb","bhmm::mea_alpha_nb","bhmm::flu_alpha_nb","bhmm::class_re_sd",
       "bhmm::hyper_log_R0_mu","bhmm::hyper_log_R0_sd"]

LIT = {"bhmm::e_R0": (1.51, 2.53, "Althaus (2014) PLOS Curr Outbreaks"),
       "bhmm::mea_R0": (10.0, 20.0, "Anderson & May (1991) Table 4.1; Guerra et al. (2017) Lancet ID"),
       "bhmm::n_R0": (1.3, 6.7, "Heijne et al. (2012) Epidemics 4:164"),
       "bhmm::flu_R0": (None, None, "CONTESTED - see docs/INFLUENZA_1978_LIMITATION.md")}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    nc = GEN/("bhmm_trace_quick.nc" if a.quick else "bhmm_trace_full.nc")
    if not nc.exists():
        sys.exit(f"No trace at {nc}. Run scripts/04_run_bhmm.py first.")
    tr = az.from_netcdf(nc)
    s = az.summary(tr)
    avail = [k for k in KEY if k in s.index]
    print(s.loc[avail, ["mean","sd","hdi_3%","hdi_97%","r_hat","ess_bulk"]].round(4).to_string()
          if "hdi_3%" in s.columns else
          s.loc[avail, ["mean","sd","r_hat","ess_bulk"]].round(4).to_string())

    print("\nLiterature comparison (posterior 89% ETI vs published range):")
    for k,(lo,hi,src) in LIT.items():
        if k not in tr.posterior: continue
        x = tr.posterior[k].values.flatten()
        l,u = np.quantile(x,.055), np.quantile(x,.945)
        if lo is None:
            print(f"  {k:14s} {np.median(x):6.3f} [{l:.3f}, {u:.3f}]   {src}")
        else:
            ov = (l <= hi) and (u >= lo)
            print(f"  {k:14s} {np.median(x):6.3f} [{l:.3f}, {u:.3f}]   lit [{lo}, {hi}]  "
                  f"{'overlaps' if ov else 'DOES NOT OVERLAP'}   {src}")

    print("\nRaw-variable movement (must differ from Normal(0,1) or the arm is unidentified):")
    for v in ["bhmm::e_log_R0_raw","bhmm::mea_log_R0_raw","bhmm::flu_log_R0_raw","bhmm::n_log_R0_raw"]:
        if v not in tr.posterior: continue
        x = tr.posterior[v].values.flatten()
        upd = abs(x.mean())>0.15 or abs(x.std()-1.0)>0.10
        print(f"  {v:26s} mean={x.mean():+.3f} sd={x.std():.3f}  "
              f"{'UPDATED' if upd else '*** NOT UPDATED ***'}")

    divs = int(tr.sample_stats.diverging.sum())
    print(f"\ndivergences={divs}  max Rhat={s.r_hat.max():.4f}  min ESS={s.ess_bulk.min():.0f}")
    s.loc[avail].to_csv(TAB/"headline_posterior.csv")
    print(f"written -> {TAB/'headline_posterior.csv'}")


if __name__ == "__main__":
    main()
