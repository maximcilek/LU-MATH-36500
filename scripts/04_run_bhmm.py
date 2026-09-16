"""
scripts/04_run_bhmm.py
======================
Production fit of the unified BHMM, with an identifiability gate before
sampling and automatic execution of every post-fit step afterwards.

Run:
    python scripts/04_run_bhmm.py                       # full pipeline
    python scripts/04_run_bhmm.py --quick               # fast sanity check
    python scripts/04_run_bhmm.py --no-chain            # fit only, no post-steps
    python scripts/04_run_bhmm.py --label my_experiment # name the archive

What runs automatically after the fit (this is the v6 change; in v5 every one
of these had to be invoked by hand):
    scripts/05_summarize.py            posterior summary + convergence tables
    scripts/07_zombie_sensitivity.py   prior sensitivity for the unknown-disease arm
    scripts/08_debug_report.py         debug figures + debug PDF
    scripts/06_build_reports.py        all publication PDFs
    scripts/09_archive_run.py          snapshot everything into archive/<run_id>/

IDENTIFIABILITY GATE
--------------------
Before sampling, the script computes d logp / d theta at the initial point for
every parameter that is supposed to be estimated. Any parameter with a zero
likelihood gradient cannot be informed by the data -- its posterior will equal
its prior regardless of chain length. The run ABORTS if the gate fails.

This gate exists because v5 shipped with exactly that defect: e_log_R0_raw and
flu_log_R0_raw had no likelihood gradient, so the reported Ebola and Influenza
R0 values were prior draws, not estimates. See core/bhmm/ode_surrogate.py.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.bhmm.bhmm_model import build_disease_grids, build_full_bhmm
from core.bhmm.prior_predictive import zombie_prior_predictive, summarise_prior_predictive
from core.bhmm.zombie_arm import (route_a_prior_predictive, route_b_from_trace,
                                  write_outputs as write_zombie_routes)
from core.mcmc_audit import write_all as write_mcmc_tables
from core.bhmm.priors import EBOLA, MEASLES, INFLUENZA, NOROVIRUS, prior_table, HEADLINE_ARMS
from core.diagnostics import (identifiability_gate, ppc_coverage, final_size_check,
                              prior_dominance, halfnormal_moments,
                              pearson_residuals, serial_correlation)

ROOT = Path(__file__).resolve().parents[1]
CAN = ROOT / "data" / "canonical"
GEN = ROOT / "data" / "generated"
TAB = ROOT / "tables"
FIGB = ROOT / "figures" / "bhmm"
for d in (GEN, TAB, FIGB):
    d.mkdir(parents=True, exist_ok=True)

NAVY, RED, TEAL, BLUE, ORANGE = "#1F4E79", "#C0392B", "#1A7B6B", "#2E75B6", "#ED7D31"
PURP = "#7B2D8B"

# Parameters that MUST be identified for the analysis to be meaningful.
def _gated():
    g = ["e_log_R0_raw", "mea_log_R0_raw", "flu_log_R0_raw", "n_log_R0_raw", "class_re_sd"]
    if EBOLA.estimate_intercept:
        g.append("log_intercept_e")
    if INFLUENZA.estimate_intercept:
        g.append("log_intercept_flu")
    return g


# ---------------------------------------------------------------------------


def load_data():
    e = pd.read_csv(CAN / "ebola_kikwit_1995_analysis_ready.csv", parse_dates=["date"])
    n = pd.read_csv(CAN / "norovirus_derbyshire_2001_analysis_ready.csv")
    f = pd.read_csv(CAN / "influenza_england_1978_analysis_ready.csv", parse_dates=["date"])
    m = pd.read_csv(CAN / "measles_hagelloch_1861_analysis_ready.csv", parse_dates=["date"])
    print(f"  Ebola     : {int(e.analysis_eligible.sum())}/{len(e)} eligible days, "
          f"{int(e.onset.sum())} onsets")
    print(f"  Norovirus : {int(n.ill.sum())}/{len(n)} ill (AR {n.ill.mean():.3f})")
    print(f"  Influenza : {len(f)} days, N={int(f.N.iloc[0])}, peak {int(f.in_bed.max())}")
    print(f"  Measles   : {len(m)} days, N={int(m.N.iloc[0])}, "
          f"{int(m.rash_onset.sum())} onsets, peak {int(m.rash_onset.max())}")
    return e, n, f, m


def run_zombie():
    """Prior-predictive arm. No data, no likelihood; posterior equals prior."""
    print("\n[ZOMBIE] prior-predictive (no data, no likelihood)")
    pp = zombie_prior_predictive(CAN / "zombie_sezr_prior_params.json",
                                 n_samples=1000, seed=20260914)
    summ = summarise_prior_predictive(pp)
    summ.to_csv(TAB / "zombie_prior_predictive_summary.csv", index=False)
    np.savez(GEN / "zombie_prior_pred.npz",
             I_t_samples=pp["I_t_samples"], R0_samples=pp["R0_samples"],
             peak_I_samples=pp["peak_I_samples"], peak_day_samples=pp["peak_day_samples"])
    r0 = pp["R0_samples"]
    print(f"  R0 prior predictive: median={np.median(r0):.2f} "
          f"95% PI=[{np.quantile(r0,.025):.2f}, {np.quantile(r0,.975):.2f}]")
    return pp


# ---------------------------------------------------------------------------


def gate(model) -> pd.DataFrame:
    print("\n[GATE] identifiability: d logp / d theta at the initial point")
    g = identifiability_gate(model, _gated())
    for _, r in g.iterrows():
        mark = "ok " if r.status == "PASS" else "!! "
        print(f"  {mark}{r.parameter:34s} |grad| = {r.max_abs_grad:12.4g}  {r.status}")
    g.to_csv(TAB / "identifiability_gate.csv", index=False)
    bad = g[g.status.str.startswith("FAIL")]
    if len(bad):
        raise RuntimeError(
            "IDENTIFIABILITY GATE FAILED for: " + ", ".join(bad.parameter) +
            "\nThese parameters do not enter the likelihood; their posteriors "
            "would equal their priors. Aborting before wasting a production run. "
            "This is the v5 defect -- see core/bhmm/ode_surrogate.py.")
    print("  gate passed: every estimated parameter reaches the likelihood")
    return g


# ---------------------------------------------------------------------------


def diagnostics(trace, e_df, n_df, f_df, m_df, model, zombie_pp, seed,
                include_influenza=True):
    summary = az.summary(trace)
    summary.to_csv(TAB / "bhmm_posterior_summary.csv")
    summary[["mean", "sd", "r_hat", "ess_bulk", "ess_tail"]].to_csv(
        TAB / "bhmm_convergence_diagnostics.csv")

    divs = int(trace.sample_stats.diverging.sum())
    bad_rhat = summary[summary.r_hat > 1.01]
    low_ess = summary[summary.ess_bulk < 400]

    print(f"\n[CONVERGENCE]")
    print(f"  divergences        : {divs}")
    print(f"  Rhat > 1.01        : {len(bad_rhat)} / {len(summary)}")
    print(f"  ESS_bulk < 400     : {len(low_ess)} / {len(summary)}")

    key = [k for k in [
        "bhmm::e_R0", "bhmm::mea_R0", "bhmm::flu_R0", "bhmm::n_R0",
        "bhmm::log_intercept_e", "bhmm::log_intercept_flu",
        "bhmm::sigma_rw_flu", "bhmm::e_alpha_nb", "bhmm::mea_alpha_nb",
        "bhmm::flu_alpha_nb",
        "bhmm::class_re_sd", "bhmm::n_attack_rate",
        "bhmm::hyper_log_R0_mu", "bhmm::hyper_log_R0_sd"] if k in summary.index]
    print("\n[KEY PARAMETERS]")
    print(summary.loc[key, ["mean", "sd", "r_hat", "ess_bulk"]].round(4).to_string())

    # ---- prior-vs-posterior movement: the check that catches a dead parameter
    print("\n[PRIOR MOVEMENT]  raw variables should differ from Normal(0,1)")
    moved = []
    for v in ["bhmm::e_log_R0_raw", "bhmm::mea_log_R0_raw",
              "bhmm::flu_log_R0_raw", "bhmm::n_log_R0_raw"]:
        if v not in trace.posterior:
            continue
        x = trace.posterior[v].values.flatten()
        upd = abs(x.mean()) > 0.15 or abs(x.std() - 1.0) > 0.10
        moved.append(dict(parameter=v, post_mean=float(x.mean()), post_sd=float(x.std()),
                          updated_by_data=bool(upd)))
        print(f"  {v:26s} mean={x.mean():+.3f} sd={x.std():.3f}  "
              f"{'UPDATED' if upd else 'NOT UPDATED - investigate'}")
    pd.DataFrame(moved).to_csv(TAB / "prior_movement_check.csv", index=False)

    # ---- prior dominance
    print("\n[PRIOR DOMINANCE]")
    dom = []
    for v, scale in [("bhmm::flu_alpha_nb", INFLUENZA.nb_alpha_scale),
                     ("bhmm::e_alpha_nb", EBOLA.nb_alpha_scale),
                     ("bhmm::sigma_rw_flu", INFLUENZA.sigma_rw_scale)]:
        if v not in trace.posterior:
            continue
        pmn, psd = halfnormal_moments(scale)
        d = prior_dominance(trace.posterior[v].values.flatten(), pmn, psd)
        d["parameter"] = v
        dom.append(d)
        print(f"  {v:22s} post/prior mean={d['mean_ratio']:.2f} "
              f"sd ratio={d['sd_ratio']:.2f}  {d['verdict']}")
    pd.DataFrame(dom).to_csv(TAB / "prior_dominance.csv", index=False)

    # ---- posterior predictive
    print("\n[PPC]")
    with model:
        ppc = pm.sample_posterior_predictive(trace, random_seed=seed, progressbar=False)
    ppd = ppc.posterior_predictive
    el = e_df[e_df.analysis_eligible]
    targets = [("Ebola", el.onset.values.astype(float), "bhmm::ebola_obs"),
               ("Measles", m_df.rash_onset.values.astype(float), "bhmm::mea_obs"),
               ("Norovirus", n_df.ill.values.astype(int), "bhmm::noro_obs")]
    if include_influenza:
        targets.insert(2, ("Influenza", f_df.in_bed.values.astype(float), "bhmm::flu_obs"))
    targets = [t for t in targets if t[2] in ppd]
    ppc_rows, resid = [], {}
    for name, obs, var in targets:
        s = ppd[var].values.reshape(-1, len(obs))
        c = ppc_coverage(obs, s)
        c["disease"] = name
        ppc_rows.append(c)
        print(f"  {name:10s} {c['coverage']:6.1%} in 90% PI  "
              f"pred_mean={c['pred_mean']:.3f} obs_mean={c['obs_mean']:.3f}  "
              f"[{'PASS' if c['passes'] else 'FAIL'}]")
        if name != "Norovirus":
            resid[name] = pearson_residuals(obs, s.mean(axis=0))
    pd.DataFrame(ppc_rows).to_csv(TAB / "ppc_results.csv", index=False)

    # ---- residual serial correlation
    print("\n[RESIDUAL SERIAL CORRELATION]  rule: |lag-1| < 0.20")
    sc_rows = []
    for name, r in resid.items():
        sc = serial_correlation(r)
        sc["disease"] = name
        sc_rows.append(sc)
        ok = abs(sc["acf_lag1"]) < 0.20
        print(f"  {name:10s} lag-1={sc['acf_lag1']:+.3f}  Ljung-Box p={sc['ljung_box_p']:.3f}  "
              f"[{'PASS' if ok else 'residual structure remains'}]")
    pd.DataFrame(sc_rows).to_csv(TAB / "residual_serial_correlation.csv", index=False)

    # ---- final-size consistency (the Influenza limitation, quantified)
    print("\n[FINAL SIZE CONSISTENCY]")
    fs_rows = []
    fs_targets = [("Norovirus", "bhmm::n_R0", float(n_df.ill.mean()))]
    if include_influenza:
        fs_targets.insert(0, ("Influenza", "bhmm::flu_R0", 512 / 763))
    # Measles AR = 188/188 = 1.0. The K-M relation is degenerate at AR = 1
    # (it gives only a lower bound on R0), so a final-size check is not
    # informative and is deliberately skipped rather than reported as a
    # spurious conflict. See docs/DATASET_SELECTION.md.
    for name, var, ar in fs_targets:
        if var not in trace.posterior:
            continue
        fs = final_size_check(trace.posterior[var].values.flatten(), ar)
        fs["disease"] = name
        fs_rows.append(fs)
        print(f"  {name:10s} implied AR={fs['implied_ar_median']:.1%} "
              f"vs reported {fs['reported_attack_rate']:.1%}  "
              f"(R0 from reported AR = {fs['R0_from_reported_ar']:.2f})  "
              f"[{'consistent' if fs['consistent'] else 'CONFLICT - see docs/INFLUENZA_1978_LIMITATION.md'}]")
    pd.DataFrame(fs_rows).to_csv(TAB / "final_size_consistency.csv", index=False)

    make_figures(trace, summary, zombie_pp, targets, ppd)
    return summary, dict(divergences=divs, n_bad_rhat=int(len(bad_rhat)),
                         n_low_ess=int(len(low_ess)))


def make_figures(trace, summary, zombie_pp, targets, ppd):
    post = trace.posterior
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 170})

    # R0 comparison
    fig, ax = plt.subplots(1, 5, figsize=(21, 4.6))
    fig.suptitle("BHMM posterior: R0 across diseases (v6 - R0 identified in every arm)",
                 fontsize=12, fontweight="bold", color=NAVY)
    items = [("bhmm::e_R0", "Ebola\nKikwit 1995", RED, (1.51, 2.53)),
             ("bhmm::mea_R0", "Measles\nHagelloch 1861", PURP, (10.0, 20.0)),
             ("bhmm::n_R0", "Norovirus\nDerbyshire 2001", BLUE, (1.3, 6.7)),
             ("bhmm::flu_R0", "Influenza 1978\n(CAVEATED)", TEAL, None),
             (None, "Zombie SEZR\nprior predictive", ORANGE, None)]
    items = [it for it in items if it[0] is None or it[0] in post]
    for a, (v, lab, col, lit) in zip(ax, items):
        vals = zombie_pp["R0_samples"] if v is None else post[v].values.flatten()
        hi = float(np.quantile(vals, 0.99))
        a.hist(np.clip(vals, 0, hi), bins=50, color=col, alpha=.75, density=True)
        a.axvline(np.median(vals), color="k", lw=2, ls="--",
                  label=f"median={np.median(vals):.2f}")
        a.axvline(1.0, color="grey", lw=1, ls=":")
        if lit:
            a.axvspan(*lit, alpha=.15, color="green", label=f"lit {lit[0]}-{lit[1]}")
        a.set(xlabel="R0", ylabel="density", title=lab)
        a.legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(FIGB / "R0_comparison.png", bbox_inches="tight"); plt.close(fig)

    # convergence
    keys = [k for k in summary.index if "::" in k and "[" not in k][:16]
    sub = summary.loc[keys]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.2))
    fig.suptitle("MCMC convergence", fontsize=12, fontweight="bold", color=NAVY)
    names = [k.replace("bhmm::", "") for k in keys]
    a1.barh(names, sub.r_hat, color=["#D5F5D5" if r <= 1.01 else "#FFF3CD" if r <= 1.05
                                      else "#FDEBEA" for r in sub.r_hat], edgecolor="grey")
    a1.axvline(1.01, color="grey", ls=":"); a1.axvline(1.05, color=RED, ls="--")
    a1.set(xlabel="Rhat", title="Rhat (target <= 1.01)")
    a2.barh(names, sub.ess_bulk, color=["#D5F5D5" if e >= 400 else "#FFF3CD" if e >= 100
                                         else "#FDEBEA" for e in sub.ess_bulk], edgecolor="grey")
    a2.axvline(400, color="green", ls="--"); a2.axvline(100, color=RED, ls=":")
    a2.set(xlabel="ESS bulk", title="ESS (target >= 400)")
    fig.tight_layout(); fig.savefig(FIGB / "convergence_panel.png", bbox_inches="tight"); plt.close(fig)

    # PPC
    ncol = max(len(targets), 1)
    fig, ax = plt.subplots(2, ncol, figsize=(5*ncol, 8), squeeze=False)
    fig.suptitle("Posterior predictive checks", fontsize=12, fontweight="bold", color=NAVY)
    for i, (name, obs, var) in enumerate(targets):
        s = ppd[var].values.reshape(-1, len(obs))
        col = [RED, PURP, TEAL, BLUE, ORANGE][i % 5]
        lo, hi = np.quantile(s, .05, axis=0), np.quantile(s, .95, axis=0)
        cov = float(np.mean((obs >= lo) & (obs <= hi)))
        pm_ = s.mean(axis=0)
        ax[0, i].scatter(obs, pm_, s=18, alpha=.6, color=col)
        lim = max(float(np.max(obs)), float(np.max(pm_))) * 1.05
        ax[0, i].plot([0, lim], [0, lim], "k--", lw=1, alpha=.5)
        ax[0, i].set(xlabel="observed", ylabel="predicted mean",
                     title=f"{name}\n{cov:.1%} in 90% PI")
        x = np.arange(len(obs))
        ax[1, i].fill_between(x, lo, hi, alpha=.3, color=col, label="90% PI")
        ax[1, i].scatter(x, obs, s=10, color=col, zorder=5, label="observed")
        ax[1, i].plot(x, pm_, color="k", lw=1, alpha=.7, label="pred mean")
        ax[1, i].set(xlabel="observation index", ylabel="count")
        ax[1, i].legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(FIGB / "ppc_panel.png", bbox_inches="tight"); plt.close(fig)
    print(f"  figures -> {FIGB}")


# ---------------------------------------------------------------------------


def chain_post_steps(run_id: str, label: str | None, skip: list[str]):
    steps = [("05_summarize.py", []),
             ("07_zombie_sensitivity.py", []),
             ("15_mcmc_audit_report.py", []),
             ("08_debug_report.py", []),
             ("06_build_reports.py", []),
             ("11_prefit_onepager.py", []),
             ("12_prefit_pages.py", []),
             ("13_disease_dossiers.py", []),
             ("14_visual_atlas.py", []),
             ("16_zombie_routes_report.py", []),
             ("17_influenza_performance.py", []),
             ("09_archive_run.py", ["--run-id", run_id] + (["--label", label] if label else []))]
    print("\n" + "=" * 64)
    print("POST-FIT PIPELINE (automatic)")
    print("=" * 64)
    for script, extra in steps:
        if script in skip:
            print(f"  skipped: {script}")
            continue
        print(f"\n--- {script} ---")
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / script)] + extra,
                           cwd=str(ROOT))
        if r.returncode != 0:
            print(f"  WARNING: {script} exited {r.returncode}; pipeline continues")


# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description="BHMM v6 production fit")
    ap.add_argument("--quick", action="store_true", help="300 draws x 2 chains")
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--tune", type=int, default=None)
    ap.add_argument("--target_accept", type=float, default=0.95)
    ap.add_argument("--max_treedepth", type=int, default=12)
    ap.add_argument("--seed", type=int, default=20260914)
    ap.add_argument("--grid", type=int, default=240, help="ODE surrogate grid nodes")
    ap.add_argument("--grw-ebola", action="store_true",
                    help="force a GRW on Ebola (pre-fit says it is not warranted)")
    ap.add_argument("--grw-influenza", action="store_true",
                    help="enable the Influenza GRW (saturated at T=14; sensitivity only)")
    ap.add_argument("--grw-measles", action="store_true",
                    help="enable the Measles GRW (pre-fit says it is not warranted)")
    ap.add_argument("--drop-influenza", action="store_true",
                    help="omit the Influenza likelihood entirely (sensitivity: "
                         "checks whether the caveated arm influences the others)")
    ap.add_argument("--no-chain", action="store_true", help="fit only; skip post-steps")
    ap.add_argument("--label", type=str, default=None, help="archive label")
    args = ap.parse_args()

    t0 = time.time()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 64)
    print("BHMM v6  -  Unified Bayesian Hierarchical Mechanistic Model")
    print("Ebola Kikwit 1995 | Norovirus Derbyshire 2001 | Influenza England 1978")
    print("Zombie SEZR: prior-predictive reference arm (no data)")
    print(f"run_id={run_id}  seed={args.seed}")
    print("=" * 64)

    e_df, n_df, f_df, m_df = load_data()
    zombie_pp = run_zombie()

    print("\n[GRIDS] building R0-indexed ODE surrogates")
    grids = build_disease_grids(e_df, f_df, m_df, G=args.grid)
    gerr = {k: g.interpolation_error(20) for k, g in grids.items()}
    for k, g in grids.items():
        print(f"  {k:10s} {g.traj.shape}  observable={g.observable}  "
              f"max interp rel err={gerr[k]['max_rel_error']:.2e}")
    (TAB / "ode_grid_interpolation_error.json").write_text(json.dumps(gerr, indent=2))

    model = build_full_bhmm(
        e_df, n_df, f_df, m_df, grids=grids,
        use_grw_ebola=args.grw_ebola,
        use_grw_influenza=args.grw_influenza,
        use_grw_measles=args.grw_measles,
        include_influenza=not args.drop_influenza)
    gate(model)

    pd.DataFrame(prior_table()).to_csv(TAB / "prior_specification.csv", index=False)

    draws = 300 if args.quick else args.draws
    chains = 2 if args.quick else args.chains
    tune = args.tune if args.tune else max(draws // 2, 1000)
    accept = 0.90 if args.quick else args.target_accept

    print(f"\n[SAMPLING] {draws} draws x {chains} chains  tune={tune}  "
          f"target_accept={accept}  max_treedepth={args.max_treedepth}")
    with model:
        trace = pm.sample(draws=draws, tune=tune, chains=chains,
                          target_accept=accept, random_seed=args.seed,
                          nuts={"max_treedepth": args.max_treedepth},
                          progressbar=True, cores=1)

    out = GEN / ("bhmm_trace_quick.nc" if args.quick else "bhmm_trace_full.nc")
    trace.to_netcdf(str(out), engine="h5netcdf")
    print(f"\n  trace -> {out}")

    summary, conv = diagnostics(trace, e_df, n_df, f_df, m_df, model, zombie_pp,
                                args.seed, include_influenza=not args.drop_influenza)

    # ---- undocumented-pathogen arm: BOTH routes, reported side by side ----
    print("\n[ZOMBIE ROUTES]")
    ra = route_a_prior_predictive(CAN/"zombie_sezr_prior_params.json", seed=args.seed)
    rb = route_b_from_trace(trace, sd_floor=0.1, seed=args.seed)
    measured = {k: trace.posterior[v].values.ravel()
                for k, v in [("Ebola","bhmm::e_R0"), ("Measles","bhmm::mea_R0"),
                             ("Norovirus","bhmm::n_R0")] if v in trace.posterior}
    cmpdf = write_zombie_routes(ra, rb, measured, TAB, GEN)
    print("  Route A (assumption restated)  : median "
          f"{np.median(ra['R0']):.3f}  [{np.quantile(ra['R0'],.055):.2f}, {np.quantile(ra['R0'],.945):.2f}]")
    print("  Route B (learned from the arms): median "
          f"{np.median(rb['R0']):.3f}  [{np.quantile(rb['R0'],.055):.2f}, {np.quantile(rb['R0'],.945):.2f}]")
    print("  -> tables/zombie_route_comparison.csv")

    # ---- sampler-process tables (auditable without reopening the trace) ----
    print("\n[MCMC AUDIT TABLES]")
    gated = [p for p in summary.index if "::" in p and "[" not in p][:20]
    w = write_mcmc_tables(trace, TAB, gated)
    print(f"  {len(w)} sampler-state tables written to tables/mcmc_*.csv")

    meta = dict(run_id=run_id, label=args.label, seed=args.seed, draws=draws,
                chains=chains, tune=tune, target_accept=accept,
                max_treedepth=args.max_treedepth, grid_nodes=args.grid,
                use_grw_ebola=bool(args.grw_ebola),
                use_grw_influenza=bool(args.grw_influenza),
                use_grw_measles=bool(args.grw_measles),
                drop_influenza=bool(args.drop_influenza),
                elapsed_sec=round(time.time() - t0, 1),
                timestamp=datetime.now().isoformat(timespec="seconds"),
                **conv)
    (TAB / "run_metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"\n[DONE] fit complete in {meta['elapsed_sec']:.0f}s")

    if not args.no_chain:
        chain_post_steps(run_id, args.label, skip=[])
        print("\n" + "=" * 64)
        print(f"PIPELINE COMPLETE  ->  archive/{run_id}" + (f"_{args.label}" if args.label else ""))
        print("=" * 64)


if __name__ == "__main__":
    main()
