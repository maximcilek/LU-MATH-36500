"""
scripts/02_prefit_diagnostics.py
================================
Exhaustive pre-fit audit. Writes figures/prefit/*.png and tables/prefit_*.csv.

These are the debug-level pre-fit figures that were missing from the v5 runs:
figures/prefit/ existed but was never populated because no script wrote to it.

Everything here runs BEFORE any model fitting and uses no posterior. Its job is
to establish, from raw data alone, which observation model and which temporal
structure each arm needs -- so that those choices are pre-registered rather than
chosen after seeing the fit.

Run:  python scripts/02_prefit_diagnostics.py
"""
import sys, json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats, optimize

from core.diagnostics import (dispersion, serial_correlation, pearson_residuals,
                              km_R0_from_attack_rate, km_attack_rate_np)
from core.bhmm.priors import EBOLA, MEASLES, INFLUENZA, NOROVIRUS
from core.bhmm.ode_surrogate import solve_window, build_grid

ROOT = Path(__file__).resolve().parents[1]
CAN, FIG, TAB = ROOT/"data"/"canonical", ROOT/"figures"/"prefit", ROOT/"tables"
FIG.mkdir(parents=True, exist_ok=True); TAB.mkdir(exist_ok=True)

RED, BLUE, TEAL, GREEN, NAVY, GREY = "#C0392B","#2E75B6","#1A7B6B","#375623","#1F4E79","#888888"
PURP = "#7B2D8B"
plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False,"figure.dpi":170})

results = {}


def _panel(fig, title):
    fig.suptitle(title, fontsize=12, fontweight="bold", color=NAVY)


# ===========================================================================
def ebola_prefit():
    df = pd.read_csv(CAN/"ebola_kikwit_1995_analysis_ready.csv", parse_dates=["date"])
    el = df[df.analysis_eligible]
    y, days = el.onset.values.astype(float), el.day.values

    disp = dispersion(y)
    # Constant-transmission reference fit: R0 + intercept only, no GRW.
    grid = build_grid(structure="seird_incidence", sigma=EBOLA.sigma,
                      removal=EBOLA.removal, mu=EBOLA.mu_point, N=EBOLA.N,
                      I0=EBOLA.I0, t_seed=EBOLA.t_seed, T=int(days.max())+1, G=120)
    def dev(p):
        m = np.clip(np.interp(p[0], grid.log_r0, np.zeros(grid.G)) * 0 +
                    solve_window(np.exp(p[0]), **grid.meta)[days] * np.exp(p[1]), 1e-9, None)
        o = np.clip(y, 1e-9, None)
        return 2*np.sum(o*np.log(o/m) - (o-m))
    r = optimize.minimize(dev, [np.log(1.8), 0.0], method="Nelder-Mead",
                          options={"maxiter":1500})
    R0hat, chat = float(np.exp(r.x[0])), float(r.x[1])
    fit = solve_window(R0hat, **grid.meta)[days]*np.exp(chat)
    res = pearson_residuals(y, fit)
    sc = serial_correlation(res)

    fig, ax = plt.subplots(2, 3, figsize=(15, 8)); _panel(fig, "Ebola Kikwit 1995 - Pre-fit Audit")
    ax[0,0].bar(df.day, df.onset, color=RED, alpha=.8)
    ax[0,0].scatter(df.day[~df.analysis_eligible], np.zeros((~df.analysis_eligible).sum()),
                    marker="|", color=GREY, s=30, label="reporting=FALSE (excluded)")
    ax[0,0].set(xlabel="Day", ylabel="Onsets", title=f"A. Epidemic curve\n{int(df.analysis_eligible.sum())}/{len(df)} eligible")
    ax[0,0].legend(frameon=False, fontsize=7)

    ax[0,1].hist(y, bins=16, color=RED, alpha=.8)
    ax[0,1].set(xlabel="Daily onsets", ylabel="Count",
                title=f"B. Distribution\nVar/Mean={disp['var_mean_ratio']:.2f}, skew={disp['skewness']:.2f}")

    ax[0,2].plot(days, y, "o", ms=3.5, color=RED, label="observed")
    ax[0,2].plot(days, fit, "-", lw=2, color=NAVY, label=f"const-$\\beta$ fit $R_0$={R0hat:.2f}")
    ax[0,2].set(xlabel="Day", ylabel="Onsets", title="C. Constant-transmission reference fit")
    ax[0,2].legend(frameon=False, fontsize=7.5)

    lags = range(1,9)
    a = [float(np.corrcoef(res[k:], res[:-k])[0,1]) for k in lags]
    ax[1,0].bar(list(lags), a, color=[RED if abs(v)>.2 else TEAL for v in a], alpha=.85)
    for h in (.2,-.2): ax[1,0].axhline(h, color=NAVY, ls="--", lw=1)
    ax[1,0].axhline(0, color="k", lw=.6)
    ax[1,0].set(xlabel="Lag (observations)", ylabel="Pearson r",
                title=f"D. Residual ACF\nlag-1={sc['acf_lag1']:+.3f}, LB p={sc['ljung_box_p']:.4f}")

    gaps = np.diff(days)
    ax[1,1].hist(gaps, bins=np.arange(.5, gaps.max()+1.5), color=TEAL, alpha=.85)
    ax[1,1].set(xlabel="Gap between eligible days", ylabel="Count",
                title=f"E. Observation spacing\nirregular: GRW needs $\\sqrt{{\\Delta t}}$ scaling")

    spread = grid.traj.std(axis=0)/(np.abs(grid.traj).mean(axis=0)+1e-12)
    ax[1,2].plot(spread, color=NAVY, lw=2)
    ax[1,2].set(xlabel="Day", ylabel="rel. spread across $R_0$ grid",
                title="F. $R_0$ identifiability\ntrajectory sensitivity to $R_0$ (must be >0)")
    fig.tight_layout(); fig.savefig(FIG/"ebola_prefit_panel.png", bbox_inches="tight"); plt.close(fig)

    results["ebola"] = dict(dispersion=disp, serial=sc, const_fit_R0=R0hat,
                            const_fit_log_intercept=chat,
                            grid_interp_error=grid.interpolation_error(20))
    print(f"  Ebola     : Var/Mean={disp['var_mean_ratio']:.2f}  lag-1={sc['acf_lag1']:+.3f}  "
          f"GRW={'YES' if sc['grw_warranted'] else 'no'}  const-fit R0={R0hat:.2f}")


# ===========================================================================
def influenza_prefit():
    df = pd.read_csv(CAN/"influenza_england_1978_analysis_ready.csv", parse_dates=["date"])
    y = df.in_bed.values.astype(float); N = int(df.N.iloc[0])
    disp = dispersion(y)

    # Profile t_seed over the documented lead-in window.
    prof = []
    for ts in [5,7,9,11,13,15]:
        meta = dict(structure="seiqr_prevalence", sigma=INFLUENZA.sigma,
                    removal=INFLUENZA.removal, gamma_q=INFLUENZA.gamma_q, mu=0.0,
                    N=N, I0=INFLUENZA.I0, t_seed=float(ts), T=len(y), dt_obs=1.0,
                    kE=INFLUENZA.kE, kI=INFLUENZA.kI, kQ=INFLUENZA.kQ)
        def dev(p):
            m = np.clip(solve_window(np.exp(p[0]), **meta)*np.exp(p[1]), 1e-9, None)
            o = np.clip(y, 1e-9, None); return 2*np.sum(o*np.log(o/m)-(o-m))
        r = optimize.minimize(dev, [np.log(3.), 0.], method="Nelder-Mead",
                              options={"maxiter":1500})
        R0, c = float(np.exp(r.x[0])), float(r.x[1])
        fit = solve_window(R0, **meta)*np.exp(c)
        prof.append(dict(t_seed=ts, R0=R0, log_intercept=c, poisson_deviance=float(r.fun),
                         peak=float(fit.max()), peak_day=int(np.argmax(fit)),
                         implied_ar=km_attack_rate_np(R0)))
    prof_df = pd.DataFrame(prof); prof_df.to_csv(TAB/"prefit_influenza_seed_profile.csv", index=False)
    best = prof_df.loc[prof_df.poisson_deviance.idxmin()]

    meta = dict(structure="seiqr_prevalence", sigma=INFLUENZA.sigma, removal=INFLUENZA.removal,
                gamma_q=INFLUENZA.gamma_q, mu=0.0, N=N, I0=INFLUENZA.I0,
                t_seed=float(best.t_seed), T=len(y), dt_obs=1.0,
                kE=INFLUENZA.kE, kI=INFLUENZA.kI, kQ=INFLUENZA.kQ)
    fit = solve_window(float(best.R0), **meta)*np.exp(float(best.log_intercept))
    res = pearson_residuals(y, fit); sc = serial_correlation(res)

    reported_ar = 512/763
    R0_finalsize = km_R0_from_attack_rate(reported_ar)

    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    _panel(fig, "Influenza England 1978 - Pre-fit Audit (SEIQR, in_bed = confinement compartment)")
    ax[0,0].bar(df.day, df.in_bed, color=TEAL, alpha=.85, label="in_bed (confined)")
    ax[0,0].bar(df.day, df.convalescent, color="#AAD4CC", alpha=.7, label="convalescent")
    ax[0,0].set(xlabel="Day", ylabel="Boys", title=f"A. Epidemic curve\npeak {int(y.max())}/{N} = {y.max()/N:.0%} on day {int(np.argmax(y))}")
    ax[0,0].legend(frameon=False, fontsize=7.5)

    ax[0,1].plot(df.day, y, "o-", ms=4, color=TEAL, label="observed")
    ax[0,1].plot(df.day, fit, "-", lw=2, color=NAVY,
                 label=f"SEIQR fit $R_0$={best.R0:.2f}\n(t_seed={int(best.t_seed)}d)")
    ax[0,1].set(xlabel="Day", ylabel="in_bed", title="B. Constant-transmission SEIQR fit")
    ax[0,1].legend(frameon=False, fontsize=7.5)

    ax[0,2].plot(prof_df.t_seed, prof_df.poisson_deviance, "o-", color=NAVY, lw=2)
    ax2 = ax[0,2].twinx(); ax2.plot(prof_df.t_seed, prof_df.R0, "s--", color=RED, lw=1.5)
    ax2.set_ylabel("fitted $R_0$", color=RED)
    ax[0,2].set(xlabel="t_seed (days before day 0)", ylabel="Poisson deviance",
                title="C. Seed-time profile\n(documented lead-in 8-12 d)")

    lags = range(1,7)
    a = [float(np.corrcoef(res[k:], res[:-k])[0,1]) for k in lags]
    ax[1,0].bar(list(lags), a, color=[RED if abs(v)>.2 else TEAL for v in a], alpha=.85)
    for h in (.2,-.2): ax[1,0].axhline(h, color=NAVY, ls="--", lw=1)
    ax[1,0].axhline(0, color="k", lw=.6)
    ax[1,0].set(xlabel="Lag (days)", ylabel="Pearson r",
                title=f"D. Residual ACF\nlag-1={sc['acf_lag1']:+.3f}, LB p={sc['ljung_box_p']:.4f}")

    # THE headline limitation figure
    rr = np.linspace(1.05, 12, 300)
    ax[1,1].plot(rr, [km_attack_rate_np(v) for v in rr], color=NAVY, lw=2)
    ax[1,1].axhline(reported_ar, color=RED, ls="--", lw=2, label=f"reported AR = {reported_ar:.1%}")
    ax[1,1].axvline(R0_finalsize, color=RED, ls=":", lw=2, label=f"$R_0$ from AR = {R0_finalsize:.2f}")
    ax[1,1].axvline(best.R0, color=TEAL, ls=":", lw=2, label=f"$R_0$ from curve = {best.R0:.2f}")
    ax[1,1].set(xlabel="$R_0$", ylabel="K-M attack rate",
                title="E. IDENTIFICATION CONFLICT\nsize vs speed disagree")
    ax[1,1].legend(frameon=False, fontsize=7)

    ax[1,2].axis("off")
    ax[1,2].text(0.02, 0.98,
        "LIMITATION (documented)\n\n"
        f"K-M $R_0$ from reported AR 512/763 : {R0_finalsize:.2f}\n"
        f"$R_0$ from prevalence time course  : {best.R0:.2f}\n"
        f"AR implied by curve-fit $R_0$      : {best.implied_ar:.1%}\n"
        f"Reported AR                        : {reported_ar:.1%}\n\n"
        "No published model reproduces both.\n"
        "Avilov et al. (2024) J R Soc Interface\n"
        "21:20240394 -- 'where the classic\n"
        "SEIR model fails'. Their DDE model\n"
        "gives R0 = 8.14, flagged by Ahmad\n"
        "et al. (2025) Sci Rep as far above\n"
        "the typical influenza range 1-4.\n\n"
        "=> flu_R0 must be reported WITH this\n"
        "   conflict, never as a clean\n"
        "   cross-disease comparator.",
        transform=ax[1,2].transAxes, va="top", fontsize=8, family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFF3CD", edgecolor="#856404"))
    fig.tight_layout(); fig.savefig(FIG/"influenza_prefit_panel.png", bbox_inches="tight"); plt.close(fig)

    results["influenza"] = dict(dispersion=disp, serial=sc,
                                const_fit_R0=float(best.R0), t_seed_best=float(best.t_seed),
                                R0_from_final_size=R0_finalsize,
                                reported_attack_rate=reported_ar,
                                implied_attack_rate=float(best.implied_ar),
                                identification_conflict=True)
    print(f"  Influenza : Var/Mean={disp['var_mean_ratio']:.1f}  lag-1={sc['acf_lag1']:+.3f}  "
          f"curve R0={best.R0:.2f} vs final-size R0={R0_finalsize:.2f}  <-- CONFLICT")


# ===========================================================================
def norovirus_prefit():
    df = pd.read_csv(CAN/"norovirus_derbyshire_2001_analysis_ready.csv")
    by = df.groupby("class").ill.agg(["sum","count"])
    by["ar"] = by["sum"]/by["count"]
    chi2, p, dof, _ = stats.chi2_contingency(
        np.column_stack([by["sum"], by["count"]-by["sum"]]))
    ar = df.ill.mean()

    fig, ax = plt.subplots(2, 3, figsize=(15, 8)); _panel(fig, "Norovirus Derbyshire 2001 - Pre-fit Audit")
    cols = [RED if i == 10 else BLUE for i in by.index]
    ax[0,0].bar(by.index.astype(str), by.ar, color=cols, alpha=.85)
    ax[0,0].axhline(ar, color=NAVY, ls="--", lw=1.5, label=f"overall {ar:.1%}")
    ax[0,0].set(xlabel="Class", ylabel="Attack rate",
                title=f"A. AR by class\nchi2={chi2:.1f}, df={dof}, p={p:.2e}")
    ax[0,0].legend(frameon=False, fontsize=7.5)

    ax[0,1].scatter(by["count"], by.ar, s=45, color=BLUE, alpha=.8)
    for i, r_ in by.iterrows():
        if i in (3, 10): ax[0,1].annotate(f"C{i}", (r_["count"], r_.ar), fontsize=8, color=RED)
    ax[0,1].set(xlabel="Class size", ylabel="Attack rate", title="B. AR vs class size")

    ax[0,2].bar(["ill\n(outcome)","absent\n(NOT outcome)","vomiting\n(too sparse)"],
                [df.ill.sum(), (df.day_absent>0).sum(), (df.day_vomiting>0).sum()],
                color=[GREEN, RED, GREY], alpha=.85)
    ax[0,2].set(ylabel="Students", title="C. Outcome definition\nill = start_illness>0")

    ill_dur = df.loc[df.ill==1, "end_illness"] - df.loc[df.ill==1, "start_illness"]
    ax[1,0].hist(ill_dur[ill_dur>=0], bins=12, color=BLUE, alpha=.85)
    ax[1,0].set(xlabel="Illness duration (days)", ylabel="Count", title="D. Illness duration")

    # Which R0 does the observed AR imply, via K-M?
    R0_km = km_R0_from_attack_rate(ar)
    rr = np.linspace(1.02, 4, 250)
    ax[1,1].plot(rr, [km_attack_rate_np(v) for v in rr], color=NAVY, lw=2)
    ax[1,1].axhline(ar, color=BLUE, ls="--", lw=2, label=f"observed AR {ar:.1%}")
    ax[1,1].axvline(R0_km, color=BLUE, ls=":", lw=2, label=f"$R_0$ = {R0_km:.2f}")
    ax[1,1].set(xlabel="$R_0$", ylabel="K-M attack rate", title="E. Final-size relation")
    ax[1,1].legend(frameon=False, fontsize=7.5)

    ax[1,2].axis("off")
    ax[1,2].text(0.02, 0.98,
        "PRE-FIT DECISIONS\n\n"
        "Outcome   : ill = (start_illness > 0)\n"
        "            NOT day_absent -- 126 of\n"
        "            417 well students were\n"
        "            absent; using absence\n"
        "            misclassifies them.\n\n"
        f"Clustering: chi2 p = {p:.1e}\n"
        "            -> 15 class random\n"
        "               intercepts\n\n"
        "Class 10  : AR 66.7% (n=24) retained;\n"
        "            absorbed by the RE, not\n"
        "            deleted. Sensitivity S2.\n"
        "Class 3   : AR 0% (n=25) retained;\n"
        "            RE shrinks to the mean.\n\n"
        "Vomiting  : 5 ill+vomit -> excluded.\n\n"
        "No GRW    : cross-sectional data;\n"
        "            no time index for\n"
        "            beta(t) to vary over.",
        transform=ax[1,2].transAxes, va="top", fontsize=8, family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#EAF2FB", edgecolor=BLUE))
    fig.tight_layout(); fig.savefig(FIG/"norovirus_prefit_panel.png", bbox_inches="tight"); plt.close(fig)

    results["norovirus"] = dict(attack_rate=float(ar), chi2=float(chi2), chi2_p=float(p),
                                R0_from_final_size=R0_km, n_classes=int(len(by)))
    print(f"  Norovirus : AR={ar:.3f}  class chi2 p={p:.2e}  K-M R0={R0_km:.3f}")


# ===========================================================================
def measles_prefit():
    df = pd.read_csv(CAN/"measles_hagelloch_1861_analysis_ready.csv", parse_dates=["date"])
    y, days, N = df.rash_onset.values.astype(float), df.day.values, int(df.N.iloc[0])
    disp = dispersion(y)

    meta = dict(structure="seird_incidence", sigma=MEASLES.sigma, removal=MEASLES.removal,
                gamma_q=0.0, mu=MEASLES.mu_point, N=N, I0=MEASLES.I0,
                t_seed=MEASLES.t_seed, T=len(y), dt_obs=1.0, kE=1, kI=1, kQ=1)
    def dev(p_):
        m = np.clip(solve_window(np.exp(p_[0]), **meta)*np.exp(p_[1]), 1e-9, None)
        o = np.clip(y, 1e-9, None); return 2*np.sum(o*np.log(o/m)-(o-m))
    r = optimize.minimize(dev, [np.log(8.), 0.], method="Nelder-Mead",
                          options={"maxiter":2000})
    R0hat, chat = float(np.exp(r.x[0])), float(r.x[1])
    fit = solve_window(R0hat, **meta)*np.exp(chat)
    res = pearson_residuals(y, fit); sc = serial_correlation(res)

    ar = float(y.sum()/N)   # 188/188 = 1.0 for this complete-cohort outbreak

    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    _panel(fig, "Measles Hagelloch 1861 - Pre-fit Audit (rash onset = incidence)")
    ax[0,0].bar(df.day, y, color=PURP, alpha=.85)
    ax[0,0].set(xlabel="Day", ylabel="Rash onsets",
                title=f"A. Epidemic curve\n{int(y.sum())}/{N} children, peak {int(y.max())} day {int(np.argmax(y))}")
    ax[0,1].hist(y, bins=15, color=PURP, alpha=.85)
    ax[0,1].set(xlabel="Daily rash onsets", ylabel="Count",
                title=f"B. Distribution\nVar/Mean={disp['var_mean_ratio']:.2f}, "
                      f"{disp['zero_frac']:.0%} zero days")
    ax[0,2].plot(days, y, "o", ms=3.5, color=PURP, label="observed")
    ax[0,2].plot(days, fit, "-", lw=2, color=NAVY, label=f"const-$\\beta$ fit $R_0$={R0hat:.2f}")
    ax[0,2].set(xlabel="Day", ylabel="Rash onsets", title="C. Constant-transmission fit")
    ax[0,2].legend(frameon=False, fontsize=7.5)

    lags = range(1, 9)
    a = [float(np.corrcoef(res[k:], res[:-k])[0,1]) for k in lags]
    ax[1,0].bar(list(lags), a, color=[RED if abs(v)>.2 else TEAL for v in a], alpha=.85)
    for h in (.2,-.2): ax[1,0].axhline(h, color=NAVY, ls="--", lw=1)
    ax[1,0].axhline(0, color="k", lw=.6)
    ax[1,0].set(xlabel="Lag (days)", ylabel="Pearson r",
                title=f"D. Residual ACF\nlag-1={sc['acf_lag1']:+.3f}, LB p={sc['ljung_box_p']:.4f}")

    ll = pd.read_csv(ROOT/"data"/"raw"/"measles"/"measles_hagelloch_1861_linelist.csv")
    cl = ll.CL.value_counts()
    ax[1,1].bar(cl.index.astype(str), cl.values, color=PURP, alpha=.85)
    ax[1,1].set(ylabel="Children", title="E. Class structure\n(drives the spiky curve)")
    ax[1,1].tick_params(axis="x", labelsize=7)

    ax[1,2].axis("off")
    ax[1,2].text(0.02, 0.98,
        "PRE-FIT NOTES\n\n"
        f"Cohort           : {N} children\n"
        f"Total infected   : {int(y.sum())} (AR {ar:.0%})\n"
        f"Deaths           : 12 (CFR 6.4%)\n"
        f"Window           : {len(y)} days\n"
        f"Zero-onset days  : {int((y==0).sum())}\n"
        f"Var/Mean         : {disp['var_mean_ratio']:.2f}\n"
        f"Const-fit R0     : {R0hat:.2f}\n\n"
        "AR = 100%: every child in the\n"
        "village was infected. The K-M\n"
        "final-size relation is therefore\n"
        "DEGENERATE -- it gives only a\n"
        "lower bound on R0, not a point\n"
        "value. This is NOT the Influenza\n"
        "conflict: size and speed do not\n"
        "disagree here, the size is simply\n"
        "uninformative above a threshold.\n"
        "R0 is identified by the time\n"
        "course alone.\n\n"
        "The observed curve is spikier than\n"
        "homogeneous mixing predicts because\n"
        "of household and classroom\n"
        "clustering (Neal & Roberts 2004).\n"
        "NB overdispersion absorbs this.",
        transform=ax[1,2].transAxes, va="top", fontsize=7.6, family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#F5EEF8", edgecolor=PURP))
    fig.tight_layout(); fig.savefig(FIG/"measles_prefit_panel.png", bbox_inches="tight"); plt.close(fig)

    results["measles"] = dict(dispersion=disp, serial=sc, const_fit_R0=R0hat,
                              const_fit_log_intercept=chat, attack_rate=ar,
                              final_size_degenerate=bool(ar >= 0.999))
    print(f"  Measles   : Var/Mean={disp['var_mean_ratio']:.2f}  lag-1={sc['acf_lag1']:+.3f}  "
          f"GRW={'YES' if sc['grw_warranted'] else 'no'}  const-fit R0={R0hat:.2f}  AR={ar:.0%}")


def overview_panel():
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    _panel(fig, "Cross-disease pre-fit overview")
    names = ["Ebola","Measles","Influenza"]
    vm = [results["ebola"]["dispersion"]["var_mean_ratio"],
          results["measles"]["dispersion"]["var_mean_ratio"],
          results["influenza"]["dispersion"]["var_mean_ratio"]]
    ax[0].bar(names, vm, color=[RED, PURP, TEAL], alpha=.85)
    ax[0].axhline(1, color=NAVY, ls="--", lw=1.5, label="Poisson (Var/Mean=1)")
    ax[0].set(ylabel="Var/Mean", yscale="log", title="Overdispersion\n-> NegBinomial for both")
    ax[0].legend(frameon=False, fontsize=7.5)

    l1 = [results["ebola"]["serial"]["acf_lag1"],
          results["measles"]["serial"]["acf_lag1"],
          results["influenza"]["serial"]["acf_lag1"]]
    ax[1].bar(names, l1, color=[RED, PURP, TEAL], alpha=.85)
    for h in (.2,-.2): ax[1].axhline(h, color=NAVY, ls="--", lw=1.5)
    ax[1].set(ylabel="Residual lag-1 ACF", title="Serial correlation\n-> GRW warranted (|r|>0.20)")

    ax[2].axis("off")
    ax[2].text(0.02, 0.98,
        "OBSERVATION MODEL BY ARM\n\n"
        "Ebola      onset = incidence flux\n"
        "           sigma*E(t); NegBinomial\n"
        "           139/192 eligible days\n\n"
        "Measles    rash onset = incidence\n"
        "           sigma*E(t); NegBinomial\n"
        "           86 days, census cohort\n\n"
        "Influenza  in_bed = confinement Q(t)\n"
        "           SEIQR; NegBinomial\n"
        "           NOT I(t) -- see Avilov\n"
        "           et al. (2024)\n\n"
        "Norovirus  ill = binary per student\n"
        "           Bernoulli + 15 class REs\n"
        "           R0 via K-M final size\n\n"
        "Zombie     prior predictive only;\n"
        "           no data, no likelihood",
        transform=ax[2].transAxes, va="top", fontsize=8.5, family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#F0F4F8", edgecolor=NAVY))
    fig.tight_layout(); fig.savefig(FIG/"overview_prefit_panel.png", bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    print("Pre-fit diagnostics (no posterior used)...")
    ebola_prefit(); measles_prefit(); influenza_prefit(); norovirus_prefit(); overview_panel()
    (TAB/"prefit_diagnostics.json").write_text(json.dumps(results, indent=2, default=float))
    rows = []
    for d, r in results.items():
        if "dispersion" in r:
            rows.append(dict(disease=d, var_mean=r["dispersion"]["var_mean_ratio"],
                             skew=r["dispersion"]["skewness"],
                             zero_frac=r["dispersion"]["zero_frac"],
                             lag1_acf=r["serial"]["acf_lag1"],
                             ljung_box_p=r["serial"]["ljung_box_p"],
                             grw_warranted=r["serial"]["grw_warranted"]))
    pd.DataFrame(rows).to_csv(TAB/"prefit_summary.csv", index=False)
    print(f"\n  figures -> {FIG}   ({len(list(FIG.glob('*.png')))} panels)")
    print(f"  tables  -> {TAB}/prefit_*.csv, prefit_diagnostics.json")
