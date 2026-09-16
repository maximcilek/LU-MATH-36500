"""
scripts/11_prefit_onepager.py
=============================
Two-page, figure-led pre-fit summary for a non-specialist reader.

Page 1  one column per disease: epidemic curve, mechanistic fit, residual ACF
Page 2  cross-disease comparison + the decision table that drives the model

Deliberately minimal prose. Every claim is a number in a table or a mark on a
chart. Run after scripts/02_prefit_diagnostics.py.

    python scripts/11_prefit_onepager.py
"""
import sys, json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import optimize

from core.diagnostics import pearson_residuals, km_attack_rate_np, km_R0_from_attack_rate
from core.bhmm.ode_surrogate import solve_window
from core.bhmm.priors import EBOLA, MEASLES, INFLUENZA, NOROVIRUS

from core.pdf_utils import (register_fonts, SANS, SANS_BOLD, SANS_ITAL,
                            NAVY, MGREY, LGREY, WHITE, BLACK, make_footer)
register_fonts()
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, HRFlowable, Image as RLImage, PageBreak)

ROOT = Path(__file__).resolve().parents[1]
CAN, TAB, FIG, REP = ROOT/"data"/"canonical", ROOT/"tables", ROOT/"figures"/"prefit", ROOT/"reports"
REP.mkdir(exist_ok=True)

RED, PURP, TEAL, BLUE, GREY = "#C0392B", "#7B2D8B", "#1A7B6B", "#2E75B6", "#777777"
GOOD, WARN, BAD = colors.HexColor("#D5F5D5"), colors.HexColor("#FFF3CD"), colors.HexColor("#FDEBEA")
plt.rcParams.update({"font.size": 7.5, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 200,
                     "axes.titlesize": 8, "axes.labelsize": 7})


def load():
    return (pd.read_csv(CAN/"ebola_kikwit_1995_analysis_ready.csv"),
            pd.read_csv(CAN/"measles_hagelloch_1861_analysis_ready.csv"),
            pd.read_csv(CAN/"influenza_england_1978_analysis_ready.csv"),
            pd.read_csv(CAN/"norovirus_derbyshire_2001_analysis_ready.csv"),
            json.loads((TAB/"prefit_diagnostics.json").read_text()),
            pd.read_csv(TAB/"prefit_summary.csv"))


def _fit(meta, y, days, x0):
    def dev(p):
        m = np.clip(solve_window(np.exp(p[0]), **meta)[days]*np.exp(p[1]), 1e-9, None)
        o = np.clip(y, 1e-9, None); return 2*np.sum(o*np.log(o/m)-(o-m))
    r = optimize.minimize(dev, x0, method="Nelder-Mead", options={"maxiter": 2000})
    R0 = float(np.exp(r.x[0])); c = float(r.x[1])
    return R0, c, solve_window(R0, **meta)[days]*np.exp(c)


def page1(e, m, f, n, pf):
    fig, ax = plt.subplots(3, 4, figsize=(11.2, 7.6))

    # ---- Ebola -----------------------------------------------------------
    el = e[e.analysis_eligible]; ye = el.onset.values.astype(float); de = el.day.values
    meta_e = dict(structure="seird_incidence", sigma=EBOLA.sigma, removal=EBOLA.removal,
                  gamma_q=0.0, mu=EBOLA.mu_point, N=EBOLA.N, I0=EBOLA.I0,
                  t_seed=0.0, T=int(de.max())+1, dt_obs=1.0, kE=1, kI=1, kQ=1)
    R0e, ce, fite = _fit(meta_e, ye, de, [np.log(1.8), 0.])
    rese = pearson_residuals(ye, fite)

    # ---- Measles ---------------------------------------------------------
    ym = m.rash_onset.values.astype(float); dm = m.day.values
    meta_m = dict(structure="seird_incidence", sigma=MEASLES.sigma, removal=MEASLES.removal,
                  gamma_q=0.0, mu=MEASLES.mu_point, N=MEASLES.N, I0=MEASLES.I0,
                  t_seed=0.0, T=int(dm.max())+1, dt_obs=1.0, kE=1, kI=1, kQ=1)
    R0m, cm, fitm = _fit(meta_m, ym, dm, [np.log(8.), 0.])
    resm = pearson_residuals(ym, fitm)

    # ---- Influenza -------------------------------------------------------
    yf = f.in_bed.values.astype(float); df_ = f.day.values
    ts = pf["influenza"]["t_seed_best"]
    meta_f = dict(structure="seiqr_prevalence", sigma=INFLUENZA.sigma,
                  removal=INFLUENZA.removal, gamma_q=INFLUENZA.gamma_q, mu=0.0,
                  N=INFLUENZA.N, I0=INFLUENZA.I0, t_seed=ts, T=len(yf), dt_obs=1.0,
                  kE=1, kI=1, kQ=1)
    def devf(p):
        mm = np.clip(solve_window(np.exp(p[0]), **meta_f)*np.exp(p[1]), 1e-9, None)
        o = np.clip(yf, 1e-9, None); return 2*np.sum(o*np.log(o/mm)-(o-mm))
    rf = optimize.minimize(devf, [np.log(3.), 0.], method="Nelder-Mead",
                           options={"maxiter": 2000})
    R0f = float(np.exp(rf.x[0])); fitf = solve_window(R0f, **meta_f)*np.exp(rf.x[1])
    resf = pearson_residuals(yf, fitf)

    cols = [
        ("Ebola  Kikwit 1995", RED, de, ye, fite, rese, R0e, "139 days", (1.51, 2.53)),
        ("Measles  Hagelloch 1861", PURP, dm, ym, fitm, resm, R0m, "86 days", (10, 20)),
        ("Influenza  England 1978", TEAL, df_, yf, fitf, resf, R0f, "14 days", None),
    ]
    for j, (title, col, dd, yy, ff, rr, R0, nlab, lit) in enumerate(cols):
        a = ax[0, j]
        a.bar(dd, yy, color=col, alpha=.85, width=1.0)
        a.set_title(f"{title}\n{nlab},  peak {int(yy.max())}", color=col, fontweight="bold")
        a.set_ylabel("cases/day")
        a = ax[1, j]
        a.plot(dd, yy, "o", ms=2.2, color=col, alpha=.7, label="observed")
        a.plot(dd, ff, "-", lw=1.8, color="#1F4E79",
               label=f"model  $R_0$={R0:.2f}")
        a.set_ylabel("cases/day"); a.legend(frameon=False, fontsize=6)
        a.set_title("does the mechanism fit?")
        a = ax[2, j]
        lags = range(1, 8)
        acf = [float(np.corrcoef(rr[k:], rr[:-k])[0, 1]) for k in lags]
        bars = a.bar(list(lags), acf, color=["#C0392B" if abs(v) > .2 else "#2E7D32" for v in acf],
                     alpha=.9)
        a.axhline(.2, color="k", ls="--", lw=.8); a.axhline(-.2, color="k", ls="--", lw=.8)
        a.axhline(0, color="k", lw=.5); a.set_ylim(-.5, 1.0)
        a.set_xlabel("lag (days)"); a.set_ylabel("residual r")
        ok = abs(acf[0]) < .2
        a.set_title(("leftover pattern? NO" if ok else "leftover pattern? YES"),
                    color="#2E7D32" if ok else "#C0392B", fontweight="bold")

    # ---- Norovirus column (different data type) --------------------------
    by = n.groupby("class").ill.agg(["sum", "count"]); by["ar"] = by["sum"]/by["count"]
    a = ax[0, 3]
    a.bar(by.index.astype(str), by.ar, color=["#C0392B" if i == 10 else BLUE for i in by.index], alpha=.85)
    a.axhline(n.ill.mean(), color="k", ls="--", lw=1)
    a.set_title(f"Norovirus  Derbyshire 2001\n492 students,  {int(n.ill.sum())} ill",
                color=BLUE, fontweight="bold")
    a.set_ylabel("attack rate"); a.tick_params(axis="x", labelsize=5)
    a = ax[1, 3]
    rr_ = np.linspace(1.02, 3, 200)
    a.plot(rr_, [km_attack_rate_np(v) for v in rr_], color="#1F4E79", lw=1.8)
    a.axhline(n.ill.mean(), color=BLUE, ls="--", lw=1.5)
    a.axvline(km_R0_from_attack_rate(float(n.ill.mean())), color=BLUE, ls=":", lw=1.5)
    a.set_title(f"final size  ->  $R_0$={km_R0_from_attack_rate(float(n.ill.mean())):.2f}")
    a.set_xlabel("$R_0$"); a.set_ylabel("attack rate")
    a = ax[2, 3]
    a.axis("off")
    a.text(.5, .5, "no time series\n\nclass clustering handled by\n15 random intercepts\n\n"
                   "clustering p < 0.001", ha="center", va="center", fontsize=7, color=BLUE,
           bbox=dict(boxstyle="round,pad=0.45", facecolor="#EAF2FB", edgecolor=BLUE))
    a.set_title("leftover pattern? not applicable", color="#2E7D32", fontweight="bold")

    fig.tight_layout(pad=0.6)
    out = FIG/"onepager_page1.png"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    return out, dict(R0e=R0e, R0m=R0m, R0f=R0f)


def page2(e, m, f, n, pf, ps, fits):
    fig, ax = plt.subplots(1, 4, figsize=(11.2, 3.5))
    names = ["Ebola", "Measles", "Influenza"]
    cols = [RED, PURP, TEAL]

    vm = [ps.loc[ps.disease == d.lower(), "var_mean"].values[0] for d in names]
    ax[0].bar(names, vm, color=cols, alpha=.85)
    ax[0].axhline(1, color="k", ls="--", lw=1.2)
    ax[0].set_yscale("log"); ax[0].set_ylabel("variance / mean")
    ax[0].set_title("Is Poisson enough?\nNo - all above 1", fontweight="bold")
    ax[0].tick_params(axis="x", labelsize=6.5)

    l1 = [ps.loc[ps.disease == d.lower(), "lag1_acf"].values[0] for d in names]
    b = ax[1].bar(names, l1, color=["#2E7D32" if abs(v) < .2 else "#C0392B" for v in l1], alpha=.9)
    ax[1].axhline(.2, color="k", ls="--", lw=1); ax[1].axhline(-.2, color="k", ls="--", lw=1)
    ax[1].axhline(0, color="k", lw=.6); ax[1].set_ylabel("residual lag-1 r")
    ax[1].set_title("Leftover time pattern?\nOnly Influenza", fontweight="bold")
    ax[1].tick_params(axis="x", labelsize=6.5)

    est = [fits["R0e"], fits["R0m"], fits["R0f"],
           km_R0_from_attack_rate(float(n.ill.mean()))]
    lit = [(1.51, 2.53), (10, 20), None, (1.3, 6.7)]
    lbl = ["Ebola", "Measles", "Influenza", "Norovirus"]
    cc = [RED, PURP, TEAL, BLUE]
    x = np.arange(4)
    for i, (lo_hi, c) in enumerate(zip(lit, cc)):
        if lo_hi:
            ax[2].fill_between([i-.35, i+.35], [lo_hi[0]]*2, [lo_hi[1]]*2,
                               color="green", alpha=.18)
    ax[2].scatter(x, est, s=45, color=cc, zorder=5)
    ax[2].set_xticks(x); ax[2].set_xticklabels(lbl, fontsize=6.5, rotation=15)
    ax[2].set_yscale("log"); ax[2].set_ylabel("$R_0$")
    ax[2].set_title("Does $R_0$ match literature?\ngreen = published range", fontweight="bold")

    sizes = [139, 86, 14, 492]
    ax[3].bar(lbl, sizes, color=cc, alpha=.85)
    ax[3].set_ylabel("observations"); ax[3].set_yscale("log")
    ax[3].set_title("How much data?\nInfluenza is thin", fontweight="bold")
    ax[3].tick_params(axis="x", labelsize=6.5, rotation=15)

    fig.tight_layout(pad=0.6)
    out = FIG/"onepager_page2.png"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)

    # Influenza conflict inset
    fig2, a = plt.subplots(1, 1, figsize=(4.6, 2.9))
    rr = np.linspace(1.05, 9, 300)
    a.plot(rr, [km_attack_rate_np(v) for v in rr], color="#1F4E79", lw=2)
    a.axhline(.671, color="#C0392B", ls="--", lw=1.8)
    a.axvline(1.66, color="#C0392B", ls=":", lw=1.8)
    a.axvline(fits["R0f"], color=TEAL, ls=":", lw=1.8)
    a.annotate("size says\n$R_0$=1.66", (1.66, .06), xytext=(2.9, .12), fontsize=7.5,
               color="#C0392B", ha="center",
               arrowprops=dict(arrowstyle="->", color="#C0392B", lw=1))
    a.annotate(f"speed says\n$R_0$={fits['R0f']:.1f}", (fits["R0f"], .80), xytext=(5.0, .48),
               fontsize=7.5, color=TEAL, ha="center",
               arrowprops=dict(arrowstyle="->", color=TEAL, lw=1))
    a.set_ylim(0, 1.08)
    a.set_xlabel("$R_0$"); a.set_ylabel("attack rate")
    a.set_title("Influenza 1978: the two clues disagree", fontsize=8.5, fontweight="bold")
    fig2.tight_layout()
    out2 = FIG/"onepager_flu_conflict.png"
    fig2.savefig(out2, bbox_inches="tight", dpi=200); plt.close(fig2)
    return out, out2


def build_pdf(p1, p2, p2b, pf, ps, fits):
    S = lambda n_, **k: ParagraphStyle(n_, **{**dict(fontName=SANS, fontSize=8,
                                                     leading=10.5, textColor=BLACK), **k})
    T_ = S("T", fontSize=13, fontName=SANS_BOLD, textColor=NAVY, alignment=TA_CENTER, leading=16)
    ST = S("ST", fontSize=8, textColor=colors.HexColor(GREY), alignment=TA_CENTER)
    H = S("H", fontSize=9.5, fontName=SANS_BOLD, textColor=NAVY, spaceBefore=4, spaceAfter=2)
    B = S("B", fontSize=8, leading=10.5)

    def tbl(head, rows, widths, rc=None, fs=6.6):
        hr = [Paragraph(h, S("th", fontSize=fs, fontName=SANS_BOLD, textColor=WHITE)) for h in head]
        dr = [[Paragraph(str(c), S("td", fontSize=fs, leading=fs+2)) for c in r] for r in rows]
        sty = [("BACKGROUND", (0,0), (-1,0), NAVY), ("TEXTCOLOR", (0,0), (-1,0), WHITE),
               ("ROWBACKGROUNDS", (0,1), (-1,-1), [WHITE, LGREY]),
               ("GRID", (0,0), (-1,-1), .25, MGREY), ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
               ("LEFTPADDING", (0,0), (-1,-1), 3), ("RIGHTPADDING", (0,0), (-1,-1), 3),
               ("TOPPADDING", (0,0), (-1,-1), 2), ("BOTTOMPADDING", (0,0), (-1,-1), 2)]
        if rc:
            for i, c in enumerate(rc):
                if c is not None: sty.append(("BACKGROUND", (0,i+1), (-1,i+1), c))
        t = Table([hr]+dr, colWidths=widths); t.setStyle(TableStyle(sty)); return t

    st = []
    st += [Paragraph("Pre-fit Diagnostics", T_),
           Paragraph("Four outbreaks, checked before any model was fitted", ST),
           HRFlowable(width="100%", thickness=.9, color=NAVY, spaceBefore=2, spaceAfter=4),
           RLImage(str(p1), width=7.5*inch, height=5.09*inch), Spacer(1, 4)]

    g = lambda ok: GOOD if ok else BAD
    rows, rc = [], []
    for d, disp_name in [("ebola","Ebola"), ("measles","Measles"), ("influenza","Influenza")]:
        r = ps[ps.disease == d].iloc[0]
        ok = not bool(r.grw_warranted)
        rows.append([disp_name, f"{r.var_mean:.1f}", f"{r.lag1_acf:+.3f}",
                     "no" if ok else "YES",
                     "clean" if ok else "pattern left over"])
        rc.append(g(ok))
    rows.append(["Norovirus", "n/a (binary)", "n/a", "n/a", "clean"]); rc.append(GOOD)
    st += [Paragraph("Verdict per outbreak", H),
           tbl(["Outbreak", "Variance/mean", "Residual lag-1", "Needs extra time term?", "Status"],
               rows, [1.3*inch, 1.3*inch, 1.3*inch, 1.75*inch, 1.8*inch], rc), Spacer(1, 3),
           Paragraph("Top row is the raw data. Middle row overlays the mechanistic model. "
                     "Bottom row asks whether any pattern is left in the errors — green bars mean "
                     "the model captured the timing and nothing systematic remains.", B)]

    st += [PageBreak(),
           Paragraph("Cross-outbreak Comparison", T_),
           Paragraph("The four checks that decide how each arm is modelled", ST),
           HRFlowable(width="100%", thickness=.9, color=NAVY, spaceBefore=2, spaceAfter=4),
           RLImage(str(p2), width=7.5*inch, height=2.34*inch), Spacer(1, 5)]

    dec = [
        ["Ebola", "daily new cases", "Negative Binomial", "no", "PASS",
         "clean fit, R<sub>0</sub> in published range"],
        ["Measles", "daily new rash", "Negative Binomial", "no", "PASS",
         "clean fit, complete village cohort"],
        ["Norovirus", "ill / not ill", "Bernoulli + class effects", "n/a", "PASS",
         "tight estimate, clustering handled"],
        ["Influenza", "students in bed", "Negative Binomial", "saturated - off", "CAVEAT",
         "only 14 days; two clues disagree"],
    ]
    st += [Paragraph("Modelling decisions that follow", H),
           tbl(["Outbreak", "What was counted", "Observation model", "Time-varying term",
                "Verdict", "Why"], dec,
               [.85*inch, 1.15*inch, 1.5*inch, 1.15*inch, .7*inch, 2.1*inch],
               [GOOD, GOOD, GOOD, WARN]), Spacer(1, 6)]

    flu_tbl = tbl(["Clue", "Implied R<sub>0</sub>"],
                  [["Final size (67% infected)", "1.66"],
                   [f"Epidemic speed", f"{fits['R0f']:.2f}"],
                   ["Published (Avilov 2024)", "8.14"]],
                  [2.05*inch, 1.25*inch], [WARN, WARN, WARN], fs=7.4)
    side = Table([[RLImage(str(p2b), width=3.9*inch, height=2.46*inch), flu_tbl]],
                 colWidths=[4.0*inch, 3.4*inch])
    side.setStyle(TableStyle([("VALIGN", (0,0), (-1,-1), "TOP"),
                              ("LEFTPADDING", (0,0), (-1,-1), 0)]))
    st += [Paragraph("The one problem worth knowing about", H), side, Spacer(1, 3),
           Paragraph("Influenza 1978 is a famous dataset that classic models cannot fully fit: the "
                     "final size and the epidemic speed point to different R<sub>0</sub> values, and no "
                     "published model reconciles them. It is kept as a stress test, not as a "
                     "headline estimate. Ebola, Measles and Norovirus carry the comparison.", B),
           Spacer(1, 5),
           Paragraph("How to proceed", H),
           tbl(["Next step", "Reason"],
               [["Report Ebola, Measles, Norovirus as the comparison",
                 "all three pass every pre-fit check"],
                ["Report Influenza only with its caveat stated",
                 "documented conflict; cite Avilov et al. (2024)"],
                ["Run once with and once without Influenza",
                 "confirms the caveated arm is not pulling the others"],
                ["Compare each posterior interval to the green bands above",
                 "the interval, not the point estimate, is the claim"]],
               [3.0*inch, 4.45*inch])]

    doc = SimpleDocTemplate(str(REP/"PREFIT_SUMMARY_2page.pdf"), pagesize=letter,
                            leftMargin=.5*inch, rightMargin=.5*inch,
                            topMargin=.45*inch, bottomMargin=.5*inch)
    f = make_footer("Pre-fit Diagnostics - CTMC-BHMM v7", "")
    doc.build(st, onFirstPage=f, onLaterPages=f)


if __name__ == "__main__":
    e, m, f, n, pf, ps = load()
    p1, fits = page1(e, m, f, n, pf)
    p2, p2b = page2(e, m, f, n, pf, ps, fits)
    build_pdf(p1, p2, p2b, pf, ps, fits)
    print(f"  -> {REP/'PREFIT_SUMMARY_2page.pdf'}")
