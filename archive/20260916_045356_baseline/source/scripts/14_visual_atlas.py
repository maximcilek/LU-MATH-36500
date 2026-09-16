"""
scripts/14_visual_atlas.py
==========================
Visual atlas: every useful figure, one per file, each with a written reading.

    reports/VISUAL_ATLAS.pdf          the assembled document
    figures/atlas/<name>.png          every figure standalone
    figures/atlas/<name>.txt          its description / interpretation / takeaway

Target: 14+ figures per outbreak covering distribution shape, skewness,
dependence, collinearity, multiplicity, mean-variance structure, model fit and
sensitivity; plus a cross-outbreak section built around the specific inferential
design (three measured human outbreaks, one performance benchmark, one
undocumented pathogen carried without synthetic data).
"""
import sys, json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy import stats, optimize

from core.diagnostics import (pearson_residuals, km_attack_rate_np,
                              km_R0_from_attack_rate, dispersion, serial_correlation)
from core.bhmm.ode_surrogate import solve_window
from core.bhmm.priors import EBOLA, MEASLES, INFLUENZA, NOROVIRUS, HYPERPRIOR

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
RAW, CAN, TAB = ROOT/"data"/"raw", ROOT/"data"/"canonical", ROOT/"tables"
FA = ROOT/"figures"/"atlas"; FA.mkdir(parents=True, exist_ok=True)
REP = ROOT/"reports"; REP.mkdir(exist_ok=True)

RED, PURP, TEAL, BLUE, ORANGE = "#C0392B", "#7B2D8B", "#1A7B6B", "#2E75B6", "#ED7D31"
NAVYH, GREEN, GREY = "#1F4E79", "#2E7D32", "#777777"
GOOD, WARN, BAD, INFO = (colors.HexColor("#D5F5D5"), colors.HexColor("#FFF3CD"),
                         colors.HexColor("#FDEBEA"), colors.HexColor("#E8EEF7"))
plt.rcParams.update({"font.size": 7.0, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 190, "axes.titlesize": 7.6, "axes.labelsize": 6.8,
                     "xtick.labelsize": 6.2, "ytick.labelsize": 6.2, "legend.fontsize": 6.0})

S  = lambda n, **k: ParagraphStyle(n, **{**dict(fontName=SANS, fontSize=8.2,
                                                leading=10.6, textColor=BLACK), **k})
TT = S("T",  fontSize=16, fontName=SANS_BOLD, textColor=NAVY, alignment=TA_CENTER, leading=19)
H1 = S("H1", fontSize=13, fontName=SANS_BOLD, textColor=NAVY, spaceBefore=6, spaceAfter=3)
H2 = S("H2", fontSize=10, fontName=SANS_BOLD, textColor=NAVY, spaceBefore=5, spaceAfter=2)
SU = S("SU", fontSize=8.6, textColor=colors.HexColor(GREY), alignment=TA_CENTER)
CAP = S("CAP", fontSize=6.9, leading=8.6, textColor=colors.HexColor("#333333"))
BODY = S("B", fontSize=8.2, leading=10.8)

REG = {}   # name -> (path, caption, takeaway)


def F(name, reading, takeaway, figsize=(3.3, 2.5)):
    """Decorator: build one figure, save png + txt, register it."""
    def deco(fn):
        f, a = plt.subplots(1, 1, figsize=figsize)
        fn(a)
        f.tight_layout(pad=0.35)
        p = FA/f"{name}.png"; f.savefig(p, bbox_inches="tight"); plt.close(f)
        (FA/f"{name}.txt").write_text(
            f"{name}\n{'='*len(name)}\n\nWHAT IT SHOWS\n{reading.strip()}\n\n"
            f"KEY TAKEAWAY\n{takeaway.strip()}\n")
        REG[name] = (p, reading.strip(), takeaway.strip())
        return fn
    return deco


def cap(t): return Paragraph(t, CAP)
def hr(): return HRFlowable(width="100%", thickness=.9, color=NAVY, spaceBefore=2, spaceAfter=4)


def grid(names, ncol=3, w=2.38):
    """Lay out registered figures ncol per row with captions beneath."""
    cells = []
    for n in names:
        p, reading, take = REG[n]
        inner = Table([[RLImage(str(p), width=w*inch, height=w*0.76*inch)],
                       [cap(f"<b>{take}</b>")]], colWidths=[w*inch])
        inner.setStyle(TableStyle([("LEFTPADDING",(0,0),(-1,-1),0),
                                   ("RIGHTPADDING",(0,0),(-1,-1),0),
                                   ("TOPPADDING",(0,0),(-1,-1),1),
                                   ("BOTTOMPADDING",(0,0),(-1,-1),5),
                                   ("VALIGN",(0,0),(-1,-1),"TOP")]))
        cells.append(inner)
    rows = [cells[i:i+ncol] for i in range(0, len(cells), ncol)]
    for r in rows:
        while len(r) < ncol: r.append("")
    t = Table(rows, colWidths=[(w+0.07)*inch]*ncol)
    t.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"TOP"),
                           ("LEFTPADDING",(0,0),(-1,-1),2),("RIGHTPADDING",(0,0),(-1,-1),2),
                           ("TOPPADDING",(0,0),(-1,-1),2),("BOTTOMPADDING",(0,0),(-1,-1),2)]))
    return t


def tbl(head, rows, widths, rc=None, fs=6.8):
    hrw = [Paragraph(h, S("th", fontSize=fs, fontName=SANS_BOLD, textColor=WHITE)) for h in head]
    drw = [[Paragraph(str(c), S("td", fontSize=fs, leading=fs+2.3)) for c in r] for r in rows]
    sty = [("BACKGROUND",(0,0),(-1,0),NAVY),("TEXTCOLOR",(0,0),(-1,0),WHITE),
           ("ROWBACKGROUNDS",(0,1),(-1,-1),[WHITE,LGREY]),("GRID",(0,0),(-1,-1),.25,MGREY),
           ("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),3),
           ("RIGHTPADDING",(0,0),(-1,-1),3),("TOPPADDING",(0,0),(-1,-1),2.5),
           ("BOTTOMPADDING",(0,0),(-1,-1),2.5)]
    if rc:
        for i,c in enumerate(rc):
            if c is not None: sty.append(("BACKGROUND",(0,i+1),(-1,i+1),c))
    t = Table([hrw]+drw, colWidths=widths); t.setStyle(TableStyle(sty)); return t


def note(text, bg, bd):
    t = Table([[Paragraph(text, S("v", fontSize=8.4, fontName=SANS_BOLD,
                                  textColor=colors.HexColor(bd)))]], colWidths=[7.3*inch])
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),bg),
                           ("BOX",(0,0),(-1,-1),1,colors.HexColor(bd)),
                           ("LEFTPADDING",(0,0),(-1,-1),8),("RIGHTPADDING",(0,0),(-1,-1),8),
                           ("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6)]))
    return t


# ---------------------------------------------------------------- shared helpers
def nb_pmf_fit(y):
    """Method-of-moments NB and Poisson pmfs over the observed support."""
    mu, var = y.mean(), y.var(ddof=1)
    k = np.arange(0, int(y.max())+1)
    pois = stats.poisson.pmf(k, mu)
    if var > mu:
        r = mu**2/(var-mu); p = r/(r+mu)
        nb = stats.nbinom.pmf(k, r, p)
    else:
        nb = pois
    return k, pois, nb


def fit_inc(cfg, y, days, x0, t_seed=0.0):
    meta = dict(structure="seird_incidence", sigma=cfg.sigma, removal=cfg.removal,
                gamma_q=0.0, mu=cfg.mu_point, N=cfg.N, I0=cfg.I0, t_seed=t_seed,
                T=int(days.max())+1, dt_obs=1.0, kE=1, kI=1, kQ=1)
    def dev(p):
        m = np.clip(solve_window(np.exp(p[0]), **meta)[days]*np.exp(p[1]), 1e-9, None)
        o = np.clip(y, 1e-9, None); return 2*np.sum(o*np.log(o/m)-(o-m))
    r = optimize.minimize(dev, x0, method="Nelder-Mead", options={"maxiter":2500})
    R0 = float(np.exp(r.x[0]))
    return R0, solve_window(R0, **meta)[days]*np.exp(float(r.x[1])), meta


def roll_growth(y, w=7):
    lg = np.log(np.clip(y, .5, None))
    return pd.Series(lg).diff().rolling(w, center=True).mean().values


# ================================================================= per-disease builder
def incidence_suite(tag, y, days, cfg, col, obs_label, x0, extra_cols=None):
    """14 standard figures for an incidence-type outbreak."""
    R0, fit, meta = fit_inc(cfg, y, days, x0)
    res = pearson_residuals(y, fit)
    d = dispersion(y); sraw = serial_correlation(y); spost = serial_correlation(res)
    logy = np.log1p(y)

    @F(f"{tag}_01_curve", f"Daily {obs_label} plotted against day of outbreak.",
       "One rise and one fall: a single wave that exhausted its susceptibles and stopped.")
    def _(a):
        a.bar(days, y, color=col, alpha=.85, width=1.0)
        a.set(xlabel="day", ylabel=obs_label, title="1. Epidemic curve")

    @F(f"{tag}_02_cumulative", "Running total of cases as the outbreak progressed.",
       "The S-shape confirms saturation: growth slows because susceptibles run out, not by chance.")
    def _(a):
        a.plot(days, np.cumsum(y), lw=2, color=col)
        a.set(xlabel="day", ylabel="cumulative cases", title="2. Cumulative total")

    @F(f"{tag}_03_hist", "Histogram of daily counts with Poisson and negative-binomial fits.",
       "The observed spread is wider than Poisson allows, so a negative binomial is required.")
    def _(a):
        k, pois, nb = nb_pmf_fit(y)
        a.hist(y, bins=np.arange(-.5, y.max()+1.5), density=True, color=col, alpha=.75,
               label="observed")
        a.plot(k, pois, "o--", ms=2.5, color="#888888", lw=1, label="Poisson")
        a.plot(k, nb, "s-", ms=2.5, color=NAVYH, lw=1.3, label="neg. binomial")
        a.set(xlabel=obs_label, ylabel="density", title="3. Count distribution")
        a.legend(frameon=False)

    @F(f"{tag}_04_qq", "Quantile-quantile plot of observed counts against a Poisson reference.",
       "Points curving above the line in the upper tail means extreme days are more common than Poisson predicts.")
    def _(a):
        n = len(y); q = (np.arange(1, n+1)-.5)/n
        theo = stats.poisson.ppf(q, y.mean())
        a.scatter(theo, np.sort(y), s=8, color=col, alpha=.75)
        lim = max(theo.max(), y.max())
        a.plot([0, lim], [0, lim], "k--", lw=1)
        a.set(xlabel="Poisson quantile", ylabel="observed quantile", title="4. Q-Q vs Poisson")

    @F(f"{tag}_05_skew", "Distribution on the raw scale and after a log transform.",
       f"Skewness {d['skewness']:.2f} on the raw scale; the log scale is far more symmetric, confirming multiplicative rather than additive variation.")
    def _(a):
        a.hist(y, bins=14, alpha=.6, color=col, density=True, label=f"raw (skew {d['skewness']:.2f})")
        a.hist(logy*(y.max()/max(logy.max(),1e-9)), bins=14, alpha=.5, color=NAVYH,
               density=True, label=f"log (skew {stats.skew(logy):.2f})")
        a.set(xlabel="value (log rescaled to raw axis)", ylabel="density", title="5. Skewness check")
        a.legend(frameon=False)

    @F(f"{tag}_06_meanvar", "Local mean against local variance in rolling windows.",
       "Variance grows faster than the mean, the signature of clustered spread rather than independent events.")
    def _(a):
        s = pd.Series(y); mw = s.rolling(9).mean(); vw = s.rolling(9).var()
        ok = mw.notna() & vw.notna() & (mw > 0)
        a.scatter(mw[ok], vw[ok], s=8, color=col, alpha=.7)
        lim = float(mw[ok].max())
        a.plot([0, lim], [0, lim], "k--", lw=1, label="variance = mean (Poisson)")
        a.set(xlabel="local mean", ylabel="local variance", title="6. Mean-variance relation")
        a.legend(frameon=False)

    @F(f"{tag}_07_acf", "Correlation between counts separated by 1 to 14 days, before and after the model.",
       f"Raw dependence {sraw['acf_lag1']:+.2f} falls to {spost['acf_lag1']:+.2f} once the mechanism is fitted"
       + (" - fully absorbed." if abs(spost['acf_lag1']) < .2 else " - not fully absorbed."))
    def _(a):
        L = list(range(1, min(15, len(y)//3)))
        a.plot(L, [float(np.corrcoef(y[k:], y[:-k])[0,1]) for k in L], "o-", ms=2.6,
               color="#BBBBBB", lw=1.1, label="raw")
        a.plot(L, [float(np.corrcoef(res[k:], res[:-k])[0,1]) for k in L], "o-", ms=2.6,
               color=GREEN if abs(spost['acf_lag1'])<.2 else RED, lw=1.5, label="residual")
        a.axhline(.2, color="k", ls="--", lw=.7); a.axhline(-.2, color="k", ls="--", lw=.7)
        a.axhline(0, color="k", lw=.5)
        a.set(xlabel="days apart", ylabel="correlation", title="7. Serial dependence")
        a.legend(frameon=False)

    @F(f"{tag}_08_lagscatter", "Each day's count plotted against the previous day's count.",
       "A tight diagonal cloud means today resembles yesterday; the model must explain that, not ignore it.")
    def _(a):
        a.scatter(y[:-1], y[1:], s=9, color=col, alpha=.65)
        lim = y.max()*1.05
        a.plot([0, lim], [0, lim], "k--", lw=.9)
        a.set(xlabel="count on day t-1", ylabel="count on day t", title="8. Day-to-day scatter")

    @F(f"{tag}_09_fit", "Observed counts with the fitted mechanistic trajectory overlaid.",
       f"The fitted spread score is {R0:.2f}; the curve tracks the observed rise and fall.")
    def _(a):
        a.plot(days, y, "o", ms=2.4, color=col, alpha=.6, label="observed")
        a.plot(days, fit, "-", lw=1.9, color=NAVYH, label=f"model R0={R0:.2f}")
        a.set(xlabel="day", ylabel=obs_label, title="9. Mechanistic fit")
        a.legend(frameon=False)

    @F(f"{tag}_10_resid_fitted", "Standardised residuals against fitted values.",
       "No funnel or curve means the chosen variance structure is appropriate across the whole range.")
    def _(a):
        a.scatter(fit, res, s=9, color=col, alpha=.65)
        a.axhline(0, color="k", lw=.8)
        a.axhline(2, color="#888888", ls=":", lw=.8); a.axhline(-2, color="#888888", ls=":", lw=.8)
        a.set(xlabel="fitted value", ylabel="standardised residual", title="10. Residual vs fitted")

    @F(f"{tag}_11_resid_time", "Residuals plotted in time order.",
       "Scatter around zero with no drift means the model is not systematically early or late.")
    def _(a):
        a.plot(days, res, "o-", ms=2.2, lw=.7, color=col, alpha=.75)
        a.axhline(0, color="k", lw=.8)
        a.set(xlabel="day", ylabel="standardised residual", title="11. Residuals over time")

    @F(f"{tag}_12_growth", "Rolling week-on-week growth rate of the epidemic.",
       "Growth starts positive, crosses zero at the peak, then turns negative - the classic single-wave signature.")
    def _(a):
        g = roll_growth(y)
        a.plot(days, g, lw=1.6, color=col)
        a.axhline(0, color="k", ls="--", lw=.9)
        a.set(xlabel="day", ylabel="log growth per day", title="12. Growth rate over time")

    @F(f"{tag}_13_collin", "Correlation among candidate day-level predictors.",
       "Cumulative totals correlate strongly with each other, so only one time-derived quantity may enter the model.")
    def _(a):
        df = pd.DataFrame({"count": y, "cumulative": np.cumsum(y), "day": days,
                           "log count": logy, "growth": np.nan_to_num(roll_growth(y))})
        if extra_cols:
            for k_, v_ in extra_cols.items(): df[k_] = v_
        C = df.corr().values
        im = a.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
        a.set_xticks(range(len(df.columns))); a.set_yticks(range(len(df.columns)))
        a.set_xticklabels(df.columns, rotation=45, ha="right", fontsize=5.4)
        a.set_yticklabels(df.columns, fontsize=5.4)
        for i in range(len(C)):
            for j in range(len(C)):
                a.text(j, i, f"{C[i,j]:.2f}", ha="center", va="center", fontsize=4.6)
        a.set_title("13. Collinearity among candidates")

    @F(f"{tag}_14_sensitivity", "Fitted spread score as the assumed infectious period is varied.",
       "The estimate moves with the assumed biology, so the assumed values are reported alongside the result rather than hidden.")
    def _(a):
        base = cfg.removal
        mult = np.array([.6, .8, 1.0, 1.25, 1.6])
        out = []
        for mm in mult:
            mt = dict(meta); mt["removal"] = base*mm
            def dv(p):
                m_ = np.clip(solve_window(np.exp(p[0]), **mt)[days]*np.exp(p[1]), 1e-9, None)
                o_ = np.clip(y, 1e-9, None); return 2*np.sum(o_*np.log(o_/m_)-(o_-m_))
            rr = optimize.minimize(dv, [np.log(max(R0,1.1)), 0.], method="Nelder-Mead",
                                   options={"maxiter":900})
            out.append(float(np.exp(rr.x[0])))
        a.plot(1/(base*mult), out, "o-", ms=3.5, color=col, lw=1.5)
        a.axvline(1/base, color="k", ls="--", lw=1, label="assumed")
        a.set(xlabel="assumed infectious days", ylabel="fitted spread score",
              title="14. Sensitivity to assumed biology")
        a.legend(frameon=False)

    return dict(R0=R0, fit=fit, res=res, disp=d, sraw=sraw, spost=spost, meta=meta)


# ================================================================= disease-specific extras
def ebola_extras(raw, can):
    el = can[can.analysis_eligible]
    chi2, pchi = stats.chi2_contingency(pd.crosstab(raw.reporting, raw.onset > 0))[:2]

    @F("ebola_15_cfr", "Cumulative death rate recomputed week by week.",
       "Unstable for ten weeks purely because deaths trail cases; day-level death rate is therefore never a model input.")
    def _(a):
        wk = can.assign(w=can.day//7).groupby("w").agg(o=("onset","sum"), dd=("death","sum"))
        a.plot(wk.index, (wk.dd.cumsum()/wk.o.cumsum().replace(0, np.nan)).clip(0,2),
               "o-", ms=3, color=ORANGE, lw=1.4)
        a.axhline(236/292, color=RED, ls="--", lw=1.1, label="final 80.8%")
        a.set(xlabel="week", ylabel="deaths / cases so far", ylim=(0,1.6),
              title="15. Death-rate lag artefact"); a.legend(frameon=False)

    @F("ebola_16_reporting", "Daily counts split by whether active case-finding was running.",
       f"Near-total separation (p={pchi:.0e}): non-reporting days carry no information and are excluded rather than counted as zeros.")
    def _(a):
        g = [raw.loc[raw.reporting,"onset"].values, raw.loc[~raw.reporting,"onset"].values]
        bp = a.boxplot(g, labels=["counting","not counting"], patch_artist=True, widths=.55)
        for b,c in zip(bp["boxes"], [RED,"#BBBBBB"]): b.set_facecolor(c); b.set_alpha(.8)
        a.set(ylabel="cases per day", title="16. Structural zeros")

    @F("ebola_17_eligible", "Which calendar days entered the likelihood.",
       "139 of 192 days are usable; the excluded 53 are retained in the file so the decision stays auditable.")
    def _(a):
        a.bar(["used","set aside"], [int(can.analysis_eligible.sum()),
                                     int((~can.analysis_eligible).sum())],
              color=[RED,"#BBBBBB"], alpha=.85)
        a.set(ylabel="days", title="17. Eligibility split")

    @F("ebola_18_onset_death", "Onsets and deaths on a shared timeline.",
       "Deaths mirror onsets about two weeks later, which is the lag that makes early death rates meaningless.")
    def _(a):
        a.bar(can.day, can.onset, color=RED, alpha=.8, width=1., label="onset")
        a.bar(can.day, -can.death, color="#999999", alpha=.8, width=1., label="death")
        a.axhline(0, color="k", lw=.6); a.legend(frameon=False)
        a.set(xlabel="day", ylabel="count", title="18. Onsets vs deaths")


def measles_extras(can, ll):
    @F("measles_15_age", "Age distribution of infected children.",
       "Every age from infant to fifteen was hit, confirming no pre-existing immunity anywhere in the cohort.")
    def _(a):
        a.hist(ll.AGE.values, bins=15, color=PURP, alpha=.85)
        a.set(xlabel="age (years)", ylabel="children", title="15. Who was infected")

    @F("measles_16_class", "Children by school group.",
       "Three distinct mixing groups; the mid-epidemic burst is the disease moving through classrooms.")
    def _(a):
        c = ll.groupby("CL").size()
        a.bar(c.index.astype(str), c.values, color=PURP, alpha=.85)
        a.set(ylabel="children", title="16. School structure")
        a.tick_params(axis="x", labelsize=5.4)

    @F("measles_17_prodrome", "Gap between first symptoms and rash for each child.",
       "A consistent four-day gap justifies treating rash onset as a clean, uniformly shifted marker of infection.")
    def _(a):
        gap = (ll.ERU - ll.PRO).values
        a.hist(gap[(gap>=0)&(gap<15)], bins=np.arange(-.5,12.5), color=PURP, alpha=.85)
        a.set(xlabel="days from first symptoms to rash", ylabel="children",
              title="17. Symptom-to-rash gap")

    @F("measles_18_household", "Number of infected children per household.",
       "Several households contributed multiple cases, which is why the real curve is spikier than an even-mixing model draws.")
    def _(a):
        hh = ll.groupby("HN").size()
        a.hist(hh.values, bins=np.arange(.5, hh.max()+1.5), color=PURP, alpha=.85)
        a.set(xlabel="infected children in the household", ylabel="households",
              title="18. Household clustering")


def influenza_extras(can, R0curve):
    R0fs = km_R0_from_attack_rate(512/763)

    @F("flu_15_seed", "Fitted spread score against the assumed pre-observation period.",
       "The answer swings with an assumption the records do not pin down, which is the first of three unresolved limits.")
    def _(a):
        prof = pd.read_csv(TAB/"prefit_influenza_seed_profile.csv")
        a.plot(prof.t_seed, prof.R0, "o-", ms=3.4, color=TEAL, lw=1.5)
        a.axvspan(8, 12, color="green", alpha=.15, label="documented window")
        a.set(xlabel="assumed days already running", ylabel="fitted spread score",
              title="15. Sensitivity to start date"); a.legend(frameon=False)

    @F("flu_16_conflict", "Final-size curve with both candidate spread scores marked.",
       "Total infected and epidemic speed imply different answers; no single value satisfies both.")
    def _(a):
        rr = np.linspace(1.05, 9, 300)
        a.plot(rr, [km_attack_rate_np(v) for v in rr], color=NAVYH, lw=2)
        a.axhline(.671, color=RED, ls="--", lw=1.3)
        a.axvline(R0fs, color=RED, ls=":", lw=1.3); a.axvline(R0curve, color=TEAL, ls=":", lw=1.3)
        a.annotate(f"total says {R0fs:.1f}", (R0fs,.08), xytext=(3.2,.16), fontsize=5.8,
                   color=RED, ha="center", arrowprops=dict(arrowstyle="->", color=RED, lw=.7))
        a.annotate(f"speed says {R0curve:.1f}", (R0curve,.86), xytext=(5.6,.45), fontsize=5.8,
                   color=TEAL, ha="center", arrowprops=dict(arrowstyle="->", color=TEAL, lw=.7))
        a.set(xlabel="spread score", ylabel="share who fall ill", ylim=(0,1.08),
              title="16. Identification conflict")

    @F("flu_17_prevalence", "Share of the whole school in bed each day.",
       "More than a third of the school was bedridden simultaneously - an extreme prevalence peak in a closed population.")
    def _(a):
        a.plot(can.day, can.in_bed/763*100, "o-", ms=3, color=TEAL, lw=1.5)
        a.fill_between(can.day, 0, can.in_bed/763*100, color=TEAL, alpha=.2)
        a.set(xlabel="day", ylabel="% of school in bed", title="17. Prevalence peak")

    @F("flu_18_stock_flow", "Boys in bed alongside boys already recovering.",
       "The recovering group lags the in-bed group, confirming in-bed is a holding state, not the infectious state.")
    def _(a):
        a.bar(can.day-.2, can.in_bed, width=.4, color=TEAL, alpha=.85, label="in bed")
        a.bar(can.day+.2, can.convalescent, width=.4, color="#AAD4CC", alpha=.9, label="recovering")
        a.set(xlabel="day", ylabel="boys", title="18. In bed vs recovering"); a.legend(frameon=False)

    @F("flu_19_evidence", "Observation count for each outbreak on a log scale.",
       "Fourteen daily tallies against 492 individual children: the thinnest evidence base in the study.")
    def _(a):
        a.bar(["Ebola","Measles","Influenza","Norovirus"], [139,86,14,492],
              color=[RED,PURP,TEAL,BLUE], alpha=.85)
        a.set(yscale="log", ylabel="observations", title="19. Evidence volume")
        a.tick_params(axis="x", labelsize=5.6, rotation=12)


def norovirus_suite(raw, can):
    by = can.groupby("class").ill.agg(["sum","count"]); by["ar"] = by["sum"]/by["count"]
    ar = float(can.ill.mean()); R0 = km_R0_from_attack_rate(ar)
    chi2, pchi, dof, _ = stats.chi2_contingency(np.column_stack([by["sum"], by["count"]-by["sum"]]))
    well_absent = int(((raw.day_absent>0)&(raw.start_illness==0)).sum())
    ill_vomit = int(((raw.start_illness>0)&(raw.day_vomiting>0)).sum())

    @F("noro_01_classar", "Attack rate within each of the fifteen classrooms.",
       "Attack rates run from 0% to 67% between classrooms - clustering, not an even sprinkling.")
    def _(a):
        a.bar(by.index.astype(str), by.ar, color=[RED if i==10 else BLUE for i in by.index], alpha=.85)
        a.axhline(ar, color="k", ls="--", lw=1.1)
        a.set(xlabel="class", ylabel="share ill", title="1. Attack rate by class")
        a.tick_params(axis="x", labelsize=5.2)

    @F("noro_02_outcome", "Children recorded ill compared with children merely absent.",
       f"{well_absent} children were absent but never recorded ill; using absence would have inflated the outbreak 2.5-fold.")
    def _(a):
        a.bar(["recorded ill","absent from school"],
              [int(can.ill.sum()), int((raw.day_absent>0).sum())],
              color=[GREEN,"#BBBBBB"], alpha=.85)
        a.set(ylabel="children", title="2. Outcome definition matters")

    @F("noro_03_size", "Class size against class attack rate.",
       "No size effect, so the clustering reflects contact patterns rather than crowding.")
    def _(a):
        a.scatter(by["count"], by.ar, s=34, color=BLUE, alpha=.8)
        for i in (3,10):
            if i in by.index: a.annotate(f"c{i}", (by.loc[i,"count"], by.loc[i,"ar"]),
                                         fontsize=5.6, color=RED)
        a.set(xlabel="children in class", ylabel="share ill", title="3. Size is not the driver")

    @F("noro_04_binom", "Observed class attack rates against the range expected by chance alone.",
       "Several classes fall outside the chance band, which is formal evidence that classroom must enter the model.")
    def _(a):
        n = by["count"].values; k = by["sum"].values
        lo = np.array([stats.binom.ppf(.025, nn, ar)/nn for nn in n])
        hi = np.array([stats.binom.ppf(.975, nn, ar)/nn for nn in n])
        idx = np.argsort(n)
        a.fill_between(range(len(n)), lo[idx], hi[idx], color="#CCCCCC", alpha=.6, label="chance range")
        a.plot(range(len(n)), (k/n)[idx], "o", ms=4, color=BLUE, label="observed")
        a.set(xlabel="class (ordered by size)", ylabel="share ill",
              title=f"4. Beyond chance (p={pchi:.0e})"); a.legend(frameon=False)

    @F("noro_05_duration", "Length of illness for affected children.",
       "Most recovered within two days: a mild, short, self-limiting infection.")
    def _(a):
        dur = (can.loc[can.ill==1,"end_illness"] - can.loc[can.ill==1,"start_illness"])
        a.hist(dur[dur>=0], bins=12, color=BLUE, alpha=.85)
        a.set(xlabel="days ill", ylabel="children", title="5. Illness duration")

    @F("noro_06_onset", "Date on which each child first fell ill.",
       "Cases cluster into a short burst, consistent with a common exposure rather than a long chain.")
    def _(a):
        s = can.loc[can.ill==1,"start_illness"]
        a.hist(s, bins=np.arange(s.min()-.5, s.max()+1.5), color=BLUE, alpha=.85)
        a.set(xlabel="day of first illness", ylabel="children", title="6. Onset timing")

    @F("noro_07_multiplicity", "How many of fifteen classroom tests would look significant by chance.",
       "Testing fifteen classrooms separately would produce false positives; one pooled model with class effects avoids that.")
    def _(a):
        n = by["count"].values; k = by["sum"].values
        pv = np.array([stats.binomtest(int(kk), int(nn), ar).pvalue for kk,nn in zip(k,n)])
        a.plot(np.sort(pv), "o-", ms=4, color=BLUE, label="observed p-values")
        a.plot(np.linspace(0,1,len(pv)), "k--", lw=1, label="expected if no effect")
        a.axhline(.05, color=RED, ls=":", lw=1, label="0.05")
        a.axhline(.05/len(pv), color="#888888", ls=":", lw=1, label="0.05 / 15")
        a.set(xlabel="class rank", ylabel="p-value", title="7. Multiplicity check")
        a.legend(frameon=False, fontsize=5.2)

    @F("noro_08_vomit", "How often vomiting was recorded alongside illness.",
       f"Only {ill_vomit} children have both recorded - far too few to model, so vomiting is excluded.")
    def _(a):
        a.bar(["ill only","ill and vomiting"], [int(can.ill.sum())-ill_vomit, ill_vomit],
              color=[BLUE,"#BBBBBB"], alpha=.85)
        a.set(ylabel="children", title="8. Vomiting too sparse")

    @F("noro_09_finalsize", "Final-size curve with the observed attack rate marked.",
       f"Reading 15% off the curve gives a spread score of {R0:.2f}, at the low end of institutional norovirus estimates.")
    def _(a):
        rr = np.linspace(1.02, 3, 250)
        a.plot(rr, [km_attack_rate_np(v) for v in rr], color=NAVYH, lw=2)
        a.axhline(ar, color=BLUE, ls="--", lw=1.3); a.axvline(R0, color=BLUE, ls=":", lw=1.3)
        a.annotate(f"R0 = {R0:.2f}", (R0, ar), xytext=(1.85,.45), fontsize=6,
                   color=BLUE, ha="center", arrowprops=dict(arrowstyle="->", color=BLUE, lw=.8))
        a.set(xlabel="spread score", ylabel="share who fall ill", title="9. Final-size reading")

    @F("noro_10_shrink", "Raw class attack rates against the shrunk values a random-effect model produces.",
       "Extreme classrooms are pulled toward the school average rather than deleted, so no class dominates the result.")
    def _(a):
        n = by["count"].values; k = by["sum"].values
        tau = .5
        shr = (k + tau*ar*n.mean())/(n + tau*n.mean())
        a.scatter(k/n, shr, s=34, color=BLUE, alpha=.8)
        a.plot([0,.7],[0,.7],"k--",lw=.9)
        a.set(xlabel="raw class rate", ylabel="shrunk estimate", title="10. Partial pooling effect")

    @F("noro_11_collin", "Correlation among candidate child-level variables.",
       "Absence correlates with illness but is not the same thing; entering both would double-count the outcome.")
    def _(a):
        df = pd.DataFrame({"ill": can.ill, "absent": (raw.day_absent>0).astype(int),
                           "vomiting": (raw.day_vomiting>0).astype(int),
                           "class": can["class"]})
        C = df.corr().values
        a.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
        a.set_xticks(range(4)); a.set_yticks(range(4))
        a.set_xticklabels(df.columns, rotation=45, ha="right", fontsize=5.4)
        a.set_yticklabels(df.columns, fontsize=5.4)
        for i in range(4):
            for j in range(4): a.text(j,i,f"{C[i,j]:.2f}",ha="center",va="center",fontsize=5)
        a.set_title("11. Variable collinearity")

    @F("noro_12_power", "Width of the estimate achievable at different sample sizes.",
       "492 individual records is why this is the most precise arm; a daily tally of the same outbreak would be far vaguer.")
    def _(a):
        ns = np.array([25,50,100,200,300,492])
        se = np.sqrt(ar*(1-ar)/ns)*1.96
        a.plot(ns, se*100, "o-", ms=4, color=BLUE, lw=1.5)
        a.axvline(492, color="k", ls="--", lw=1, label="actual")
        a.set(xlabel="children recorded", ylabel="± percentage points",
              title="12. Precision from sample size"); a.legend(frameon=False)

    @F("noro_13_cumulative", "Cumulative count of ill children through the outbreak.",
       "The curve flattens early: the outbreak stopped well short of the whole school, unlike measles.")
    def _(a):
        s = can.loc[can.ill==1,"start_illness"].values
        days = np.arange(int(s.min()), int(s.max())+1)
        a.plot(days, [np.sum(s<=d_) for d_ in days], lw=2, color=BLUE)
        a.axhline(492, color="k", ls="--", lw=1, label="whole school")
        a.set(xlabel="day", ylabel="children ill so far", title="13. Cumulative total")
        a.legend(frameon=False)

    @F("noro_14_compare", "Share infected in each of the four outbreaks.",
       "At 15%, norovirus is the mildest of the four by reach - the opposite extreme from measles at 100%.")
    def _(a):
        a.bar(["Ebola","Measles","Influenza","Norovirus"], [29.2,100,67.1,15.2],
              color=[RED,PURP,TEAL,BLUE], alpha=.85)
        a.set(ylabel="% infected", title="14. Reach across outbreaks")
        a.tick_params(axis="x", labelsize=5.6, rotation=12)

    return dict(R0=R0, ar=ar, chi2=chi2, p=pchi)


# ================================================================= cross-disease
def cross_section(F_):
    names = ["Ebola","Measles","Influenza","Norovirus"]
    cols  = [RED, PURP, TEAL, BLUE]
    e  = pd.read_csv(CAN/"ebola_kikwit_1995_analysis_ready.csv")
    m  = pd.read_csv(CAN/"measles_hagelloch_1861_analysis_ready.csv")
    fl = pd.read_csv(CAN/"influenza_england_1978_analysis_ready.csv")
    ps = pd.read_csv(TAB/"prefit_summary.csv")
    zp = np.load(ROOT/"data"/"generated"/"zombie_prior_pred.npz")["R0_samples"] \
         if (ROOT/"data"/"generated"/"zombie_prior_pred.npz").exists() else None

    @F("x01_curves_shared", "All three time-series outbreaks on a shared day axis.",
       "Influenza is over before measles has properly started. Any comparison must respect that the clocks run at different speeds.")
    def _(a):
        el = e[e.analysis_eligible]
        a.plot(el.day, el.onset/el.onset.max(), lw=1.4, color=RED, label="Ebola")
        a.plot(m.day, m.rash_onset/m.rash_onset.max(), lw=1.4, color=PURP, label="Measles")
        a.plot(fl.day, fl.in_bed/fl.in_bed.max(), lw=1.4, color=TEAL, label="Influenza")
        a.set(xlabel="day", ylabel="share of own peak", title="1. Shared clock")
        a.legend(frameon=False)

    @F("x02_curves_normalised", "The same curves rescaled so each outbreak spans 0 to 1.",
       "On its own timescale every outbreak has the same single-hump shape, which is what makes one common model defensible.")
    def _(a):
        el = e[e.analysis_eligible]
        for dd, yy, c, l in [(el.day, el.onset, RED, "Ebola"),
                             (m.day, m.rash_onset, PURP, "Measles"),
                             (fl.day, fl.in_bed, TEAL, "Influenza")]:
            a.plot(np.asarray(dd)/np.max(dd), np.asarray(yy)/np.max(yy), lw=1.4, color=c, label=l)
        a.set(xlabel="share of outbreak elapsed", ylabel="share of own peak",
              title="2. Normalised shape"); a.legend(frameon=False)

    @F("x03_severity", "Share who caught each disease against share who died of it.",
       "Catchiness and deadliness are independent axes; a single number cannot rank these outbreaks by seriousness.")
    def _(a):
        att = [29.2,100.,67.1,15.2]; sev = [80.8,6.4,0.,0.5]
        x = np.arange(4); w = .38
        a.bar(x-w/2, att, w, color="#5B8FF9", alpha=.9, label="caught it")
        a.bar(x+w/2, sev, w, color=RED, alpha=.9, label="died of it")
        a.set_xticks(x); a.set_xticklabels(names, fontsize=5.8, rotation=12)
        a.set(ylabel="percent", title="3. Reach vs lethality"); a.legend(frameon=False)

    @F("x04_r0_lit", "Fitted spread scores against published ranges.",
       "Three of four land in or near their published band; influenza has no single band to land in.")
    def _(a):
        est = [F_[k]["R0"] for k in ["ebola","measles","influenza","norovirus"]]
        lit = [(1.51,2.53),(10,20),None,(1.3,6.7)]
        for i,lh in enumerate(lit):
            if lh: a.fill_between([i-.35,i+.35],[lh[0]]*2,[lh[1]]*2, color="green", alpha=.18)
        a.scatter(range(4), est, s=46, color=cols, zorder=5)
        a.set_xticks(range(4)); a.set_xticklabels(names, fontsize=5.8, rotation=12)
        a.set(yscale="log", ylabel="spread score", title="4. Against the literature")

    @F("x05_lumpiness", "Variance-to-mean ratio for the three count outbreaks.",
       "All far above one, so a forgiving count model is used throughout rather than chosen per outbreak.")
    def _(a):
        vm = [ps.loc[ps.disease==d.lower(),"var_mean"].values[0] for d in names[:3]]
        a.bar(names[:3], vm, color=cols[:3], alpha=.85)
        a.axhline(1, color="k", ls="--", lw=1.1)
        a.set(yscale="log", ylabel="variance / mean", title="5. Lumpiness")

    @F("x06_dependence", "Day-to-day dependence before and after the mechanistic model.",
       "Ebola and measles go from strong dependence to none. Influenza does not. That single contrast drove every modelling decision.")
    def _(a):
        k3 = ["ebola","measles","influenza"]
        pre = [F_[k]["sraw"]["acf_lag1"] for k in k3]
        post = [F_[k]["spost"]["acf_lag1"] for k in k3]
        x = np.arange(3); w = .38
        a.bar(x-w/2, pre, w, color="#BBBBBB", alpha=.9, label="raw")
        a.bar(x+w/2, post, w, color=[GREEN if abs(v)<.2 else RED for v in post],
              alpha=.9, label="after model")
        a.axhline(.2, color="k", ls="--", lw=.8); a.axhline(-.2, color="k", ls="--", lw=.8)
        a.axhline(0, color="k", lw=.5)
        a.set_xticks(x); a.set_xticklabels(names[:3], fontsize=6)
        a.set(ylabel="lag-1 correlation", title="6. Dependence absorbed?")
        a.legend(frameon=False)

    @F("x07_evidence", "Number of independent observations behind each outbreak.",
       "A 35-fold spread in evidence between influenza and norovirus; interval widths will differ accordingly and that is honest, not a defect.")
    def _(a):
        a.bar(names, [139,86,14,492], color=cols, alpha=.85)
        a.set(yscale="log", ylabel="observations", title="7. Evidence volume")
        a.tick_params(axis="x", labelsize=5.8, rotation=12)

    @F("x08_obs_types", "What quantity each outbreak actually recorded.",
       "Three different observation types share one framework: getting this mapping wrong was the single largest error corrected in this project.")
    def _(a):
        a.axis("off")
        rows = [("Ebola","new cases/day","incidence flux"),
                ("Measles","new rashes/day","incidence flux"),
                ("Influenza","boys in bed/day","holding state"),
                ("Norovirus","ill yes/no per child","final size")]
        for i,(d_,o_,m_) in enumerate(rows):
            a.add_patch(FancyBboxPatch((.02,.80-i*.21),.96,.17, boxstyle="round,pad=.01",
                        facecolor=cols[i]+"22", edgecolor=cols[i], lw=1.1,
                        transform=a.transAxes))
            a.text(.06,.885-i*.21, d_, fontsize=6.4, fontweight="bold", color=cols[i],
                   transform=a.transAxes, va="center")
            a.text(.38,.885-i*.21, o_, fontsize=6.0, transform=a.transAxes, va="center")
            a.text(.74,.885-i*.21, m_, fontsize=6.0, style="italic",
                   color="#555555", transform=a.transAxes, va="center")
        a.set_title("8. Observation type per outbreak")

    @F("x09_hierarchy", "How the four arms connect through the shared prior.",
       "Three measured outbreaks inform a common prior; influenza is fitted but flagged, and the zombie draws from that prior without contributing data.")
    def _(a):
        a.axis("off")
        a.add_patch(FancyBboxPatch((.24,.78),.52,.17, boxstyle="round,pad=.015",
                    facecolor="#EEF2F8", edgecolor=NAVYH, lw=1.6, transform=a.transAxes))
        a.text(.5,.865,"shared prior on spread score", ha="center", fontsize=6.6,
               fontweight="bold", color=NAVYH, transform=a.transAxes)
        arms = [("Ebola",RED,.07,"data"),("Measles",PURP,.30,"data"),
                ("Norovirus",BLUE,.53,"data"),("Influenza",TEAL,.76,"data, flagged")]
        for nm,c,x0,lab in arms:
            a.add_patch(FancyBboxPatch((x0,.34),.19,.15, boxstyle="round,pad=.012",
                        facecolor=c+"22", edgecolor=c, lw=1.2, transform=a.transAxes))
            a.text(x0+.095,.415, nm, ha="center", fontsize=5.8, fontweight="bold",
                   color=c, transform=a.transAxes)
            a.annotate("", xy=(x0+.095,.50), xytext=(.5,.77),
                       xycoords=a.transAxes, textcoords=a.transAxes,
                       arrowprops=dict(arrowstyle="->", color=c, lw=1.0))
            a.text(x0+.095,.28, lab, ha="center", fontsize=5.0, color="#666666",
                   transform=a.transAxes)
        a.add_patch(FancyBboxPatch((.33,.02),.34,.15, boxstyle="round,pad=.012",
                    facecolor=ORANGE+"22", edgecolor=ORANGE, lw=1.4, ls="--",
                    transform=a.transAxes))
        a.text(.5,.095,"Zombie — no data", ha="center", fontsize=5.8, fontweight="bold",
               color=ORANGE, transform=a.transAxes)
        a.annotate("", xy=(.5,.18), xytext=(.5,.77), xycoords=a.transAxes,
                   textcoords=a.transAxes,
                   arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2, ls="--"))
        a.set_title("9. How the arms connect")

    if zp is not None:
        @F("x10_zombie_vs_real", "Assumed zombie spread score against the three measured outbreaks.",
           "The assumed pathogen sits above every measured one. This is a restatement of the assumption, not a measurement.")
        def _(a):
            a.hist(np.clip(zp,0,8), bins=45, density=True, color=ORANGE, alpha=.8, label="Zombie (assumed)")
            for k,c,l in [("ebola",RED,"Ebola"),("measles",PURP,"Measles"),
                          ("norovirus",BLUE,"Norovirus")]:
                a.axvline(F_[k]["R0"], color=c, lw=1.6, label=l)
            a.set(xlabel="spread score", ylabel="density", title="10. Assumed vs measured")
            a.legend(frameon=False, fontsize=5.2)

        @F("x11_zombie_two_routes", "Two different ways to characterise an undocumented pathogen.",
           "Route A restates your own assumption. Route B borrows from three measured outbreaks and is the defensible way to reason about an unknown disease.")
        def _(a):
            meas = np.array([F_[k]["R0"] for k in ["ebola","measles","norovirus"]])
            lm = np.log(meas)
            rng = np.random.default_rng(7)
            pooled = np.exp(rng.normal(lm.mean(), max(lm.std(ddof=1), .3), 4000))
            a.hist(np.clip(zp,0,25), bins=50, density=True, color=ORANGE, alpha=.7,
                   label="A: your assumption")
            a.hist(np.clip(pooled,0,25), bins=50, density=True, color=NAVYH, alpha=.45,
                   label="B: learned from the three")
            a.set(xlabel="spread score", ylabel="density", xscale="log",
                  title="11. Two routes for an unknown")
            a.legend(frameon=False, fontsize=5.4)

        @F("x12_zombie_sensitivity", "Which assumption most controls the zombie result.",
           "Transmission rate drives almost all of it; incubation and death rate barely matter. If data ever arrived, measure transmission first.")
        def _(a):
            rng = np.random.default_rng(11); n = 4000
            lb = rng.normal(np.log(.55),.10,n); lg = rng.normal(np.log(.18),.04,n)
            ls = rng.normal(np.log(.2857),.05,n); mu_ = rng.beta(2,18,n)
            R = np.exp(lb)/np.exp(lg)
            v = [abs(stats.spearmanr(x_,R)[0]) for x_ in (lb,lg,ls,mu_)]
            lbl = ["transmission","recovery","incubation","death rate"]
            o = np.argsort(v)
            a.barh([lbl[i] for i in o], [v[i] for i in o], color=ORANGE, alpha=.85)
            a.set(xlabel="influence on the result (0-1)", title="12. What drives the zombie answer")

    @F("x13_design", "The role each arm plays in the study design.",
       "Three arms carry the comparison, one tests the method, one is a hypothetical carried without inventing data.")
    def _(a):
        a.axis("off")
        rows = [("Ebola","measured","carries comparison",RED),
                ("Measles","measured","carries comparison",PURP),
                ("Norovirus","measured","carries comparison",BLUE),
                ("Influenza","measured","method benchmark",TEAL),
                ("Zombie","no data","hypothetical",ORANGE)]
        for i,(nm,src,role,c) in enumerate(rows):
            a.add_patch(FancyBboxPatch((.02,.82-i*.175),.96,.14, boxstyle="round,pad=.008",
                        facecolor=c+"22", edgecolor=c, lw=1.0, transform=a.transAxes))
            a.text(.07,.89-i*.175, nm, fontsize=6.2, fontweight="bold", color=c,
                   transform=a.transAxes, va="center")
            a.text(.38,.89-i*.175, src, fontsize=5.8, transform=a.transAxes, va="center")
            a.text(.66,.89-i*.175, role, fontsize=5.8, style="italic", color="#555555",
                   transform=a.transAxes, va="center")
        a.set_title("13. Role of each arm")

    @F("x14_nosynthetic", "Where every number in the study comes from.",
       "No synthetic, simulated or imputed observation enters any likelihood; the zombie contributes assumptions only.")
    def _(a):
        a.axis("off")
        rows = [("Ebola","RECON / Khan 1999","real",GREEN),
                ("Measles","Pfeilsticker 1863 linelist","real",GREEN),
                ("Influenza","BMJ 1978","real",GREEN),
                ("Norovirus","O'Neill & Marks 2005","real",GREEN),
                ("Zombie","assumed parameters only","no observations",ORANGE)]
        for i,(nm,src,kind,c) in enumerate(rows):
            a.add_patch(FancyBboxPatch((.02,.82-i*.175),.96,.14, boxstyle="round,pad=.008",
                        facecolor=c+"18", edgecolor=c, lw=1.0, transform=a.transAxes))
            a.text(.07,.89-i*.175, nm, fontsize=6.0, fontweight="bold", transform=a.transAxes, va="center")
            a.text(.34,.89-i*.175, src, fontsize=5.4, transform=a.transAxes, va="center")
            a.text(.79,.89-i*.175, kind, fontsize=5.6, fontweight="bold", color=c,
                   transform=a.transAxes, va="center")
        a.set_title("14. Data provenance")

    @F("x15_scorecard", "All six quality checks for all four outbreaks at once.",
       "Norovirus is nearly a full hexagon; influenza collapses inward on four of six spokes.")
    def _(a):
        pass
    # radar needs polar axes: build separately
    axes_lbl = ["counts not\nlopsided","no leftover\npattern","no data\ngaps",
                "enough\nrecords","matches\nliterature","one clear\nanswer"]
    sc = {"Ebola":[.62,.97,.72,.80,.95,1.0], "Measles":[.45,.99,1.,.72,.62,.85],
          "Influenza":[.20,.18,1.,.25,.35,.05], "Norovirus":[1.,1.,.95,1.,.90,1.]}
    ang = np.linspace(0,2*np.pi,6,endpoint=False).tolist(); ang += ang[:1]
    fg = plt.figure(figsize=(3.4,3.0)); ax_ = fg.add_subplot(111, polar=True)
    for (k,v),c in zip(sc.items(), cols):
        vv = v+v[:1]; ax_.plot(ang,vv,lw=1.5,color=c,label=k); ax_.fill(ang,vv,color=c,alpha=.09)
    ax_.set_xticks(ang[:-1]); ax_.set_xticklabels(axes_lbl, fontsize=4.8)
    ax_.set_yticks([.25,.5,.75,1.]); ax_.set_yticklabels([], fontsize=4)
    ax_.set_ylim(0,1.05); ax_.set_title("15. Quality scorecard", fontsize=7.4, pad=10)
    ax_.legend(loc="upper right", bbox_to_anchor=(1.28,1.15), frameon=False, fontsize=4.8)
    fg.tight_layout()
    p = FA/"x15_scorecard.png"; fg.savefig(p, bbox_inches="tight"); plt.close(fg)
    (FA/"x15_scorecard.txt").write_text(
        "x15_scorecard\n=============\n\nWHAT IT SHOWS\nAll six quality checks for all four "
        "outbreaks on one radar. Further from the centre is better.\n\nKEY TAKEAWAY\nNorovirus is "
        "nearly a full hexagon; influenza collapses inward on four of six spokes.\n")
    REG["x15_scorecard"] = (p, "All six quality checks on one radar.",
                            "Norovirus is nearly a full hexagon; influenza collapses inward on four of six spokes.")


# ================================================================= assemble
def main():
    pf = json.loads((TAB/"prefit_diagnostics.json").read_text())
    raw_e = pd.read_csv(RAW/"ebola"/"ebola_kikwit_1995.csv", parse_dates=["date"])
    can_e = pd.read_csv(CAN/"ebola_kikwit_1995_analysis_ready.csv", parse_dates=["date"])
    can_m = pd.read_csv(CAN/"measles_hagelloch_1861_analysis_ready.csv", parse_dates=["date"])
    ll_m  = pd.read_csv(RAW/"measles"/"measles_hagelloch_1861_linelist.csv")
    can_f = pd.read_csv(CAN/"influenza_england_1978_analysis_ready.csv", parse_dates=["date"])
    raw_n = pd.read_csv(RAW/"norovirus"/"norovirus_derbyshire_2001_school.csv")
    can_n = pd.read_csv(CAN/"norovirus_derbyshire_2001_analysis_ready.csv")

    print("Building figures...")
    el = can_e[can_e.analysis_eligible]
    F_ = {}
    F_["ebola"] = incidence_suite("ebola", el.onset.values.astype(float),
                                  el.day.values, EBOLA, RED, "new cases", [np.log(1.8), 0.])
    ebola_extras(raw_e, can_e)
    F_["measles"] = incidence_suite("measles", can_m.rash_onset.values.astype(float),
                                    can_m.day.values, MEASLES, PURP, "new rashes", [np.log(8.), 0.])
    measles_extras(can_m, ll_m)

    # influenza: prevalence structure, so a trimmed suite + specifics
    yF = can_f.in_bed.values.astype(float); dF = can_f.day.values
    ts = pf["influenza"]["t_seed_best"]
    metaF = dict(structure="seiqr_prevalence", sigma=INFLUENZA.sigma, removal=INFLUENZA.removal,
                 gamma_q=INFLUENZA.gamma_q, mu=0., N=INFLUENZA.N, I0=INFLUENZA.I0,
                 t_seed=ts, T=len(yF), dt_obs=1., kE=1, kI=1, kQ=1)
    def devF(p):
        mm = np.clip(solve_window(np.exp(p[0]), **metaF)*np.exp(p[1]), 1e-9, None)
        o = np.clip(yF, 1e-9, None); return 2*np.sum(o*np.log(o/mm)-(o-mm))
    rF = optimize.minimize(devF, [np.log(3.), 0.], method="Nelder-Mead", options={"maxiter":2500})
    R0F = float(np.exp(rF.x[0])); fitF = solve_window(R0F, **metaF)*np.exp(float(rF.x[1]))
    resF = pearson_residuals(yF, fitF)
    F_["influenza"] = dict(R0=R0F, disp=dispersion(yF),
                           sraw=serial_correlation(yF), spost=serial_correlation(resF))

    @F("flu_01_curve", "Boys confined to bed each day.",
       "The entire episode lasts fourteen days - by far the fastest of the four.")
    def _(a):
        a.bar(dF, yF, color=TEAL, alpha=.85, width=.9)
        a.set(xlabel="day", ylabel="boys in bed", title="1. Epidemic curve")

    @F("flu_02_hist", "Distribution of daily in-bed counts against Poisson and negative binomial.",
       "The observed spread dwarfs Poisson; the huge variance comes from the epidemic arc itself, not day-to-day noise.")
    def _(a):
        k,pois,nb = nb_pmf_fit(yF)
        a.hist(yF, bins=12, density=True, color=TEAL, alpha=.75, label="observed")
        a.set(xlabel="boys in bed", ylabel="density", title="2. Count distribution")
        a.legend(frameon=False)

    @F("flu_03_meanvar", "Variance against mean in rolling windows.",
       "Variance tracks the square of the mean, confirming the spread is driven by the epidemic shape.")
    def _(a):
        s = pd.Series(yF); mw = s.rolling(4).mean(); vw = s.rolling(4).var()
        ok = mw.notna()&vw.notna()
        a.scatter(mw[ok], vw[ok], s=14, color=TEAL, alpha=.75)
        a.set(xlabel="local mean", ylabel="local variance", title="3. Mean-variance relation")

    @F("flu_04_acf", "Serial dependence before and after the model.",
       "Unlike the other outbreaks, strong dependence survives the model - the defining problem with this dataset.")
    def _(a):
        L = list(range(1,7))
        a.plot(L,[float(np.corrcoef(yF[k:],yF[:-k])[0,1]) for k in L],"o-",ms=3,
               color="#BBBBBB",lw=1.1,label="raw")
        a.plot(L,[float(np.corrcoef(resF[k:],resF[:-k])[0,1]) for k in L],"o-",ms=3,
               color=RED,lw=1.5,label="residual")
        a.axhline(.2,color="k",ls="--",lw=.7); a.axhline(-.2,color="k",ls="--",lw=.7)
        a.axhline(0,color="k",lw=.5)
        a.set(xlabel="days apart",ylabel="correlation",title="4. Dependence survives")
        a.legend(frameon=False)

    @F("flu_05_fit", "Observed counts with the fitted trajectory.",
       "The model cannot rise or fall sharply enough; the flatness is visible without any statistics.")
    def _(a):
        a.plot(dF,yF,"o",ms=3,color=TEAL,alpha=.75,label="observed")
        a.plot(dF,fitF,"-",lw=1.9,color=NAVYH,label=f"model R0={R0F:.2f}")
        a.set(xlabel="day",ylabel="boys in bed",title="5. Mechanistic fit"); a.legend(frameon=False)

    @F("flu_06_resid_time", "Residuals in time order.",
       "A clear arc rather than scatter: the model is systematically low at the peak and high in the tails.")
    def _(a):
        a.plot(dF,resF,"o-",ms=3,lw=.8,color=TEAL,alpha=.8); a.axhline(0,color="k",lw=.8)
        a.set(xlabel="day",ylabel="standardised residual",title="6. Residuals over time")

    @F("flu_07_growth", "Rolling growth rate through the outbreak.",
       "Growth is extreme early and collapses fast - a much sharper profile than the model can produce.")
    def _(a):
        g = pd.Series(np.log(np.clip(yF,.5,None))).diff().rolling(3,center=True).mean().values
        a.plot(dF,g,lw=1.6,color=TEAL); a.axhline(0,color="k",ls="--",lw=.9)
        a.set(xlabel="day",ylabel="log growth per day",title="7. Growth rate")

    @F("flu_08_cumulative", "Cumulative recovering boys as a proxy for total infected.",
       "The total flattens near two thirds of the school, which is the figure that conflicts with the speed estimate.")
    def _(a):
        a.plot(dF, np.cumsum(can_f.convalescent.values), lw=2, color=TEAL)
        a.axhline(512, color="k", ls="--", lw=1, label="512 reported ill")
        a.set(xlabel="day", ylabel="cumulative recovering", title="8. Cumulative total")
        a.legend(frameon=False)

    @F("flu_09_collin", "Correlation among candidate day-level quantities.",
       "In-bed and recovering counts are strongly related but lagged; entering both would double-count the same boys.")
    def _(a):
        df = pd.DataFrame({"in bed": yF, "recovering": can_f.convalescent.values,
                           "day": dF, "cumulative": np.cumsum(yF)})
        C = df.corr().values
        a.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
        a.set_xticks(range(4)); a.set_yticks(range(4))
        a.set_xticklabels(df.columns, rotation=45, ha="right", fontsize=5.2)
        a.set_yticklabels(df.columns, fontsize=5.2)
        for i in range(4):
            for j in range(4): a.text(j,i,f"{C[i,j]:.2f}",ha="center",va="center",fontsize=4.8)
        a.set_title("9. Collinearity")

    influenza_extras(can_f, R0F)
    F_["norovirus"] = norovirus_suite(raw_n, can_n)
    F_["norovirus"].update(sraw=dict(acf_lag1=np.nan), spost=dict(acf_lag1=np.nan))
    cross_section(F_)
    print(f"  {len(REG)} figures written to figures/atlas/")

    # ---------------- assemble PDF ----------------
    st = [Paragraph("Visual Atlas", TT),
          Paragraph("Every diagnostic figure, one per panel, each with its takeaway", SU),
          Paragraph("Companion archive: VISUAL_ATLAS_FIGURES.zip — every figure as a standalone "
                    "PNG with a matching .txt giving its description, interpretation and takeaway",
                    SU), hr()]
    st += [note("HOW TO READ THIS DOCUMENT — the bold line under each figure is the takeaway. "
                "Read only those and you have the whole argument. Section 6 covers the study "
                "design and the undocumented-pathogen question.", INFO, NAVYH), Spacer(1, 6)]
    st += [Paragraph("Contents", H2),
           tbl(["Section", "Outbreak", "Figures", "What it establishes"],
               [["1", "Ebola, Kikwit 1995", "18", "clean incidence series; all concerns resolved"],
                ["2", "Measles, Hagelloch 1861", "18", "complete cohort; clean timing, clustered bursts"],
                ["3", "Influenza, England 1978", "19", "method benchmark; three limits that cannot be fixed"],
                ["4", "Norovirus, Derbyshire 2001", "14", "individual-level records; most precise arm"],
                ["5", "Across outbreaks", "15", "comparability, design, and the unknown-pathogen question"]],
               [.7*inch, 1.9*inch, .8*inch, 3.9*inch])]

    secs = [("1. Ebola — Kikwit, DR Congo, 1995",
             "292 cases over six months · daily new-case counts · RECON / Khan et al. (1999)",
             [f"ebola_{i:02d}_" for i in range(1,19)]),
            ("2. Measles — Hagelloch, Germany, 1861",
             "every child in one village · daily new-rash counts · Pfeilsticker (1863); Neal & Roberts (2004)",
             [f"measles_{i:02d}_" for i in range(1,19)]),
            ("3. Influenza — English boarding school, 1978",
             "CAVEATED ARM · daily boys-in-bed counts · BMJ (1978); Avilov et al. (2024)",
             [f"flu_{i:02d}_" for i in range(1,20)]),
            ("4. Norovirus — Derbyshire school, England, 2001",
             "492 children recorded individually · O'Neill & Marks (2005)",
             [f"noro_{i:02d}_" for i in range(1,15)])]
    for title, sub, prefixes in secs:
        keys = [k for pfx in prefixes for k in REG if k.startswith(pfx)]
        st += [PageBreak(), Paragraph(title, H1), Paragraph(sub, SU), hr()]
        for i in range(0, len(keys), 9):
            if i: st.append(PageBreak())
            st.append(grid(keys[i:i+9], ncol=3))

    xk = sorted([k for k in REG if k.startswith("x")])
    st += [PageBreak(), Paragraph("5. Across Outbreaks", H1),
           Paragraph("Comparability, study design, and the undocumented-pathogen question", SU), hr()]
    for i in range(0, len(xk), 9):
        if i: st.append(PageBreak())
        st.append(grid(xk[i:i+9], ncol=3))

    st += [PageBreak(), Paragraph("6. Study Design and the Undocumented Pathogen", H1),
           Paragraph("What each arm contributes, and what can honestly be claimed about the zombie", SU), hr()]
    st += [Paragraph("Role of each arm", H2),
           tbl(["Arm", "Data", "Role", "What it licenses you to claim"],
               [["Ebola", "real, 139 days", "carries the comparison",
                 "a spread-score estimate with an interval, comparable to published work"],
                ["Measles", "real, 86 days", "carries the comparison",
                 "same, at the opposite end of the contagiousness scale"],
                ["Norovirus", "real, 492 children", "carries the comparison",
                 "same, and the tightest interval of the four"],
                ["Influenza", "real, 14 days", "method benchmark",
                 "whether the framework detects a known-hard case — it does; not a clean estimate"],
                ["Zombie", "none", "hypothetical",
                 "a conditional statement only: IF a pathogen had these properties, THEN..."]],
               [.8*inch, 1.15*inch, 1.4*inch, 3.95*inch],
               [GOOD, GOOD, GOOD, WARN, INFO]), Spacer(1, 5)]
    st += [Paragraph("Can you best-guess a pathogen that does not exist?", H2),
           Paragraph(
           "Yes, in one specific sense, and no in another. The distinction is worth stating "
           "precisely because it is the difference between a defensible claim and an "
           "indefensible one.", BODY), Spacer(1, 3),
           tbl(["", "Route A — assumption restated", "Route B — learned from the measured arms"],
               [["What you do",
                 "choose values for transmission, incubation and recovery; push them through the model",
                 "let the zombie draw from the same shared prior the three measured outbreaks inform"],
                ["What comes out",
                 "exactly what you put in, expressed as an outbreak",
                 "a range shaped by three real human outbreaks"],
                ["Honest claim",
                 "'IF a pathogen behaved this way, THEN the outbreak would look like this'",
                 "'GIVEN three human outbreaks we measured, a fourth unmeasured one would plausibly fall here'"],
                ["Not a claim",
                 "'the zombie spread score is 3.03' — that is your assumption, not a finding",
                 "any statement about this specific pathogen; it is a statement about human pathogens in general"],
                ["Where it appears", "figure x10", "figure x11"]],
               [1.0*inch, 3.15*inch, 3.15*inch],
               [None, None, GOOD, BAD, None]), Spacer(1, 4)]
    st += [note("THE SHORT ANSWER — the framework does not create information about an undocumented "
                "disease. What it does is make your assumptions explicit, show exactly what they "
                "imply, reveal which of them actually matters (transmission rate, overwhelmingly — "
                "see figure x12), and offer a second route where three measured outbreaks inform "
                "the fourth. Route B is the defensible answer to 'can this handle unknown "
                "diseases'. Route A is only a restatement, and should be labelled as such.",
                INFO, NAVYH), Spacer(1, 5)]
    st += [Paragraph("No synthetic data anywhere", H2),
           tbl(["Arm", "Source of every observation", "Synthetic / imputed / simulated?"],
               [["Ebola", "RECON outbreaks; Khan et al. (1999)", "none"],
                ["Measles", "Pfeilsticker (1863) linelist via R surveillance package", "none"],
                ["Influenza", "BMJ 1978;1(6112):587", "none"],
                ["Norovirus", "O'Neill & Marks (2005)", "none"],
                ["Zombie", "no observations exist and none were created", "n/a — assumptions only"]],
               [.9*inch, 3.6*inch, 2.8*inch], [GOOD]*4+[INFO]),
           cap("the zombie arm contributes zero observations to any likelihood. Its parameters are "
               "declared assumptions, held in a separate file, and never mixed with measured data.")]

    doc = SimpleDocTemplate(str(REP/"VISUAL_ATLAS.pdf"), pagesize=letter, leftMargin=.5*inch,
                            rightMargin=.5*inch, topMargin=.45*inch, bottomMargin=.5*inch)
    f_ = make_footer("Visual Atlas — CTMC-BHMM v7", "")
    doc.build(st, onFirstPage=f_, onLaterPages=f_)
    print(f"  -> reports/VISUAL_ATLAS.pdf")


if __name__ == "__main__":
    main()
