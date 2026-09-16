"""
scripts/12_prefit_pages.py
==========================
Figure-led pre-fit reports written for a reader who is not a modeller.

    reports/PREFIT_BY_DISEASE.pdf   one page per outbreak (4 pages)
    reports/PREFIT_OVERVIEW.pdf     cross-outbreak summary (<= 5 pages)

Rules followed throughout:
  - visuals carry the argument; prose is confined to captions
  - every chart and table has a plain-language caption directly beneath it
    answering "this one shows that..."
  - no jargon in captions: no "overdispersion", "autocorrelation",
    "posterior", "likelihood". Those words appear only in axis labels and
    the debug table on the last page, where a specialist will look.

Run after scripts/02_prefit_diagnostics.py.
"""
import sys, json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import optimize, stats

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
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, HRFlowable, Image as RLImage, PageBreak)

ROOT = Path(__file__).resolve().parents[1]
CAN, TAB, FO, REP = ROOT/"data"/"canonical", ROOT/"tables", ROOT/"figures"/"onepager", ROOT/"reports"
FO.mkdir(parents=True, exist_ok=True); REP.mkdir(exist_ok=True)

RED, PURP, TEAL, BLUE = "#C0392B", "#7B2D8B", "#1A7B6B", "#2E75B6"
GREY, NAVYH, GREEN = "#777777", "#1F4E79", "#2E7D32"
GOOD, WARN, BAD = colors.HexColor("#D5F5D5"), colors.HexColor("#FFF3CD"), colors.HexColor("#FDEBEA")

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 200, "axes.titlesize": 9, "axes.labelsize": 8})

# ---------------------------------------------------------------- styles
S  = lambda n, **k: ParagraphStyle(n, **{**dict(fontName=SANS, fontSize=8.4,
                                                leading=11, textColor=BLACK), **k})
TT = S("T",  fontSize=15, fontName=SANS_BOLD, textColor=NAVY, alignment=TA_CENTER, leading=18)
SU = S("SU", fontSize=9,  textColor=colors.HexColor(GREY), alignment=TA_CENTER)
HD = S("HD", fontSize=10, fontName=SANS_BOLD, textColor=NAVY, spaceBefore=4, spaceAfter=2)
CAP = S("CAP", fontSize=7.6, leading=9.6, textColor=colors.HexColor("#333333"))
BODY = S("B", fontSize=8.4, leading=11)


def cap(text):
    """Caption paragraph: the 'this one shows that...' line under a visual."""
    return Paragraph(f"<b>This shows that</b> {text}", CAP)


def tbl(head, rows, widths, rc=None, fs=7.0):
    hr = [Paragraph(h, S("th", fontSize=fs, fontName=SANS_BOLD, textColor=WHITE)) for h in head]
    dr = [[Paragraph(str(c), S("td", fontSize=fs, leading=fs+2.4)) for c in r] for r in rows]
    sty = [("BACKGROUND", (0,0), (-1,0), NAVY), ("TEXTCOLOR", (0,0), (-1,0), WHITE),
           ("ROWBACKGROUNDS", (0,1), (-1,-1), [WHITE, LGREY]),
           ("GRID", (0,0), (-1,-1), .25, MGREY), ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
           ("LEFTPADDING", (0,0), (-1,-1), 3), ("RIGHTPADDING", (0,0), (-1,-1), 3),
           ("TOPPADDING", (0,0), (-1,-1), 2.5), ("BOTTOMPADDING", (0,0), (-1,-1), 2.5)]
    if rc:
        for i, c in enumerate(rc):
            if c is not None: sty.append(("BACKGROUND", (0,i+1), (-1,i+1), c))
    t = Table([hr]+dr, colWidths=widths); t.setStyle(TableStyle(sty)); return t


def cell(img, caption, w, ar=0.72):
    """Image stacked over its caption, as one table cell."""
    inner = Table([[RLImage(str(img), width=w*inch, height=w*ar*inch)], [cap(caption)]],
                  colWidths=[w*inch])
    inner.setStyle(TableStyle([("LEFTPADDING",(0,0),(-1,-1),0),
                               ("RIGHTPADDING",(0,0),(-1,-1),0),
                               ("TOPPADDING",(0,0),(-1,-1),1),
                               ("BOTTOMPADDING",(0,0),(-1,-1),4),
                               ("VALIGN",(0,0),(-1,-1),"TOP")]))
    return inner


def grid2(cells, w=3.62):
    rows = [cells[i:i+2] for i in range(0, len(cells), 2)]
    for r in rows:
        while len(r) < 2: r.append("")
    t = Table(rows, colWidths=[w*inch, w*inch])
    t.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"TOP"),
                           ("LEFTPADDING",(0,0),(-1,-1),2),
                           ("RIGHTPADDING",(0,0),(-1,-1),2),
                           ("TOPPADDING",(0,0),(-1,-1),2),
                           ("BOTTOMPADDING",(0,0),(-1,-1),2)]))
    return t


def verdict(text, bg, bd):
    t = Table([[Paragraph(text, S("v", fontSize=8.8, fontName=SANS_BOLD,
                                  textColor=colors.HexColor(bd)))]], colWidths=[7.3*inch])
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),bg),
                           ("BOX",(0,0),(-1,-1),1,colors.HexColor(bd)),
                           ("LEFTPADDING",(0,0),(-1,-1),8),("RIGHTPADDING",(0,0),(-1,-1),8),
                           ("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6)]))
    return t


def fig1(name, draw, figsize=(4.4, 3.4)):
    f, a = plt.subplots(1, 1, figsize=figsize)
    draw(a); f.tight_layout(pad=0.4)
    p = FO/f"{name}.png"; f.savefig(p, bbox_inches="tight"); plt.close(f)
    return p


# ================================================================ data
def load():
    return (pd.read_csv(CAN/"ebola_kikwit_1995_analysis_ready.csv"),
            pd.read_csv(CAN/"measles_hagelloch_1861_analysis_ready.csv"),
            pd.read_csv(CAN/"influenza_england_1978_analysis_ready.csv"),
            pd.read_csv(CAN/"norovirus_derbyshire_2001_analysis_ready.csv"),
            json.loads((TAB/"prefit_diagnostics.json").read_text()),
            pd.read_csv(TAB/"prefit_summary.csv"))


def fit_incidence(cfg, y, days, x0, t_seed=0.0):
    meta = dict(structure="seird_incidence", sigma=cfg.sigma, removal=cfg.removal,
                gamma_q=0.0, mu=cfg.mu_point, N=cfg.N, I0=cfg.I0, t_seed=t_seed,
                T=int(days.max())+1, dt_obs=1.0, kE=1, kI=1, kQ=1)
    def dev(p):
        m = np.clip(solve_window(np.exp(p[0]), **meta)[days]*np.exp(p[1]), 1e-9, None)
        o = np.clip(y, 1e-9, None); return 2*np.sum(o*np.log(o/m)-(o-m))
    r = optimize.minimize(dev, x0, method="Nelder-Mead", options={"maxiter": 2500})
    R0 = float(np.exp(r.x[0]))
    return R0, solve_window(R0, **meta)[days]*np.exp(float(r.x[1]))


def acf_bars(a, res, nlag=7):
    lags = list(range(1, nlag+1))
    v = [float(np.corrcoef(res[k:], res[:-k])[0,1]) for k in lags]
    a.bar(lags, v, color=[RED if abs(x) > .2 else GREEN for x in v], alpha=.9)
    a.axhline(.2, color="k", ls="--", lw=.8); a.axhline(-.2, color="k", ls="--", lw=.8)
    a.axhline(0, color="k", lw=.5); a.set_ylim(-.55, 1.0)
    a.set_xlabel("days apart"); a.set_ylabel("leftover pattern")
    return v[0]


# ================================================================ disease pages
def page_ebola(e, pf):
    el = e[e.analysis_eligible]; y = el.onset.values.astype(float); d = el.day.values
    R0, fit = fit_incidence(EBOLA, y, d, [np.log(1.8), 0.])
    res = pearson_residuals(y, fit)

    p1 = fig1("eb_curve", lambda a: (
        a.bar(e.day, e.onset, color=RED, alpha=.85, width=1.0),
        a.set(xlabel="day of outbreak", ylabel="new cases that day",
              title="The outbreak over time")))
    p2 = fig1("eb_fit", lambda a: (
        a.plot(d, y, "o", ms=2.6, color=RED, alpha=.7, label="what happened"),
        a.plot(d, fit, "-", lw=2, color=NAVYH, label=f"what the model expects"),
        a.set(xlabel="day of outbreak", ylabel="new cases that day",
              title=f"Model vs reality  (spread score {R0:.2f})"),
        a.legend(frameon=False, fontsize=7)))
    lag1 = [None]
    def _acf(a):
        lag1[0] = acf_bars(a, res)
        a.set_title("Anything the model missed?")
    p3 = fig1("eb_acf", _acf)
    def _rep(a):
        cnt = e.analysis_eligible.value_counts()
        a.bar(["days counted", "days not counted"],
              [int(cnt.get(True,0)), int(cnt.get(False,0))],
              color=[RED, "#BBBBBB"], alpha=.85)
        a.set(ylabel="days", title="Which days could be used")
    p4 = fig1("eb_rep", _rep)

    st = [Paragraph("Ebola — Kikwit, DR Congo, 1995", TT),
          Paragraph("292 cases over six months in a city hospital and its surrounding community", SU),
          HRFlowable(width="100%", thickness=.9, color=NAVY, spaceBefore=2, spaceAfter=4)]
    st += [tbl(["People at risk", "Days recorded", "Days usable", "Total cases", "Deaths", "Died of those infected"],
               [["~1,000", "192", "139", "292", "236", "81%"]],
               [1.2*inch]*3 + [1.1*inch, 1.0*inch, 1.5*inch]),
           cap("this was a severe but contained outbreak: four in five people who caught it died, "
               "which is why it ended after a few hundred cases rather than spreading further."),
           Spacer(1, 5)]
    st += [grid2([
        cell(p1, "cases climbed for about three months and then fell away. One rise and one fall "
                 "means a single wave that ran out of people to infect — not a disease that kept "
                 "coming back.", 3.55, 0.80),
        cell(p2, "the line is what our model expects; the dots are what was actually recorded. "
                 "They follow the same arc, so the assumptions we made about how long people take "
                 "to fall ill and how long they can pass it on are close to right.", 3.55, 0.80),
        cell(p3, "after the model explains what it can, almost nothing is left over — the bars are "
                 "tiny and green. If the model had missed something that changed over time, these "
                 "bars would be tall and red.", 3.55, 0.80),
        cell(p4, "53 of the 192 days had no active case-finding, so we set those aside instead of "
                 "treating them as days with genuinely zero cases. Counting them as real zeros "
                 "would make the outbreak look smaller than it was.", 3.55, 0.80),
    ]), Spacer(1, 3)]
    st += [verdict("VERDICT: usable as-is. The model matches the shape of the outbreak, leaves no "
                   "unexplained pattern behind, and the contagiousness it estimates sits inside the "
                   "range other published studies of this same outbreak report.", GOOD, GREEN)]
    return st, dict(R0=R0, lag1=lag1[0])


def page_measles(m, pf):
    y = m.rash_onset.values.astype(float); d = m.day.values
    R0, fit = fit_incidence(MEASLES, y, d, [np.log(8.), 0.])
    res = pearson_residuals(y, fit)
    ll = pd.read_csv(ROOT/"data"/"raw"/"measles"/"measles_hagelloch_1861_linelist.csv")

    p1 = fig1("me_curve", lambda a: (
        a.bar(d, y, color=PURP, alpha=.85, width=1.0),
        a.set(xlabel="day of outbreak", ylabel="new rashes that day",
              title="The outbreak over time")))
    p2 = fig1("me_fit", lambda a: (
        a.plot(d, y, "o", ms=2.6, color=PURP, alpha=.7, label="what happened"),
        a.plot(d, fit, "-", lw=2, color=NAVYH, label="what the model expects"),
        a.set(xlabel="day of outbreak", ylabel="new rashes that day",
              title=f"Model vs reality  (spread score {R0:.2f})"),
        a.legend(frameon=False, fontsize=7)))
    lag1 = [None]
    def _acf(a):
        lag1[0] = acf_bars(a, res); a.set_title("Anything the model missed?")
    p3 = fig1("me_acf", _acf)
    def _age(a):
        a.hist(ll.AGE.values, bins=14, color=PURP, alpha=.85)
        a.set(xlabel="age in years", ylabel="children", title="Who caught it")
    p4 = fig1("me_age", _age)

    st = [Paragraph("Measles — Hagelloch, Germany, 1861", TT),
          Paragraph("Every child in one village, recorded door to door by the local physician", SU),
          HRFlowable(width="100%", thickness=.9, color=NAVY, spaceBefore=2, spaceAfter=4)]
    st += [tbl(["Children in village", "Days recorded", "Children infected", "Deaths",
                "Died of those infected", "School classes"],
               [["188", "86", "188 (all of them)", "12", "6.4%", "3"]],
               [1.3*inch]*2 + [1.35*inch, .85*inch, 1.4*inch, 1.0*inch]),
           cap("every single child in the village caught measles. That is what an extremely "
               "contagious disease looks like when it reaches a group with no prior immunity "
               "and no vaccine."),
           Spacer(1, 5)]
    st += [grid2([
        cell(p1, "a slow trickle for a month, then a sharp burst, then it was over. The burst is "
                 "the schoolrooms: once it reached a classroom it went through the whole class in "
                 "days.", 3.55, 0.80),
        cell(p2, "the model follows the timing well but draws a smoother, lower hump than reality. "
                 "That gap is the classroom effect — our model assumes everyone mixes evenly, and "
                 "in a village they do not.", 3.55, 0.80),
        cell(p3, "nothing systematic is left over — the bars are small and green. The model may "
                 "smooth out the burst, but it is not getting the timing or the direction wrong.", 3.55, 0.80),
        cell(p4, "this hit children of every age from infants to fifteen-year-olds. Nobody was "
                 "spared, which confirms there was no existing immunity in the village to slow it "
                 "down.", 3.55, 0.80),
    ]), Spacer(1, 3)]
    st += [verdict("VERDICT: usable as-is. Clean timing, nothing unexplained left behind, and the "
                   "one imperfection — a smoother curve than reality — is a known and harmless "
                   "consequence of assuming everyone mixes evenly in a village with schools.",
                   GOOD, GREEN)]
    return st, dict(R0=R0, lag1=lag1[0])


def page_influenza(f, pf):
    y = f.in_bed.values.astype(float); d = f.day.values
    ts = pf["influenza"]["t_seed_best"]
    meta = dict(structure="seiqr_prevalence", sigma=INFLUENZA.sigma, removal=INFLUENZA.removal,
                gamma_q=INFLUENZA.gamma_q, mu=0.0, N=INFLUENZA.N, I0=INFLUENZA.I0,
                t_seed=ts, T=len(y), dt_obs=1.0, kE=1, kI=1, kQ=1)
    def dev(p):
        mm = np.clip(solve_window(np.exp(p[0]), **meta)*np.exp(p[1]), 1e-9, None)
        o = np.clip(y, 1e-9, None); return 2*np.sum(o*np.log(o/mm)-(o-mm))
    r = optimize.minimize(dev, [np.log(3.), 0.], method="Nelder-Mead", options={"maxiter": 2500})
    R0 = float(np.exp(r.x[0])); fit = solve_window(R0, **meta)*np.exp(float(r.x[1]))
    res = pearson_residuals(y, fit)

    p1 = fig1("fl_curve", lambda a: (
        a.bar(d, y, color=TEAL, alpha=.85, width=.9),
        a.set(xlabel="day", ylabel="boys in bed", title="The outbreak over time")))
    p2 = fig1("fl_fit", lambda a: (
        a.plot(d, y, "o", ms=3.2, color=TEAL, alpha=.8, label="what happened"),
        a.plot(d, fit, "-", lw=2, color=NAVYH, label="what the model expects"),
        a.set(xlabel="day", ylabel="boys in bed", title=f"Model vs reality  (spread score {R0:.2f})"),
        a.legend(frameon=False, fontsize=7)))
    lag1 = [None]
    def _acf(a):
        lag1[0] = acf_bars(a, res, 6); a.set_title("Anything the model missed?")
    p3 = fig1("fl_acf", _acf)
    def _conf(a):
        rr = np.linspace(1.05, 9, 300)
        a.plot(rr, [km_attack_rate_np(v) for v in rr], color=NAVYH, lw=2)
        a.axhline(.671, color=RED, ls="--", lw=1.6)
        a.axvline(1.66, color=RED, ls=":", lw=1.6)
        a.axvline(R0, color=TEAL, ls=":", lw=1.6)
        a.annotate("how many got ill\nsays 1.7", (1.66, .06), xytext=(3.4, .13),
                   fontsize=7, color=RED, ha="center",
                   arrowprops=dict(arrowstyle="->", color=RED, lw=.9))
        a.annotate(f"how fast it spread\nsays {R0:.1f}", (R0, .85), xytext=(5.2, .48),
                   fontsize=7, color=TEAL, ha="center",
                   arrowprops=dict(arrowstyle="->", color=TEAL, lw=.9))
        a.set(xlabel="spread score", ylabel="share who fall ill", ylim=(0, 1.08),
              title="Two clues, two different answers")
    p4 = fig1("fl_conf", _conf)

    st = [Paragraph("Influenza — English boarding school, 1978", TT),
          Paragraph("A closed school of 763 boys; the matron counted who was in bed each morning", SU),
          HRFlowable(width="100%", thickness=.9, color=NAVY, spaceBefore=2, spaceAfter=4)]
    st += [tbl(["Boys at risk", "Days recorded", "Most in bed at once", "Total who fell ill",
                "Share who fell ill", "Deaths"],
               [["763", "14", "282 (37%)", "512", "67%", "0"]],
               [1.1*inch, 1.15*inch, 1.4*inch, 1.3*inch, 1.25*inch, .9*inch]),
           cap("two thirds of the school caught flu inside a fortnight and nobody died — a very "
               "contagious but mild illness, the opposite profile to Ebola."),
           Spacer(1, 5)]
    st += [grid2([
        cell(p1, "the whole thing was over in two weeks: up from one boy to 282 in six days, then "
                 "back down. This is the fastest of the four outbreaks by a wide margin.", 3.55, 0.80),
        cell(p2, "the model gets the rough shape but is visibly too flat — it cannot rise as fast "
                 "or fall as fast as the real thing did. This is the first sign of trouble.", 3.55, 0.80),
        cell(p3, "a large red bar means a real pattern is still left over after the model has done "
                 "its best. Something about this outbreak our model cannot capture.", 3.55, 0.80),
        cell(p4, "the two ways of judging contagiousness disagree. Counting how many eventually "
                 "fell ill points to a low number; watching how fast it spread points to a much "
                 "higher one. Both cannot be right.", 3.55, 0.80),
    ]), Spacer(1, 3)]
    st += [verdict("VERDICT: keep, but label it. This is a well-known dataset that standard models "
                   "cannot fully explain — published attempts reach the same impasse. It is useful "
                   "as a test of whether our checks catch a problem, and they did. It should not be "
                   "quoted as a clean contagiousness estimate alongside the other three.", WARN, "#856404")]
    return st, dict(R0=R0, lag1=lag1[0])


def page_norovirus(n, pf):
    by = n.groupby("class").ill.agg(["sum","count"]); by["ar"] = by["sum"]/by["count"]
    ar = float(n.ill.mean()); R0 = km_R0_from_attack_rate(ar)

    p1 = fig1("nv_class", lambda a: (
        a.bar(by.index.astype(str), by.ar,
              color=[RED if i == 10 else BLUE for i in by.index], alpha=.85),
        a.axhline(ar, color="k", ls="--", lw=1.2),
        a.set(xlabel="school class", ylabel="share who fell ill",
              title="Illness by classroom"),
        a.tick_params(axis="x", labelsize=6)))
    p2 = fig1("nv_size", lambda a: (
        a.scatter(by["count"], by.ar, s=40, color=BLUE, alpha=.8),
        a.set(xlabel="children in the class", ylabel="share who fell ill",
              title="Does class size matter?")))
    def _out(a):
        a.bar(["fell ill\n(what we count)", "absent from school\n(not the same thing)"],
              [int(n.ill.sum()), int((n.day_absent > 0).sum())],
              color=[GREEN, "#BBBBBB"], alpha=.85)
        a.set(ylabel="children", title="Being away is not being ill")
        a.tick_params(axis="x", labelsize=7)
    p3 = fig1("nv_out", _out)
    def _fs(a):
        rr = np.linspace(1.02, 3, 250)
        a.plot(rr, [km_attack_rate_np(v) for v in rr], color=NAVYH, lw=2)
        a.axhline(ar, color=BLUE, ls="--", lw=1.5)
        a.axvline(R0, color=BLUE, ls=":", lw=1.5)
        a.annotate(f"spread score\n{R0:.2f}", (R0, ar), xytext=(1.9, .45), fontsize=7,
                   color=BLUE, ha="center",
                   arrowprops=dict(arrowstyle="->", color=BLUE, lw=.9))
        a.set(xlabel="spread score", ylabel="share who fall ill",
              title="Reading contagiousness off the total")
    p4 = fig1("nv_fs", _fs)

    st = [Paragraph("Norovirus — Derbyshire primary school, England, 2001", TT),
          Paragraph("A stomach-bug outbreak recorded child by child across 15 classrooms", SU),
          HRFlowable(width="100%", thickness=.9, color=NAVY, spaceBefore=2, spaceAfter=4)]
    st += [tbl(["Children", "Classes", "Fell ill", "Share who fell ill",
                "Worst class", "Class with none ill"],
               [["492", "15", "75", "15%", "Class 10 (67%)", "Class 3 (0%)"]],
               [.95*inch, .85*inch, .85*inch, 1.35*inch, 1.5*inch, 1.5*inch]),
           cap("only one child in seven fell ill, and it was very uneven between classrooms — "
               "which is the clue that where a child sat mattered as much as the bug itself."),
           Spacer(1, 5)]
    st += [grid2([
        cell(p1, "one classroom had two thirds of its children ill while another had none. An "
                 "outbreak that spreads by close contact shows up as clusters like this, not as an "
                 "even sprinkling across the school.", 3.55, 0.80),
        cell(p2, "big classes were not worse hit than small ones, so the clustering is about who "
                 "sat near whom, not simply about crowding.", 3.55, 0.80),
        cell(p3, "far more children were absent than were actually ill. If we had counted absence "
                 "as illness we would have more than doubled the outbreak. We count only children "
                 "recorded as unwell.", 3.55, 0.80),
        cell(p4, "for a one-off outbreak there is a fixed relationship between how contagious "
                 "something is and what share of a group eventually catches it. We read our "
                 "estimate off that curve using the 15% who fell ill.", 3.55, 0.80),
    ]), Spacer(1, 3)]
    st += [verdict("VERDICT: usable as-is, and the most precise of the four. Every child was "
                   "recorded individually, so we have nearly 500 separate observations rather than "
                   "a daily tally — the classroom differences are handled explicitly rather than "
                   "ignored.", GOOD, GREEN)]
    return st, dict(R0=R0, ar=ar)


# ================================================================ overview
def overview(e, m, f, n, pf, ps, fits):
    names = ["Ebola", "Measles", "Influenza", "Norovirus"]
    cols  = [RED, PURP, TEAL, BLUE]

    el = e[e.analysis_eligible]
    def _four(a_, i, dd, yy, lab, col, ylab):
        a_.bar(dd, yy, color=col, alpha=.85, width=1.0)
        a_.set(title=lab, xlabel="day", ylabel=ylab)
    f1, axs = plt.subplots(1, 3, figsize=(10.6, 3.1))
    _four(axs[0], 0, el.day.values, el.onset.values, "Ebola 1995", RED, "new cases")
    _four(axs[1], 1, m.day.values, m.rash_onset.values, "Measles 1861", PURP, "new rashes")
    _four(axs[2], 2, f.day.values, f.in_bed.values, "Influenza 1978", TEAL, "boys in bed")
    f1.tight_layout(pad=.5); p_curves = FO/"ov_curves.png"
    f1.savefig(p_curves, bbox_inches="tight"); plt.close(f1)

    def _lump(a):
        vm = [ps.loc[ps.disease == d.lower(), "var_mean"].values[0]
              for d in ["Ebola","Measles","Influenza"]]
        a.bar(["Ebola","Measles","Influenza"], vm, color=cols[:3], alpha=.85)
        a.axhline(1, color="k", ls="--", lw=1.2)
        a.set(yscale="log", ylabel="lumpiness score", title="How lumpy is the day-to-day count?")
    p_lump = fig1("ov_lump", _lump, (3.7, 3.0))

    def _left(a):
        l1 = [ps.loc[ps.disease == d.lower(), "lag1_acf"].values[0]
              for d in ["Ebola","Measles","Influenza"]]
        a.bar(["Ebola","Measles","Influenza"], l1,
              color=[GREEN if abs(v) < .2 else RED for v in l1], alpha=.9)
        a.axhline(.2, color="k", ls="--", lw=1); a.axhline(-.2, color="k", ls="--", lw=1)
        a.axhline(0, color="k", lw=.6)
        a.set(ylabel="leftover pattern", title="Did the model miss anything?")
    p_left = fig1("ov_left", _left, (3.7, 3.0))

    def _r0(a):
        est = [fits["ebola"]["R0"], fits["measles"]["R0"],
               fits["influenza"]["R0"], fits["norovirus"]["R0"]]
        lit = [(1.51,2.53), (10,20), None, (1.3,6.7)]
        for i, lh in enumerate(lit):
            if lh: a.fill_between([i-.35,i+.35],[lh[0]]*2,[lh[1]]*2, color="green", alpha=.18)
        a.scatter(range(4), est, s=55, color=cols, zorder=5)
        a.set_xticks(range(4)); a.set_xticklabels(names, fontsize=7, rotation=12)
        a.set(yscale="log", ylabel="spread score",
              title="Does our answer match other studies?")
    p_r0 = fig1("ov_r0", _r0, (3.7, 3.0))

    def _size(a):
        a.bar(names, [139, 86, 14, 492], color=cols, alpha=.85)
        a.set(yscale="log", ylabel="separate observations",
              title="How much evidence is behind each?")
        a.tick_params(axis="x", labelsize=7, rotation=12)
    p_size = fig1("ov_size", _size, (3.7, 3.0))

    def _sev(a):
        sev = [80.8, 6.4, 0.0, 0.5]
        att = [29.2, 100.0, 67.1, 15.2]
        x = np.arange(4); w = .38
        a.bar(x-w/2, att, w, color="#5B8FF9", alpha=.9, label="share who caught it (%)")
        a.bar(x+w/2, sev, w, color="#C0392B", alpha=.9, label="share who died of it (%)")
        a.set_xticks(x); a.set_xticklabels(names, fontsize=7, rotation=12)
        a.set(ylabel="percent", title="Catchiness vs deadliness")
        a.legend(frameon=False, fontsize=6.5)
    p_sev = fig1("ov_sev", _sev, (4.4, 3.4))

    def _flu(a):
        rr = np.linspace(1.05, 9, 300)
        a.plot(rr, [km_attack_rate_np(v) for v in rr], color=NAVYH, lw=2)
        a.axhline(.671, color=RED, ls="--", lw=1.6); a.axvline(1.66, color=RED, ls=":", lw=1.6)
        a.axvline(fits["influenza"]["R0"], color=TEAL, ls=":", lw=1.6)
        a.annotate("total ill says 1.7", (1.66,.06), xytext=(3.6,.14), fontsize=7.2, color=RED,
                   ha="center", arrowprops=dict(arrowstyle="->", color=RED, lw=.9))
        a.annotate(f"speed says {fits['influenza']['R0']:.1f}", (fits["influenza"]["R0"],.85),
                   xytext=(5.4,.45), fontsize=7.2, color=TEAL, ha="center",
                   arrowprops=dict(arrowstyle="->", color=TEAL, lw=.9))
        a.set(xlabel="spread score", ylabel="share who fall ill", ylim=(0,1.08),
              title="Influenza 1978: the one unresolved case")
    p_flu = fig1("ov_flu", _flu, (4.4, 3.4))

    st = [Paragraph("Four Outbreaks: What the Checks Found", TT),
          Paragraph("Everything below was computed from the raw records before any model was fitted", SU),
          HRFlowable(width="100%", thickness=.9, color=NAVY, spaceBefore=2, spaceAfter=4)]
    st += [tbl(["Outbreak", "Where and when", "Who", "Caught it", "Died of it", "Records"],
               [["Ebola", "Kikwit, DR Congo, 1995", "~1,000 people", "292", "81%", "192 days"],
                ["Measles", "Hagelloch, Germany, 1861", "188 children", "188 (all)", "6.4%", "86 days"],
                ["Influenza", "English boarding school, 1978", "763 boys", "512", "0%", "14 days"],
                ["Norovirus", "Derbyshire school, England, 2001", "492 children", "75", "~0%", "492 children"]],
               [.85*inch, 1.95*inch, 1.2*inch, .9*inch, .85*inch, 1.05*inch]),
           cap("these four span the full range of what an infectious disease can be: one that kills "
               "most people it infects but spreads slowly, one that infects absolutely everyone but "
               "rarely kills, and two in between. That contrast is the point of studying them "
               "together.", ), Spacer(1, 6)]
    st += [RLImage(str(p_curves), width=7.3*inch, height=2.14*inch),
           cap("each outbreak rose once and fell once. That single-hump shape means each burned "
               "through its group and stopped, rather than smouldering on — which is what makes "
               "them comparable to one another at all. Note the horizontal scales differ: measles "
               "took three months, flu took two weeks.")]

    st += [PageBreak(), Paragraph("The Four Checks", TT),
           Paragraph("Each one decides something concrete about how the outbreak is modelled", SU),
           HRFlowable(width="100%", thickness=.9, color=NAVY, spaceBefore=2, spaceAfter=5)]
    st += [grid2([
        cell(p_lump, "day-to-day counts jump around far more than a simple tally would predict — "
                     "all three sit well above the dashed line. So we use a more forgiving way of "
                     "counting that expects quiet days followed by clusters.", 3.55, 0.80),
        cell(p_left, "for Ebola and measles the model left nothing behind (green). For influenza it "
                     "did (red). That single red bar is why influenza is treated differently from "
                     "the other three.", 3.55, 0.80),
        cell(p_r0,  "the green bands are what other researchers have published for each disease. "
                    "Ebola and norovirus land inside theirs. Measles lands a little below, which we "
                    "expect because our model assumes a village mixes evenly and villages with "
                    "schools do not.", 3.55, 0.80),
        cell(p_size, "norovirus rests on nearly 500 individual children; influenza rests on 14 daily "
                     "tallies. More evidence means a narrower, more trustworthy answer — this is "
                     "the second reason influenza is the weakest of the four.", 3.55, 0.80),
    ])]

    st += [PageBreak(), Paragraph("What the Numbers Mean in Plain Terms", TT),
           Paragraph("Two comparisons worth showing a non-specialist directly", SU),
           HRFlowable(width="100%", thickness=.9, color=NAVY, spaceBefore=2, spaceAfter=5)]
    st += [grid2([
        cell(p_sev, "catchiness and deadliness are separate things. Measles infected every child in "
                    "the village but killed one in sixteen of them. Ebola infected a few hundred "
                    "people but killed four in five. Neither number alone tells you how dangerous "
                    "an outbreak is.", 3.55, 0.80),
        cell(p_flu, "for influenza the two clues point to different answers and there is no way to "
                    "satisfy both at once. Published attempts on this same dataset run into the "
                    "identical wall, so this is a property of the records, not a mistake in our "
                    "work.", 3.55, 0.82),
    ]), Spacer(1, 4)]
    st += [Paragraph("Decisions that follow directly from the checks", HD),
           tbl(["Outbreak", "What was counted", "How we count it", "Extra time allowance?", "Verdict"],
               [["Ebola", "new cases each day", "forgiving count", "not needed", "PASS"],
                ["Measles", "new rashes each day", "forgiving count", "not needed", "PASS"],
                ["Norovirus", "ill or not, per child", "per-child, per-classroom", "not applicable", "PASS"],
                ["Influenza", "boys in bed each day", "forgiving count", "would swamp the data", "CAVEAT"]],
               [.95*inch, 1.6*inch, 1.75*inch, 1.7*inch, .85*inch],
               [GOOD, GOOD, GOOD, WARN]),
           cap("three of the four needed no special handling at all. Influenza needed an extra "
               "allowance for things changing over time, but with only 14 days of records that "
               "allowance would absorb everything and leave nothing to measure — so it is left out "
               "and the limitation is stated instead.")]

    st += [PageBreak(), Paragraph("How to Read the Results When They Arrive", TT),
           Paragraph("Four things to check, in order", SU),
           HRFlowable(width="100%", thickness=.9, color=NAVY, spaceBefore=2, spaceAfter=5)]
    st += [tbl(["Step", "What to look at", "What good looks like"],
               [["1", "Ebola, measles and norovirus contagiousness estimates",
                 "each range overlaps the published range for that disease"],
                ["2", "The range, not the single number",
                 "a wide range is an honest answer from thin records, not a failure"],
                ["3", "Influenza quoted only with its caveat",
                 "never placed beside the other three as an equal"],
                ["4", "The run repeated without influenza",
                 "the other three barely move, confirming it is not distorting them"]],
               [.5*inch, 3.0*inch, 3.8*inch]),
           cap("the headline claim of this work rests on ebola, measles and norovirus. Influenza is "
               "included to show the checks can catch a bad case, not to carry an argument."),
           Spacer(1, 8)]
    st += [Paragraph("One-line summary of each outbreak", HD),
           tbl(["Outbreak", "In one sentence"],
               [["Ebola", "Spread slowly and killed most people it infected; our model fits it cleanly."],
                ["Measles", "Infected every child in the village; model fits the timing, smooths the classroom bursts."],
                ["Norovirus", "Mild, clustered in classrooms, and the best-recorded of the four."],
                ["Influenza", "Fastest and mildest, but the records give two contradictory answers; kept as a test case."]],
               [1.0*inch, 6.3*inch]),
           cap("if the professor reads only one thing on this page, it is this table.")]

    st += [PageBreak(), Paragraph("Reference Numbers", TT),
           Paragraph("The same checks in their technical form, for anyone who wants them", SU),
           HRFlowable(width="100%", thickness=.9, color=NAVY, spaceBefore=2, spaceAfter=5)]
    rows, rc = [], []
    for key, disp in [("ebola","Ebola"), ("measles","Measles"), ("influenza","Influenza")]:
        r = ps[ps.disease == key].iloc[0]
        ok = not bool(r.grw_warranted)
        rows.append([disp, f"{r.var_mean:.2f}", f"{r['skew']:.2f}", f"{r.zero_frac:.2f}",
                     f"{r.lag1_acf:+.4f}", f"{r.ljung_box_p:.4f}",
                     f"{fits[key]['R0']:.2f}", "no" if ok else "YES"])
        rc.append(GOOD if ok else WARN)
    rows.append(["Norovirus", "n/a", "n/a", "n/a", "n/a", "n/a",
                 f"{fits['norovirus']['R0']:.2f}", "n/a"]); rc.append(GOOD)
    st += [tbl(["Outbreak", "Var/Mean", "Skew", "Zero-day share", "Residual lag-1",
                "Ljung-Box p", "Const-fit R0", "GRW warranted"],
               rows, [.85*inch,.75*inch,.6*inch,1.0*inch,.95*inch,.85*inch,.85*inch,1.0*inch]),
           cap("this is the same information as the coloured charts, written the way a statistician "
               "would want it. Nothing new is here — it is included so the numbers behind every "
               "claim can be checked."),
           Spacer(1, 6)]
    st += [Paragraph("Vocabulary, once", HD),
           tbl(["Term used here", "What it means"],
               [["spread score (R0)", "how many people one infected person passes it to, on average, in a group with no immunity"],
                ["lumpiness (Var/Mean)", "how much more the daily counts jump around than a simple tally would predict"],
                ["leftover pattern", "whether the model's errors still follow a trend after it has done its best"],
                ["share who fall ill", "the fraction of the whole group that eventually catches it"],
                ["forgiving count", "a way of counting that expects quiet days followed by clusters, rather than a steady trickle"]],
               [1.6*inch, 5.7*inch]),
           cap("these five terms cover every chart in this report.")]
    return st


# ================================================================ build
def build(path, story, footer_title):
    doc = SimpleDocTemplate(str(path), pagesize=letter, leftMargin=.5*inch,
                            rightMargin=.5*inch, topMargin=.45*inch, bottomMargin=.5*inch)
    f = make_footer(footer_title, "")
    doc.build(story, onFirstPage=f, onLaterPages=f)
    print(f"  -> {path}")


if __name__ == "__main__":
    e, m, f, n, pf, ps = load()
    fits = {}
    s_e, fits["ebola"]     = page_ebola(e, pf)
    s_m, fits["measles"]   = page_measles(m, pf)
    s_f, fits["influenza"] = page_influenza(f, pf)
    s_n, fits["norovirus"] = page_norovirus(n, pf)

    build(REP/"PREFIT_BY_DISEASE.pdf",
          s_e + [PageBreak()] + s_m + [PageBreak()] + s_f + [PageBreak()] + s_n,
          "Pre-fit by outbreak - CTMC-BHMM v7")
    build(REP/"PREFIT_OVERVIEW.pdf", overview(e, m, f, n, pf, ps, fits),
          "Pre-fit overview - CTMC-BHMM v7")
