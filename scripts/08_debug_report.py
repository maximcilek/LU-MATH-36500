"""
scripts/08_debug_report.py
==========================
Debug-level post-fit figures and a rated debug PDF.

Writes figures/debug/*.png and reports/DEBUG_model_performance.pdf.
Runs automatically at the end of scripts/04_run_bhmm.py.

The rating is computed from the diagnostic tables, not hand-assigned, so it
changes when the model changes. Weights and thresholds are printed in the PDF.
"""
import sys, json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd, arviz as az
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.pdf_utils import (register_fonts, SANS, SANS_BOLD, SANS_ITAL, NAVY, RED,
                            GREEN, MGREY, WHITE, BLACK, LGREY, make_footer)
register_fonts()
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, HRFlowable, Image as RLImage, PageBreak)

ROOT = Path(__file__).resolve().parents[1]
GEN, TAB, FIGD, REP = ROOT/"data"/"generated", ROOT/"tables", ROOT/"figures"/"debug", ROOT/"reports"
FIGD.mkdir(parents=True, exist_ok=True); REP.mkdir(exist_ok=True)
TEAL, BLUE, ORANGE, GREY = "#1A7B6B", "#2E75B6", "#ED7D31", "#555555"
PASSBG, FAILBG, PARTBG = colors.HexColor("#D5F5D5"), colors.HexColor("#FDEBEA"), colors.HexColor("#FFF3CD")


def load():
    nc = GEN/"bhmm_trace_full.nc"
    if not nc.exists(): nc = GEN/"bhmm_trace_quick.nc"
    if not nc.exists(): sys.exit("no trace; run scripts/04_run_bhmm.py")
    tr = az.from_netcdf(nc)
    def rd(n, **kw):
        p = TAB/n
        return pd.read_csv(p, **kw) if p.exists() else None
    def rj(n):
        p = TAB/n
        return json.loads(p.read_text()) if p.exists() else {}
    return tr, az.summary(tr), dict(
        ppc=rd("ppc_results.csv"), dom=rd("prior_dominance.csv"),
        sc=rd("residual_serial_correlation.csv"), fs=rd("final_size_consistency.csv"),
        mv=rd("prior_movement_check.csv"), gate=rd("identifiability_gate.csv"),
        meta=rj("run_metadata.json"), gerr=rj("ode_grid_interpolation_error.json"))


def score(summary, t):
    """Component scores computed from the tables. Weights printed in the PDF."""
    s, w = {}, {}
    divs = t["meta"].get("divergences", 0)
    maxr, mine = float(summary.r_hat.max()), float(summary.ess_bulk.min())
    conv = 100.0
    conv -= min(40, divs * 2.0)
    conv -= 0 if maxr <= 1.01 else (15 if maxr <= 1.05 else 40)
    conv -= 0 if mine >= 400 else (10 if mine >= 100 else 30)
    s["Convergence"], w["Convergence"] = max(0.0, conv), 0.20

    if t["ppc"] is not None:
        cov = t["ppc"].coverage.values
        s["PPC calibration"] = float(np.clip(np.mean([min(c/0.90, 1.0) for c in cov])*100, 0, 100))
    else: s["PPC calibration"] = 0.0
    w["PPC calibration"] = 0.22

    # identification: did every raw variable move off its prior?
    if t["mv"] is not None and len(t["mv"]):
        s["R0 identification"] = float(t["mv"].updated_by_data.mean()*100)
    else: s["R0 identification"] = 0.0
    w["R0 identification"] = 0.20

    # residual structure
    if t["sc"] is not None and len(t["sc"]):
        ok = np.mean([1.0 if abs(v) < 0.20 else max(0.0, 1.0 - (abs(v)-0.20)/0.6)
                      for v in t["sc"].acf_lag1.values])
        s["Residual structure"] = float(ok*100)
    else: s["Residual structure"] = 100.0
    w["Residual structure"] = 0.12

    # prior dominance
    if t["dom"] is not None and len(t["dom"]):
        s["Prior independence"] = float((1 - t["dom"].prior_dominated.mean())*100)
    else: s["Prior independence"] = 100.0
    w["Prior independence"] = 0.08

    # literature consistency
    lit = {"bhmm::e_R0": (1.51, 2.53), "bhmm::mea_R0": (10.0, 20.0), "bhmm::n_R0": (1.3, 6.7)}
    ok = []
    for k,(lo,hi) in lit.items():
        if k in summary.index:
            m = float(summary.loc[k,"mean"])
            ok.append(100.0 if lo <= m <= hi else max(0.0, 100 - 100*min(abs(m-lo),abs(m-hi))/max(hi,1e-9)))
    s["Literature consistency"] = float(np.mean(ok)) if ok else 0.0
    w["Literature consistency"] = 0.10

    # final size consistency
    if t["fs"] is not None and len(t["fs"]):
        s["Final-size consistency"] = float(t["fs"].consistent.mean()*100)
    else: s["Final-size consistency"] = 100.0
    w["Final-size consistency"] = 0.08

    overall = sum(s[k]*w[k] for k in s)
    return s, w, round(overall)


def figures(tr, summary, t, s, overall):
    plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False,"figure.dpi":170})
    post = tr.posterior

    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("DEBUG: model performance", fontsize=12, fontweight="bold", color=NAVY)

    names, vals = list(s.keys()), list(s.values())
    cols = ["#D5F5D5" if v>=85 else "#FFF3CD" if v>=65 else "#FDEBEA" for v in vals]
    ax[0,0].barh(names, vals, color=cols, edgecolor="grey")
    ax[0,0].axvline(85, color="green", ls="--", lw=1); ax[0,0].axvline(65, color="orange", ls=":", lw=1)
    ax[0,0].set(xlim=(0,112), xlabel="score", title=f"Component scores (overall {overall}/100)")
    for i,v in enumerate(vals): ax[0,0].text(v+1.5, i, f"{v:.0f}", va="center", fontsize=8)

    r = summary.r_hat.values
    ax[0,1].hist(r, bins=30, color=BLUE, alpha=.8)
    ax[0,1].axvline(1.01, color="orange", ls="--"); ax[0,1].axvline(1.05, color=RED, ls="--")
    ax[0,1].set(xlabel="Rhat", title=f"Rhat (max={r.max():.4f})")

    e = summary.ess_bulk.values
    ax[0,2].hist(np.clip(e,0,12000), bins=30, color=GREEN, alpha=.8)
    ax[0,2].axvline(400, color="orange", ls="--"); ax[0,2].axvline(100, color=RED, ls=":")
    ax[0,2].set(xlabel="ESS bulk", title=f"ESS (min={e.min():.0f})")

    # R0 precision
    prec = []
    for k,lab,c in [("bhmm::e_R0","Ebola",RED),("bhmm::mea_R0","Measles","#7B2D8B"),
                    ("bhmm::flu_R0","Influenza",TEAL),("bhmm::n_R0","Norovirus",BLUE)]:
        if k in post:
            x = post[k].values.flatten(); prec.append((lab, float(x.std()/x.mean()*100), c))
    if prec:
        ax[1,0].bar([p[0] for p in prec], [p[1] for p in prec],
                    color=[p[2] for p in prec], alpha=.85)
        ax[1,0].axhline(20, color="green", ls="--", lw=1.2, label="target <20%")
        ax[1,0].axhline(100, color=RED, ls="--", lw=1.2, label="problem >100%")
        ax[1,0].set(ylabel="CV of R0 (%)", yscale="log", title="R0 precision")
        ax[1,0].legend(frameon=False, fontsize=7.5)

    # raw movement: the identification check
    if t["mv"] is not None and len(t["mv"]):
        mv = t["mv"]
        ax[1,1].bar(range(len(mv)), mv.post_sd, color=[GREEN if u else RED for u in mv.updated_by_data], alpha=.85)
        ax[1,1].axhline(1.0, color="k", ls="--", lw=1.5, label="prior sd = 1 (unidentified)")
        ax[1,1].set_xticks(range(len(mv)))
        ax[1,1].set_xticklabels([p.replace("bhmm::","").replace("_log_R0_raw","") for p in mv.parameter], fontsize=8)
        ax[1,1].set(ylabel="posterior sd of raw", title="Identification check\n(sd below 1 = data informed)")
        ax[1,1].legend(frameon=False, fontsize=7.5)

    ax[1,2].axis("off")
    lines = ["DIAGNOSTIC SUMMARY", ""]
    lines.append(f"divergences         {t['meta'].get('divergences','-')}")
    lines.append(f"max Rhat            {summary.r_hat.max():.4f}")
    lines.append(f"min ESS bulk        {summary.ess_bulk.min():.0f}")
    if t["ppc"] is not None:
        for _,row in t["ppc"].iterrows():
            lines.append(f"PPC {row.disease:11s} {row.coverage:.1%}")
    if t["sc"] is not None:
        for _,row in t["sc"].iterrows():
            lines.append(f"lag-1 {row.disease:9s} {row.acf_lag1:+.3f}")
    if t["fs"] is not None:
        for _,row in t["fs"].iterrows():
            lines.append(f"AR {row.disease:12s} {row.implied_ar_median:.0%} vs {row.reported_attack_rate:.0%}")
    if t["gerr"]:
        for k,v in t["gerr"].items():
            lines.append(f"grid err {k:9s} {v['max_rel_error']:.1e}")
    ax[1,2].text(0.02, 0.98, "\n".join(lines), transform=ax[1,2].transAxes, va="top",
                 fontsize=8.5, family="monospace",
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="#F0F4F8", edgecolor=NAVY))
    fig.tight_layout(); fig.savefig(FIGD/"debug_scorecard.png", bbox_inches="tight"); plt.close(fig)
    print(f"  figure -> {FIGD/'debug_scorecard.png'}")


def pdf(summary, t, s, w, overall):
    st = lambda n,**k: ParagraphStyle(n, **{**dict(fontName=SANS,fontSize=9,leading=12,
                                                   textColor=BLACK), **k})
    T  = st("T",fontSize=14,fontName=SANS_BOLD,textColor=NAVY,alignment=TA_CENTER,leading=18)
    H  = st("H",fontSize=11,fontName=SANS_BOLD,textColor=NAVY,spaceBefore=8,spaceAfter=3)
    B  = st("B",fontSize=8.8,leading=12,alignment=TA_JUSTIFY)
    SM = st("SM",fontSize=7.5,leading=10)
    story = [Paragraph("BHMM v6 - Debug Model Performance Report", T),
             Paragraph(f"run_id {t['meta'].get('run_id','-')} | seed {t['meta'].get('seed','-')} | "
                       f"{t['meta'].get('draws','-')} draws x {t['meta'].get('chains','-')} chains",
                       st("s",fontSize=8.5,textColor=colors.HexColor(GREY),alignment=TA_CENTER)),
             HRFlowable(width="100%", thickness=1, color=NAVY, spaceBefore=3, spaceAfter=5)]

    col = GREEN if overall>=85 else colors.HexColor("#856404") if overall>=70 else RED
    box = Table([[Paragraph(f"<b>OVERALL RATING: {overall}/100</b>", st("o",fontSize=12,
                  fontName=SANS_BOLD,textColor=col))]], colWidths=[7.0*inch])
    box.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),colors.HexColor("#EEF2F8")),
        ("BOX",(0,0),(-1,-1),1,NAVY),("LEFTPADDING",(0,0),(-1,-1),10),
        ("TOPPADDING",(0,0),(-1,-1),7),("BOTTOMPADDING",(0,0),(-1,-1),7)]))
    story += [box, Spacer(1,6)]

    def tbl(head, rows, widths, rc=None):
        hr = [Paragraph(h, st("th",fontSize=7.5,fontName=SANS_BOLD,textColor=WHITE)) for h in head]
        dr = [[Paragraph(str(c), SM) for c in r] for r in rows]
        sty = [("BACKGROUND",(0,0),(-1,0),NAVY),("TEXTCOLOR",(0,0),(-1,0),WHITE),
               ("ROWBACKGROUNDS",(0,1),(-1,-1),[WHITE,LGREY]),("GRID",(0,0),(-1,-1),.3,MGREY),
               ("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),4),
               ("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3)]
        if rc:
            for i,c in enumerate(rc): sty.append(("BACKGROUND",(0,i+1),(-1,i+1),c))
        tt=Table([hr]+dr,colWidths=widths); tt.setStyle(TableStyle(sty)); return tt

    story += [Paragraph("1. Component scores (computed from diagnostic tables, not assigned)", H)]
    rows = [[k, f"{w[k]:.0%}", f"{s[k]:.0f}/100",
             "strong" if s[k]>=85 else "acceptable" if s[k]>=65 else "weak"] for k in s]
    rc = [PASSBG if s[k]>=85 else PARTBG if s[k]>=65 else FAILBG for k in s]
    story += [tbl(["Component","Weight","Score","Band"], rows,
                  [2.6*inch,0.8*inch,0.9*inch,2.7*inch], rc), Spacer(1,6)]

    story += [Paragraph("2. Diagnostic detail", H)]
    det = []
    det.append(["Divergences", str(t["meta"].get("divergences","-")), "target 0"])
    det.append(["Max Rhat", f"{summary.r_hat.max():.4f}", "target <= 1.01"])
    det.append(["Min ESS bulk", f"{summary.ess_bulk.min():.0f}", "target >= 400"])
    if t["ppc"] is not None:
        for _,r in t["ppc"].iterrows():
            det.append([f"PPC {r.disease}", f"{r.coverage:.1%}", "target >= 80% of 90% PI"])
    if t["sc"] is not None:
        for _,r in t["sc"].iterrows():
            det.append([f"Residual lag-1 {r.disease}", f"{r.acf_lag1:+.3f}", "target |r| < 0.20"])
    if t["mv"] is not None:
        for _,r in t["mv"].iterrows():
            det.append([f"Raw movement {r.parameter.replace('bhmm::','')}",
                        f"mean {r.post_mean:+.3f}, sd {r.post_sd:.3f}",
                        "UPDATED" if r.updated_by_data else "NOT UPDATED - unidentified"])
    if t["fs"] is not None:
        for _,r in t["fs"].iterrows():
            det.append([f"Final size {r.disease}",
                        f"implied {r.implied_ar_median:.1%} vs reported {r.reported_attack_rate:.1%}",
                        "consistent" if r.consistent else "CONFLICT - see docs/INFLUENZA_1978_LIMITATION.md"])
    if t["gerr"]:
        for k,v in t["gerr"].items():
            det.append([f"ODE grid error {k}", f"{v['max_rel_error']:.2e}", "surrogate error, target < 1e-2"])
    story += [tbl(["Quantity","Value","Criterion"], det,
                  [2.1*inch,2.2*inch,2.7*inch]), Spacer(1,6)]

    p = FIGD/"debug_scorecard.png"
    if p.exists():
        story += [Paragraph("3. Debug panel", H),
                  RLImage(str(p), width=7.1*inch, height=4.0*inch)]

    story += [PageBreak(), Paragraph("4. Standing limitations (disclose in any write-up)", H),
        Paragraph(
        "<b>Influenza England 1978 identification conflict.</b> Kermack-McKendrick final size "
        "with the reported attack rate 512/763 = 67.1% implies R0 = 1.66. Fitting the prevalence "
        "time course implies a substantially larger R0. Avilov KK et al. (2024) J R Soc Interface "
        "21(220):20240394 show no classic compartmental model reproduces both; their "
        "delay-differential model gives R0 = 8.14, which Ahmad et al. (2025) Sci Rep flag as far "
        "above the typical influenza range of 1-4. Any reported flu_R0 must state which target it "
        "matches. See docs/INFLUENZA_1978_LIMITATION.md.", B), Spacer(1,4),
        Paragraph(
        "<b>flu_alpha_nb is not identified.</b> The Influenza Var/Mean of 103.6 reflects the "
        "epidemic arc across 14 days, not within-day count noise, so the likelihood is flat in the "
        "dispersion parameter (Gelman et al. 2020 BDA3 sec 13.3). The prior is deliberately wide so "
        "the posterior cannot be mistaken for an estimate.", B), Spacer(1,4),
        Paragraph(
        "<b>Generation time is fixed, not estimated.</b> The ODE surrogate grid varies R0 at fixed "
        "sigma and removal rates. With 14 and 139 observations, jointly identifying R0 and the "
        "generation-interval distribution is not feasible. Sensitivity to the assumed rates is "
        "prespecified in core/bhmm/priors.py.", B), Spacer(1,4),
        Paragraph(
        "<b>Zombie SEZR is prior-predictive only.</b> No data, no likelihood, posterior equals "
        "prior. It is a structural reference point, not an estimated arm.", B)]

    doc = SimpleDocTemplate(str(REP/"DEBUG_model_performance.pdf"), pagesize=letter,
                            leftMargin=.6*inch, rightMargin=.6*inch,
                            topMargin=.55*inch, bottomMargin=.65*inch)
    f = make_footer("BHMM v6 - Debug Report", t["meta"].get("timestamp","")[:10])
    doc.build(story, onFirstPage=f, onLaterPages=f)
    print(f"  report -> {REP/'DEBUG_model_performance.pdf'}")


if __name__ == "__main__":
    tr, summary, t = load()
    s, w, overall = score(summary, t)
    print(f"  overall rating: {overall}/100")
    for k,v in s.items(): print(f"    {k:26s} {v:6.1f}  (weight {w[k]:.0%})")
    figures(tr, summary, t, s, overall)
    pdf(summary, t, s, w, overall)
    (TAB/"debug_scores.json").write_text(json.dumps(dict(components=s, weights=w, overall=overall), indent=2))
