"""
scripts/06_build_reports.py
===========================
Build all publication PDFs from the current run's tables and figures.

Every report is generated from files on disk, so the PDFs always reflect the
most recent fit. Nothing is hard-coded from a previous run -- that was a defect
in v5, where 03_Rabies_Prefit_Report.pdf and 07_Rabies_Variable_Schema.pdf kept
being produced long after the Rabies arm had been removed.

Outputs (reports/):
    01_Prefit_Audit.pdf              pre-fit diagnostics for all three arms
    02_Prior_Specification.pdf       every prior with its epistemic label
    03_Posterior_Results.pdf         posterior summary + PPC + convergence
    04_Model_Changelog_v6.pdf        what changed from v5 and why
    DEBUG_model_performance.pdf      written by scripts/08

Run:  python scripts/06_build_reports.py
"""
import sys, json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd
from core.pdf_utils import (register_fonts, SANS, SANS_BOLD, SANS_ITAL,
                            NAVY, RED, GREEN, MGREY, LGREY, WHITE, BLACK, make_footer)
register_fonts()
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, HRFlowable, Image as RLImage, PageBreak)

ROOT = Path(__file__).resolve().parents[1]
TAB, REP = ROOT/"tables", ROOT/"reports"
FP, FB, FD = ROOT/"figures"/"prefit", ROOT/"figures"/"bhmm", ROOT/"figures"/"debug"
REP.mkdir(exist_ok=True)
GREY = colors.HexColor("#555555")
PASSBG, FAILBG, PARTBG = colors.HexColor("#D5F5D5"), colors.HexColor("#FDEBEA"), colors.HexColor("#FFF3CD")

S  = lambda n,**k: ParagraphStyle(n, **{**dict(fontName=SANS,fontSize=9,leading=12,textColor=BLACK), **k})
T  = S("T",fontSize=14,fontName=SANS_BOLD,textColor=NAVY,alignment=TA_CENTER,leading=18)
ST = S("ST",fontSize=8.5,textColor=GREY,alignment=TA_CENTER)
H  = S("H",fontSize=11,fontName=SANS_BOLD,textColor=NAVY,spaceBefore=8,spaceAfter=3)
B  = S("B",fontSize=8.8,leading=12,alignment=TA_JUSTIFY)
SM = S("SM",fontSize=7.3,leading=9.5)
NO = S("NO",fontSize=7.5,fontName=SANS_ITAL,textColor=GREY)

def rd(n, **kw):
    p = TAB/n
    return pd.read_csv(p, **kw) if p.exists() else None
def rj(n):
    p = TAB/n
    return json.loads(p.read_text()) if p.exists() else {}
def hr(): return HRFlowable(width="100%", thickness=1, color=NAVY, spaceBefore=3, spaceAfter=5)
def fig(p, w=7.1, r=.55):
    return RLImage(str(p), width=w*inch, height=w*r*inch) if Path(p).exists() \
        else Paragraph(f"[missing figure: {Path(p).name} - run scripts/02 and 04]", NO)
def tbl(head, rows, widths, rc=None):
    hrow=[Paragraph(h,S("th",fontSize=7.3,fontName=SANS_BOLD,textColor=WHITE)) for h in head]
    drow=[[Paragraph(str(c),SM) for c in r] for r in rows]
    sty=[("BACKGROUND",(0,0),(-1,0),NAVY),("TEXTCOLOR",(0,0),(-1,0),WHITE),
         ("ROWBACKGROUNDS",(0,1),(-1,-1),[WHITE,LGREY]),("GRID",(0,0),(-1,-1),.3,MGREY),
         ("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),4),
         ("RIGHTPADDING",(0,0),(-1,-1),4),("TOPPADDING",(0,0),(-1,-1),3),
         ("BOTTOMPADDING",(0,0),(-1,-1),3)]
    if rc:
        for i,c in enumerate(rc): sty.append(("BACKGROUND",(0,i+1),(-1,i+1),c))
    t=Table([hrow]+drow,colWidths=widths); t.setStyle(TableStyle(sty)); return t
def build(name, story, foot):
    d=SimpleDocTemplate(str(REP/name),pagesize=letter,leftMargin=.6*inch,rightMargin=.6*inch,
                        topMargin=.55*inch,bottomMargin=.65*inch)
    f=make_footer(foot,"")
    d.build(story,onFirstPage=f,onLaterPages=f); print(f"  built: {name}")

meta = rj("run_metadata.json")
RUNLINE = (f"run_id {meta.get('run_id','-')} | seed {meta.get('seed','-')} | "
           f"{meta.get('draws','-')} draws x {meta.get('chains','-')} chains")


# --------------------------------------------------------------------------
def report_prefit():
    pf = rj("prefit_diagnostics.json"); ps = rd("prefit_summary.csv")
    st = [Paragraph("Pre-fit Audit", T),
          Paragraph("Diagnostics computed from raw data only, before any model fitting", ST), hr()]
    st += [Paragraph(
        "Every observation-model and temporal-structure choice is fixed here, from the data "
        "alone, so that those choices are pre-registered rather than selected after seeing a fit. "
        "Decision rules are stated with each test.", B), Spacer(1,5)]
    if ps is not None:
        st += [Paragraph("1. Summary of decision rules", H),
               tbl(["Disease","Var/Mean","Skew","Zero frac","Residual lag-1","Ljung-Box p","GRW warranted?"],
                   [[r.disease, f"{r.var_mean:.2f}", f"{r['skew']:.2f}", f"{r.zero_frac:.2f}",
                     f"{r.lag1_acf:+.3f}", f"{r.ljung_box_p:.3f}",
                     "YES" if r.grw_warranted else "no"] for _,r in ps.iterrows()],
                   [1.3*inch,.8*inch,.6*inch,.75*inch,1.0*inch,.85*inch,1.0*inch],
                   [PARTBG if r.grw_warranted else PASSBG for _,r in ps.iterrows()]),
               Spacer(1,3),
               Paragraph("Rule: a GRW on log beta(t) is warranted only if the residuals of the "
                         "constant-transmission mechanistic fit show |lag-1 ACF| > 0.20 or "
                         "Ljung-Box p < 0.05. Ljung &amp; Box (1978) Biometrika 65(2):297-303.", NO),
               Spacer(1,6)]
    for nm, f_ in [("Ebola Kikwit 1995","ebola_prefit_panel.png"),
                   ("Influenza England 1978","influenza_prefit_panel.png"),
                   ("Norovirus Derbyshire 2001","norovirus_prefit_panel.png"),
                   ("Cross-disease overview","overview_prefit_panel.png")]:
        st += [Paragraph(nm, H), fig(FP/f_, 7.1, .53), Spacer(1,4)]
    if pf.get("influenza",{}).get("identification_conflict"):
        i = pf["influenza"]
        st += [PageBreak(), Paragraph("Influenza identification conflict", H),
               Paragraph(
               f"R0 implied by the reported attack rate 512/763 = "
               f"{i['reported_attack_rate']:.1%} is <b>{i['R0_from_final_size']:.2f}</b>. "
               f"R0 implied by fitting the prevalence time course is "
               f"<b>{i['const_fit_R0']:.2f}</b>, which in turn implies an attack rate of "
               f"{i['implied_attack_rate']:.1%}. These cannot both hold. "
               "Avilov KK et al. (2024) J R Soc Interface 21(220):20240394 show that no classic "
               "compartmental model reproduces both the time course and the final size for this "
               "dataset. See docs/INFLUENZA_1978_LIMITATION.md.", B)]
    build("01_Prefit_Audit.pdf", st, "BHMM v6 - Pre-fit Audit")


# --------------------------------------------------------------------------
def report_priors():
    pt_ = rd("prior_specification.csv")
    st = [Paragraph("Prior Specification", T),
          Paragraph("Every prior with its epistemic label and source", ST), hr(),
          Paragraph(
          "Labels: <b>LITERATURE</b> traceable to a citation; <b>STRUCTURAL</b> required for "
          "well-posedness; <b>WEAK</b> weakly informative regularisation justified by prior "
          "predictive checking; <b>DISCLOSURE</b> deliberately non-informative because the "
          "parameter is known to be unidentified.", B), Spacer(1,5)]
    if pt_ is not None:
        for dis in pt_.disease.unique():
            sub = pt_[pt_.disease==dis]
            st += [Paragraph(dis.title(), H),
                   tbl(["Parameter","Value","Label","Source"],
                       [[r.parameter, r.value, r.label, r.source] for _,r in sub.iterrows()],
                       [1.25*inch,1.35*inch,.85*inch,3.5*inch]),
                   Spacer(1,5)]
    build("02_Prior_Specification.pdf", st, "BHMM v6 - Prior Specification")


# --------------------------------------------------------------------------
def report_posterior():
    hp = rd("headline_posterior.csv", index_col=0)
    ppc, sc, fs, mv, dom = rd("ppc_results.csv"), rd("residual_serial_correlation.csv"), \
                           rd("final_size_consistency.csv"), rd("prior_movement_check.csv"), \
                           rd("prior_dominance.csv")
    st = [Paragraph("Posterior Results", T), Paragraph(RUNLINE, ST), hr()]
    if hp is not None:
        cols = [c for c in ["mean","sd","r_hat","ess_bulk"] if c in hp.columns]
        st += [Paragraph("1. Headline posterior", H),
               tbl(["Parameter"]+cols,
                   [[i.replace("bhmm::","")] + [f"{hp.loc[i,c]:.4g}" for c in cols]
                    for i in hp.index],
                   [2.0*inch]+[1.2*inch]*len(cols)), Spacer(1,5)]
    if mv is not None:
        st += [Paragraph("2. Identification check (raw variables vs Normal(0,1) prior)", H),
               tbl(["Parameter","Posterior mean","Posterior sd","Updated by data?"],
                   [[r.parameter.replace("bhmm::",""), f"{r.post_mean:+.4f}",
                     f"{r.post_sd:.4f}", "YES" if r.updated_by_data else "NO - UNIDENTIFIED"]
                    for _,r in mv.iterrows()],
                   [2.0*inch,1.4*inch,1.4*inch,2.1*inch],
                   [PASSBG if r.updated_by_data else FAILBG for _,r in mv.iterrows()]),
               Spacer(1,3),
               Paragraph("A raw variable whose posterior still has mean 0 and sd 1 never entered "
                         "the likelihood. This is the check that identified the v5 defect.", NO),
               Spacer(1,5)]
    if ppc is not None:
        st += [Paragraph("3. Posterior predictive checks", H),
               tbl(["Disease","Coverage of 90% PI","Pred mean","Obs mean","Verdict"],
                   [[r.disease, f"{r.coverage:.1%}", f"{r.pred_mean:.3f}", f"{r.obs_mean:.3f}",
                     "PASS" if r.passes else "FAIL"] for _,r in ppc.iterrows()],
                   [1.4*inch,1.6*inch,1.2*inch,1.2*inch,1.5*inch],
                   [PASSBG if r.passes else FAILBG for _,r in ppc.iterrows()]), Spacer(1,5)]
    if sc is not None:
        st += [Paragraph("4. Residual serial correlation", H),
               tbl(["Disease","lag-1 ACF","lag-2","Ljung-Box p","Verdict (rule |r|<0.20)"],
                   [[r.disease, f"{r.acf_lag1:+.3f}", f"{r.acf_lag2:+.3f}",
                     f"{r.ljung_box_p:.3f}",
                     "PASS" if abs(r.acf_lag1)<0.20 else "structure remains"]
                    for _,r in sc.iterrows()],
                   [1.3*inch,1.1*inch,1.0*inch,1.2*inch,2.3*inch]), Spacer(1,5)]
    if fs is not None:
        st += [Paragraph("5. Final-size consistency", H),
               tbl(["Disease","Implied AR","Reported AR","R0 from reported AR","Verdict"],
                   [[r.disease, f"{r.implied_ar_median:.1%}", f"{r.reported_attack_rate:.1%}",
                     f"{r.R0_from_reported_ar:.2f}",
                     "consistent" if r.consistent else "CONFLICT"] for _,r in fs.iterrows()],
                   [1.3*inch,1.2*inch,1.2*inch,1.6*inch,1.6*inch],
                   [PASSBG if r.consistent else PARTBG for _,r in fs.iterrows()]), Spacer(1,5)]
    if dom is not None:
        st += [Paragraph("6. Prior dominance", H),
               tbl(["Parameter","Post/prior mean","sd ratio","Verdict"],
                   [[r.parameter.replace("bhmm::",""), f"{r.mean_ratio:.2f}",
                     f"{r.sd_ratio:.2f}", r.verdict] for _,r in dom.iterrows()],
                   [1.8*inch,1.4*inch,1.0*inch,2.7*inch],
                   [FAILBG if r.prior_dominated else PASSBG for _,r in dom.iterrows()]),
               Spacer(1,5)]
    st += [PageBreak(), Paragraph("7. Figures", H),
           fig(FB/"R0_comparison.png", 7.1, .28), Spacer(1,4),
           fig(FB/"ppc_panel.png", 7.1, .53), Spacer(1,4),
           fig(FB/"convergence_panel.png", 7.1, .37)]
    build("03_Posterior_Results.pdf", st, "BHMM v6 - Posterior Results")


# --------------------------------------------------------------------------
def report_changelog():
    ch = ROOT/"docs"/"PARAMETER_CHANGE_AUDIT.md"
    st = [Paragraph("Model Changelog: v5 to v6", T),
          Paragraph("Every change, its evidence, and its academic precedent", ST), hr()]
    if ch.exists():
        for ln in ch.read_text().splitlines():
            l = ln.rstrip()
            if not l: st.append(Spacer(1,3)); continue
            if l.startswith("### "): st.append(Paragraph(l[4:], S("h3",fontSize=9.5,
                fontName=SANS_BOLD,textColor=NAVY,spaceBefore=5,spaceAfter=2)))
            elif l.startswith("## "): st.append(Paragraph(l[3:], H))
            elif l.startswith("# "): continue
            elif l.startswith("|"): st.append(Paragraph(l.replace("|"," "), SM))
            elif l.startswith("- ") or l.startswith("* "):
                st.append(Paragraph("&bull; "+l[2:], S("li",fontSize=8.8,leading=12,leftIndent=12)))
            else: st.append(Paragraph(l, B))
    else:
        st.append(Paragraph("docs/PARAMETER_CHANGE_AUDIT.md not found.", NO))
    build("04_Model_Changelog_v6.pdf", st, "BHMM v6 - Changelog")


if __name__ == "__main__":
    print(f"Building reports -> {REP}")
    report_prefit(); report_priors(); report_posterior(); report_changelog()
    print(f"Done. {len(list(REP.glob('*.pdf')))} PDFs in {REP}")
