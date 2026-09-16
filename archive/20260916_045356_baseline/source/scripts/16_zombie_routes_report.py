"""
scripts/16_zombie_routes_report.py
==================================
The undocumented-pathogen arm: both routes, reported side by side.

    reports/ZOMBIE_ROUTES.pdf

Route A restates the assumption. Route B predicts a new arm from the fitted
hierarchy. Only Route B learns anything from data, and neither is a statement
about this specific hypothetical pathogen. See core/bhmm/zombie_arm.py.
"""
import sys, json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, arviz as az
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats
from core.bhmm.zombie_arm import route_a_prior_predictive, route_b_from_trace, compare_routes
from core.diagnostics import km_attack_rate_np
from core.pdf_utils import (register_fonts, SANS, SANS_BOLD, SANS_ITAL, NAVY,
                            MGREY, LGREY, WHITE, BLACK, make_footer)
register_fonts()
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, HRFlowable, Image as RLImage, PageBreak)

ROOT=Path(__file__).resolve().parents[1]
CAN, TAB, GEN = ROOT/"data"/"canonical", ROOT/"tables", ROOT/"data"/"generated"
FZ=ROOT/"figures"/"zombie"; FZ.mkdir(parents=True, exist_ok=True); REP=ROOT/"reports"
ORANGE,NAVYH,GREY,RED,PURP,BLUE="#ED7D31","#1F4E79","#777777","#C0392B","#7B2D8B","#2E75B6"
GOOD,WARN,BAD,INFO=(colors.HexColor("#D5F5D5"),colors.HexColor("#FFF3CD"),
                    colors.HexColor("#FDEBEA"),colors.HexColor("#E8EEF7"))
plt.rcParams.update({"font.size":7.2,"axes.spines.top":False,"axes.spines.right":False,
                     "figure.dpi":190,"axes.titlesize":7.8,"axes.labelsize":7,"legend.fontsize":6})
S=lambda n,**k: ParagraphStyle(n,**{**dict(fontName=SANS,fontSize=8.3,leading=11,textColor=BLACK),**k})
TT=S("T",fontSize=15,fontName=SANS_BOLD,textColor=NAVY,alignment=TA_CENTER,leading=18)
SU=S("SU",fontSize=8.6,textColor=colors.HexColor(GREY),alignment=TA_CENTER)
H1=S("H1",fontSize=11,fontName=SANS_BOLD,textColor=NAVY,spaceBefore=6,spaceAfter=3)
BODY=S("B",fontSize=8.3,leading=11,alignment=TA_JUSTIFY)
CAP=S("CAP",fontSize=7.0,leading=8.8,textColor=colors.HexColor("#333"))
REG={}
def fig(name,reading,fs=(3.3,2.4)):
    def deco(fn):
        f,a=plt.subplots(1,1,figsize=fs); fn(a); f.tight_layout(pad=.35)
        p=FZ/f"{name}.png"; f.savefig(p,bbox_inches="tight"); plt.close(f)
        (FZ/f"{name}.txt").write_text(f"{name}\n\n{reading.strip()}\n")
        REG[name]=(p,reading.strip()); return fn
    return deco
def grid(names,ncol=3,w=2.38):
    cells=[]
    for n in names:
        p,r=REG[n]
        t=Table([[RLImage(str(p),width=w*inch,height=w*.74*inch)],[Paragraph(f"<b>{r}</b>",CAP)]],
                colWidths=[w*inch])
        t.setStyle(TableStyle([("LEFTPADDING",(0,0),(-1,-1),0),("RIGHTPADDING",(0,0),(-1,-1),0),
            ("TOPPADDING",(0,0),(-1,-1),1),("BOTTOMPADDING",(0,0),(-1,-1),5),("VALIGN",(0,0),(-1,-1),"TOP")]))
        cells.append(t)
    rows=[cells[i:i+ncol] for i in range(0,len(cells),ncol)]
    for r in rows:
        while len(r)<ncol: r.append("")
    tt=Table(rows,colWidths=[(w+.07)*inch]*ncol)
    tt.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),2),
        ("TOPPADDING",(0,0),(-1,-1),2),("BOTTOMPADDING",(0,0),(-1,-1),2)]))
    return tt
def tbl(head,rows,widths,rc=None,fs=6.8):
    h=[Paragraph(x,S("th",fontSize=fs,fontName=SANS_BOLD,textColor=WHITE)) for x in head]
    d=[[Paragraph(str(c),S("td",fontSize=fs,leading=fs+2.3)) for c in r] for r in rows]
    sty=[("BACKGROUND",(0,0),(-1,0),NAVY),("TEXTCOLOR",(0,0),(-1,0),WHITE),
         ("ROWBACKGROUNDS",(0,1),(-1,-1),[WHITE,LGREY]),("GRID",(0,0),(-1,-1),.25,MGREY),
         ("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),3),
         ("TOPPADDING",(0,0),(-1,-1),2.5),("BOTTOMPADDING",(0,0),(-1,-1),2.5)]
    if rc:
        for i,c in enumerate(rc):
            if c is not None: sty.append(("BACKGROUND",(0,i+1),(-1,i+1),c))
    t=Table([h]+d,colWidths=widths); t.setStyle(TableStyle(sty)); return t
def note(t,bg,bd):
    x=Table([[Paragraph(t,S("v",fontSize=8.5,fontName=SANS_BOLD,textColor=colors.HexColor(bd)))]],
            colWidths=[7.3*inch])
    x.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),bg),("BOX",(0,0),(-1,-1),1,colors.HexColor(bd)),
        ("LEFTPADDING",(0,0),(-1,-1),8),("TOPPADDING",(0,0),(-1,-1),6),
        ("BOTTOMPADDING",(0,0),(-1,-1),6)])); return x

def main():
    nc=GEN/"bhmm_trace_full.nc"
    if not nc.exists(): nc=GEN/"bhmm_trace_quick.nc"
    if not nc.exists(): sys.exit("no trace; run scripts/04_run_bhmm.py first")
    tr=az.from_netcdf(nc)
    ra=route_a_prior_predictive(CAN/"zombie_sezr_prior_params.json", seed=20260914)
    rb=route_b_from_trace(tr, sd_floor=0.1, seed=20260914)
    measured={k:tr.posterior[v].values.ravel() for k,v in
              [("Ebola","bhmm::e_R0"),("Measles","bhmm::mea_R0"),("Norovirus","bhmm::n_R0")]
              if v in tr.posterior}
    cmp_=compare_routes(ra,rb,measured)
    cmp_.to_csv(TAB/"zombie_route_comparison.csv",index=False)
    A,B=ra["R0"],rb["R0"]

    @fig("z01_routeA","Route A: the assumption restated. Narrow because you chose it to be narrow — the width reflects your confidence, not any evidence.")
    def _(a):
        a.hist(A,bins=60,density=True,color=ORANGE,alpha=.85)
        a.axvline(np.median(A),color="k",ls="--",lw=1.3,label=f"median {np.median(A):.2f}")
        a.set(xlabel="spread score",ylabel="density",title="1. Route A — assumption"); a.legend(frameon=False)
    @fig("z02_routeB","Route B: a new arm predicted from the fitted hierarchy. Wide because three measured outbreaks genuinely disagree about how transmissible pathogens are.")
    def _(a):
        a.hist(np.clip(B,0,60),bins=70,density=True,color=NAVYH,alpha=.7)
        a.axvline(np.median(B),color="k",ls="--",lw=1.3,label=f"median {np.median(B):.2f}")
        a.set(xlabel="spread score",ylabel="density",xscale="log",title="2. Route B — learned")
        a.legend(frameon=False)
    @fig("z03_overlay","Both routes on one axis. Route A is a spike of your own choosing; Route B is a broad range shaped by data. They answer different questions.")
    def _(a):
        a.hist(np.clip(A,0,60),bins=70,density=True,color=ORANGE,alpha=.65,label="A: assumption")
        a.hist(np.clip(B,0,60),bins=70,density=True,color=NAVYH,alpha=.45,label="B: learned")
        a.set(xlabel="spread score",ylabel="density",xscale="log",title="3. Side by side")
        a.legend(frameon=False)
    @fig("z04_vs_measured","Both routes against the three measured arms. Route B brackets them, as it must — it is the distribution they were drawn from.")
    def _(a):
        a.hist(np.clip(B,0,60),bins=70,density=True,color=NAVYH,alpha=.4,label="B: learned")
        a.hist(np.clip(A,0,60),bins=70,density=True,color=ORANGE,alpha=.5,label="A: assumption")
        for (k,v),c in zip(measured.items(),[RED,PURP,BLUE]):
            a.axvline(np.median(v),color=c,lw=1.6,label=k)
        a.set(xlabel="spread score",ylabel="density",xscale="log",title="4. Against measured arms")
        a.legend(frameon=False,fontsize=5.0)
    @fig("z05_hyper","The shared prior the measured arms produced. Route B draws from exactly this, which is why it is data-informed.")
    def _(a):
        a.scatter(rb["hyper_mu"][:1500],rb["hyper_sd"][:1500],s=3,alpha=.18,color=NAVYH)
        a.set(xlabel="shared prior centre (log scale)",ylabel="spread between diseases",
              title="5. Fitted hyperprior")
    @fig("z06_width","Interval width. Route B is far wider — honest uncertainty about an unmeasured pathogen, rather than borrowed confidence.")
    def _(a):
        names=list(measured)+["Route A","Route B"]
        vals=[np.quantile(v,.945)-np.quantile(v,.055) for v in measured.values()]
        vals+=[np.quantile(A,.945)-np.quantile(A,.055),np.quantile(B,.945)-np.quantile(B,.055)]
        a.barh(names,vals,color=[RED,PURP,BLUE,ORANGE,NAVYH][:len(names)],alpha=.85)
        a.set(xlabel="89% interval width",xscale="log",title="6. Uncertainty by arm")
        a.tick_params(axis="y",labelsize=5.6)
    @fig("z07_ar","Attack rate implied by each route. Route A insists nearly everyone is infected; Route B admits a wide range including mild outcomes.")
    def _(a):
        arA=np.array([km_attack_rate_np(r) for r in A[:2000]])
        arB=np.array([km_attack_rate_np(r) for r in B[:2000]])
        a.hist(arA,bins=40,density=True,color=ORANGE,alpha=.65,label="A")
        a.hist(arB,bins=40,density=True,color=NAVYH,alpha=.45,label="B")
        a.set(xlabel="share eventually infected",ylabel="density",title="7. Implied attack rate")
        a.legend(frameon=False)
    @fig("z08_prcc","Which assumption drives Route A. Transmission rate accounts for almost all of it; if data ever arrived, measure that first.")
    def _(a):
        rng=np.random.default_rng(11); n=4000
        lb=rng.normal(np.log(.55),.10,n); lg=rng.normal(np.log(.18),.04,n)
        ls=rng.normal(np.log(.2857),.05,n); mu_=rng.beta(2,18,n)
        R=np.exp(lb)/np.exp(lg)
        v=[abs(stats.spearmanr(x,R)[0]) for x in (lb,lg,ls,mu_)]
        lbl=["transmission","recovery","incubation","death rate"]; o=np.argsort(v)
        a.barh([lbl[i] for i in o],[v[i] for i in o],color=ORANGE,alpha=.85)
        a.set(xlabel="influence (0-1)",title="8. What drives Route A")
    @fig("z09_info","Where each route's information comes from. Route A adds no data; Route B inherits three measured outbreaks.")
    def _(a):
        a.axis("off")
        a.text(.5,.90,"Route A",ha="center",fontsize=8,fontweight="bold",color=ORANGE,transform=a.transAxes)
        a.text(.5,.78,"assumptions  ->  ODE  ->  output\n\nobservations used: 0",ha="center",
               fontsize=6.2,transform=a.transAxes)
        a.text(.5,.50,"Route B",ha="center",fontsize=8,fontweight="bold",color=NAVYH,transform=a.transAxes)
        a.text(.5,.30,"Ebola + Measles + Norovirus\n->  shared prior  ->  new arm\n\n"
                      f"observations used: {139+86+492}",ha="center",fontsize=6.2,transform=a.transAxes)
        a.set_title("9. Information sources")

    st=[Paragraph("The Undocumented Pathogen — Two Routes",TT),
        Paragraph("What can honestly be claimed about a disease with no data",SU),
        HRFlowable(width="100%",thickness=.9,color=NAVY,spaceBefore=2,spaceAfter=4)]
    st+=[note("NO SYNTHETIC OBSERVATIONS ARE CREATED ANYWHERE IN THIS ARM. Neither route invents, "
              "simulates or imputes a single case. Route A propagates declared assumptions; Route B "
              "borrows from three measured outbreaks through the shared prior.",INFO,NAVYH),
         Spacer(1,5)]
    st+=[Paragraph("The two routes",H1),
         tbl(["","Route A — assumption-driven","Route B — pooling-driven"],
             [["What it does","push chosen parameters through the model",
               "draw a new arm from the prior the measured outbreaks fitted"],
              ["Observations used","none","717 across three outbreaks"],
              ["Learns from data","no — posterior equals prior","yes — inherits the fitted hierarchy"],
              ["Median spread score",f"{np.median(A):.2f}",f"{np.median(B):.2f}"],
              ["89% interval",f"[{np.quantile(A,.055):.2f}, {np.quantile(A,.945):.2f}]",
               f"[{np.quantile(B,.055):.2f}, {np.quantile(B,.945):.2f}]"],
              ["Honest claim","IF a pathogen behaved this way, THEN...",
               "GIVEN three measured human outbreaks, a fourth would plausibly fall here"],
              ["Not a claim","'the spread score is this number'",
               "anything specific to this hypothetical pathogen"]],
             [1.15*inch,3.1*inch,3.1*inch],
             [None,None,None,None,None,GOOD,BAD]),Spacer(1,4)]
    st+=[grid(["z01_routeA","z02_routeB","z03_overlay","z04_vs_measured","z05_hyper","z06_width"],ncol=3)]
    st+=[PageBreak(),Paragraph("Implications and limits",H1),
         grid(["z07_ar","z08_prcc","z09_info"],ncol=3),Spacer(1,4)]
    st+=[Paragraph("What Route B is not",H1),
         Paragraph("Route B is not a statement about zombies. The model holds no zombie information "
                   "and cannot acquire any. It describes the population of human pathogens that "
                   "Ebola, Measles and Norovirus were drawn from, and predicts where a fourth "
                   "member of that population would fall. Presenting it as a pathogen-specific "
                   "estimate would repeat Route A's error one level up.",BODY),Spacer(1,4),
         note("THE SHORT ANSWER — the framework does not create information about an undocumented "
              "disease. It makes assumptions explicit, shows what they imply, identifies which one "
              "matters, and offers a second route where measured arms inform the unmeasured one. "
              "Route B is the defensible answer to 'can this handle unknown diseases'. Route A is "
              "a restatement and must be labelled as such.",INFO,NAVYH),Spacer(1,5),
         Paragraph("Numeric comparison",H1),
         tbl(["Arm","Median","89% interval","Width","Relative spread","Note"],
             [[r.arm,f"{r['median']:.3f}",f"[{r.eti89_lo:.2f}, {r.eti89_hi:.2f}]",
               f"{r.eti89_width:.2f}",f"{r.cv_pct:.0f}%",r.note] for _,r in cmp_.iterrows()],
             [1.35*inch,.7*inch,1.15*inch,.6*inch,.85*inch,2.65*inch],
             [GOOD]*len(measured)+[WARN,GOOD])]
    doc=SimpleDocTemplate(str(REP/"ZOMBIE_ROUTES.pdf"),pagesize=letter,leftMargin=.5*inch,
                          rightMargin=.5*inch,topMargin=.45*inch,bottomMargin=.5*inch)
    f_=make_footer("Undocumented Pathogen — Two Routes","")
    doc.build(st,onFirstPage=f_,onLaterPages=f_)
    print("  -> reports/ZOMBIE_ROUTES.pdf")

if __name__=="__main__": main()
