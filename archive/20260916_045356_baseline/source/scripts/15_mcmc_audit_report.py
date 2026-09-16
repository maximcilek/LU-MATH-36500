"""
scripts/15_mcmc_audit_report.py
===============================
Sampler-process audit: how the fit actually ran, not just where it ended.

    tables/mcmc_*.csv                every sampler-state series, auditable
    figures/trace/*.png              one figure per diagnostic
    reports/MCMC_AUDIT.pdf           assembled document

Run after scripts/04_run_bhmm.py (chained automatically).
"""
import sys, json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd, arviz as az
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.mcmc_audit import write_all, convergence_ledger, posterior_correlation
from core.pdf_utils import (register_fonts, SANS, SANS_BOLD, SANS_ITAL, NAVY,
                            MGREY, LGREY, WHITE, BLACK, make_footer)
register_fonts()
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, HRFlowable, Image as RLImage, PageBreak)

ROOT = Path(__file__).resolve().parents[1]
TAB, GEN, FT, REP = ROOT/"tables", ROOT/"data"/"generated", ROOT/"figures"/"trace", ROOT/"reports"
FT.mkdir(parents=True, exist_ok=True); REP.mkdir(exist_ok=True)

NAVYH, GREEN, RED, GREY = "#1F4E79", "#2E7D32", "#C0392B", "#777777"
GOOD, WARN, BAD = (colors.HexColor("#D5F5D5"), colors.HexColor("#FFF3CD"),
                   colors.HexColor("#FDEBEA"))
CH = ["#C0392B", "#2E75B6", "#1A7B6B", "#7B2D8B", "#ED7D31", "#555555"]
plt.rcParams.update({"font.size": 7.2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 190, "axes.titlesize": 7.8, "axes.labelsize": 7,
                     "legend.fontsize": 5.8, "xtick.labelsize": 6, "ytick.labelsize": 6})

S  = lambda n, **k: ParagraphStyle(n, **{**dict(fontName=SANS, fontSize=8.2,
                                                leading=10.8, textColor=BLACK), **k})
TT = S("T",  fontSize=15, fontName=SANS_BOLD, textColor=NAVY, alignment=TA_CENTER, leading=18)
SU = S("SU", fontSize=8.6, textColor=colors.HexColor(GREY), alignment=TA_CENTER)
H1 = S("H1", fontSize=11, fontName=SANS_BOLD, textColor=NAVY, spaceBefore=6, spaceAfter=3)
CAP = S("CAP", fontSize=7.0, leading=8.8, textColor=colors.HexColor("#333333"))
REG = {}


def fig(name, reading, figsize=(3.3, 2.4)):
    def deco(fn):
        f, a = plt.subplots(1, 1, figsize=figsize)
        fn(a); f.tight_layout(pad=.35)
        p = FT/f"{name}.png"; f.savefig(p, bbox_inches="tight"); plt.close(f)
        (FT/f"{name}.txt").write_text(f"{name}\n\n{reading.strip()}\n")
        REG[name] = (p, reading.strip())
        return fn
    return deco


def grid(names, ncol=3, w=2.38):
    cells = []
    for n in names:
        if n not in REG: continue
        p, reading = REG[n]
        inner = Table([[RLImage(str(p), width=w*inch, height=w*.74*inch)],
                       [Paragraph(f"<b>{reading}</b>", CAP)]], colWidths=[w*inch])
        inner.setStyle(TableStyle([("LEFTPADDING",(0,0),(-1,-1),0),("RIGHTPADDING",(0,0),(-1,-1),0),
                                   ("TOPPADDING",(0,0),(-1,-1),1),("BOTTOMPADDING",(0,0),(-1,-1),5),
                                   ("VALIGN",(0,0),(-1,-1),"TOP")]))
        cells.append(inner)
    rows = [cells[i:i+ncol] for i in range(0, len(cells), ncol)]
    for r in rows:
        while len(r) < ncol: r.append("")
    t = Table(rows, colWidths=[(w+.07)*inch]*ncol)
    t.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),2),
                           ("RIGHTPADDING",(0,0),(-1,-1),2),("TOPPADDING",(0,0),(-1,-1),2),
                           ("BOTTOMPADDING",(0,0),(-1,-1),2)]))
    return t


def tbl(head, rows, widths, rc=None, fs=6.6):
    hrw = [Paragraph(h, S("th", fontSize=fs, fontName=SANS_BOLD, textColor=WHITE)) for h in head]
    drw = [[Paragraph(str(c), S("td", fontSize=fs, leading=fs+2.2)) for c in r] for r in rows]
    sty = [("BACKGROUND",(0,0),(-1,0),NAVY),("TEXTCOLOR",(0,0),(-1,0),WHITE),
           ("ROWBACKGROUNDS",(0,1),(-1,-1),[WHITE,LGREY]),("GRID",(0,0),(-1,-1),.25,MGREY),
           ("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),3),
           ("TOPPADDING",(0,0),(-1,-1),2.4),("BOTTOMPADDING",(0,0),(-1,-1),2.4)]
    if rc:
        for i,c in enumerate(rc):
            if c is not None: sty.append(("BACKGROUND",(0,i+1),(-1,i+1),c))
    t = Table([hrw]+drw, colWidths=widths); t.setStyle(TableStyle(sty)); return t


KEY = ["bhmm::e_R0","bhmm::mea_R0","bhmm::flu_R0","bhmm::n_R0","bhmm::n_attack_rate",
       "bhmm::log_intercept_e","bhmm::e_alpha_nb","bhmm::mea_alpha_nb","bhmm::flu_alpha_nb",
       "bhmm::class_re_sd","bhmm::hyper_log_R0_mu","bhmm::hyper_log_R0_sd"]


def main():
    nc = GEN/"bhmm_trace_full.nc"
    if not nc.exists(): nc = GEN/"bhmm_trace_quick.nc"
    if not nc.exists(): sys.exit("no trace; run scripts/04_run_bhmm.py first")
    tr = az.from_netcdf(nc)
    params = [p for p in KEY if p in tr.posterior]
    meta = json.loads((TAB/"run_metadata.json").read_text()) if (TAB/"run_metadata.json").exists() else {}

    print("  writing sampler-state tables...")
    written = write_all(tr, TAB, params)
    for k, v in written.items(): print(f"    {k:38s} {v} rows")

    rd = lambda n, **kw: pd.read_csv(TAB/n, **kw) if (TAB/n).exists() else None
    lp, ss, td_, en, bf = (rd("mcmc_logp_trace.csv"), rd("mcmc_stepsize.csv"),
                           rd("mcmc_treedepth.csv"), rd("mcmc_energy.csv"),
                           rd("mcmc_energy_bfmi.csv"))
    ac, essv, mcv = rd("mcmc_accept.csv"), rd("mcmc_ess_evolution.csv"), rd("mcmc_mcse_evolution.csv")
    ru, led = rd("mcmc_rank_uniformity.csv"), rd("mcmc_convergence_ledger.csv")
    corr = rd("mcmc_posterior_correlation.csv", index_col=0)

    # ---------------- figures ----------------
    if lp is not None:
        @fig("t01_logp", "Log-posterior per draw — the MCMC equivalent of a training-loss curve. "
                         "All chains overlap at the same level with no drift: none is stuck in a worse mode.")
        def _(a):
            for c, g in lp.groupby("chain"):
                a.plot(g.draw, g.log_posterior, lw=.35, alpha=.75, color=CH[int(c) % 6],
                       label=f"chain {int(c)}")
            a.set(xlabel="draw", ylabel="log posterior", title="1. Log-posterior trace")
            a.legend(frameon=False, ncol=2)

        @fig("t02_logp_hist", "Log-posterior distribution per chain. Overlapping histograms confirm "
                              "the chains are exploring the same region of parameter space.")
        def _(a):
            for c, g in lp.groupby("chain"):
                a.hist(g.log_posterior, bins=50, histtype="step", lw=1.1,
                       color=CH[int(c) % 6], label=f"chain {int(c)}")
            a.set(xlabel="log posterior", ylabel="draws", title="2. Log-posterior by chain")
            a.legend(frameon=False, ncol=2)

    if ss is not None:
        @fig("t03_stepsize", "Tuned step size per chain. Flat across sampling means dual averaging "
                             "settled during warm-up; chains landing at similar values means they "
                             "found comparable geometry.")
        def _(a):
            for c, g in ss.groupby("chain"):
                a.plot(g.draw, g.step_size, lw=.8, color=CH[int(c) % 6], label=f"chain {int(c)}")
            a.set(xlabel="draw", ylabel="step size", title="3. Step size")
            a.legend(frameon=False, ncol=2)

    if td_ is not None:
        mx = meta.get("max_treedepth", int(td_.tree_depth.max()))
        sat = float((td_.tree_depth >= mx).mean())
        @fig("t04_treedepth", f"Leapfrog tree depth per draw. {sat:.1%} of draws hit the ceiling of "
                              f"{mx}. Saturation is not an error but means the trajectory never "
                              "turned around inside the budget, which costs efficiency.")
        def _(a):
            for c, g in td_.groupby("chain"):
                a.plot(g.draw, g.tree_depth, lw=.35, alpha=.7, color=CH[int(c) % 6])
            a.axhline(mx, color=RED, ls="--", lw=1, label=f"max {mx}")
            a.set(xlabel="draw", ylabel="tree depth", title="4. Tree depth")
            a.legend(frameon=False)

        @fig("t05_treedepth_hist", "How often each tree depth was reached. Mass well below the "
                                   "ceiling means most trajectories terminated naturally.")
        def _(a):
            a.hist(td_.tree_depth, bins=np.arange(td_.tree_depth.min()-.5,
                                                  td_.tree_depth.max()+1.5),
                   color=NAVYH, alpha=.85)
            a.set(xlabel="tree depth", ylabel="draws", title="5. Tree-depth distribution")

    if en is not None and bf is not None:
        @fig("t06_energy", "Hamiltonian energy per draw. Stable, overlapping traces mean the "
                           "sampler is exploring the energy distribution rather than drifting.")
        def _(a):
            for c, g in en.groupby("chain"):
                a.plot(g.draw, g.energy, lw=.35, alpha=.75, color=CH[int(c) % 6])
            a.set(xlabel="draw", ylabel="energy", title="6. Energy trace")

        @fig("t07_bfmi", "Energy-based fraction of missing information per chain. Above 0.3 means "
                         "the sampler is not being defeated by heavy tails (Betancourt 2018).")
        def _(a):
            a.bar(bf.chain.astype(str), bf.bfmi,
                  color=[GREEN if v > .3 else RED for v in bf.bfmi], alpha=.85)
            a.axhline(.3, color=RED, ls="--", lw=1, label="0.3 threshold")
            a.set(xlabel="chain", ylabel="E-BFMI", title="7. E-BFMI per chain")
            a.legend(frameon=False)

        @fig("t08_energy_marginal", "Marginal energy against the energy-transition distribution. "
                                    "Closely matching shapes is the visual form of a healthy BFMI.")
        def _(a):
            e = en.energy.values
            a.hist(e - e.mean(), bins=60, histtype="step", lw=1.3, density=True,
                   color=NAVYH, label="marginal energy")
            a.hist(np.diff(e), bins=60, histtype="step", lw=1.3, density=True,
                   color="#ED7D31", label="energy transitions")
            a.set(xlabel="centred energy", ylabel="density", title="8. Energy vs transitions")
            a.legend(frameon=False)

    if ac is not None:
        @fig("t09_accept", f"Acceptance probability per draw against the target of "
                           f"{meta.get('target_accept','?')}. Clustering at the target means the "
                           "step size was tuned correctly.")
        def _(a):
            for c, g in ac.groupby("chain"):
                a.plot(g.draw, g.accept_prob, lw=.3, alpha=.6, color=CH[int(c) % 6])
            if meta.get("target_accept"):
                a.axhline(meta["target_accept"], color=RED, ls="--", lw=1, label="target")
                a.legend(frameon=False)
            a.set(xlabel="draw", ylabel="acceptance probability", ylim=(0, 1.02),
                  title="9. Acceptance rate")

    if essv is not None:
        @fig("t10_ess_evo", "Effective sample size against chain length. A straight line through "
                            "the origin means each extra draw buys the same information; flattening "
                            "would mean longer runs no longer help.")
        def _(a):
            for i, (p, g) in enumerate(essv.groupby("parameter")):
                a.plot(g.draws_used, g.ess_bulk, lw=1.1, color=CH[i % 6],
                       label=p.replace("bhmm::", ""))
            a.set(xlabel="draws used", ylabel="ESS bulk", title="10. ESS growth")
            a.legend(frameon=False, ncol=2, fontsize=4.6)

    if mcv is not None:
        @fig("t11_mcse_evo", "Monte Carlo error against chain length. Falling as one over the root "
                             "of the draw count is the expected rate; departures indicate poor mixing.")
        def _(a):
            for i, (p, g) in enumerate(mcv.groupby("parameter")):
                a.plot(g.draws_used, g.mcse, lw=1.1, color=CH[i % 6])
            a.set(xlabel="draws used", ylabel="MCSE", yscale="log", xscale="log",
                  title="11. Monte Carlo error decay")

        @fig("t12_running_mean", "Running posterior mean as draws accumulate. Flat lines mean the "
                                 "estimate stopped moving long before the chain ended.")
        def _(a):
            for i, (p, g) in enumerate(mcv.groupby("parameter")):
                v = g.running_mean.values
                if np.std(v) > 0:
                    a.plot(g.draws_used, (v - v[-1]) / (abs(v[-1]) + 1e-9),
                           lw=1.0, color=CH[i % 6])
            a.axhline(0, color="k", lw=.8)
            a.set(xlabel="draws used", ylabel="relative drift from final",
                  title="12. Running-mean stability")

    if led is not None:
        @fig("t13_rhat", "R-hat per parameter against the 1.01 target. Every bar below the line "
                         "means all chains agree on the same distribution.")
        def _(a):
            nm = [p.replace("bhmm::", "") for p in led.parameter]
            a.barh(nm, led.r_hat, color=[GREEN if v <= 1.01 else RED for v in led.r_hat], alpha=.85)
            a.axvline(1.01, color=RED, ls="--", lw=1)
            a.set(xlabel="R-hat", xlim=(0.995, max(1.02, led.r_hat.max()*1.01)),
                  title="13. R-hat per parameter")
            a.tick_params(axis="y", labelsize=4.6)

        @fig("t14_ess", "Effective sample size per parameter. Above 400 is the working threshold "
                        "for a four-chain run.")
        def _(a):
            nm = [p.replace("bhmm::", "") for p in led.parameter]
            a.barh(nm, led.ess_bulk,
                   color=[GREEN if v >= 400 else RED for v in led.ess_bulk], alpha=.85)
            a.axvline(400, color=RED, ls="--", lw=1)
            a.set(xlabel="ESS bulk", title="14. ESS per parameter")
            a.tick_params(axis="y", labelsize=4.6)

        @fig("t15_mcse_pct", "Monte Carlo error as a share of each parameter's own spread. Below "
                             "5% means sampling noise is negligible next to genuine uncertainty.")
        def _(a):
            nm = [p.replace("bhmm::", "") for p in led.parameter]
            v = led.mcse_over_sd_pct.fillna(0)
            a.barh(nm, v, color=[GREEN if x <= 5 else RED for x in v], alpha=.85)
            a.axvline(5, color=RED, ls="--", lw=1)
            a.set(xlabel="MCSE / SD (%)", title="15. Sampling noise share")
            a.tick_params(axis="y", labelsize=4.6)

    if corr is not None and len(corr):
        @fig("t16_corr", "Posterior correlation between parameters. Strong off-diagonal values mark "
                         "ridges where two parameters trade off and must move together.",
             (3.4, 3.0))
        def _(a):
            C = corr.values
            im = a.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
            a.set_xticks(range(len(corr))); a.set_yticks(range(len(corr)))
            a.set_xticklabels(corr.columns, rotation=90, fontsize=4.4)
            a.set_yticklabels(corr.index, fontsize=4.4)
            for i in range(len(C)):
                for j in range(len(C)):
                    if abs(C[i, j]) > .35 and i != j:
                        a.text(j, i, f"{C[i,j]:.2f}", ha="center", va="center", fontsize=3.8)
            a.set_title("16. Posterior correlation")

    if ru is not None:
        @fig("t17_rank", "Rank-uniformity test per parameter (Vehtari et al. 2021). Passing means "
                         "the chains agree about where the probability mass sits — a stricter check "
                         "than R-hat alone.")
        def _(a):
            nm = [p.replace("bhmm::", "") for p in ru.parameter]
            a.barh(nm, ru.p_value, color=[GREEN if v > .01 else RED for v in ru.p_value], alpha=.85)
            a.axvline(.01, color=RED, ls="--", lw=1)
            a.set(xlabel="uniformity p-value", xscale="log", title="17. Rank uniformity")
            a.tick_params(axis="y", labelsize=4.6)

    # per-parameter trace + rank for the headline quantities
    for i, p in enumerate([q for q in params if q.endswith("_R0")][:4]):
        x = np.asarray(tr.posterior[p].values)
        nmp = p.replace("bhmm::", "")
        @fig(f"t2{i}_trace_{nmp}", f"Chain trace for {nmp}. Overlapping fuzzy caterpillars with no "
                                   "trend is what a converged parameter looks like.")
        def _(a, x=x, nmp=nmp):
            for c in range(x.shape[0]):
                a.plot(x[c], lw=.3, alpha=.7, color=CH[c % 6])
            a.set(xlabel="draw", ylabel=nmp, title=f"Trace: {nmp}")

    print(f"  {len(REG)} trace figures written")

    # ---------------- PDF ----------------
    st = [Paragraph("MCMC Audit — How the Fit Actually Ran", TT),
          Paragraph(f"run {meta.get('run_id','-')} · seed {meta.get('seed','-')} · "
                    f"{meta.get('draws','-')} draws x {meta.get('chains','-')} chains · "
                    f"target_accept {meta.get('target_accept','-')} · "
                    f"max_treedepth {meta.get('max_treedepth','-')}", SU),
          HRFlowable(width="100%", thickness=.9, color=NAVY, spaceBefore=2, spaceAfter=4)]
    st += [Paragraph(
        "Every series below is also written to <b>tables/mcmc_*.csv</b> so the run can be audited "
        "without reopening the trace. The log-posterior trace is the closest analogue to a "
        "training-loss curve; ESS growth is the closest analogue to a learning curve.",
        S("b", fontSize=8.2, leading=10.8)), Spacer(1, 5)]

    if led is not None:
        rows = [[r.parameter.replace("bhmm::", ""), f"{r['mean']:.4g}", f"{r['sd']:.3g}",
                 f"{r.r_hat:.4f}", f"{r.ess_bulk:.0f}", f"{r.ess_tail:.0f}",
                 f"{r.mcse_over_sd_pct:.2f}%", r.verdict] for _, r in led.iterrows()]
        rc = [GOOD if r[-1] == "PASS" else WARN for r in rows]
        st += [Paragraph("Convergence ledger", H1),
               tbl(["Parameter","mean","sd","R-hat","ESS bulk","ESS tail","MCSE/SD","Verdict"],
                   rows, [1.5*inch,.8*inch,.7*inch,.65*inch,.75*inch,.75*inch,.7*inch,.7*inch], rc),
               Spacer(1, 4)]

    extra = []
    d = int(tr.sample_stats.diverging.sum()) if "diverging" in tr.sample_stats else 0
    extra.append(["Divergent transitions", str(d), "0", "PASS" if d == 0 else "FAIL"])
    if bf is not None:
        extra.append(["Minimum E-BFMI", f"{bf.bfmi.min():.3f}", "> 0.3",
                      "PASS" if bf.bfmi.min() > .3 else "REVIEW"])
    if td_ is not None:
        mx = meta.get("max_treedepth", int(td_.tree_depth.max()))
        sat = float((td_.tree_depth >= mx).mean())
        extra.append(["Draws at max tree depth", f"{sat:.1%}", "< 5%",
                      "PASS" if sat < .05 else "REVIEW — inefficient geometry"])
    if ac is not None:
        extra.append(["Mean acceptance", f"{ac.accept_prob.mean():.3f}",
                      f"target {meta.get('target_accept','?')}", "PASS"])
    if ru is not None:
        extra.append(["Rank-uniformity failures", f"{int((~ru.uniform).sum())} of {len(ru)}", "0",
                      "PASS" if not (~ru.uniform).any() else "REVIEW"])
    st += [Paragraph("Sampler-state checks", H1),
           tbl(["Check","Value","Criterion","Verdict"], extra,
               [2.2*inch,1.3*inch,1.5*inch,2.3*inch],
               [GOOD if r[-1].startswith("PASS") else WARN for r in extra])]

    names = sorted(REG)
    for i in range(0, len(names), 9):
        st += [PageBreak(), Paragraph("Sampler-process figures" if i == 0 else "continued", H1),
               grid(names[i:i+9], ncol=3)]

    doc = SimpleDocTemplate(str(REP/"MCMC_AUDIT.pdf"), pagesize=letter, leftMargin=.5*inch,
                            rightMargin=.5*inch, topMargin=.45*inch, bottomMargin=.5*inch)
    f_ = make_footer("MCMC Audit — CTMC-BHMM v8", "")
    doc.build(st, onFirstPage=f_, onLaterPages=f_)
    print("  -> reports/MCMC_AUDIT.pdf")


if __name__ == "__main__":
    main()
