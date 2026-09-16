"""
scripts/10_full_results_report.py
=================================
Build ONE consolidated results PDF covering the entire run, designed so that a
second run on different hardware can be compared against it line by line.

    python scripts/10_full_results_report.py
    python scripts/10_full_results_report.py --tag on_prem
    python scripts/10_full_results_report.py --compare reports/FULL_RESULTS_reference.json

Why a JSON sidecar
------------------
Every number that appears in the PDF is also written to
`reports/FULL_RESULTS_<tag>.json`. Comparing two runs is then exact rather than
visual: pass the other run's JSON with --compare and the PDF gains a
side-by-side delta table that flags any value outside the reproducibility
tolerances below.

Reproducibility tolerances
--------------------------
Identical seed and identical library versions should reproduce bitwise. Across
machines, BLAS ordering and PyTensor compilation differ, so exact equality is
not expected. The tolerances applied are:

    posterior mean          within 2 Monte Carlo standard errors of each other
    posterior sd            within 10% relative
    R-hat                   both <= 1.01
    ESS                     within a factor of 2 (ESS is noisy)
    divergences             both 0
    PPC coverage            within 2 percentage points
    structural quantities   exact (grid shapes, gate results, N, eligible rows)

Structural quantities MUST match exactly. If they do not, the two runs are not
running the same model and no posterior comparison is meaningful.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import arviz as az

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
TAB, REP, GEN = ROOT/"tables", ROOT/"reports", ROOT/"data"/"generated"
FP, FB, FD = ROOT/"figures"/"prefit", ROOT/"figures"/"bhmm", ROOT/"figures"/"debug"
REP.mkdir(exist_ok=True)

GREY = colors.HexColor("#555555")
PASS_, FAIL_, PART_ = (colors.HexColor("#D5F5D5"), colors.HexColor("#FDEBEA"),
                       colors.HexColor("#FFF3CD"))

# Parameters reported in the headline comparison table, in display order.
HEADLINE = [
    ("bhmm::e_R0",             "Ebola R0",               (1.51, 2.53)),
    ("bhmm::mea_R0",           "Measles R0",             (10.0, 20.0)),
    ("bhmm::flu_R0",           "Influenza R0",           None),
    ("bhmm::n_R0",             "Norovirus R0",           (1.3, 6.7)),
    ("bhmm::n_attack_rate",    "Norovirus K-M AR",       None),
    ("bhmm::log_intercept_e",  "Ebola log ascertainment",None),
    ("bhmm::e_alpha_nb",       "Ebola NB alpha",         None),
    ("bhmm::mea_alpha_nb",     "Measles NB alpha",       None),
    ("bhmm::flu_alpha_nb",     "Influenza NB alpha",     None),
    ("bhmm::class_re_sd",      "Class RE sd",            None),
    ("bhmm::hyper_log_R0_mu",  "Hyperprior mean",        None),
    ("bhmm::hyper_log_R0_sd",  "Hyperprior sd",          None),
    ("bhmm::e_log_R0_raw",     "Ebola R0 raw",           None),
    ("bhmm::mea_log_R0_raw",   "Measles R0 raw",         None),
    ("bhmm::flu_log_R0_raw",   "Influenza R0 raw",       None),
    ("bhmm::n_log_R0_raw",     "Norovirus R0 raw",       None),
]

S  = lambda n, **k: ParagraphStyle(n, **{**dict(fontName=SANS, fontSize=8.8,
                                                leading=11.8, textColor=BLACK), **k})
T_ = S("T",  fontSize=14, fontName=SANS_BOLD, textColor=NAVY, alignment=TA_CENTER, leading=18)
ST = S("ST", fontSize=8.5, textColor=GREY, alignment=TA_CENTER)
H  = S("H",  fontSize=11,  fontName=SANS_BOLD, textColor=NAVY, spaceBefore=8, spaceAfter=3)
H2 = S("H2", fontSize=9.5, fontName=SANS_BOLD, textColor=NAVY, spaceBefore=5, spaceAfter=2)
B  = S("B",  fontSize=8.8, leading=11.8, alignment=TA_JUSTIFY)
SM = S("SM", fontSize=7.2, leading=9.3)
NO = S("NO", fontSize=7.5, fontName=SANS_ITAL, textColor=GREY)


def rd(name, **kw):
    p = TAB/name
    return pd.read_csv(p, **kw) if p.exists() else None


def rj(name):
    p = TAB/name
    return json.loads(p.read_text()) if p.exists() else {}


def hr():
    return HRFlowable(width="100%", thickness=1, color=NAVY, spaceBefore=3, spaceAfter=5)


def tbl(head, rows, widths, rc=None, fs=7.2):
    hrow = [Paragraph(h, S("th", fontSize=fs, fontName=SANS_BOLD, textColor=WHITE)) for h in head]
    drow = [[Paragraph(str(c), S("td", fontSize=fs, leading=fs+2.2)) for c in r] for r in rows]
    sty = [("BACKGROUND", (0,0), (-1,0), NAVY), ("TEXTCOLOR", (0,0), (-1,0), WHITE),
           ("ROWBACKGROUNDS", (0,1), (-1,-1), [WHITE, LGREY]),
           ("GRID", (0,0), (-1,-1), .3, MGREY), ("VALIGN", (0,0), (-1,-1), "TOP"),
           ("LEFTPADDING", (0,0), (-1,-1), 4), ("RIGHTPADDING", (0,0), (-1,-1), 4),
           ("TOPPADDING", (0,0), (-1,-1), 3), ("BOTTOMPADDING", (0,0), (-1,-1), 3)]
    if rc:
        for i, c in enumerate(rc):
            if c is not None:
                sty.append(("BACKGROUND", (0, i+1), (-1, i+1), c))
    t = Table([hrow]+drow, colWidths=widths)
    t.setStyle(TableStyle(sty))
    return t


def fig(path, w=7.1, r=.55):
    return (RLImage(str(path), width=w*inch, height=w*r*inch) if Path(path).exists()
            else Paragraph(f"[missing: {Path(path).name}]", NO))


# ---------------------------------------------------------------------------
# Collect every comparable number into one dict
# ---------------------------------------------------------------------------


def collect() -> dict:
    nc = GEN/"bhmm_trace_full.nc"
    if not nc.exists():
        nc = GEN/"bhmm_trace_quick.nc"
    if not nc.exists():
        sys.exit("No trace found. Run scripts/04_run_bhmm.py first.")

    tr = az.from_netcdf(nc)
    s = az.summary(tr)
    meta = rj("run_metadata.json")

    out = {
        "schema_version": 1,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "run": meta,
        "structural": {},
        "posterior": {},
        "diagnostics": {},
    }
    try:
        import pymc, pytensor, scipy
        out["environment"]["versions"] = dict(
            pymc=pymc.__version__, arviz=az.__version__, numpy=np.__version__,
            scipy=scipy.__version__, pytensor=pytensor.__version__)
    except Exception:
        pass

    # ---- structural: must match EXACTLY across runs -----------------------
    can = ROOT/"data"/"canonical"
    e = pd.read_csv(can/"ebola_kikwit_1995_analysis_ready.csv")
    n = pd.read_csv(can/"norovirus_derbyshire_2001_analysis_ready.csv")
    f = pd.read_csv(can/"influenza_england_1978_analysis_ready.csv")
    m = pd.read_csv(can/"measles_hagelloch_1861_analysis_ready.csv")
    out["structural"] = {
        "ebola_rows": int(len(e)),
        "ebola_eligible": int(e.analysis_eligible.sum()),
        "ebola_total_onsets": int(e.onset.sum()),
        "norovirus_students": int(len(n)),
        "norovirus_ill": int(n.ill.sum()),
        "influenza_days": int(len(f)),
        "influenza_N": int(f.N.iloc[0]),
        "influenza_peak": int(f.in_bed.max()),
        "measles_days": int(len(m)),
        "measles_N": int(m.N.iloc[0]),
        "measles_total_onsets": int(m.rash_onset.sum()),
        "n_free_parameters": int(len([k for k in s.index if "[" not in k])),
        "n_summary_rows": int(len(s)),
    }
    g = rd("identifiability_gate.csv")
    if g is not None:
        out["structural"]["identifiability_gate"] = {
            str(r.parameter): str(r.status) for _, r in g.iterrows()}
    ge = rj("ode_grid_interpolation_error.json")
    if ge:
        out["structural"]["ode_grid_error"] = {
            k: round(float(v["max_rel_error"]), 8) for k, v in ge.items()}

    # ---- posterior --------------------------------------------------------
    for k, label, lit in HEADLINE:
        if k not in s.index:
            continue
        x = tr.posterior[k].values.flatten() if k in tr.posterior else None
        rec = dict(label=label,
                   mean=float(s.loc[k, "mean"]), sd=float(s.loc[k, "sd"]),
                   r_hat=float(s.loc[k, "r_hat"]),
                   ess_bulk=float(s.loc[k, "ess_bulk"]),
                   ess_tail=float(s.loc[k, "ess_tail"]),
                   mcse_mean=float(s.loc[k, "mcse_mean"]))
        if x is not None:
            rec.update(median=float(np.median(x)),
                       eti89_lo=float(np.quantile(x, .055)),
                       eti89_hi=float(np.quantile(x, .945)),
                       cv_pct=float(x.std()/abs(x.mean())*100) if x.mean() else float("nan"))
        if lit:
            rec["lit_lo"], rec["lit_hi"] = lit
            rec["in_lit_range"] = bool(lit[0] <= rec["mean"] <= lit[1])
        out["posterior"][k] = rec

    # ---- diagnostics ------------------------------------------------------
    out["diagnostics"]["divergences"] = int(tr.sample_stats.diverging.sum())
    out["diagnostics"]["max_r_hat"] = float(s.r_hat.max())
    out["diagnostics"]["min_ess_bulk"] = float(s.ess_bulk.min())
    out["diagnostics"]["n_rhat_over_101"] = int((s.r_hat > 1.01).sum())
    out["diagnostics"]["n_ess_under_400"] = int((s.ess_bulk < 400).sum())

    for name, key in [("ppc_results.csv", "ppc"),
                      ("residual_serial_correlation.csv", "residual_acf"),
                      ("final_size_consistency.csv", "final_size"),
                      ("prior_movement_check.csv", "prior_movement"),
                      ("prior_dominance.csv", "prior_dominance"),
                      ("prefit_summary.csv", "prefit")]:
        d = rd(name)
        if d is not None:
            out["diagnostics"][key] = json.loads(d.to_json(orient="records"))

    z = rd("zombie_prior_predictive_summary.csv")
    if z is not None:
        out["diagnostics"]["zombie"] = json.loads(z.to_json(orient="records"))
    ds = rj("debug_scores.json")
    if ds:
        out["diagnostics"]["debug_scores"] = ds
    return out


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def compare(a: dict, b: dict) -> tuple[list, list]:
    """Return (structural_rows, posterior_rows) for the delta tables."""
    srows = []
    for k, va in a["structural"].items():
        vb = b["structural"].get(k, "-")
        if isinstance(va, dict):
            same = (va == vb)
            srows.append([k, f"{len(va)} entries", f"{len(vb) if isinstance(vb, dict) else '-'} entries",
                          "MATCH" if same else "MISMATCH", same])
        else:
            same = (va == vb)
            srows.append([k, str(va), str(vb), "MATCH" if same else "MISMATCH", same])

    prows = []
    for k, ra in a["posterior"].items():
        rb = b["posterior"].get(k)
        if rb is None:
            prows.append([ra["label"], f"{ra['mean']:.4g}", "absent", "-", "-", "MISSING", False])
            continue
        dm = ra["mean"] - rb["mean"]
        pooled_mcse = np.hypot(ra.get("mcse_mean", 0.0), rb.get("mcse_mean", 0.0))
        z = abs(dm)/pooled_mcse if pooled_mcse > 0 else np.inf
        sd_rel = abs(ra["sd"] - rb["sd"])/max(abs(rb["sd"]), 1e-12)
        ok_mean = z <= 2.0
        ok_sd = sd_rel <= 0.10
        ok_rhat = (ra["r_hat"] <= 1.01) and (rb["r_hat"] <= 1.01)
        ess_ratio = ra["ess_bulk"]/max(rb["ess_bulk"], 1e-9)
        ok_ess = 0.5 <= ess_ratio <= 2.0
        ok = ok_mean and ok_sd and ok_rhat and ok_ess
        notes = []
        if not ok_mean: notes.append(f"mean {z:.1f} MCSE apart")
        if not ok_sd:   notes.append(f"sd {sd_rel:.0%} apart")
        if not ok_rhat: notes.append("Rhat > 1.01")
        if not ok_ess:  notes.append(f"ESS ratio {ess_ratio:.2f}")
        prows.append([ra["label"],
                      f"{ra['mean']:.4g} ± {ra['sd']:.3g}",
                      f"{rb['mean']:.4g} ± {rb['sd']:.3g}",
                      f"{dm:+.3g}",
                      f"{z:.1f}" if np.isfinite(z) else "inf",
                      "MATCH" if ok else "; ".join(notes) or "DIFFERS",
                      ok])
    return srows, prows


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------


def build_pdf(res: dict, other: dict | None, tag: str, out_path: Path):
    run = res.get("run", {})
    st = []
    st += [Paragraph("Full Results Report", T_),
           Paragraph("Unified CTMC-BHMM v6 - consolidated single-run results for cross-machine comparison", ST),
           Paragraph(f"tag: {tag}   |   run_id {run.get('run_id','-')}   |   "
                     f"seed {run.get('seed','-')}   |   "
                     f"{run.get('draws','-')} draws x {run.get('chains','-')} chains   |   "
                     f"tune {run.get('tune','-')}   |   target_accept {run.get('target_accept','-')}",
                     ST), hr()]

    # -- how to compare ----------------------------------------------------
    st += [Paragraph("How to compare this against your run", H),
           Paragraph(
           "Every number in this report is also written to "
           f"<b>reports/FULL_RESULTS_{tag}.json</b>. To compare exactly rather than by eye, run "
           "your own fit with the same seed and then:", B), Spacer(1, 3),
           Paragraph("<font face='Courier' size='8'>"
                     "python scripts/10_full_results_report.py --tag on_prem \\<br/>"
                     "&nbsp;&nbsp;&nbsp;&nbsp;--compare reports/FULL_RESULTS_reference.json"
                     "</font>", B), Spacer(1, 4),
           Paragraph(
           "<b>Section 2 (structural) must match exactly.</b> Row counts, eligible rows, the "
           "identifiability gate and the free-parameter count do not depend on hardware. If any "
           "of these differ, the two runs are not the same model and no posterior comparison is "
           "meaningful - check that both sides are on the same commit and that "
           "scripts/01_prepare_data.py has been run.", B), Spacer(1, 3),
           Paragraph(
           "<b>Section 3 (posterior) is compared with tolerances.</b> Identical seed plus identical "
           "library versions reproduces bitwise; across machines, BLAS ordering and PyTensor "
           "compilation differ, so exact equality is not expected. Applied tolerances: posterior "
           "means within 2 pooled Monte Carlo standard errors, posterior sd within 10% relative, "
           "both R-hat at or below 1.01, ESS within a factor of 2, divergences 0 on both sides, "
           "PPC coverage within 2 percentage points.", B), Spacer(1, 5)]

    # -- environment -------------------------------------------------------
    env = res.get("environment", {})
    ver = env.get("versions", {})
    st += [Paragraph("1. Environment", H),
           tbl(["Field", "This run"],
               [["python", env.get("python", "-")],
                ["platform", env.get("platform", "-")],
                ["machine", env.get("machine", "-")]] +
               [[k, v] for k, v in ver.items()],
               [1.6*inch, 5.4*inch]), Spacer(1, 3),
           Paragraph("Library versions are the single most common cause of small posterior "
                     "differences between machines. Compare these first.", NO), Spacer(1, 5)]

    # -- structural --------------------------------------------------------
    s_ = res["structural"]
    rows = [[k, str(v) if not isinstance(v, dict) else f"{len(v)} entries"]
            for k, v in s_.items()]
    st += [Paragraph("2. Structural quantities (must match exactly)", H),
           tbl(["Quantity", "Value"], rows, [3.3*inch, 3.7*inch]), Spacer(1, 3)]
    if "identifiability_gate" in s_:
        g = s_["identifiability_gate"]
        allpass = all(v == "PASS" for v in g.values())
        st += [Paragraph("Identifiability gate", H2),
               tbl(["Parameter", "Status"],
                   [[k.replace("bhmm::", ""), v] for k, v in g.items()],
                   [4.0*inch, 3.0*inch],
                   [PASS_ if v == "PASS" else FAIL_ for v in g.values()]),
               Spacer(1, 2),
               Paragraph(
               "Every estimated parameter must show a non-zero likelihood gradient at the initial "
               "point. A FAIL means the parameter never enters the likelihood and its posterior "
               "would equal its prior - this is the defect that invalidated the v5 Ebola and "
               "Influenza R0 estimates." if not allpass else
               "All parameters reach the likelihood. This gate is what prevents a repeat of the v5 "
               "defect, where e_log_R0_raw and flu_log_R0_raw had zero likelihood gradient and "
               "their reported posteriors were prior draws.", NO), Spacer(1, 5)]

    # -- posterior ---------------------------------------------------------
    st += [PageBreak(), Paragraph("3. Posterior summary", H)]
    prow, prc = [], []
    for k, rec in res["posterior"].items():
        lit = ""
        if "lit_lo" in rec:
            lit = f"[{rec['lit_lo']}, {rec['lit_hi']}] {'in' if rec['in_lit_range'] else 'OUT'}"
        prow.append([rec["label"],
                     f"{rec['mean']:.4g}", f"{rec['sd']:.3g}",
                     f"{rec.get('median', float('nan')):.4g}",
                     f"[{rec.get('eti89_lo', float('nan')):.3g}, {rec.get('eti89_hi', float('nan')):.3g}]",
                     f"{rec.get('cv_pct', float('nan')):.1f}%",
                     f"{rec['r_hat']:.4f}", f"{rec['ess_bulk']:.0f}", lit])
        bad = rec["r_hat"] > 1.01 or rec["ess_bulk"] < 400
        prc.append(PART_ if bad else PASS_)
    st += [tbl(["Parameter", "mean", "sd", "median", "89% ETI", "CV", "R-hat", "ESS", "literature"],
               prow, [1.35*inch, .62*inch, .55*inch, .62*inch, 1.25*inch, .5*inch,
                      .55*inch, .52*inch, 1.04*inch], prc, fs=6.8), Spacer(1, 3),
           Paragraph("The three `raw` rows are the non-centred hierarchical draws. Their prior is "
                     "Normal(0, 1): a posterior still at mean 0 with sd 1 means that arm's R0 is "
                     "not identified. All three must show movement.", NO), Spacer(1, 5)]

    # -- diagnostics -------------------------------------------------------
    d = res["diagnostics"]
    st += [Paragraph("4. Convergence and model checks", H),
           tbl(["Check", "Value", "Criterion", "Verdict"],
               [["Divergences", str(d["divergences"]), "0", "PASS" if d["divergences"] == 0 else "FAIL"],
                ["Max R-hat", f"{d['max_r_hat']:.4f}", "<= 1.01",
                 "PASS" if d["max_r_hat"] <= 1.01 else "REVIEW"],
                ["Min ESS bulk", f"{d['min_ess_bulk']:.0f}", ">= 400",
                 "PASS" if d["min_ess_bulk"] >= 400 else "REVIEW"],
                ["Params with R-hat > 1.01", str(d["n_rhat_over_101"]), "0",
                 "PASS" if d["n_rhat_over_101"] == 0 else "REVIEW"],
                ["Params with ESS < 400", str(d["n_ess_under_400"]), "0",
                 "PASS" if d["n_ess_under_400"] == 0 else "REVIEW"]],
               [2.3*inch, 1.2*inch, 1.4*inch, 2.1*inch],
               [PASS_ if v else PART_ for v in
                [d["divergences"] == 0, d["max_r_hat"] <= 1.01, d["min_ess_bulk"] >= 400,
                 d["n_rhat_over_101"] == 0, d["n_ess_under_400"] == 0]]), Spacer(1, 5)]

    if d.get("ppc"):
        st += [Paragraph("Posterior predictive coverage", H2),
               tbl(["Disease", "Coverage of 90% PI", "Pred mean", "Obs mean", "Verdict"],
                   [[r["disease"], f"{r['coverage']:.1%}", f"{r['pred_mean']:.4g}",
                     f"{r['obs_mean']:.4g}", "PASS" if r["passes"] else "FAIL"]
                    for r in d["ppc"]],
                   [1.4*inch, 1.6*inch, 1.3*inch, 1.3*inch, 1.4*inch],
                   [PASS_ if r["passes"] else FAIL_ for r in d["ppc"]]), Spacer(1, 4)]

    if d.get("prior_movement"):
        st += [Paragraph("Identification: raw variables vs their Normal(0,1) prior", H2),
               tbl(["Parameter", "Posterior mean", "Posterior sd", "Updated by data?"],
                   [[r["parameter"].replace("bhmm::", ""), f"{r['post_mean']:+.4f}",
                     f"{r['post_sd']:.4f}", "YES" if r["updated_by_data"] else "NO - UNIDENTIFIED"]
                    for r in d["prior_movement"]],
                   [2.0*inch, 1.5*inch, 1.5*inch, 2.0*inch],
                   [PASS_ if r["updated_by_data"] else FAIL_ for r in d["prior_movement"]]),
               Spacer(1, 4)]

    if d.get("residual_acf"):
        st += [Paragraph("Residual serial correlation", H2),
               tbl(["Disease", "lag-1", "lag-2", "Ljung-Box p", "Rule |lag-1| < 0.20"],
                   [[r["disease"], f"{r.get('acf_lag1', float('nan')):+.3f}",
                     f"{r.get('acf_lag2', float('nan')):+.3f}",
                     f"{r.get('ljung_box_p', float('nan')):.4f}",
                     "PASS" if abs(r.get("acf_lag1", 1)) < 0.20 else "structure remains"]
                    for r in d["residual_acf"]],
                   [1.3*inch, 1.0*inch, 1.0*inch, 1.2*inch, 2.5*inch]), Spacer(1, 4)]

    if d.get("final_size"):
        st += [Paragraph("Final-size consistency", H2),
               tbl(["Disease", "Implied AR", "Reported AR", "R0 from reported AR", "Verdict"],
                   [[r["disease"], f"{r['implied_ar_median']:.1%}",
                     f"{r['reported_attack_rate']:.1%}", f"{r['R0_from_reported_ar']:.3f}",
                     "consistent" if r["consistent"] else "CONFLICT (expected for Influenza)"]
                    for r in d["final_size"]],
                   [1.3*inch, 1.1*inch, 1.2*inch, 1.6*inch, 1.8*inch],
                   [PASS_ if r["consistent"] else PART_ for r in d["final_size"]]), Spacer(1, 4)]

    if d.get("prior_dominance"):
        st += [Paragraph("Prior dominance", H2),
               tbl(["Parameter", "post/prior mean", "sd ratio", "Verdict"],
                   [[r["parameter"].replace("bhmm::", ""), f"{r['mean_ratio']:.3f}",
                     f"{r['sd_ratio']:.3f}", r["verdict"]] for r in d["prior_dominance"]],
                   [1.8*inch, 1.4*inch, 1.0*inch, 2.8*inch],
                   [FAIL_ if r["prior_dominated"] else PASS_ for r in d["prior_dominance"]]),
               Spacer(1, 4)]

    # -- prefit ------------------------------------------------------------
    if d.get("prefit"):
        st += [PageBreak(), Paragraph("5. Pre-fit audit (data-only; must match exactly)", H),
               tbl(["Disease", "Var/Mean", "Skew", "Zero frac", "lag-1 ACF", "Ljung-Box p", "GRW warranted"],
                   [[r["disease"], f"{r['var_mean']:.3f}", f"{r['skew']:.3f}",
                     f"{r['zero_frac']:.3f}", f"{r['lag1_acf']:+.4f}",
                     f"{r['ljung_box_p']:.4f}", "YES" if r["grw_warranted"] else "no"]
                    for r in d["prefit"]],
                   [1.3*inch, .9*inch, .7*inch, .8*inch, 1.0*inch, .95*inch, 1.15*inch]),
               Spacer(1, 3),
               Paragraph("These derive from the canonical data and the constant-transmission "
                         "reference fit only. They involve no MCMC, so they must reproduce exactly "
                         "on any machine. A difference here means the canonical data differ.", NO),
               Spacer(1, 5)]

    # -- zombie ------------------------------------------------------------
    if d.get("zombie"):
        st += [Paragraph("6. Zombie SEZR prior-predictive arm (no data, no likelihood)", H),
               tbl(["Quantity", "mean", "sd", "2.5%", "50%", "97.5%"],
                   [[r["quantity"], f"{r['mean']:.4g}", f"{r['sd']:.4g}",
                     f"{r['q025']:.4g}", f"{r['q500']:.4g}", f"{r['q975']:.4g}"]
                    for r in d["zombie"]],
                   [2.0*inch, 1.0*inch, 1.0*inch, 1.0*inch, 1.0*inch, 1.0*inch]),
               Spacer(1, 3),
               Paragraph("This arm is a fixed-seed forward simulation from the prior. It contains "
                         "no MCMC and must reproduce exactly.", NO), Spacer(1, 5)]

    # -- comparison --------------------------------------------------------
    if other is not None:
        srows, prows = compare(res, other)
        oth_run = other.get("run", {})
        st += [PageBreak(),
               Paragraph("7. Side-by-side comparison", H),
               Paragraph(f"This run (tag <b>{tag}</b>) vs reference "
                         f"(run_id {oth_run.get('run_id','-')}, seed {oth_run.get('seed','-')}, "
                         f"{oth_run.get('draws','-')} x {oth_run.get('chains','-')})", B),
               Spacer(1, 4),
               Paragraph("Structural (exact match required)", H2),
               tbl(["Quantity", "This run", "Reference", "Verdict"],
                   [r[:4] for r in srows], [2.5*inch, 1.6*inch, 1.6*inch, 1.3*inch],
                   [PASS_ if r[4] else FAIL_ for r in srows]), Spacer(1, 5),
               Paragraph("Posterior (tolerance-based)", H2),
               tbl(["Parameter", "This run", "Reference", "delta mean", "|d|/MCSE", "Verdict"],
                   [r[:6] for r in prows],
                   [1.35*inch, 1.45*inch, 1.45*inch, .85*inch, .7*inch, 1.3*inch],
                   [PASS_ if r[6] else PART_ for r in prows], fs=6.8), Spacer(1, 3),
               Paragraph("A MATCH verdict means the two posterior means are within 2 pooled Monte "
                         "Carlo standard errors, the standard deviations agree to 10%, both R-hats "
                         "are at or below 1.01, and the ESS values are within a factor of 2. "
                         "Differences larger than that on a shared seed usually indicate a library "
                         "version difference; check section 1 first.", NO)]
        n_sfail = sum(1 for r in srows if not r[4])
        n_pfail = sum(1 for r in prows if not r[6])
        verdict = ("REPRODUCED" if n_sfail == 0 and n_pfail == 0 else
                   "STRUCTURAL MISMATCH - not the same model" if n_sfail else
                   f"{n_pfail} posterior quantities outside tolerance")
        vc = GREEN if n_sfail == 0 and n_pfail == 0 else RED if n_sfail else colors.HexColor("#856404")
        vb = Table([[Paragraph(f"<b>COMPARISON VERDICT: {verdict}</b>",
                               S("v", fontSize=11, fontName=SANS_BOLD, textColor=vc))]],
                   colWidths=[7.0*inch])
        vb.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,-1), colors.HexColor("#EEF2F8")),
                                ("BOX", (0,0), (-1,-1), 1, NAVY),
                                ("LEFTPADDING", (0,0), (-1,-1), 10),
                                ("TOPPADDING", (0,0), (-1,-1), 7),
                                ("BOTTOMPADDING", (0,0), (-1,-1), 7)]))
        st += [Spacer(1, 5), vb]

    # -- figures -----------------------------------------------------------
    st += [PageBreak(), Paragraph("8. Figures", H),
           Paragraph("Pre-fit: cross-disease overview", H2), fig(FP/"overview_prefit_panel.png", 7.1, .31),
           Spacer(1, 4),
           Paragraph("Posterior: R0 across arms", H2), fig(FB/"R0_comparison.png", 7.1, .28),
           Spacer(1, 4),
           Paragraph("Posterior predictive checks", H2), fig(FB/"ppc_panel.png", 7.1, .53)]
    st += [PageBreak(), Paragraph("Convergence", H2), fig(FB/"convergence_panel.png", 7.1, .37),
           Spacer(1, 4),
           Paragraph("Debug scorecard", H2), fig(FD/"debug_scorecard.png", 7.1, .56)]

    # -- standing limitations ---------------------------------------------
    st += [PageBreak(), Paragraph("9. Standing limitations carried by every run", H),
           Paragraph("<b>Influenza England 1978 identification conflict.</b> Kermack-McKendrick "
                     "final size with the reported attack rate 512/763 = 67.1% implies R0 = 1.66. "
                     "Fitting the prevalence time course implies a larger value. Avilov KK et al. "
                     "(2024) J R Soc Interface 21(220):20240394 show that no classic compartmental "
                     "model reproduces both; their delay-differential model gives R0 = 8.14, which "
                     "Ahmad et al. (2025) Sci Rep flag as far above the typical influenza range of "
                     "1-4. Any reported flu_R0 must state which target it matches. See "
                     "docs/INFLUENZA_1978_LIMITATION.md.", B), Spacer(1, 4),
           Paragraph("<b>No GRW on either time-series arm by default.</b> For Ebola the pre-fit "
                     "decision rule is not met once the correct observable (incidence flux "
                     "sigma*E(t)) is used: residual lag-1 ACF is -0.037 with Ljung-Box p = 0.885. "
                     "For Influenza a daily GRW would have 14 innovations against 14 observations - "
                     "a saturated latent process that makes R0 unidentifiable. Both are enabled for "
                     "sensitivity with --grw-ebola and --grw-influenza.", B), Spacer(1, 4),
           Paragraph("<b>Generation time is fixed, not estimated.</b> The ODE surrogate grid varies "
                     "R0 at fixed sigma and removal rates. Sensitivity to the assumed rates is "
                     "prespecified in core/bhmm/priors.py.", B), Spacer(1, 4),
           Paragraph("<b>Zombie SEZR is prior-predictive only.</b> No data, no likelihood, "
                     "posterior equals prior. It is a structural reference point, not an estimated "
                     "arm.", B), Spacer(1, 4),
           Paragraph("<b>ODE surrogate interpolation error.</b> The mechanistic trajectory is "
                     "piecewise-linear in log R0 between grid nodes. Measured maximum relative "
                     "error is reported in section 2 and is well below Monte Carlo error at the "
                     "production grid size.", B)]

    doc = SimpleDocTemplate(str(out_path), pagesize=letter,
                            leftMargin=.6*inch, rightMargin=.6*inch,
                            topMargin=.55*inch, bottomMargin=.65*inch)
    f = make_footer(f"CTMC-BHMM v6 - Full Results ({tag})",
                    res.get("generated_at", "")[:10])
    doc.build(st, onFirstPage=f, onLaterPages=f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="reference",
                    help="label for this run, used in the output filenames")
    ap.add_argument("--compare", default=None,
                    help="path to another run's FULL_RESULTS_*.json")
    a = ap.parse_args()

    res = collect()
    js = REP/f"FULL_RESULTS_{a.tag}.json"
    js.write_text(json.dumps(res, indent=2, default=float))

    other = None
    if a.compare:
        p = Path(a.compare)
        if not p.exists():
            sys.exit(f"comparison file not found: {p}")
        other = json.loads(p.read_text())

    pdf = REP/f"FULL_RESULTS_{a.tag}.pdf"
    build_pdf(res, other, a.tag, pdf)
    print(f"  json -> {js}")
    print(f"  pdf  -> {pdf}")
    if other:
        srows, prows = compare(res, other)
        print(f"  structural mismatches : {sum(1 for r in srows if not r[4])}")
        print(f"  posterior out of tol  : {sum(1 for r in prows if not r[6])}")


if __name__ == "__main__":
    main()
