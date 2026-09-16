"""
core/bhmm/priors.py
===================
Complete prior specification for the unified BHMM (v6).

Every prior carries an explicit epistemic label:

    LITERATURE      value traceable to a cited publication
    STRUCTURAL      fixed by model well-posedness (positivity, conservation)
    WEAK            weakly informative regularisation, justified by prior
                    predictive checking rather than by a single source
    DISCLOSURE      deliberately non-informative because the parameter is
                    known to be unidentified; the width is a disclosure
                    choice, not a calibration

This file is the pre-registration document for the analysis. Nothing about the
prior specification is implicit or set elsewhere.

-------------------------------------------------------------------------------
v6 CHANGES vs v5 (all documented in docs/PARAMETER_CHANGE_AUDIT.md)
-------------------------------------------------------------------------------
1. INFLUENZA_PRIOR restructured from SEIR-prevalence to SEIQR-confinement.
   The 1978 boarding-school series counts boys *confined to bed*, which is not
   the infectious compartment. Treating it as I(t) is a documented modelling
   error:
       Kalachev L et al., discussed in Avilov KK et al. (2024),
       J R Soc Interface 21:20240394, sec. 3 -- "it is erroneous to interpret
       the 'confined-to-bed' persons (B) as the 'infected and infectious' (I)
       in the classical SEIR model".
   v6 maps in_bed to a confinement compartment Q downstream of I.

2. Seed offset t_seed added. The BMJ record begins on 22 Jan 1978, but the
   index case was infected around 10 Jan and febrile 15-18 Jan (RECON
   `influenza_england_1978_school` documentation; BMJ 1978;1(6112):587).
   Forcing model time 0 to coincide with observation day 0 is what produced the
   v5 timing failure: the fitted ODE peaked at day 36 while the data peaks at
   day 6.

3. The sigma_rw priors are no longer justified by an author-derived formula
   presented as a textbook lemma. See `docs/PRIOR_CALIBRATION.md` and
   scripts/03_prior_predictive_calibration.py -- the justification is now prior
   predictive checking, which is a standard and citable procedure.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Literal, Optional


# ===========================================================================
# Disease configuration
# ===========================================================================


@dataclass
class DiseaseConfig:
    """
    Complete structural + prior specification for one disease arm.

    All rate parameters are in day^-1 regardless of the native resolution of
    the data, so that R0, generation times and GRW innovation SDs are directly
    comparable across arms.
    """

    name: str

    # ---- mechanistic structure -------------------------------------------
    structure: Literal["seird_incidence", "seiqr_prevalence", "final_size"]
    structure_source: str

    # ---- rate parameters (fixed at literature means; see module docstring) --
    sigma: float                 # E -> I rate (1/latent period)
    sigma_source: str
    removal: float               # I -> next compartment rate; R0 = beta/removal
    removal_source: str
    gamma_q: float = 0.0         # Q -> R rate (confinement duration); SEIQR only
    gamma_q_source: str = "N/A"
    # Erlang shape parameters (linear chain trick). k=1 => exponential.
    kE: int = 1
    kI: int = 1
    kQ: int = 1
    erlang_source: str = "STRUCTURAL: k=1 (exponential residence times) in the primary analysis"

    # ---- population and seeding ------------------------------------------
    N: int = 1000
    N_source: str = ""
    I0: float = 1.0
    t_seed: float = 0.0          # days between epidemic seed and observation day 0
    t_seed_source: str = ""

    # ---- CFR -------------------------------------------------------------
    mu_point: float = 0.0
    mu_alpha: float = 1.0
    mu_beta: float = 1.0
    mu_source: str = ""

    # ---- literature R0 reference (used for reporting + calibration inputs) --
    R0_lit_lo: Optional[float] = None
    R0_lit_hi: Optional[float] = None
    R0_lit_source: str = ""

    # ---- observation-model priors ----------------------------------------
    nb_alpha_scale: float = 3.0            # HalfNormal scale for NB dispersion
    nb_alpha_label: str = "WEAK"
    nb_alpha_source: str = ""

    log_intercept_mu: float = 0.0          # Normal prior on log ascertainment
    log_intercept_sd: float = 0.5
    log_intercept_source: str = ""
    # Estimate an ascertainment intercept at all? Only meaningful when the
    # observation process can miss cases. A complete census cannot, and
    # estimating an intercept against a census creates a ridge with R0.
    estimate_intercept: bool = True
    ascertainment_source: str = ""

    sigma_rw_scale: float = 0.15           # HalfNormal scale for GRW innovation SD
    sigma_rw_label: str = "WEAK"
    sigma_rw_source: str = ""

    # ---- ODE surrogate grid bounds ---------------------------------------
    # The grid must comfortably contain every R0 the posterior can reach.
    # If the posterior presses against a bound the interpolation clips and the
    # gradient flattens, silently re-creating the v5 non-identifiability.
    # scripts/04 asserts the posterior stays inside these bounds.
    grid_R0_lo: float = 0.3
    grid_R0_hi: float = 15.0

    # ---- prespecified sensitivity grid -----------------------------------
    sensitivity: Dict[str, list] = field(default_factory=dict)


# ===========================================================================
# Ebola -- Kikwit, DRC, 1995
# ===========================================================================
EBOLA = DiseaseConfig(
    name="Ebola_Kikwit_1995",

    # Observable is the daily count of NEW symptom onsets: an incidence flux,
    # not a prevalence. v5 compared it to I(t), which has the wrong shape
    # (I(t) integrates onsets and lags them by the infectious period).
    # The correct observable is the flux into the symptomatic compartment,
    # sigma * E(t).
    structure="seird_incidence",
    structure_source=(
        "LITERATURE: Legrand J et al. (2007) Epidemics 21(3):183-197 -- "
        "SEIHFR structure for Ebola with isolation; we use the reduced SEIRD "
        "form with isolation folded into the removal rate. Observable is daily "
        "onset incidence (Khan AS et al. (1999) J Infect Dis 179:S76-86)."
    ),

    # sigma: WHO Ebola fact sheet 2-21 day incubation; Legrand 2007 uses 10 d.
    # 1/9 d^-1 = 0.1111.
    sigma=1.0 / 9.0,
    sigma_source="LITERATURE: WHO Ebola fact sheet (2-21 d incubation); Legrand et al. (2007) 10 d mean",

    # removal: Legrand 2007 Table 2 gives gamma_I = 1/6 d^-1 in the community.
    # Kikwit had active case isolation from late May 1995 (delta approx 1/7 d^-1
    # once the international team arrived), so effective removal is faster than
    # gamma alone. We use the community value 1/6 as the reference removal rate
    # and let the GRW absorb the intervention-driven change in transmission.
    removal=1.0 / 6.0,
    removal_source=(
        "LITERATURE: Legrand et al. (2007) Table 2, gamma_I = 0.167 d^-1; "
        "WHO 4-9 d symptomatic period. Post-intervention isolation is absorbed "
        "by the GRW on log beta(t) rather than by a time-varying delta."
    ),

    N=1000,
    N_source=(
        "STRUCTURAL: Kikwit General Hospital catchment. The absolute value is "
        "weakly identified; log_intercept_e absorbs the scale. Sensitivity over "
        "N in {500, 1000, 2000} is prespecified."
    ),
    I0=1.0,
    t_seed=0.0,
    t_seed_source="STRUCTURAL: day 0 = first recorded onset (6 Jan 1995); no seed offset applied",

    mu_point=0.808, mu_alpha=12.0, mu_beta=3.0,
    mu_source="LITERATURE: Khan et al. (1999) CFR 80.6%; RECON canonical 236/292 = 80.8%",

    R0_lit_lo=1.51, R0_lit_hi=2.53,
    R0_lit_source="LITERATURE: Althaus CL (2014) PLOS Curr Outbreaks, R0 = 1.51-2.53 (central 1.83)",

    # Pre-fit Var/Mean = 2.60 on analysis-eligible rows. Calibrated to the data
    # being fitted, therefore STATIC and labelled as such -- a formula here
    # would be circular (prior set from the likelihood's own data).
    nb_alpha_scale=3.0,
    nb_alpha_label="WEAK",
    nb_alpha_source="WEAK: HalfNormal(3.0), calibrated to pre-fit Var/Mean = 2.60 (static by design; see docs/PRIOR_CALIBRATION.md sec 4)",

    # With R0 now driving the trajectory (v6), the intercept absorbs only
    # ascertainment. A tight prior prevents it from re-absorbing the scale
    # information that identifies R0.
    log_intercept_mu=0.0, log_intercept_sd=0.5,
    log_intercept_source="WEAK: Normal(0, 0.5) on the log ascertainment ratio; tightened from v5 Normal(-1.8, 1.5) because the v6 trajectory is R0-dependent and a wide intercept re-absorbs the scale that identifies R0",
    estimate_intercept=True,
    ascertainment_source=(
        "LITERATURE: Ebola surveillance in Kikwit was retrospective and "
        "incomplete -- Khan et al. (1999) J Infect Dis 179:S76 describe case "
        "finding that improved sharply once the international team arrived in "
        "May 1995. Under-ascertainment is real, so an ascertainment intercept "
        "is estimated."),

    # Prior predictive calibration (scripts/03) selects 0.02 for T=139 with the
    # observed irregular spacing: larger scales let beta(t) swing by orders of
    # magnitude across the 6-month window, which is not credible. Only used when
    # --grw-ebola is passed; the pre-fit decision rule says no GRW is warranted.
    sigma_rw_scale=0.02,
    sigma_rw_label="WEAK",
    sigma_rw_source="WEAK: prior predictive calibrated (scripts/03); p95 of the implied beta(t) range stays under 3x. Gelman et al. (2020) BDA3 sec 6.4; Gabry et al. (2019) JRSSA 182:389",

    sensitivity={
        "N":              [500, 1000, 2000],
        "sigma_rw_scale": [0.05, 0.10, 0.20],
        "sigma":          [1.0 / 12.0, 1.0 / 9.0, 1.0 / 6.0],
    },
)


# ===========================================================================
# Influenza A/H1N1 -- boarding school, England, 1978
# ===========================================================================
INFLUENZA = DiseaseConfig(
    name="Influenza_England_1978",

    # in_bed counts boys CONFINED TO BED. Boys were infectious for roughly two
    # days before confinement and remained in bed roughly five days, during
    # which they were largely withdrawn from contact. in_bed is therefore a
    # confinement compartment Q downstream of the infectious compartment I,
    # not I itself.
    structure="seiqr_prevalence",
    structure_source=(
        "LITERATURE: Avilov KK, Li Q, Lin L, Demirhan H, Stone L, He D (2024). "
        "The 1978 English boarding school influenza outbreak: where the classic "
        "SEIR model fails. J R Soc Interface 21(220):20240394 -- documents that "
        "confined-to-bed (B) must be modelled as a compartment distinct from "
        "infectious (I), and that SEIR models treating them as identical "
        "significantly overestimate the outbreak timespan."
    ),

    # sigma: influenza A latent period ~1-2 days.
    sigma=1.0 / 1.5,
    sigma_source="LITERATURE: influenza A/H1N1 latent period 1-2 d; Ferguson NM et al. (2003) Science 300:1966",

    # removal == delta: rate of I -> Q. Infectious/circulating period before
    # confinement is ~2 days. R0 = beta / delta.
    removal=1.0 / 2.0,
    removal_source=(
        "LITERATURE: boys 'remained infectious, i.e. capable of transmitting the "
        "disease, for about 2 days' before confinement (BMJ 1978;1(6112):587 and "
        "standard treatments of this dataset). delta = 0.5 d^-1."
    ),

    # gamma_q: duration of confinement ~5 days.
    gamma_q=1.0 / 5.0,
    gamma_q_source="LITERATURE: 'the average duration of illness was 5 days' (BMJ 1978;1(6112):587)",

    N=763,
    N_source="LITERATURE: 763 boys at risk (BMJ 1978;1(6112):587); known exactly, no assumption required",
    I0=1.0,

    # Index case infected ~10 Jan, febrile 15-18 Jan; the series begins 22 Jan.
    # 12 days of unobserved epidemic precede observation day 0.
    t_seed=11.0,
    t_seed_source=(
        "LITERATURE + PROFILED: 'The index case was infected by 1978-01-10, and "
        "had febrile illness from 1978-01-15 to 1978-01-18' (RECON outbreaks, "
        "influenza_england_1978_school). The series begins 1978-01-22, giving a "
        "documented 8-12 day unobserved lead-in. Within that documented window "
        "t_seed=11 minimises the Poisson deviance of the constant-beta fit "
        "(scripts/02). Sensitivity over {7, 9, 11, 13} is prespecified. NOTE: "
        "t_seed is fixed, not estimated -- with 14 observations, jointly "
        "identifying t_seed and R0 reintroduces the non-identifiability that "
        "v6 exists to remove."
    ),

    # Primary analysis uses exponential residence times (k=1) so that R0 retains
    # its conventional meaning and stays in a range comparable to the other
    # arms. Erlang shapes fit the prevalence curve far better (Poisson deviance
    # 88.5 vs 505.7 at k=5) but drive R0 to ~15, well outside any published
    # influenza range. This trade-off is the subject of
    # docs/INFLUENZA_1978_LIMITATION.md and is run as a prespecified sensitivity,
    # not hidden.
    kE=1, kI=1, kQ=1,
    erlang_source=(
        "STRUCTURAL: k=1 (exponential) in the primary analysis. Avilov et al. "
        "(2024) J R Soc Interface 21:20240394 show non-exponential residence "
        "times are required for a good fit to this series; our own profiling "
        "reproduces that (Poisson deviance 505.7 at k=1 vs 88.5 at k=5) but also "
        "reproduces the accompanying R0 inflation (3.3 -> 15.5). Erlang shapes "
        "are run as sensitivity S_FLU_ERLANG."
    ),

    mu_point=0.0, mu_alpha=1.0, mu_beta=999.0,
    mu_source="LITERATURE: no deaths reported among 763 boys (BMJ 1978); CFR fixed at ~0 for trajectory purposes",

    # NOTE: the literature R0 for THIS dataset is contested and model-dependent.
    # See docs/INFLUENZA_1978_LIMITATION.md. The range below spans the
    # final-size-consistent value and the typical influenza range; it is used
    # only as a *reporting* reference, never as a prior on R0.
    R0_lit_lo=1.5, R0_lit_hi=4.0,
    R0_lit_source=(
        "CONTESTED -- reporting reference only, not a prior. "
        "Kermack-McKendrick final size with the reported attack rate 512/763 = "
        "0.671 implies R0 = 1.66. Avilov et al. (2024) J R Soc Interface "
        "21:20240394 fit the prevalence time course with a delay-differential "
        "model and obtain R0 = 8.14. Ahmad et al. (2025) Sci Rep 15 (s41598-025-"
        "26072-3) flag 8.14 as far above the typical influenza range of 1-4. "
        "No published model reproduces both the time course and the final size; "
        "see docs/INFLUENZA_1978_LIMITATION.md."
    ),

    # Pre-fit Var/Mean = 103.6 arises from the epidemic ARC across 14 days, not
    # from within-day count noise. The v5 production run confirmed the parameter
    # is prior-dominated (posterior/prior mean ratio 1.27 at HalfNormal(50), and
    # 1.13 at HalfNormal(200)). Widening further does not identify it. The value
    # below is a DISCLOSURE choice: wide enough that the posterior cannot be
    # mistaken for a data-driven estimate.
    nb_alpha_scale=200.0,
    nb_alpha_label="DISCLOSURE",
    nb_alpha_source=(
        "DISCLOSURE: HalfNormal(200). Posterior/prior-mean ratio was 1.27 at "
        "HalfNormal(50) and 1.13 at HalfNormal(200) -- the likelihood is flat in "
        "this parameter (Gelman et al. (2020) BDA3 sec 13.3). Reported as "
        "unidentified, not as an estimate."
    ),

    log_intercept_mu=0.0, log_intercept_sd=0.0,
    log_intercept_source="STRUCTURAL: fixed at 0 (ascertainment = 1). Not estimated.",
    estimate_intercept=False,
    ascertainment_source=(
        "STRUCTURAL: in_bed is a CENSUS, not a sample. The school recorded every "
        "boy confined to bed each day across a closed population of 763 "
        "(BMJ 1978;1(6112):587). There is no under-ascertainment for the "
        "observation process to absorb, so ascertainment is fixed at 1. "
        "Estimating an intercept against a census is not merely redundant: a "
        "Poisson-deviance profile over (R0, log_intercept) shows a diagonal "
        "ridge -- the deviance minimum moves from intercept +0.6 at R0=2.0 to "
        "-0.6 at R0=3.5 -- because both parameters scale the same trajectory. "
        "That ridge saturated the NUTS tree depth and was the cause of the slow, "
        "poorly mixing v6 trial runs."),

    # Prior predictive calibration (scripts/03) selects 0.10 for T=14: median
    # implied beta(t) range 1.32x, 95th percentile 2.73x, both under the 3x
    # plausibility ceiling for a closed two-week outbreak.
    sigma_rw_scale=0.10,
    sigma_rw_label="WEAK",
    sigma_rw_source="WEAK: prior predictive calibrated (scripts/03); median implied beta(t) range 1.32x, p95 2.73x, ceiling 3x. Gelman et al. (2020) BDA3 sec 6.4; Gabry et al. (2019) JRSSA 182:389. NOTE: the Influenza GRW is DISABLED by default -- 14 innovations against 14 observations is a saturated latent process. This scale applies only when --grw-influenza is passed for sensitivity.",

    sensitivity={
        "t_seed":         [7.0, 9.0, 11.0, 13.0],
        "erlang_kEIQ":    [(1, 1, 1), (2, 2, 2), (3, 3, 3), (5, 5, 5)],
        "sigma_rw_scale": [0.05, 0.10, 0.20],
        "removal":        [1.0 / 3.0, 1.0 / 2.0, 1.0 / 1.5],
        "gamma_q":        [1.0 / 6.0, 1.0 / 5.0, 1.0 / 4.0],
    },
)


# ===========================================================================
# Norovirus GII -- Derbyshire primary school, England, 2001
# ===========================================================================
NOROVIRUS = DiseaseConfig(
    name="Norovirus_Derbyshire_2001",

    # Cross-sectional binary outcome per student. R0 enters through the
    # Kermack-McKendrick final-size relation, which is exact for SEIR-type
    # models with a single wave and depends only on R0 (not on the
    # generation-interval distribution).
    structure="final_size",
    structure_source=(
        "LITERATURE: Kermack WO & McKendrick AG (1927) Proc Roy Soc A "
        "115(772):700-721 final size relation; O'Neill PD & Marks PJ (2005) "
        "Stat Med 24(13):2011-2024 for this outbreak."
    ),

    sigma=1.0 / 1.2,
    sigma_source="LITERATURE: CDC norovirus 12-48 h incubation; Heijne JCM et al. (2012) Epidemics 4(4):164-171 mean 1.2 d",
    removal=1.0 / 2.0,
    removal_source="LITERATURE: CDC norovirus 1-3 d illness; Heijne et al. (2012) mean 2 d",

    N=492,
    N_source="LITERATURE: complete school roll, 492 students (O'Neill & Marks 2005)",

    mu_point=0.0, mu_alpha=1.0, mu_beta=200.0,
    mu_source="LITERATURE: CDC -- norovirus rarely fatal in healthy populations",

    R0_lit_lo=1.3, R0_lit_hi=6.7,
    R0_lit_source="LITERATURE: Heijne et al. (2012) Epidemics 4:164-171, R0 = 1.3-6.7 in institutional settings",

    nb_alpha_scale=1.0,
    nb_alpha_label="STRUCTURAL",
    nb_alpha_source="STRUCTURAL: unused -- Norovirus uses a Bernoulli likelihood",

    log_intercept_mu=0.0, log_intercept_sd=1.0,
    log_intercept_source="STRUCTURAL: unused -- attack rate enters directly through the K-M relation",

    sigma_rw_scale=0.0,
    sigma_rw_label="STRUCTURAL",
    sigma_rw_source="STRUCTURAL: no GRW -- the data are cross-sectional, with no time index for beta(t) to vary over",

    sensitivity={
        "exclude_class": [None, 10],
        "class_re_sd_scale": [0.5, 1.0, 2.0],
    },
)




# ===========================================================================
# Measles -- Hagelloch, Germany, 1861   (added in v7)
# ===========================================================================
MEASLES = DiseaseConfig(
    name="Measles_Hagelloch_1861",

    # ERU (rash onset) is recorded for every one of the 188 children. It is an
    # INCIDENCE series -- new symptomatic cases per day -- so it is compared to
    # the flux into the symptomatic compartment, sigma * E(t), exactly as for
    # Ebola. Comparing an onset series to prevalence I(t) is the v5 error that
    # v6 corrected.
    structure="seird_incidence",
    structure_source=(
        "LITERATURE: Neal PJ & Roberts GO (2004) Biostatistics 5(2):249-261 "
        "model this outbreak as an SEIR process on the 188-child cohort. "
        "Meyer S, Held L & Hoehle M (2017) J Stat Softw 77(11) distribute the "
        "linelist. Observable is daily rash-onset incidence."
    ),

    # Measles latent period: ~8-10 days from infection to prodrome, and the
    # rash follows the prodrome by ~3-4 days. Modelling rash onset as the
    # symptomatic transition gives an effective latent period of ~11 days.
    # The Hagelloch linelist itself shows a median prodrome-to-rash gap of
    # 4 days, consistent with this.
    sigma=1.0 / 11.0,
    sigma_source=(
        "LITERATURE: measles incubation 10-14 days to rash onset (CDC Pink Book, "
        "ch. 13); Neal & Roberts (2004) use a latent period of this order for "
        "this dataset. 1/11 d^-1."
    ),

    # Infectious period for measles is roughly 8 days spanning the 4 days
    # before and 4 days after rash onset. Removal rate 1/8 d^-1.
    removal=1.0 / 8.0,
    removal_source=(
        "LITERATURE: measles infectious from ~4 days before to ~4 days after "
        "rash onset (CDC Pink Book, ch. 13), i.e. ~8 days. 1/8 d^-1."
    ),

    N=188,
    N_source=(
        "LITERATURE: complete village child cohort, 188 children "
        "(Pfeilsticker 1863; Oesterle 1992). Known exactly -- every child in "
        "Hagelloch is in the linelist, so N carries no assumption."
    ),
    I0=1.0,
    t_seed=0.0,
    t_seed_source=(
        "STRUCTURAL: day 0 = first recorded rash onset (1861-11-03). The "
        "linelist records the index case, so unlike the Influenza series there "
        "is no undocumented lead-in to offset. Sensitivity over {0, 3, 6} is "
        "prespecified."
    ),

    mu_point=0.0638, mu_alpha=12.0, mu_beta=176.0,
    mu_source=(
        "LITERATURE: 12 deaths among 188 children = CFR 6.38%, counted directly "
        "from the linelist DEAD column. Beta(12, 176) is the exact "
        "Beta-Binomial posterior for that count under a uniform prior."
    ),

    R0_lit_lo=10.0, R0_lit_hi=20.0,
    R0_lit_source=(
        "LITERATURE: measles R0 is classically 12-18 in unvaccinated "
        "populations (Anderson RM & May RM (1991), Infectious Diseases of "
        "Humans, Table 4.1); Guerra FM et al. (2017) Lancet Infect Dis "
        "17(12):e420-e428 review 1-770 with a median of 15.9 and argue the "
        "12-18 canon is over-narrow. Used for REPORTING ONLY, never as a prior "
        "on R0. NOTE: Hagelloch is a single village with intense household and "
        "classroom mixing, so a value at or above the upper end is plausible."
    ),

    # Pre-fit Var/Mean is computed by scripts/02 and reported. Measles onset
    # counts in a 188-child village are small (0-26/day) with many zero days,
    # so a NegBinomial is used as for the other incidence arm. STATIC by design:
    # a formula keyed to the data being fitted would be circular.
    nb_alpha_scale=3.0,
    nb_alpha_label="WEAK",
    nb_alpha_source=(
        "WEAK: HalfNormal(3.0), the same weakly informative scale used for the "
        "Ebola incidence arm. Static by design -- see "
        "docs/PRIOR_CALIBRATION.md and PARAMETER_CHANGE_AUDIT.md sec 10."
    ),

    # Pfeilsticker recorded the rash date for every child in the village, so
    # this is a CENSUS of the cohort, not a sample. Ascertainment is 1 and the
    # intercept is fixed, exactly as for the Influenza in_bed census. Estimating
    # an intercept against a census creates a ridge with R0 (v6 audit sec 12).
    log_intercept_mu=0.0, log_intercept_sd=0.0,
    log_intercept_source="STRUCTURAL: fixed at 0 (ascertainment = 1). Not estimated.",
    estimate_intercept=False,
    ascertainment_source=(
        "STRUCTURAL: the linelist is a CENSUS. Pfeilsticker (1863) recorded a "
        "rash date for all 188 children in Hagelloch, and the 188 rash onsets "
        "in the daily series sum to exactly the cohort size. There is no "
        "under-ascertainment for an intercept to absorb."
    ),

    # GRW disabled by default, as for the other two time-series arms. Whether it
    # is warranted is decided by scripts/02 from the residuals of the
    # constant-transmission fit, not assumed here.
    sigma_rw_scale=0.05,
    sigma_rw_label="WEAK",
    sigma_rw_source=(
        "WEAK: prior predictive calibrated (scripts/03). Only applies when "
        "--grw-measles is passed; the default is no GRW, subject to the "
        "scripts/02 decision rule."
    ),

    # Measles R0 is classically 12-18 and Hagelloch is a dense village cohort,
    # so the default ceiling of 15 would clip. Widened to 40.
    grid_R0_lo=0.5, grid_R0_hi=40.0,

    sensitivity={
        "t_seed":         [0.0, 3.0, 6.0],
        "sigma":          [1.0 / 13.0, 1.0 / 11.0, 1.0 / 9.0],
        "removal":        [1.0 / 10.0, 1.0 / 8.0, 1.0 / 6.0],
        "sigma_rw_scale": [0.02, 0.05, 0.10],
    },
)


# ===========================================================================
# Cross-disease hyperprior
# ===========================================================================


@dataclass(frozen=True)
class Hyperprior:
    """
    Partial pooling of log R0 across the three human disease arms.

    WEAK by design. Normal(0.5, 1.0) on the mean places 95% prior mass on
    R0 in roughly [0.23, 11.8]; HalfNormal(0.5) on the SD allows anything from
    near-complete pooling to effectively independent arms, letting the data
    choose. Deliberately NOT formula-calibrated: a hyperprior is a
    regularisation device, and tuning it to external quantities would encode
    domain knowledge at the wrong level of the hierarchy.

    Reference: Gelman A et al. (2020). Bayesian Data Analysis, 3e, Ch. 5.
    Gelman A (2006). Bayesian Analysis 1(3):515-534 (HalfNormal for scale
    parameters in hierarchical models).

    IMPORTANT (v6): pooling is only meaningful when every arm routes R0 into its
    own likelihood. In v5 two of three arms did not, so the hyperprior was
    driven entirely by Norovirus and then imposed on Ebola and Influenza. See
    core/bhmm/ode_surrogate.py for the diagnosis and fix.
    """
    log_R0_mu_loc: float = 0.5
    log_R0_mu_scale: float = 1.0
    log_R0_sd_scale: float = 0.5
    sd_floor: float = 0.1     # STRUCTURAL: prevents prior collapse to a point mass

    label: str = "WEAK"
    source: str = "WEAK: Gelman et al. (2020) BDA3 Ch. 5; Gelman (2006) Bayesian Analysis 1(3):515"


HYPERPRIOR = Hyperprior()


# ===========================================================================
# Registry
# ===========================================================================

ALL_DISEASES: Dict[str, DiseaseConfig] = {
    "ebola": EBOLA,
    "measles": MEASLES,
    "influenza": INFLUENZA,
    "norovirus": NOROVIRUS,
}

# Arms whose R0 is a clean cross-disease comparator. Influenza is excluded:
# its final-size and time-course identification targets disagree and no
# published model reconciles them (docs/INFLUENZA_1861_LIMITATION is for
# measles; the influenza one is docs/INFLUENZA_1978_LIMITATION.md). It is kept
# in the model as a diagnostic stress test, not as a headline estimate.
HEADLINE_ARMS = ("ebola", "measles", "norovirus")

# Influenza England 1978 is a BENCHMARK arm, not a result-producing arm.
#
# It is included because the literature already establishes that compartmental
# models cannot fit it (Avilov KK et al. (2024) J R Soc Interface 21:20240394,
# "where the classic SEIR model fails"). Its job here is to answer the question
# a reviewer will ask: "how do you know your diagnostics would catch a bad fit?"
#
# In the v7 production run it answered that question decisively. Every sampler
# diagnostic passed -- R-hat 1.0002, ESS 6254, zero divergences, 100% of
# observations inside the 90% predictive band -- while the model demonstrably
# did not fit: residual lag-1 correlation +0.83 and a fitted spread score
# implying 91% of the school infected against 67% observed.
#
# Reporting rules for this arm:
#   - never quote flu_R0 beside the headline arms as an equal
#   - never cite its predictive coverage as evidence of good fit
#   - always cite Avilov et al. (2024) when its spread score is mentioned
#   - always run the model once with and once without it (--drop-influenza)
BENCHMARK_ARMS = ("influenza",)
CAVEATED_ARMS = BENCHMARK_ARMS          # retained for backward compatibility


def prior_table() -> list[dict]:
    """Flat table of every prior with its epistemic label, for the audit PDF."""
    rows: list[dict] = []
    for key, cfg in ALL_DISEASES.items():
        rows += [
            dict(disease=key, parameter="structure", value=cfg.structure,
                 label="STRUCTURAL", source=cfg.structure_source),
            dict(disease=key, parameter="sigma", value=f"{cfg.sigma:.4f} /day",
                 label="LITERATURE", source=cfg.sigma_source),
            dict(disease=key, parameter="removal", value=f"{cfg.removal:.4f} /day",
                 label="LITERATURE", source=cfg.removal_source),
            dict(disease=key, parameter="N", value=str(cfg.N),
                 label="LITERATURE", source=cfg.N_source),
            dict(disease=key, parameter="nb_alpha_scale", value=f"HalfNormal({cfg.nb_alpha_scale})",
                 label=cfg.nb_alpha_label, source=cfg.nb_alpha_source),
            dict(disease=key, parameter="log_intercept",
                 value=f"Normal({cfg.log_intercept_mu}, {cfg.log_intercept_sd})",
                 label="WEAK", source=cfg.log_intercept_source),
            dict(disease=key, parameter="sigma_rw_scale", value=f"HalfNormal({cfg.sigma_rw_scale})",
                 label=cfg.sigma_rw_label, source=cfg.sigma_rw_source),
        ]
        if cfg.gamma_q:
            rows.append(dict(disease=key, parameter="gamma_q", value=f"{cfg.gamma_q:.4f} /day",
                             label="LITERATURE", source=cfg.gamma_q_source))
        if cfg.t_seed:
            rows.append(dict(disease=key, parameter="t_seed", value=f"{cfg.t_seed} days",
                             label="LITERATURE", source=cfg.t_seed_source))
    rows.append(dict(disease="shared", parameter="hyper_log_R0_mu",
                     value=f"Normal({HYPERPRIOR.log_R0_mu_loc}, {HYPERPRIOR.log_R0_mu_scale})",
                     label=HYPERPRIOR.label, source=HYPERPRIOR.source))
    rows.append(dict(disease="shared", parameter="hyper_log_R0_sd",
                     value=f"HalfNormal({HYPERPRIOR.log_R0_sd_scale})",
                     label=HYPERPRIOR.label, source=HYPERPRIOR.source))
    return rows


# Backwards-compatible aliases (v5 import names)
EBOLA_PRIOR = EBOLA
INFLUENZA_PRIOR = INFLUENZA
NOROVIRUS_PRIOR = NOROVIRUS

MEASLES_PRIOR = MEASLES
