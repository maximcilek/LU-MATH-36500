"""
core/bhmm/bhmm_model.py
=======================
Unified Bayesian Hierarchical Mechanistic Model (BHMM) -- v6.

Three human disease arms sharing a hyperprior on log R0, plus a zombie SEZR
prior-predictive reference arm handled separately in prior_predictive.py.

    Ebola Kikwit 1995        SEIRD, observable = daily onset incidence sigma*E(t)
    Measles Hagelloch 1861   SEIRD, observable = daily rash-onset incidence sigma*E(t)
    Influenza England 1978   SEIQR, observable = daily confined-to-bed Q(t)
    Norovirus Derbyshire     final size, observable = per-student binary outcome

v7 adds the Measles arm. Rationale in docs/DATASET_SELECTION.md: it restores a
third clean, well-identified comparator so that the headline cross-disease R0
comparison does not rest on Influenza, whose identification targets are known to
conflict (docs/INFLUENZA_1978_LIMITATION.md). Influenza is retained as a
diagnostic stress test rather than removed.

-------------------------------------------------------------------------------
WHAT CHANGED IN v6 AND WHY  (full detail in docs/PARAMETER_CHANGE_AUDIT.md)
-------------------------------------------------------------------------------
[CRITICAL] R0 now enters every likelihood.
    v5 solved the ODE once at prior-mean rates, outside the model, so e_R0 and
    flu_R0 never appeared in the likelihood. Confirmed from the v5 trace: the
    raw variables sat exactly at their Normal(0,1) priors
    (e_log_R0_raw mean -0.0157 sd 1.0105; flu_log_R0_raw mean -0.0005 sd 1.0124)
    while n_log_R0_raw moved to mean -0.6206 sd 0.8949. v6 interpolates an
    R0-indexed ODE grid inside the model -- see core/bhmm/ode_surrogate.py.

[CRITICAL] Correct observable per arm.
    Ebola  : `onset` is an incidence flux; v5 compared it to the prevalence
             I(t), which lags and integrates onsets. v6 uses sigma*E(t).
    Influenza: `in_bed` counts confined boys, not infectious boys. v6 uses Q(t)
             from an SEIQR structure, per Avilov et al. (2024) J R Soc Interface
             21:20240394.

[IMPORTANT] Seed offset for Influenza.
    The BMJ series begins 12 days after the documented index infection. v5
    forced model time 0 onto observation day 0, which is why the v5 ODE peaked
    at day 36 against data peaking at day 6.

[IMPORTANT] Ebola GRW indexed by observation, with sqrt(dt) innovation scaling.
    Reduces the latent dimension from 193 to 139 (ratio to observations 1.39 ->
    1.00) and correctly handles the irregular spacing created by excluding the
    53 non-reporting days.

[IMPORTANT] log_intercept priors tightened to Normal(0, 0.5).
    Now that the trajectory is R0-dependent, a wide intercept would re-absorb
    exactly the scale information that identifies R0.

-------------------------------------------------------------------------------
KNOWN LIMITATION, DISCLOSED NOT HIDDEN
-------------------------------------------------------------------------------
For the 1978 boarding-school data, no published model reproduces both the
prevalence time course and the 67% attack rate simultaneously. This model fits
the prevalence curve; the implied final size will exceed the reported attack
rate. See docs/INFLUENZA_1978_LIMITATION.md, which must be cited in any write-up
that reports flu_R0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from core.bhmm.priors import EBOLA, MEASLES, INFLUENZA, NOROVIRUS, HYPERPRIOR
from core.bhmm.ode_surrogate import ODEGrid, build_grid, grw_sigma_vector


# ---------------------------------------------------------------------------
# Kermack-McKendrick final size
# ---------------------------------------------------------------------------


def km_attack_rate(R0, n_iter: int = 80):
    """
    Solve the Kermack-McKendrick final-size relation  AR = 1 - exp(-R0 * AR)
    by damped fixed-point iteration from above.

    Exact for SEIR-type models with a single wave and a fully susceptible start;
    the final size depends only on R0, not on the generation-interval
    distribution (Kermack & McKendrick 1927; Ma J & Earn DJD (2006) Bull Math
    Biol 68:679-702).

    -----------------------------------------------------------------------
    v8 CORRECTION -- this replaces a 12-step Newton iteration that was WRONG
    below R0 = 1 and silently corrupted the Norovirus arm.
    -----------------------------------------------------------------------
    For R0 <= 1 the relation has the single root AR = 0: a subcritical pathogen
    cannot sustain an outbreak. Newton started from AR_0 = 1 - exp(-R0)
    approaches that boundary root only LINEARLY, so 12 steps stopped far short
    of it:

        R0      12-step Newton      true final size
        0.850       0.0933                0
        0.919       0.1373                0
        1.000       0.1983                0

    In the v7 production run (seed 20260914, 4 chains x 2000 draws) the
    Norovirus posterior settled at n_R0 = 0.919 with a reported attack rate of
    0.139 -- close to the observed 0.152, so every convergence diagnostic
    passed. But 0.919 is subcritical: the true final size there is exactly
    zero. R0 was being identified by the solver's truncation error rather than
    by the final-size relation. `tables/final_size_consistency.csv` flagged the
    contradiction (implied AR 0.0 vs reported 0.152) but the model itself did
    not.

    The correct R0 for an attack rate of 0.152 is 1.085.

    Damped fixed-point iteration from AR = 0.999 converges to the true root
    from above for every R0, including collapsing to ~0 when R0 <= 1, and is
    differentiable throughout so NUTS gradients still reach R0.
    """
    R0c = pt.clip(R0, 0.01, 40.0)
    AR = pt.ones_like(R0c) * 0.999
    for _ in range(n_iter):
        AR = pt.clip(1.0 - pt.exp(-R0c * AR), 1e-12, 1.0 - 1e-9)
    return AR


# ---------------------------------------------------------------------------
# Grid construction (done once, outside the sampler)
# ---------------------------------------------------------------------------


def build_disease_grids(ebola_df: pd.DataFrame,
                        flu_df: pd.DataFrame,
                        measles_df: pd.DataFrame | None = None,
                        *, G: int = 160) -> dict[str, ODEGrid]:
    """
    Build the R0-indexed ODE grids for the two time-series arms.

    Returned grids are plain data; they contain no PyMC objects and can be
    inspected, plotted, or error-checked before sampling. scripts/02 does
    exactly that and writes the interpolation error into the pre-fit audit.
    """
    elig = ebola_df[ebola_df.analysis_eligible]
    e_days = elig.day.values.astype(np.int64)

    ebola_grid = build_grid(
        structure="seird_incidence",
        sigma=EBOLA.sigma,
        removal=EBOLA.removal,
        mu=EBOLA.mu_point,
        N=EBOLA.N,
        I0=EBOLA.I0,
        t_seed=EBOLA.t_seed,
        T=int(e_days.max()) + 1,        # solve over the full calendar span
        R0_lo=EBOLA.grid_R0_lo, R0_hi=EBOLA.grid_R0_hi,
        G=G,
    )

    flu_grid = build_grid(
        structure="seiqr_prevalence",
        sigma=INFLUENZA.sigma,
        removal=INFLUENZA.removal,
        gamma_q=INFLUENZA.gamma_q,
        N=INFLUENZA.N,
        I0=INFLUENZA.I0,
        t_seed=INFLUENZA.t_seed,
        T=len(flu_df),
        kE=INFLUENZA.kE, kI=INFLUENZA.kI, kQ=INFLUENZA.kQ,
        R0_lo=INFLUENZA.grid_R0_lo, R0_hi=INFLUENZA.grid_R0_hi,
        G=G,
    )
    grids = {"ebola": ebola_grid, "influenza": flu_grid}

    if measles_df is not None:
        m_days = measles_df.day.values.astype(np.int64)
        grids["measles"] = build_grid(
            structure="seird_incidence",
            sigma=MEASLES.sigma,
            removal=MEASLES.removal,
            mu=MEASLES.mu_point,
            N=MEASLES.N,
            I0=MEASLES.I0,
            t_seed=MEASLES.t_seed,
            T=int(m_days.max()) + 1,
            R0_lo=MEASLES.grid_R0_lo, R0_hi=MEASLES.grid_R0_hi,
            G=G,
        )
    return grids


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def build_full_bhmm(ebola_df: pd.DataFrame,
                    noro_df: pd.DataFrame,
                    flu_df: pd.DataFrame,
                    measles_df: pd.DataFrame | None = None,
                    *,
                    n_class: int = 15,
                    grids: dict[str, ODEGrid] | None = None,
                    sigma_rw_e_scale: float | None = None,
                    sigma_rw_flu_scale: float | None = None,
                    sigma_rw_measles_scale: float | None = None,
                    use_grw_ebola: bool = False,
                    use_grw_influenza: bool = False,
                    use_grw_measles: bool = False,
                    include_influenza: bool = True,
                    exclude_noro_class: int | None = None) -> pm.Model:
    """
    Build the unified BHMM.

    Parameters
    ----------
    ebola_df, noro_df, flu_df : analysis-ready canonical datasets
    n_class                   : number of Norovirus classes (15)
    grids                     : precomputed ODE grids; built if omitted
    sigma_rw_*_scale          : override the HalfNormal scale (sensitivity runs)
    use_grw_ebola             : include a GRW on log beta(t) for Ebola.
                                DEFAULT False. The pre-fit audit (scripts/02)
                                finds residual lag-1 ACF = -0.037 with
                                Ljung-Box p = 0.885 once the correct observable
                                (incidence flux sigma*E(t)) is used, so the
                                decision rule |lag-1| > 0.20 is NOT met and a
                                GRW is not warranted. The v5 lag-1 of +0.678 was
                                an artefact of comparing onset counts to the
                                prevalence I(t). Including a GRW here would add
                                139 latent parameters that explain nothing --
                                the direct cause of the v5 Ebola R0 CV of 216%.
    use_grw_influenza         : include a GRW for Influenza. DEFAULT False.
                                The residual lag-1 ACF is +0.80 even with the
                                corrected SEIQR structure, so the ACF rule alone
                                would call for a GRW. It is nevertheless
                                disabled by default because with T = 14 a daily
                                GRW has 14 free innovations against 14
                                observations: the latent process is SATURATED
                                and can reproduce any data vector exactly. The
                                consequence is measured, not assumed -- with the
                                GRW on, a 2-chain trial gave Rhat 1.80-1.83 and
                                ESS 2-3 on flu_R0, sigma_rw_flu and
                                flu_alpha_nb (chains locked into different
                                (R0, sigma_rw) modes), and flu_alpha_nb was
                                prior-dominated. With it off, all three reach
                                Rhat <= 1.01 with ESS >= 321, 0 divergences, and
                                flu_alpha_nb becomes data-informed (posterior
                                0.93 against a prior mean of 160).
                                The remaining residual structure is a real,
                                published misfit for this dataset rather than
                                something to absorb with 14 free parameters --
                                see docs/INFLUENZA_1978_LIMITATION.md. Enable
                                with --grw-influenza for sensitivity.
    exclude_noro_class        : drop one class (sensitivity run S2)

    Returns
    -------
    pm.Model ready for pm.sample().
    """
    if grids is None:
        grids = build_disease_grids(ebola_df, flu_df, measles_df)
    e_grid = grids["ebola"]
    f_grid = grids["influenza"]
    m_grid = grids.get("measles")
    if measles_df is not None and m_grid is None:
        raise ValueError("measles_df supplied but no 'measles' grid was built")

    # ---- Ebola data ------------------------------------------------------
    # Only analysis_eligible rows enter the likelihood. reporting=FALSE rows are
    # structural zeros and a perfect separator (chi-square 52.2, p < 0.001);
    # they are retained in the canonical file but excluded here.
    elig_e = ebola_df[ebola_df.analysis_eligible].copy()
    e_obs = elig_e.onset.values.astype(float)
    e_days = elig_e.day.values.astype(np.int64)
    n_e = len(e_obs)

    # ---- Norovirus data --------------------------------------------------
    nd = noro_df
    if exclude_noro_class is not None:
        nd = nd[nd["class"] != exclude_noro_class].copy()
    n_ill = nd.ill.values.astype(int)
    n_cls_raw = nd["class"].values.astype(int)
    uniq = np.sort(np.unique(n_cls_raw))
    remap = {c: i for i, c in enumerate(uniq)}
    n_cls = np.array([remap[c] for c in n_cls_raw], dtype=np.int64)
    n_class_eff = len(uniq)

    # ---- Influenza data --------------------------------------------------
    f_obs = flu_df.in_bed.values.astype(float)
    T_flu = len(f_obs)

    # ---- Measles data (v7) -----------------------------------------------
    # rash_onset is an INCIDENCE series: new symptomatic cases per day, so it is
    # compared to the flux sigma*E(t), not to prevalence. Every day in the
    # window is retained, including the 51 zero-onset days -- unlike Ebola,
    # these are genuine observed zeros in a village under continuous
    # observation, not a surveillance artefact, so there is no eligibility gate.
    if measles_df is not None:
        m_obs = measles_df.rash_onset.values.astype(float)
        m_days = measles_df.day.values.astype(np.int64)
        n_m = len(m_obs)

    # ---- prior scale overrides ------------------------------------------
    s_rw_e = EBOLA.sigma_rw_scale if sigma_rw_e_scale is None else sigma_rw_e_scale
    s_rw_f = INFLUENZA.sigma_rw_scale if sigma_rw_flu_scale is None else sigma_rw_flu_scale

    with pm.Model(name="bhmm") as model:

        # ==== Level 1: cross-disease hyperprior ==========================
        # Partial pooling of log R0. Meaningful in v6 because all three arms
        # now route R0 into their own likelihood.
        hyper_mu = pm.Normal("hyper_log_R0_mu",
                             mu=HYPERPRIOR.log_R0_mu_loc,
                             sigma=HYPERPRIOR.log_R0_mu_scale)
        hyper_sd = pm.HalfNormal("hyper_log_R0_sd",
                                 sigma=HYPERPRIOR.log_R0_sd_scale)

        # ==== Level 2: disease-specific R0, non-centred ==================
        # Non-centred form (Betancourt & Girolami 2015): the conditional
        # geometry of the raw variable is Normal(0,1) regardless of hyper_sd,
        # so no funnel forms and NUTS step sizes stay consistent.
        e_raw = pm.Normal("e_log_R0_raw", mu=0.0, sigma=1.0)
        f_raw = pm.Normal("flu_log_R0_raw", mu=0.0, sigma=1.0)
        n_raw = pm.Normal("n_log_R0_raw", mu=0.0, sigma=1.0)
        if measles_df is not None:
            m_raw = pm.Normal("mea_log_R0_raw", mu=0.0, sigma=1.0)

        sd_eff = hyper_sd + HYPERPRIOR.sd_floor
        e_log_R0 = pm.Deterministic("e_log_R0", hyper_mu + sd_eff * e_raw)
        f_log_R0 = pm.Deterministic("flu_log_R0", hyper_mu + sd_eff * f_raw)
        n_log_R0 = pm.Deterministic("n_log_R0", hyper_mu + sd_eff * n_raw)

        e_R0 = pm.Deterministic("e_R0", pt.exp(e_log_R0))
        f_R0 = pm.Deterministic("flu_R0", pt.exp(f_log_R0))
        n_R0 = pm.Deterministic("n_R0", pt.exp(n_log_R0))
        if measles_df is not None:
            m_log_R0 = pm.Deterministic("mea_log_R0", hyper_mu + sd_eff * m_raw)
            m_R0 = pm.Deterministic("mea_R0", pt.exp(m_log_R0))

        # NOTE (v6): the v5 code added log(R0_base_d) as a per-disease offset
        # inside the Deterministic. That offset shifted the *prior* location of
        # each arm toward its literature value, which was harmless only because
        # R0 was unidentified. Now that R0 is identified, a literature offset
        # would double-count literature information (once in the offset, once in
        # the reader's interpretation). It is removed. Each arm's R0 is now
        # driven by its own likelihood plus the shared hyperprior only.

        # ==== Level 3: observation-model parameters ======================
        e_alpha = pm.HalfNormal("e_alpha_nb", sigma=EBOLA.nb_alpha_scale)
        f_alpha = pm.HalfNormal("flu_alpha_nb", sigma=INFLUENZA.nb_alpha_scale)
        if measles_df is not None:
            m_alpha = pm.HalfNormal("mea_alpha_nb", sigma=MEASLES.nb_alpha_scale)

        # Ascertainment intercepts are estimated only where the observation
        # process can actually miss cases (see priors.ascertainment_source).
        #   Ebola     : retrospective, incomplete surveillance -> estimated
        #   Influenza : daily census of a closed 763-boy school -> fixed at 1
        # Estimating an intercept against a census creates a diagonal ridge with
        # R0 (both scale the same trajectory), which saturates the NUTS tree
        # depth. Deviance profile: the minimum moves from intercept +0.6 at
        # R0=2.0 to -0.6 at R0=3.5.
        if EBOLA.estimate_intercept:
            e_int = pm.Normal("log_intercept_e",
                              mu=EBOLA.log_intercept_mu, sigma=EBOLA.log_intercept_sd)
        else:
            e_int = pt.constant(EBOLA.log_intercept_mu)
        if INFLUENZA.estimate_intercept:
            f_int = pm.Normal("log_intercept_flu",
                              mu=INFLUENZA.log_intercept_mu, sigma=INFLUENZA.log_intercept_sd)
        else:
            f_int = pt.constant(INFLUENZA.log_intercept_mu)

        class_re_sd = pm.HalfNormal("class_re_sd", sigma=1.0)
        class_re = pm.Normal("class_re", mu=0.0, sigma=class_re_sd, shape=n_class_eff)

        # ==== Level 4: time-varying transmission (GRW on log beta_t) =====
        # NON-CENTRED GRW (v6). The centred form used in v5,
        #     pm.GaussianRandomWalk(sigma=sigma_rw, init_dist=Normal(0, sigma_rw))
        # puts sigma_rw in the conditional scale of every latent state, creating
        # a Neal funnel between sigma_rw and the T-dimensional walk -- the exact
        # pathology that Theorem 4 (Betancourt & Girolami 2015) removes from the
        # R0 hierarchy. It was harmless in v5 only because R0 was not in the
        # likelihood and the walk had nothing to compete with. With R0 now
        # identified, the walk and R0 both shape the trajectory, the funnel
        # becomes active, and the centred form produced divergences with
        # ESS ~ 3 on flu_R0 and sigma_rw_flu in a 2-chain trial.
        #
        # Non-centred equivalent: draw standardised innovations and scale them.
        #     eps_t ~ Normal(0, 1)
        #     log_rw_t = sigma_rw * sum_{k<=t} sqrt(dt_k) * eps_k
        # gives Var[log_rw_0] = sigma_rw^2 (init matches innovation) and
        # Var[log_rw_t - log_rw_{t-1}] = sigma_rw^2 * dt_t, so this is the same
        # process as the centred form, with Normal(0,1) sampling geometry.
        #
        # Reference: Betancourt M & Girolami M (2015), Hamiltonian Monte Carlo
        # for Hierarchical Models; Papaspiliopoulos O, Roberts GO & Skold M
        # (2007) Stat Sci 22(1):59-73 (general non-centred parameterisations for
        # hierarchical and state space models).

        # Ebola: indexed by OBSERVATION (n_e = 139 steps, not 193 calendar
        # days), with per-step innovation scaled by sqrt(day gap) so the walk is
        # a correctly discretised Wiener process on an irregular grid.
        if use_grw_ebola:
            sigma_rw_e = pm.HalfNormal("sigma_rw_e", sigma=s_rw_e)
            e_gaps = np.concatenate([[1.0], np.diff(e_days).astype(np.float64)])
            e_sqrt_dt = pt.as_tensor_variable(np.sqrt(e_gaps))
            e_eps = pm.Normal("log_rw_e_raw", mu=0.0, sigma=1.0, shape=n_e)
            log_rw_e = pm.Deterministic(
                "log_rw_e", sigma_rw_e * pt.cumsum(e_eps * e_sqrt_dt))
        else:
            # No GRW: the pre-fit decision rule is not met (see docstring).
            log_rw_e = pt.zeros(n_e)

        # Influenza: 14 consecutive days, uniform spacing, so sqrt(dt) = 1.
        if use_grw_influenza:
            sigma_rw_flu = pm.HalfNormal("sigma_rw_flu", sigma=s_rw_f)
            f_eps = pm.Normal("log_rw_flu_raw", mu=0.0, sigma=1.0, shape=T_flu)
            log_rw_flu = pm.Deterministic(
                "log_rw_flu", sigma_rw_flu * pt.cumsum(f_eps))
        else:
            log_rw_flu = pt.zeros(T_flu)

        # ==== Level 5: mechanistic trajectories, R0-dependent ============
        # THIS is the v6 fix. traj depends on the sampled R0 through a
        # differentiable interpolation of the precomputed grid, so the
        # likelihood gradient reaches R0.
        e_traj_full = e_grid.interpolate(e_log_R0)          # (T_calendar,)
        e_traj = e_traj_full[pt.as_tensor_variable(e_days)]  # (n_e,)
        f_traj = f_grid.interpolate(f_log_R0)                # (T_flu,)

        e_mu = pm.Deterministic(
            "e_mu_obs",
            pt.clip(pt.exp(e_int + log_rw_e) * e_traj, 1e-3, 1e9))
        f_mu = pm.Deterministic(
            "flu_mu_obs",
            pt.clip(pt.exp(f_int + log_rw_flu) * f_traj, 1e-3, 1e9))

        # Measles: same incidence structure as Ebola. Ascertainment is fixed at
        # 1 because the linelist is a census of the village child cohort.
        if measles_df is not None:
            if use_grw_measles:
                s_rw_m = (MEASLES.sigma_rw_scale if sigma_rw_measles_scale is None
                          else sigma_rw_measles_scale)
                sigma_rw_mea = pm.HalfNormal("sigma_rw_mea", sigma=s_rw_m)
                m_eps = pm.Normal("log_rw_mea_raw", mu=0.0, sigma=1.0, shape=n_m)
                log_rw_mea = pm.Deterministic(
                    "log_rw_mea", sigma_rw_mea * pt.cumsum(m_eps))
            else:
                log_rw_mea = pt.zeros(n_m)
            m_int = pt.constant(MEASLES.log_intercept_mu)
            m_traj_full = m_grid.interpolate(m_log_R0)
            m_traj = m_traj_full[pt.as_tensor_variable(m_days)]
            m_mu = pm.Deterministic(
                "mea_mu_obs",
                pt.clip(pt.exp(m_int + log_rw_mea) * m_traj, 1e-3, 1e9))

        # Norovirus: R0 enters through the exact final-size relation.
        n_AR = pm.Deterministic("n_attack_rate", km_attack_rate(n_R0))
        n_logit = pt.log(n_AR) - pt.log(1.0 - n_AR) + class_re[n_cls]

        # ==== Level 6: likelihoods =======================================
        pm.NegativeBinomial("ebola_obs", mu=e_mu, alpha=e_alpha, observed=e_obs)
        if include_influenza:
            pm.NegativeBinomial("flu_obs", mu=f_mu, alpha=f_alpha, observed=f_obs)
        if measles_df is not None:
            pm.NegativeBinomial("mea_obs", mu=m_mu, alpha=m_alpha, observed=m_obs)
        pm.Bernoulli("noro_obs", logit_p=n_logit, observed=n_ill)

    return model


# ---------------------------------------------------------------------------
# Back-compat shim
# ---------------------------------------------------------------------------


def ode_prevalence(log_beta, log_sigma, log_gamma, mu, N, I0, T_end, n_steps=None):
    """
    Retained so that core/bhmm/prior_predictive.py (zombie arm) keeps working
    unchanged. NOT used by the BHMM likelihood any more -- see build_full_bhmm,
    which goes through core/bhmm/ode_surrogate.py instead.
    """
    from scipy.integrate import solve_ivp
    beta, sigma, gamma = np.exp(float(log_beta)), np.exp(float(log_sigma)), np.exp(float(log_gamma))
    mu = float(mu)
    n_steps = n_steps or int(T_end) + 1
    t_eval = np.linspace(0, T_end, n_steps)
    y0 = [N - I0, 0.0, float(I0), 0.0, 0.0]

    def rhs(t, y):
        S, E, I, R, D = y
        f = beta * I / N
        return [-f * S, f * S - sigma * E, sigma * E - gamma * I,
                gamma * (1 - mu) * I, gamma * mu * I]

    sol = solve_ivp(rhs, [0, T_end], y0, t_eval=t_eval,
                    method="RK45", rtol=1e-6, atol=1e-8)
    return np.clip(sol.y[2], 0, None)
