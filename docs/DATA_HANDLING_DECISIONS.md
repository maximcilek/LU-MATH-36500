# Data Handling Decisions — BHMM v5

This document is the preregistered, immutable record of every data-handling
choice made before model fitting. Changes require explicit versioning.

**Version:** v6  |  **Date:** 2026-09-15  |  **Seed:** 20260914

---

## Q: Is any synthetic, fake, or generated data used in the BHMM?

**No.** The BHMM likelihood uses only real published observations:

| Disease | Source | Data type | Reference |
|---------|--------|-----------|-----------|
| Ebola Kikwit 1995 | RECON outbreaks R package | Daily onset + death counts | Khan et al. (1999) J Infect Dis 179:S76 |
| Norovirus Derbyshire 2001 | RECON outbreaks R package | Per-student illness records | O'Neill & Marks (2005) Stat Med 24:2011 |
| Influenza England 1978 | BMJ 1978; RECON outbreaks | Daily in_bed (I(t)) prevalence | Murray (1978) BMJ 1(6112):587 |

The zombie SEZR arm uses **no data whatsoever**. It is a prior-predictive
arm: the model propagates assumed parameter distributions through the ODE
to produce a distribution over epidemic trajectories. No observations are
generated. No likelihood is computed. The zombie "posterior" equals its prior.

The files `data/raw/zombie/sezr_initial_population.csv` and
`data/raw/zombie/sezr_person_day_observations.csv` are synthetic SEZR
simulation outputs from the original Math 36500 model. They are **never read
by any BHMM script**. They are retained in `data/raw/zombie/` with a README
for provenance only.

---

## §1 Ebola Kikwit 1995

**Outcome variable:** `onset` (new confirmed symptom-onset cases per day).
This is an incidence measure: E[onset_t] ≈ γ·I(t) (removal flux from I).

**Eligibility:**
- `analysis_eligible = (reporting == TRUE)` (n=139 of 192 days)
- `reporting=FALSE` rows have onset=0 by design (structural zeros; perfect separator)
- These 53 rows are **retained** in the canonical file but **excluded** from the likelihood
- Rationale: χ²=52.2, p<0.001; including them as predictors causes complete separation

**Onset-death lag:**
- Same-day CFR is structurally undefined because deaths follow onset by ~14 days
- `cfr_cumulative_lag14 = cum_deaths[t] / cum_onset[t-14]` provided for descriptive use only
- Never co-model onset and cfr_cumulative_lag14 as joint outcomes

**Collinearity:**
- `onset` and `cumulative_cases` have r=0.94 (VIF > 10)
- Never enter both in the same model component

---

## §2 Norovirus Derbyshire 2001

**Outcome variable:** `ill = (start_illness > 0)` (binary per student).

**Critical:** Do NOT use `day_absent` as the outcome. 126 of 417 well students
were absent (35.4%). Using absence misclassifies well students as ill.
`day_absent` is a mediator on the path illness → absence, not the outcome.

**Class heterogeneity:**
- χ²=62.7, df=14, p<0.001 across classes
- Class 10: AR=16/24=66.7% (vs overall 15.2%) — absorbed by class random intercept
- Class 3: AR=0/25=0.0% (structural zero) — RE shrinks toward global mean; no imputation

**Excluded from primary model:**
- `day_vomiting`: only 5 students are both ill and have vomiting records (φ=0.09)
- `day_absent`: mediator; including it as predictor induces collider bias

**Sensitivity S2:** Rerun without Class 10 → ΔR₀=0.009 (stable)

---

## §3 Influenza England 1978 (replaces Rabies CAR)

**Why Rabies was removed:**
Rabies CAR 2003-2012 describes dog-to-dog transmission in the I_d compartment
(animal reservoir, not human cases). This violates the core BHMM assumption
that all real-disease arms model human person-to-person transmission in a
closed population.

**Influenza 1978 (Murray 1978 BMJ):**
- 763 boarding school boys; H1N1 influenza; 14 days (Jan 22 – Feb 4, 1978)
- Human disease; closed population; N=763 known exactly (no assumption required)
- Community (person-to-person) transmission within the school

**Outcome variable:** `in_bed[t]` — students currently confined to bed.
This is a **prevalence** measure: in_bed[t] ≈ I(t) (infectious compartment stock).
This differs from Ebola (incidence = flux) and Norovirus (binary per-person).

**Observation model:** NB(μ = exp(log_intercept_flu + log_rw_flu[t]) × ODE_I(t))
No gamma-flux conversion required: the ODE directly predicts I(t).
(Theorem 6, mathematical_framework.pdf: Prevalence vs Incidence Observation)

**Pre-fit diagnostics:**
- Var/Mean = 103.6 — reflects epidemic trajectory shape; NB with HalfNormal(50) prior
- Lag-1 ACF = 0.847 — GRW on log(β_t) required (same as Ebola)
- Zero in-bed days = 0/14 — standard NB; no ZIP needed
- Literature R₀ = 2.0-2.5 — immediate posterior cross-check available

**Sensitivity S_flu1:** Rerun with days 0-6 only (growth phase) — prespecified

---

## §4 Zombie SEZR

**No real zombie outbreak data exists.** The zombie arm is prior-predictive only.

**What `zombie_sezr_prior_params.json` contains:**
Prior hyperparameter specifications (β, σ, γ, μ distributions) derived from
the Math 36500 SEZR simulation parameters. These are treated as expert-elicited
priors representing assumed knowledge about a hypothetical pathogen.
(Gelman et al. 2020 BDA3 Ch. 6: prior predictive checks)

**What the zombie arm does:**
1. Draws 1,000 parameter sets from the assumed prior distributions
2. Propagates each through the SEIRD ODE (forward simulation)
3. Summarises the resulting distribution of epidemic trajectories and R₀ values

**What the zombie arm does NOT do:**
- Generate fake observation records
- Compute a likelihood
- Update the prior with any data

**Inferential analysis of unknown disease (FAQ):**
The zombie arm demonstrates how to characterise an unknown disease with only
assumptions. The correct academic framework is:

1. **Prior predictive simulation** (current approach): propagate expert
   beliefs through the mechanistic model to understand what the assumptions imply.
2. **Prior sensitivity analysis**: systematically vary prior parameters over
   plausible ranges; examine how R₀ distribution changes.
   See scripts/07_zombie_sensitivity.py.
3. **Structural comparison**: compare the unknown-disease prior predictive
   against posteriors from known diseases (Mann-Whitney test; already implemented).
4. **Sequential Bayesian updating**: if any future observations become available
   (even partial data), the prior predictive becomes the prior for a full BHMM update.
   The zombie model is ready for this: adding a likelihood term is a one-line change.

The zombie comparison demonstrates that the assumed SEZR parameters describe
a substantially more explosive outbreak than any of the three reference diseases,
suggesting the parameters are not calibrated to real epidemic dynamics.
This is a valid and publishable scientific finding from a prior predictive analysis.

---

## §5 SHA-256 provenance

All canonical files are registered in `data/SHA256SUMS.txt`.
Verify before any analysis: `sha256sum -c data/SHA256SUMS.txt`


---

## v6 addendum: observable definitions corrected

The data-handling decisions above (eligibility, outcome definitions, exclusions)
are unchanged and remain correct. What changed in v6 is how each observed
quantity is MAPPED to the mechanistic model. Both mappings were wrong in v5.

| Arm | Observed column | v5 mapping | v6 mapping | Source for the correction |
|---|---|---|---|---|
| Ebola | `onset` | prevalence I(t) | incidence flux sigma*E(t) | onset counts new cases, which is a flux, not a stock |
| Influenza | `in_bed` | prevalence I(t) | confinement compartment Q(t), SEIQR | Avilov et al. (2024) J R Soc Interface 21:20240394 |
| Norovirus | `ill` | K-M final size | unchanged | O'Neill & Marks (2005) |

Neither correction changes which rows enter the likelihood, only what the model
predicts for them. See docs/PARAMETER_CHANGE_AUDIT.md sections 2 and 3.

## v6 addendum: synthetic data guard is now enforced in code

The statement that no synthetic data enters any likelihood is now tested, not
just asserted. `tests/test_unified_engine.py::test_no_stale_arms` fails if
`core/bhmm/bhmm_model.py` references any zombie raw data file. The zombie arm
remains prior-predictive only: no observations, no likelihood, posterior equals
prior.
