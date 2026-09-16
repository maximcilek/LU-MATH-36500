# Parameter and Structure Change Audit: v5 to v6

**Date:** 2026-09-16
**Basis:** v5 production trace, seed 20260914, 4 chains x 2000 draws
**Status:** every change below is evidence-driven; the evidence is reproducible from the v5 trace

---

## Summary of changes

| # | Change | Severity | Evidence |
|---|---|---|---|
| 1 | R0 routed into every likelihood via an R0-indexed ODE surrogate | CRITICAL | v5 raw variables sat exactly at their N(0,1) priors |
| 2 | Ebola observable corrected from prevalence I(t) to incidence sigma*E(t) | CRITICAL | constant-beta R0 moves 1.02 -> 1.92; residual lag-1 ACF 0.678 -> -0.037 |
| 3 | Influenza observable corrected from I(t) to confinement Q(t) via SEIQR | CRITICAL | Avilov et al. (2024); v5 ODE peaked day 36 vs data day 6 |
| 4 | Influenza seed offset t_seed = 11 days | IMPORTANT | index case documented 12 days before the series starts |
| 5 | Ebola GRW removed by default | IMPORTANT | pre-fit decision rule not met once the observable is correct |
| 6 | GRW re-parameterised non-centred | IMPORTANT | centred form produced divergences and ESS ~ 3 once R0 was live |
| 7 | Ebola GRW (when enabled) indexed by observation with sqrt(dt) scaling | IMPORTANT | 53 excluded days make the series irregularly spaced |
| 8 | log_intercept priors tightened to Normal(0, 0.5) | IMPORTANT | a wide intercept re-absorbs the scale that identifies R0 |
| 9 | Literature R0 offset removed from the non-centred Deterministic | MODERATE | double-counts literature once R0 is identified |
| 10 | sigma_rw justification changed to prior predictive checking | MODERATE | the v5 "Lemma 1 (Durbin & Koopman sec 2.3)" citation was overstated |
| 11 | flu_alpha_nb documented as unidentified rather than estimated | MODERATE | posterior/prior mean ratio 1.27 at HN(50), 1.13 at HN(200) |

---

## 1. CRITICAL: R0 never entered the likelihood (Ebola, Influenza)

### Evidence

v5 solved the ODE once, outside the model, at prior-mean rates:

```python
e_base = ode_prevalence(EBOLA_PRIOR.log_beta_mu, ...)        # fixed numpy array
e_mu_obs = exp(log_intercept_e + log_rw_e[t]) * e_base[t]    # R0 absent
```

The sampled `e_R0` and `flu_R0` were `pm.Deterministic` transformations of the
hyperprior and a raw variable that appeared nowhere in the likelihood. From the
v5 production trace:

```
e_log_R0_raw     mean = -0.0157   sd = 1.0105     prior N(0,1)   NOT UPDATED
flu_log_R0_raw   mean = -0.0005   sd = 1.0124     prior N(0,1)   NOT UPDATED
n_log_R0_raw     mean = -0.6206   sd = 0.8949     prior N(0,1)   updated
```

Only Norovirus routed R0 into a likelihood, through the Kermack-McKendrick
final-size relation. The reported "Ebola R0 = 2.2" and "Influenza R0 = 1.7" were
therefore the Norovirus-driven hyperprior rescaled by each arm's literature
offset, not estimates from Ebola or Influenza data.

This also explains every anomaly previously attributed to other causes:

- ESS near 7,000 for e_R0 and flu_R0 -- sampling an unconstrained prior is easy
- CV of 216% -- that is the hyperprior width, not a posterior width
- flu_R0 barely moving when sigma_rw_flu was halved from 0.30 to 0.15 -- the
  parameter was never connected to the data in the first place

### Fix

`core/bhmm/ode_surrogate.py` precomputes trajectories on a log-uniform R0 grid
and interpolates differentiably inside the model:

```
idx = (log R0 - log R0_min) / dlog
i0  = floor(idx);  w = idx - i0
traj(R0) = (1-w) * GRID[i0] + w * GRID[i0+1]
```

Verified post-fix gradients at the initial point:

```
e_log_R0_raw    |d logp/d theta| =   27.1    identified
flu_log_R0_raw  |d logp/d theta| = 4002      identified
n_log_R0_raw    |d logp/d theta| =  542.4    identified
```

### Precedent

Emulating an expensive simulator on a parameter grid and interpolating is
standard in Bayesian calibration of computer models: Kennedy MC & O'Hagan A
(2001) J R Stat Soc B 63(3):425-464; Conti S & O'Hagan A (2010) J Stat Plan
Inference 140(3):640-651. We use exact ODE solutions rather than a GP emulator,
so there is no emulator uncertainty to propagate, only bounded interpolation
error, which is measured and written to `tables/ode_grid_interpolation_error.json`.

### Permanent guard

`core/diagnostics.identifiability_gate` computes d logp / d theta for every
estimated parameter before sampling; `scripts/04_run_bhmm.py` aborts if any
gradient is zero. `tests/test_unified_engine.py::test_r0_identifiable` enforces
the same condition in CI.

---

## 2. CRITICAL: wrong observable for Ebola

`onset` counts NEW symptom onsets per day: an incidence flux. v5 compared it to
the prevalence I(t), which integrates and lags onsets by the infectious period.
The correct observable is the flux into the symptomatic compartment, sigma*E(t).

Measured on the eligible rows with a constant-transmission fit:

| Observable | Fitted R0 | Residual lag-1 ACF | Ljung-Box p |
|---|---|---|---|
| prevalence I(t)  (v5) | 1.02 | +0.678 | < 1e-10 |
| incidence sigma*E(t)  (v6) | 1.92 | -0.037 | 0.885 |

The v6 value of 1.92 falls inside the Althaus (2014) range of 1.51-2.53
(central 1.83). The v5 value did not.

---

## 3. CRITICAL: wrong observable for Influenza

`in_bed` counts boys CONFINED TO BED. Boys were infectious roughly two days
before confinement and remained in bed roughly five days, largely withdrawn from
contact. in_bed is therefore a confinement compartment downstream of the
infectious compartment, not the infectious compartment itself.

> "it is erroneous to interpret the 'confined-to-bed' persons (B) as the
> 'infected and infectious' (I) in the classical SEIR model"
> -- Kalachev et al., discussed in Avilov KK, Li Q, Lin L, Demirhan H, Stone L,
> He D (2024). The 1978 English boarding school influenza outbreak: where the
> classic SEIR model fails. J R Soc Interface 21(220):20240394.

v6 uses S -> E -> I -> Q -> R with in_bed mapped to Q, and Q excluded from the
force of infection.

Symptom of the v5 error: the v5 ODE base trajectory rose monotonically from 1.0
to 6.85 across the 14-day window and peaked at day 36, while the data peaks at
day 6 with 282. The GRW was therefore doing 100% of the shape work.

---

## 4. IMPORTANT: Influenza seed offset

The index case was infected by 1978-01-10 and febrile 1978-01-15 to 01-18; the
series begins 1978-01-22 (RECON `influenza_england_1978_school`; BMJ
1978;1(6112):587). Forcing model time 0 onto observation day 0 assumes the
epidemic began when recording began.

t_seed is fixed at 11 days: inside the documented 8-12 day window, and the value
that minimises the Poisson deviance of the constant-beta fit within it. It is
NOT estimated -- with 14 observations, jointly identifying t_seed and R0
reintroduces the non-identifiability this version exists to remove. Sensitivity
over {7, 9, 11, 13} is prespecified.

---

## 5. IMPORTANT: Ebola GRW removed

The decision rule stated in the project's own GRW documentation is: include a
GRW only if the residuals of the constant-transmission fit show |lag-1 ACF| >
0.20 or Ljung-Box p < 0.05.

v5 measured the ACF on the RAW data (lag-1 = +0.678) rather than on residuals
from a fitted mechanistic model. With the corrected observable, the residual
lag-1 ACF is -0.037 with Ljung-Box p = 0.885. The rule is not met.

The Ebola GRW was compensating for the wrong observable, not for genuine
time-varying transmission. Removing it deletes 139 latent parameters and takes
the latent-to-observation ratio from 193/139 = 1.39 down to 0. This addresses
the v5 Ebola R0 CV of 216% at its cause rather than by tightening a prior.

A GRW can still be forced with `--grw-ebola` for sensitivity.

---

## 6. IMPORTANT: non-centred GRW

The centred form puts sigma_rw in the conditional scale of every latent state,
creating a funnel between sigma_rw and the walk -- the same pathology that the
non-centred R0 reparameterisation removes. It was harmless in v5 only because R0
was not in the likelihood and the walk had nothing to compete with.

With R0 live, a 2-chain trial of the centred form gave 4 divergences, Rhat 1.46
and ESS 3 on flu_R0 and sigma_rw_flu. The non-centred equivalent:

```
eps_t ~ Normal(0, 1)
log_rw_t = sigma_rw * sum_{k<=t} sqrt(dt_k) * eps_k
```

gives Var[log_rw_0] = sigma_rw^2 (init matches innovation) and
Var[log_rw_t - log_rw_{t-1}] = sigma_rw^2 * dt_t, i.e. the same process with
Normal(0,1) sampling geometry.

Precedent: Betancourt M & Girolami M (2015), Hamiltonian Monte Carlo for
Hierarchical Models; Papaspiliopoulos O, Roberts GO & Skold M (2007) Stat Sci
22(1):59-73.

---

## 9. MODERATE: literature R0 offset removed

v5 added `log(R0_base_d)` inside the non-centred Deterministic, shifting each
arm's prior location toward its literature value. That was harmless only while
R0 was unidentified. Now that the data informs R0, retaining the offset would
double-count the literature -- once in the prior location and again when a
reader compares the posterior to the same literature range. Each arm's R0 is now
driven by its own likelihood plus the shared hyperprior only.

---

## 10. MODERATE: honest prior justification for sigma_rw

v5 justified sigma_rw with

```
sigma_max = log(R0_hi / R0_lo) / sqrt(T)
```

presented as "Lemma 1 (Durbin & Koopman 2012, sec 2.3)" with a proof. **That
citation is overstated.** Durbin & Koopman sec 2.3 discusses the signal-to-noise
ratio of the local level model; it does not contain that lemma. A reviewer
checking the reference would not find it. The formula is a reasonable heuristic
but it is the author's own construction.

v6 justifies the prior by prior predictive checking, which is a standard,
citable procedure with a defined place in the Bayesian workflow:

- Gelman A et al. (2020). Bayesian Data Analysis, 3e. CRC Press, sec 6.4.
- Gabry J, Simpson D, Vehtari A, Betancourt M & Gelman A (2019). Visualization
  in Bayesian workflow. J R Stat Soc A 182(2):389-402.
- Schad DJ, Betancourt M & Vasishth S (2021). Toward a principled Bayesian
  workflow in cognitive science. Psychological Methods 26(1):103-126.

`scripts/03_prior_predictive_calibration.py` simulates the GRW forward from each
candidate prior and reports the implied multiplicative range of beta(t) across
the observation window. A prior is accepted when the 95th percentile of that
range stays below a 3x plausibility ceiling.

Results:

| Arm | T | Selected | Median beta range | p95 | Author heuristic (reported, not cited) |
|---|---|---|---|---|---|
| Ebola | 139 | HalfNormal(0.02) | 1.27x | 2.22x | 0.0438 |
| Influenza | 14 | HalfNormal(0.10) | 1.32x | 2.73x | 0.2621 |

The heuristic is still printed alongside, explicitly labelled as an
author-derived rule of thumb, so the two lines of reasoning can be compared.

---

## 11. MODERATE: flu_alpha_nb is unidentified, not estimated

The Influenza Var/Mean of 103.6 reflects the epidemic arc across 14 days, not
within-day count noise, so the NB log-likelihood is nearly flat in the
dispersion parameter (Cameron AC & Trivedi PK (2013), Regression Analysis of
Count Data 2e, sec 3.3).

| Prior | Prior mean | Posterior mean | Ratio |
|---|---|---|---|
| HalfNormal(50) | 39.9 | 50.8 | 1.27 |
| HalfNormal(200) | 159.6 | 180 | 1.13 |

Widening the prior did not identify the parameter, exactly as predicted. v6
keeps HalfNormal(200) as a DISCLOSURE choice and reports the parameter as
unidentified (Gelman et al. 2020 BDA3 sec 13.3). `tables/prior_dominance.csv`
records this automatically each run.

---

## Verification

```bash
python tests/test_unified_engine.py     # includes the identifiability gate
python scripts/02_prefit_diagnostics.py # regenerates all pre-fit evidence
python scripts/03_prior_predictive_calibration.py
```

---

## 12. IMPORTANT: ascertainment intercept conditioned on the observation process

### Evidence

A Poisson-deviance profile over (R0, log_intercept) for the Influenza arm shows
a clean diagonal ridge -- the two parameters trade off almost exactly because
both scale the same trajectory:

| R0 \ log_intercept | -0.60 | -0.30 | 0.00 | +0.30 | +0.60 |
|---|---|---|---|---|---|
| 2.0 | 4081 | 3395 | 2787 | 2286 | **1928** |
| 2.5 | 1823 | 1495 | **1370** | 1520 | 2042 |
| 3.0 | 976 | **947** | 1226 | 1922 | 3181 |
| 3.5 | **863** | 1000 | 1503 | 2502 | 4169 |
| 4.0 | **1050** | 1263 | 1871 | 3010 | 4866 |

The per-row minimum walks from +0.6 to -0.6 as R0 rises. This ridge saturated
the NUTS tree depth: a quick run took over 20 minutes and still had not
finished, versus 157 s after the fix.

### Fix

An ascertainment intercept is estimated only where the observation process can
actually miss cases.

| Arm | Observation process | Intercept | Justification |
|---|---|---|---|
| Ebola | retrospective, incomplete surveillance | estimated, Normal(0, 0.5) | Khan et al. (1999) describe case finding improving sharply once the international team arrived in May 1995 |
| Influenza | daily census of every boy confined to bed in a closed 763-boy school | fixed at 0 (ascertainment = 1) | BMJ 1978;1(6112):587 -- the school counted everyone; there is no under-ascertainment to absorb |

This is a data-collection fact, not a tuning choice.

---

## 13. IMPORTANT: Influenza GRW disabled by default (saturated latent process)

### Evidence

With T = 14 a daily GRW has 14 free innovations against 14 observations. The
latent process can reproduce any data vector exactly, so R0, sigma_rw and
alpha_nb become mutually unidentifiable.

Measured, 3-chain trials, identical seed:

| Configuration | flu_R0 Rhat | flu_R0 ESS | sigma_rw_flu Rhat | flu_alpha_nb | divergences |
|---|---|---|---|---|---|
| GRW on | 1.80 | 3 | 1.82 | prior-dominated | 0 |
| GRW off | **1.00** | **1446** | n/a | **0.93, data-informed** | 0 |

With the GRW off, every parameter in the model reaches Rhat <= 1.01 with
ESS >= 321. Notably `flu_alpha_nb` stops being prior-dominated: its posterior
mean is 0.93 against a prior mean of 160, because there is now a real
mechanistic trajectory to measure deviation from rather than a latent process
that already fits perfectly.

### Interpretation

The residual lag-1 ACF of +0.80 remains. That is a genuine, published misfit for
this dataset (Avilov et al. 2024), not something that should be absorbed by 14
free parameters at the cost of making R0 unidentifiable. It is reported in
`tables/residual_serial_correlation.csv` every run and carried into the debug
PDF.

Enable with `--grw-influenza` for sensitivity. Precedent for coarsening rather
than saturating a time-varying transmission term: Flaxman et al. (2020) Nature
584:257-261 and Abbott et al. (EpiNow2) both use weekly rather than daily steps
for exactly this reason.

---

## Net effect on the default model

| | v5 | v6 |
|---|---|---|
| Free parameters | 240 | 10 |
| R0 arms identified | 1 of 3 | 3 of 3 |
| GRW latent variables | 207 | 0 |
| Ebola R0 | 2.2, CV 216% (prior draw) | 1.93, sd 0.034 (Althaus range 1.51-2.53) |
| flu_alpha_nb | prior-dominated | data-informed |
| Quick-run time | ~35 min | ~4 min |


---

# v6 to v7

| # | Change | Severity | Evidence |
|---|---|---|---|
| 14 | Measles Hagelloch 1861 added as a fourth arm | MAJOR | v6 rested its three-way comparison on Influenza, which has a documented identification conflict and 18x worse precision than the other arms |
| 15 | Per-disease ODE grid bounds | IMPORTANT | measles R0 is 12-18; the shared ceiling of 15 would have clipped and silently flattened the gradient |
| 16 | HEADLINE_ARMS / CAVEATED_ARMS split | IMPORTANT | prevents flu_R0 being reported as a clean comparator by accident |
| 17 | Final-size check skipped for Measles | MODERATE | AR = 188/188 = 100% makes the K-M relation degenerate; reporting a "conflict" there would be spurious |
| 18 | `--drop-influenza` sensitivity switch | MODERATE | lets the caveated arm's influence on the hyperprior be measured directly |

## 14. Measles Hagelloch 1861 added

### Data provenance

Extracted from `hagelloch.df` in the R `surveillance` package via the CRAN source
mirror, parsed with the pure-Python `rdata` reader. No values were transcribed,
generated or imputed. `scripts/00_extract_measles.py --verify` re-derives the
files from the source object and compares hashes, so the committed CSVs are
auditable. Full citation chain in `data/raw/measles/PROVENANCE.md`.

Verified facts: 188 children (complete village cohort), 12 deaths (CFR 6.38%),
rash onsets 1861-11-03 to 1862-01-27, 86-day window, peak 26 on day 34, classes
preschool 90 / 1st 30 / 2nd 68.

### Pre-fit verdict

| Quantity | Value | Decision |
|---|---|---|
| Var/Mean | 10.64 | NegBinomial indicated |
| Residual lag-1 ACF | -0.005 | GRW NOT warranted (rule: \|r\| > 0.20) |
| Ljung-Box p | ~0.99 | no residual temporal structure |
| Constant-transmission R0 | 8.01 | below the classical 12-18; see caveat |
| Attack rate | 188/188 = 100% | final-size relation degenerate |
| Identifiability gradient | 221 | PASS |

The arm behaves like Ebola: correct incidence observable, clean residuals, no
GRW, ascertainment fixed at 1 because the linelist is a census.

### Caveats carried forward

The constant-transmission fit peaks at about 7 against an observed 26, because
homogeneous mixing cannot reproduce household and classroom bursts (Neal PJ &
Roberts GO (2004) Biostatistics 5(2):249-261 model the contact structure
explicitly). NegBinomial dispersion absorbs the spikiness and the residual ACF
confirms no trend remains. Guerra FM et al. (2017) Lancet Infect Dis
17(12):e420-e428 find measles R0 estimates spanning 1-770 with median 15.9 and
argue the 12-18 canon is over-narrow, so a village estimate below it is not
automatically wrong. Compare the posterior 89% ETI, not the point fit.

## 15. Per-disease ODE grid bounds

`DiseaseConfig` now carries `grid_R0_lo` / `grid_R0_hi`. Measles uses
[0.5, 40.0]; the others keep [0.3, 15.0]. A posterior pressed against a grid
bound clips in `ODEGrid.interpolate` and the gradient flattens, which would
re-create the v5 non-identifiability silently.
`tests/test_unified_engine.py::test_measles_grid_covers_literature` enforces the
ceiling.

---

# v7 to v8

| # | Change | Severity | Evidence |
|---|---|---|---|
| 19 | Kermack-McKendrick solver replaced | **CRITICAL** | 12-step Newton was wrong below R0=1 and corrupted the Norovirus arm |
| 20 | Route B added for the undocumented pathogen | MAJOR | prior-predictive-only arm learns nothing; hierarchical borrowing does |
| 21 | Full sampler-process auditing | MAJOR | the run reported where it ended, not how it got there |
| 22 | Influenza reclassified as a BENCHMARK arm | IMPORTANT | every convergence check passed while the model demonstrably did not fit |
| 23 | All figure generation chained automatically | IMPORTANT | figure scripts had to be invoked by hand |

## 19. CRITICAL: the final-size solver was wrong below R0 = 1

### Evidence

The in-model solver used 12 Newton steps from AR_0 = 1 - exp(-R0). For R0 <= 1
the relation AR = 1 - exp(-R0 * AR) has the single root AR = 0, and Newton
approaches a boundary root only LINEARLY. Twelve steps stopped far short:

| R0 | 12-step Newton | true final size |
|---|---|---|
| 0.850 | 0.0933 | 0 |
| 0.900 | 0.1243 | 0 |
| 0.919 | 0.1373 | 0 |
| 1.000 | 0.1983 | 0 |

In the v7 production run the Norovirus posterior settled at **n_R0 = 0.919**
with a reported attack rate of 0.139, close to the observed 0.152. Every
convergence diagnostic passed: R-hat 1.0019, ESS 2546, zero divergences.

But 0.919 is subcritical. The true final size there is exactly zero. **R0 was
being identified by the solver's truncation error rather than by the
Kermack-McKendrick relation.** The correct R0 for an attack rate of 0.152 is
**1.085**.

`tables/final_size_consistency.csv` flagged the contradiction (implied AR 0.000
against reported 0.152) but the model itself did not, and the flag was not
treated as blocking.

### Fix

Damped fixed-point iteration from AR = 0.999, 80 steps, which converges to the
true root from above for every R0 and collapses to ~0 when R0 <= 1:

| R0 | fixed-point (80) | reference |
|---|---|---|
| 0.919 | 0.000919 | 0 |
| 1.085 | 0.153282 | 0.152480 |
| 1.500 | 0.582812 | 0.582812 |
| 8.000 | 0.999664 | 0.999664 |

Differentiable throughout, so NUTS gradients still reach R0.

### Consequence

**The v7 Norovirus R0 of 0.919 must be discarded.** Expect approximately 1.085
after the refit. Ebola, Measles and Influenza are unaffected: none of them route
R0 through the final-size relation.

## 20. Route B for the undocumented pathogen

`core/bhmm/zombie_arm.py`. Route A (prior predictive) restates the assumption
and learns nothing. Route B draws a new arm from the fitted hierarchy, so its
range is informed by the measured outbreaks without inventing a single
observation.

Measured on the v7 trace:

| Route | Median | 89% interval | Width | Relative spread |
|---|---|---|---|---|
| A — assumption restated | 3.04 | [2.57, 3.60] | 1.03 | 11% |
| B — learned from the arms | 1.70 | [0.56, 5.08] | 4.52 | 137% |

Route B is 4.4x wider and its median sits between the measured arms. That width
is honest uncertainty about an unmeasured pathogen; Route A's narrowness is
borrowed confidence from a choice.

Reference: Gelman A et al. (2020) BDA3 sec 5.5 (prediction for a new group).

## 21. Sampler-process auditing

`core/mcmc_audit.py` plus `scripts/15_mcmc_audit_report.py` write twelve CSVs
covering step size, tree depth, energy and E-BFMI, acceptance rate,
log-posterior trace, mass matrix, ESS growth, MCSE decay, rank uniformity,
posterior correlation and a per-parameter convergence ledger.

Verified on the v7 trace: E-BFMI 0.82-0.88 across four chains, tree-depth
saturation 0.05%, mean acceptance 0.945 against a target of 0.95, zero
rank-uniformity failures, strongest posterior correlation n_R0 x n_attack_rate
at 0.996 (expected — the attack rate is a deterministic function of R0).

The log-posterior trace is the closest analogue to a training-loss curve and ESS
growth to a learning curve.
