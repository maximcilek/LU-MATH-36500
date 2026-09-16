# Influenza England 1978: a documented identification conflict

**This document must be cited in any write-up that reports `flu_R0`.**

---

## The conflict, in one table

| Identification target | Implied R0 | Implied attack rate |
|---|---|---|
| Reported attack rate 512/763 = 67.1% (Kermack-McKendrick) | **1.66** | 67.1% (by construction) |
| Prevalence time course, exponential residence times | **~3.3** | ~96% |
| Prevalence time course, Erlang(5,5,5) residence times | **~15.5** | ~100% |
| Avilov et al. (2024) delay-differential model | **8.14** | fitted jointly |
| Typical influenza literature range | 1-4 | - |

No setting reproduces both the epidemic's speed and its final size.

## This is a known, published open problem

Avilov KK, Li Q, Lin L, Demirhan H, Stone L, He D (2024). *The 1978 English
boarding school influenza outbreak: where the classic SEIR model fails.*
J R Soc Interface 21(220):20240394.

Their findings, which our own profiling reproduces independently:

- SIR and SEIR models with a naive starting population converge to a ~100%
  attack rate against the reported 67%.
- SIR and SEIR models with estimated non-zero initial immunity can reach the
  correct attack rate, but fit the prevalence curve poorly (RMSE 16-16.5
  persons) and significantly overestimate the outbreak timespan.
- Non-exponential residence times are required for a good fit. Their final model
  needed constant delays in E and I and Erlang-distributed residence in B and C.
- Their central R0 estimate is 8.14.

Quoted therein: "there is no known model of any kind in the literature that has
correctly described both the time course of the epidemic and the final size."

Ahmad et al. (2025), *Model fit vs. predictive reliability: a case study of the
1978 influenza outbreak*, Sci Rep 15 (s41598-025-26072-3), flag the 8.14 figure
as "substantially higher than the typical range of R0 in [1,4] for influenza
epidemics" and note the discrepancy "warrants further scrutiny".

## What we reproduced independently

With the corrected SEIQR structure and t_seed = 11:

| Erlang (kE,kI,kQ) | Poisson deviance | Fitted R0 | Residual lag-1 ACF |
|---|---|---|---|
| (1,1,1) exponential | 505.7 | 3.3 | +0.79 |
| (3,3,3) | 638.1 | 3.3 | +0.81 |
| (5,5,5) at t_seed=3 | 88.5 | 15.5 | +0.59 |

Narrower generation intervals sharpen the curve but require a higher R0 for the
same growth rate, which pushes the implied attack rate further above the
reported 67%. The trade-off is structural, not a tuning failure.

## What this repository does about it

1. **Primary analysis uses exponential residence times (k=1)** so R0 keeps its
   conventional meaning and stays comparable to the other arms. Erlang shapes
   run as prespecified sensitivity `S_FLU_ERLANG`.
2. **`scripts/02_prefit_diagnostics.py` computes and plots the conflict**
   before any fitting, in panel E of `figures/prefit/influenza_prefit_panel.png`.
3. **`scripts/04_run_bhmm.py` runs a final-size consistency check** every run and
   writes `tables/final_size_consistency.csv`, which flags CONFLICT rather than
   silently reporting a single number.
4. **`scripts/08_debug_report.py` carries the limitation into the debug PDF**
   under "standing limitations".
5. **`core/bhmm/priors.py` marks `R0_lit_lo`/`R0_lit_hi` as CONTESTED** and uses
   them for reporting only, never as a prior on R0.

## Recommendation

This dataset is famous precisely because it breaks compartmental models. It is a
legitimate arm if the conflict is stated, and it can be framed as a strength:
the suite includes a dataset known to challenge classic models, and the
diagnostics correctly detect the misfit rather than hiding it.

If the goal is a clean three-way R0 comparison, consider replacing it. Candidate
replacements, all human, all closed-population, none with a known
SEIR-breaking pathology:

| Candidate | Setting | Why it fits the suite |
|---|---|---|
| SARS Singapore 2003 | Hospital + community linelist | Human, well-characterised R0 2.2-3.6, onset and discharge dates. Needs the quarantine switch ON. |
| Measles Hagelloch 1861 | 188 children, one German village | Classic closed-population dataset with individual-level data; in the `outbreaks` R package. |
| H1N1 school outbreak 2009 | Various documented school clusters | Directly comparable to the Norovirus school setting. |

Swapping the arm is a configuration change: add a `DiseaseConfig` to
`core/bhmm/priors.py` and a grid entry in `build_disease_grids`.
