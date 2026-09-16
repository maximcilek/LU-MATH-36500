# Dataset selection and the role of each arm (v7)

## The four arms

| Arm | N | Observations | Observable | Role |
|---|---|---|---|---|
| Ebola Kikwit 1995 | ~1000 | 139 eligible days | daily onset incidence | headline comparator |
| Measles Hagelloch 1861 | 188 | 86 days | daily rash-onset incidence | headline comparator |
| Norovirus Derbyshire 2001 | 492 | 492 students | binary per student | headline comparator |
| Influenza England 1978 | 763 | 14 days | daily confined-to-bed | **caveated / stress test** |
| Zombie SEZR | — | none | none | prior-predictive reference |

`core/bhmm/priors.py` exposes `HEADLINE_ARMS = ("ebola", "measles", "norovirus")`
and `CAVEATED_ARMS = ("influenza",)`. Reporting code uses these so the
distinction cannot be lost by accident.

## Why Measles was added in v7 rather than replacing Influenza

The v6 suite rested its three-way R0 comparison on one arm with a documented
identification conflict and 18x less precision than the others:

| Arm (v6) | Observations | R0 | CV |
|---|---|---|---|
| Norovirus | 492 | 0.916 ± 0.048 | 5.2% |
| Ebola | 139 | 1.933 ± 0.034 | 1.8% |
| Influenza | 14 | 2.76 ± 0.90 | 33% |

Adding Measles restores a third clean, well-identified comparator. Keeping
Influenza costs nothing scientifically once it is labelled, and it earns its
place: it is a dataset *known* to break classic compartmental models, so it
demonstrates that the suite's diagnostics detect a misfit rather than hide it.
That is a methods contribution, not a liability — provided `flu_R0` is never
presented as a clean comparator.

If you would rather drop it entirely, `scripts/04_run_bhmm.py --drop-influenza`
omits the Influenza likelihood and leaves everything else intact. Running with
and without is also the cleanest test that the caveated arm is not influencing
the others through the shared hyperprior.

## Why Hagelloch specifically

- **Human, closed, complete.** 188 children, the entire child cohort of one
  German village. N is known exactly; nothing is assumed.
- **Sample size sits between the other arms.** 86 daily observations, against
  139 for Ebola and 492 for Norovirus. No precision outlier.
- **Census, not a sample.** Pfeilsticker recorded a rash date for every child;
  the 188 daily rash onsets sum to exactly the cohort size. Ascertainment is 1,
  so the intercept is fixed and there is no R0-intercept ridge (v6 audit sec 12).
- **No known model-breaking pathology.** Unlike the 1978 influenza series, there
  is no published result showing that compartmental models cannot fit it. It has
  been analysed repeatedly, most influentially by Neal & Roberts (2004).
- **Clean pre-fit verdict.** Residual lag-1 ACF of the constant-transmission fit
  is −0.005 with Ljung-Box p ≈ 0.99, so no GRW is warranted, exactly as for
  Ebola. Var/Mean is 10.6, so a NegBinomial observation model is indicated.
- **A genuinely different R0 regime.** Measles sits an order of magnitude above
  the other arms, which stretches the hierarchical hyperprior and makes the
  partial-pooling structure do real work instead of pooling three similar values.

## Two honest caveats on the Measles arm

**1. The final-size relation is degenerate here.** All 188 children were
infected, so the attack rate is 100%. Kermack–McKendrick gives R0 → ∞ as AR → 1,
meaning the final size provides only a *lower bound* on R0, not a point value.
R0 is therefore identified by the time course alone. This is **not** the
Influenza conflict: size and speed do not disagree, the size is simply
uninformative above a threshold. `scripts/04` skips the final-size check for
this arm rather than reporting a spurious conflict.

**2. The constant-transmission fit gives R0 ≈ 8.0, below the classical 12–18.**
The fitted trajectory also has a lower peak (≈7) than observed (26), because
homogeneous mixing cannot reproduce the household- and classroom-driven bursts
that Neal & Roberts (2004) model explicitly with contact structure. The
NegBinomial dispersion absorbs this spikiness, and the residual ACF confirms no
temporal trend remains. Two things to keep in mind when reporting:

- Guerra FM et al. (2017) *Lancet Infect Dis* 17(12):e420–e428 reviewed measles
  R0 estimates and found a range of 1–770 with a median of 15.9, arguing that
  the 12–18 canon is over-narrow and setting-dependent. A village estimate below
  it is not automatically wrong.
- The 8.0 figure is a Poisson-deviance point fit used only for the pre-fit
  audit. The posterior under the NegBinomial likelihood and the shared
  hyperprior will differ, and the 89% ETI is what should be compared against the
  literature range, not the point estimate.

If the posterior ETI excludes 12–18 entirely, say so plainly and attribute it to
homogeneous-mixing misspecification in a strongly clustered village population,
citing Neal & Roberts (2004).

## Alternatives considered and not taken

**SARS Singapore 2003.** Human, 238 cases, R0 well characterised at 2.2–3.6.
Rejected as the primary addition because its strong nosocomial component
requires the quarantine switch ON, making it structurally a second Ebola rather
than a distinct regime. Still a reasonable fourth or fifth arm if wanted.

**2009 H1N1 school clusters.** Directly comparable to the Norovirus school
setting, but the published series are mostly small and heterogeneous in
definition, which reintroduces the precision problem Measles was chosen to fix.

**Replacing Influenza outright.** Available at any time via `--drop-influenza`,
but discarding a dataset because it is hard is a weaker position than including
it with the difficulty measured and reported.

## References

- Anderson RM & May RM (1991). *Infectious Diseases of Humans.* Oxford UP, Table 4.1.
- Guerra FM, Bolotin S, Lim G, et al. (2017). The basic reproduction number (R0)
  of measles: a systematic review. *Lancet Infect Dis* 17(12):e420–e428.
- Meyer S, Held L & Höhle M (2017). Spatio-temporal analysis of epidemic
  phenomena using the R package surveillance. *J Stat Softw* 77(11):1–55.
- Neal PJ & Roberts GO (2004). Statistical inference and model selection for the
  1861 Hagelloch measles epidemic. *Biostatistics* 5(2):249–261.
- Oesterle H (1992). *Statistische Reanalyse einer Masernepidemie 1861 in
  Hagelloch.* MD Thesis, Universität Tübingen.
- Pfeilsticker A (1863). *Beiträge zur Pathologie der Masern.* MD Thesis,
  Universität Tübingen.
