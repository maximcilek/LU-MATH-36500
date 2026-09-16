# Measles Hagelloch 1861 — data provenance

## Source

Extracted from `hagelloch.df` in the R **surveillance** package, obtained from the
CRAN source mirror at

    https://raw.githubusercontent.com/cran/surveillance/master/data/hagelloch.RData

and parsed with the pure-Python `rdata` reader. No values were transcribed by
hand, generated, imputed or simulated.

## Original source of the observations

Pfeilsticker A (1863). *Beiträge zur Pathologie der Masern mit besonderer
Berücksichtigung der statistischen Verhältnisse.* MD Thesis, Eberhard Karls
Universität Tübingen.

Oesterle H (1992). *Statistische Reanalyse einer Masernepidemie 1861 in
Hagelloch.* MD Thesis, Eberhard Karls Universität Tübingen.

Made machine-readable by:
Neal PJ & Roberts GO (2004). Statistical inference and model selection for the
1861 Hagelloch measles epidemic. *Biostatistics* 5(2):249–261.
doi:10.1093/biostatistics/5.2.249

Distributed in:
Meyer S, Held L & Höhle M (2017). Spatio-temporal analysis of epidemic phenomena
using the R package surveillance. *Journal of Statistical Software* 77(11):1–55.
doi:10.18637/jss.v077.i11

## Files

| File | Contents |
|---|---|
| `measles_hagelloch_1861_linelist.csv` | 188 children, individual level, as extracted |
| `measles_hagelloch_1861_daily.csv` | daily counts derived from the linelist by `scripts/00_extract_measles.py` |

## Verified facts

| Quantity | Value |
|---|---|
| Children | 188 (complete village cohort) |
| Deaths | 12 (CFR 6.38%) |
| Rash-onset date range | 1861-11-03 to 1862-01-27 |
| Prodrome date range | 1861-10-30 to 1862-01-24 |
| Observation window | 86 days |
| Peak daily rash onset | 26, on day 34 |
| School classes | preschool 90, 1st class 30, 2nd class 68 |
| Age | median 7.0 years, range 0.5–15.0 |

## Columns in the linelist

`PN` person number, `NAME` family name, `FN` family number, `HN` household
number, `AGE` years, `SEX`, `PRO` prodrome date (R Date), `ERU` rash date
(R Date), `CL` school class, `DEAD` death date or NA, `IFTO` inferred infector
person number, `tPRO`/`tERU`/`tDEAD`/`tR`/`tI` continuous-time versions used by
Neal & Roberts, `x.loc`/`y.loc` household coordinates in metres.

## Why the rash-onset series is the modelled observable

`ERU` (rash onset) is the reliably recorded event: Pfeilsticker recorded the
rash date for every one of the 188 children. It is an INCIDENCE series, so the
model compares it to the flux into the symptomatic compartment, not to
prevalence — the same correction applied to Ebola in v6.
