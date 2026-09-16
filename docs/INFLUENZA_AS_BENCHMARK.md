# Influenza 1978 is a benchmark arm, not a result arm

## Why it is in the study

The 1978 English boarding-school series is one of the most frequently used test
cases for compartmental epidemic models. It is also one that those models
demonstrably cannot fit: Avilov KK, Li Q, Lin L, Demirhan H, Stone L & He D
(2024), *The 1978 English boarding school influenza outbreak: where the classic
SEIR model fails*, J R Soc Interface 21(220):20240394.

That makes it ideal for a job the other three arms cannot do — answering the
question *"how do you know your diagnostics would catch a bad fit?"*

## What it demonstrated

From the v7 production run (seed 20260914, 4 chains x 2000 draws):

| Dimension | Result | Verdict |
|---|---|---|
| R-hat | 1.0002 | excellent |
| ESS bulk | 6,254 | excellent |
| Divergences | 0 | excellent |
| Predictive coverage | 100% of points in the 90% band | passes the stated gate |
| Dispersion parameter | moved from prior mean 160 to 0.93 | now data-informed |
| **Residual lag-1 correlation** | **+0.83** | **FAIL (threshold 0.20)** |
| **Ljung-Box p** | **0.0017** | **FAIL** |
| **Implied vs reported attack rate** | **91% vs 67%** | **FAIL — 24 point gap** |

**Every convergence diagnostic passed and the model still did not fit.** That is
the finding. Convergence measures whether the sampler explored the posterior
correctly; it says nothing about whether the model is right. This arm separates
those two questions cleanly, on a dataset where the answer is known in advance.

It also shows that 100% predictive coverage is weak evidence: a band wide enough
always covers.

## Reporting rules

| Do | Do not |
|---|---|
| Present it as a performance benchmark for the framework | Quote its spread score beside Ebola, Measles and Norovirus as an equal |
| State that residual structure and the final-size gap persist | Describe 100% predictive coverage as evidence of good fit |
| Cite Avilov et al. (2024) whenever the spread score is mentioned | Attribute the misfit to sampling, tuning or implementation |
| Run the model with and without it (`--drop-influenza`) | Assume it is harmless to the other arms without checking |

## Where this is enforced in code

- `core/bhmm/priors.py` — `BENCHMARK_ARMS = ("influenza",)`, separate from
  `HEADLINE_ARMS`
- `scripts/04_run_bhmm.py` — `--drop-influenza` omits its likelihood entirely
- `scripts/17_influenza_performance.py` — generates
  `reports/INFLUENZA_PERFORMANCE.pdf` every run
- `tables/final_size_consistency.csv` — flags the contradiction automatically

## Why not simply remove it

Discarding a dataset because it is hard is a weaker position than including it
with the difficulty measured and reported. The arm costs nothing — the shared
prior is wide (fitted spread between diseases ~0.50 on the log scale), so
pooling is weak and the other arms are driven by their own likelihoods. The
`--drop-influenza` run confirms this directly rather than assuming it.
