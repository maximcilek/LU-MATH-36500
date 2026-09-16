# Unified CTMC-BHMM Suite - v6

A Bayesian Hierarchical Mechanistic Model fitting a compartmental epidemic model
simultaneously to three real human outbreaks, with a zombie SEZR prior-predictive
arm for structural comparison.

## Read this first (v8)

**v8 fixes a critical solver bug that invalidates the v7 Norovirus R0.** The
Kermack-McKendrick final-size solver used 12 Newton steps, which is wrong below
R0 = 1: the v7 posterior settled at n_R0 = 0.919 (subcritical, true final size
zero) identified purely by the solver's truncation error. Correct value is
approximately 1.085. Ebola, Measles and Influenza are unaffected. Full evidence
in [`docs/PARAMETER_CHANGE_AUDIT.md`](docs/PARAMETER_CHANGE_AUDIT.md) section 19.

Also new in v8: Route B for the undocumented pathogen, full sampler-process
auditing, influenza reclassified as a benchmark arm, and every figure generated
automatically.

## Read this first (v6/v7 history)

v6 fixes a **critical defect in v5**: `R0` never entered the likelihood for the
Ebola or Influenza arms. Their reported posteriors were prior draws. Any results
from v5 for those two arms should be discarded. Full evidence and fix in
[`docs/PARAMETER_CHANGE_AUDIT.md`](docs/PARAMETER_CHANGE_AUDIT.md).

## Disease arms

| Arm | Data | Observable | Model | Source |
|---|---|---|---|---|
| Ebola Zaire | Kikwit DRC 1995, 139 eligible days | daily onset = incidence flux sigma*E(t) | SEIRD, NegBinomial | Khan et al. (1999) J Infect Dis 179:S76 |
| Measles | Hagelloch Germany 1861, 86 days, N=188 | daily rash onset = incidence flux sigma*E(t) | SEIRD, NegBinomial | Pfeilsticker (1863); Neal & Roberts (2004) Biostatistics 5:249 |
| Norovirus GII | Derbyshire school 2001, 492 students | binary ill per student | K-M final size, Bernoulli + 15 class REs | O'Neill & Marks (2005) Stat Med 24:2011 |
| Influenza H1N1 | Boarding school England 1978, 14 days, N=763 | daily in_bed = confinement Q(t) | SEIQR, NegBinomial | BMJ 1978;1(6112):587 |
| Zombie SEZR | none | none | prior-predictive only | Math 36500 simulation parameters |

**Four arms, three of them headline comparators.** `HEADLINE_ARMS = (ebola,
measles, norovirus)`; Influenza is in `CAVEATED_ARMS` and is retained as a
diagnostic stress test, not as a clean R0 estimate. Rationale in
[`docs/DATASET_SELECTION.md`](docs/DATASET_SELECTION.md). Run
`--drop-influenza` to omit it entirely.

**Influenza carries a documented identification conflict.** See
[`docs/INFLUENZA_1978_LIMITATION.md`](docs/INFLUENZA_1978_LIMITATION.md). Cite it
in any write-up that reports `flu_R0`.

## Data integrity

No synthetic, generated or simulated data enters any likelihood. All three real
arms use published peer-reviewed data. The zombie CSVs in `data/raw/zombie/` are
synthetic SEZR simulation output, are labelled as such, and are never read by any
model script -- enforced by `tests/test_unified_engine.py::test_no_stale_arms`.

## Quick start

```bash
pip install -r requirements.txt
pip install h5py            # NetCDF4 trace writing

python run_all.py --quick                 # end-to-end smoke test, ~10 min
python run_all.py --label baseline        # full production pipeline
```

or with make:

```bash
make quick
make all LABEL=baseline
make help
```

`run_all.py` chains: prepare data -> integrity tests -> pre-fit diagnostics ->
prior predictive calibration -> BHMM fit. The fit step then chains the entire
post-fit pipeline itself (summary, zombie sensitivity, debug report, publication
PDFs, archive) with no manual invocation.

## Pipeline

| Step | Script | Produces |
|---|---|---|
| 00 | `scripts/00_extract_measles.py` | re-derives the measles raw files from the R source object (`--verify` / `--refetch`); not part of the normal pipeline |
| 01 | `scripts/01_prepare_data.py` | `data/canonical/*.csv`, `data/SHA256SUMS.txt` |
| -- | `tests/test_unified_engine.py` | integrity gate incl. R0 identifiability |
| 02 | `scripts/02_prefit_diagnostics.py` | `figures/prefit/*.png`, `tables/prefit_*` |
| 03 | `scripts/03_prior_predictive_calibration.py` | GRW prior calibration + figure |
| 04 | `scripts/04_run_bhmm.py` | trace, all diagnostic tables, `figures/bhmm/*` |
| 05 | `scripts/05_summarize.py` | `tables/headline_posterior.csv` |
| 06 | `scripts/06_build_reports.py` | `reports/01..04_*.pdf` |
| 07 | `scripts/07_zombie_sensitivity.py` | prior sensitivity for the unknown-disease arm |
| 08 | `scripts/08_debug_report.py` | `figures/debug/*`, `reports/DEBUG_*.pdf` |
| 09 | `scripts/09_archive_run.py` | `archive/<run_id>[_<label>]/` |
| 10 | `scripts/10_full_results_report.py` | `reports/FULL_RESULTS_<tag>.pdf` + `.json` for cross-machine comparison |
| 11 | `scripts/11_prefit_onepager.py` | 2-page visual pre-fit summary |
| 12 | `scripts/12_prefit_pages.py` | one page per outbreak + 5-page overview |
| 13 | `scripts/13_disease_dossiers.py` | per-outbreak dossier PDFs + figure archive |
| 14 | `scripts/14_visual_atlas.py` | 76-figure visual atlas, each with its takeaway |
| 15 | `scripts/15_mcmc_audit_report.py` | `tables/mcmc_*.csv` + `reports/MCMC_AUDIT.pdf` |
| 16 | `scripts/16_zombie_routes_report.py` | `reports/ZOMBIE_ROUTES.pdf` — both routes side by side |
| 17 | `scripts/17_influenza_performance.py` | `reports/INFLUENZA_PERFORMANCE.pdf` — benchmark arm |

**Steps 05-17 all run automatically after step 04.** Nothing needs manual invocation.

Steps 05-09 run automatically after 04. Use `--no-chain` to fit only.

## Run archives

Every run snapshots into `archive/<run_id>[_<label>]/` containing the trace,
every table and figure, every report, the exact model source that produced them,
resolved metadata with library versions, and a SHA-256 manifest.

```bash
python scripts/09_archive_run.py --list        # list runs
python scripts/09_archive_run.py --prune 10    # keep newest 10
cd archive/<name> && sha256sum -c MANIFEST.sha256
```

## The identifiability gate

Before sampling, `scripts/04_run_bhmm.py` computes `d logp / d theta` at the
initial point for every estimated parameter. A zero gradient means the parameter
cannot be informed by the data, so its posterior will equal its prior. The run
**aborts** rather than burning a production fit. This gate exists because v5
shipped without it.

## Repository layout

```
core/
  bhmm/
    ode_surrogate.py   R0-indexed ODE grid + differentiable interpolation
    bhmm_model.py      the unified model
    priors.py          every prior with an epistemic label and source
    prior_predictive.py zombie arm (no data, no likelihood)
  diagnostics.py       reusable pre-fit and post-fit tests with decision rules
  pdf_utils.py         DejaVu Unicode fonts, shared PDF helpers
data/
  raw/                 published raw files + labelled synthetic zombie CSVs
    measles/           linelist + daily series + PROVENANCE.md (citation chain)
  canonical/           analysis-ready, built by scripts/01
docs/                  audit trail, data decisions, known limitations
figures/{prefit,bhmm,debug}
reports/               generated PDFs
archive/               per-run snapshots
```

## Key references

- Anderson RM & May RM (1991). Infectious Diseases of Humans. Oxford UP.
- Avilov KK et al. (2024). J R Soc Interface 21(220):20240394.
- Guerra FM et al. (2017). Lancet Infect Dis 17(12):e420-e428.
- Neal PJ & Roberts GO (2004). Biostatistics 5(2):249-261.
- Betancourt M & Girolami M (2015). HMC for Hierarchical Models.
- Diekmann O, Heesterbeek JAP & Roberts MG (2010). J R Soc Interface 7(47):873.
- Gabry J et al. (2019). J R Stat Soc A 182(2):389-402.
- Gelman A et al. (2020). Bayesian Data Analysis, 3e.
- Kennedy MC & O'Hagan A (2001). J R Stat Soc B 63(3):425-464.
- Kermack WO & McKendrick AG (1927). Proc Roy Soc A 115(772):700-721.
- Lloyd AL (2001). Theor Popul Biol 60(1):59-71.
- Papaspiliopoulos O, Roberts GO & Skold M (2007). Stat Sci 22(1):59-73.
