.PHONY: all quick data prefit calib fit reports archive clean test help verify-measles

PY ?= python
SEED ?= 20260914
DRAWS ?= 2000
CHAINS ?= 4
LABEL ?=

help:
	@echo "make all      - full pipeline (data -> tests -> prefit -> calib -> fit -> reports -> archive)"
	@echo "make quick    - fast end-to-end smoke test"
	@echo "make data     - rebuild canonical datasets"
	@echo "make test     - integrity tests only"
	@echo "make prefit   - pre-fit diagnostics + figures/prefit"
	@echo "make calib    - prior predictive calibration"
	@echo "make fit      - BHMM fit (auto-chains all post-steps)"
	@echo "make reports  - rebuild PDFs from existing tables"
	@echo "make archive  - snapshot the current run"
	@echo "make clean    - remove generated artefacts (archives kept)"

all:
	$(PY) run_all.py --seed $(SEED) --draws $(DRAWS) --chains $(CHAINS) $(if $(LABEL),--label $(LABEL),)

quick:
	$(PY) run_all.py --quick --seed $(SEED) $(if $(LABEL),--label $(LABEL),)

data:    ; $(PY) scripts/01_prepare_data.py
test:    ; $(PY) tests/test_unified_engine.py
verify-measles: ; $(PY) scripts/00_extract_measles.py --verify
prefit:  ; $(PY) scripts/02_prefit_diagnostics.py
calib:   ; $(PY) scripts/03_prior_predictive_calibration.py
fit:     ; $(PY) scripts/04_run_bhmm.py --seed $(SEED) --draws $(DRAWS) --chains $(CHAINS) $(if $(LABEL),--label $(LABEL),)
reports: ; $(PY) scripts/06_build_reports.py
archive: ; $(PY) scripts/09_archive_run.py $(if $(LABEL),--label $(LABEL),)

clean:
	rm -rf tables/*.csv tables/*.json figures/prefit/*.png figures/bhmm/*.png \
	       figures/debug/*.png reports/*.pdf data/generated/*.nc data/generated/*.npz
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
	@echo "cleaned (archive/ preserved)"
