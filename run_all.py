#!/usr/bin/env python3
"""
run_all.py
==========
Single-command pipeline for the unified CTMC-BHMM suite (v6).

    python run_all.py                          # full production pipeline
    python run_all.py --quick                  # fast end-to-end smoke test
    python run_all.py --label baseline         # name the archive
    python run_all.py --from 04                # resume from a given step

Steps:
    01  prepare canonical data from raw            scripts/01_prepare_data.py
    02  pre-fit diagnostics + figures/prefit       scripts/02_prefit_diagnostics.py
    03  prior predictive calibration               scripts/03_prior_predictive_calibration.py
    04  BHMM fit (chains 05/07/08/06/09 itself)    scripts/04_run_bhmm.py
    TEST verify model integrity before fitting     tests/test_unified_engine.py

Step 04 runs every post-fit step automatically, so nothing needs to be invoked
by hand. Pass --no-chain to 04 directly if you want the fit only.
"""
import argparse, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = [
    ("01", "prepare data",              "scripts/01_prepare_data.py",                 []),
    ("TEST", "integrity tests",         "tests/test_unified_engine.py",               []),
    ("02", "pre-fit diagnostics",       "scripts/02_prefit_diagnostics.py",           []),
    ("03", "prior predictive calib.",   "scripts/03_prior_predictive_calibration.py", []),
    ("04", "BHMM fit + full post-pipeline", "scripts/04_run_bhmm.py",                 []),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--label", default=None)
    ap.add_argument("--seed", type=int, default=20260914)
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--from", dest="start", default=None, help="resume from step id")
    ap.add_argument("--drop-influenza", action="store_true",
                    help="omit the caveated Influenza arm from the likelihood")
    a = ap.parse_args()

    started = a.start is None
    t0 = time.time()
    print("=" * 66)
    print("CTMC-BHMM v6  -  full pipeline")
    print("=" * 66)

    for sid, desc, script, extra in STEPS:
        if not started:
            if sid == a.start: started = True
            else:
                print(f"  [{sid}] skipped ({desc})"); continue
        args = list(extra)
        if sid == "04":
            args += ["--seed", str(a.seed)]
            if a.quick: args += ["--quick"]
            else: args += ["--draws", str(a.draws), "--chains", str(a.chains)]
            if a.label: args += ["--label", a.label]
            if a.drop_influenza: args += ["--drop-influenza"]
        print(f"\n{'='*66}\n[{sid}] {desc}\n{'='*66}")
        r = subprocess.run([sys.executable, str(ROOT/script)] + args, cwd=str(ROOT))
        if r.returncode != 0:
            print(f"\nFAILED at step {sid} (exit {r.returncode}). Pipeline stopped.")
            sys.exit(r.returncode)

    print(f"\n{'='*66}")
    print(f"PIPELINE COMPLETE in {time.time()-t0:.0f}s")
    print("  reports/   publication PDFs")
    print("  archive/   per-run snapshots (python scripts/09_archive_run.py --list)")
    print("=" * 66)


if __name__ == "__main__":
    main()
