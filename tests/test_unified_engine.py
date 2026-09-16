"""
tests/test_unified_engine.py
============================
Integrity tests. Run before every fit (run_all.py does this automatically).

The most important test is test_r0_identifiable: it asserts that every R0
parameter has a non-zero likelihood gradient. v5 shipped without this check and
consequently reported prior draws as posterior estimates for two of three arms.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd, pytensor.tensor as pt

ROOT = Path(__file__).resolve().parents[1]
CAN = ROOT/"data"/"canonical"


def _data():
    return (pd.read_csv(CAN/"ebola_kikwit_1995_analysis_ready.csv"),
            pd.read_csv(CAN/"norovirus_derbyshire_2001_analysis_ready.csv"),
            pd.read_csv(CAN/"influenza_england_1978_analysis_ready.csv"),
            pd.read_csv(CAN/"measles_hagelloch_1861_analysis_ready.csv"))


def test_data_shapes():
    e, n, f, m = _data()
    assert len(e) == 192 and int(e.analysis_eligible.sum()) == 139
    assert len(n) == 492 and int(n.ill.sum()) == 75
    assert len(f) == 14 and int(f.N.iloc[0]) == 763 and int(f.in_bed.max()) == 282
    assert (f.in_bed > 0).all(), "no structural zeros expected in influenza"
    assert len(m) == 86 and int(m.N.iloc[0]) == 188
    assert int(m.rash_onset.sum()) == 188, (
        "daily rash onsets must sum to the 188-child cohort; if not, the "
        "measles raw files have been altered -- rerun scripts/00_extract_measles.py --verify")
    assert int(m.rash_onset.max()) == 26
    print("  data shapes ok (4 arms)")


def test_km_attack_rate():
    from core.bhmm.bhmm_model import km_attack_rate
    ar = float(km_attack_rate(pt.as_tensor_variable(2.0)).eval())
    assert abs(ar - 0.7968) < 0.01, f"K-M at R0=2 wrong: {ar}"
    from core.diagnostics import km_R0_from_attack_rate
    r = km_R0_from_attack_rate(512/763)
    assert abs(r - 1.657) < 0.01, f"K-M inverse wrong: {r}"
    print(f"  K-M relation ok (R0=2 -> AR={ar:.4f}; AR=0.671 -> R0={r:.3f})")


def test_grid_is_not_flat_in_r0():
    """If the ODE grid does not move with R0, R0 cannot be identified."""
    from core.bhmm.bhmm_model import build_disease_grids
    e, n, f, m = _data()
    # G=160 is the minimum production grid. Coarser grids raise the
    # piecewise-linear surrogate error above the 2% bar (measured: 8% at G=60,
    # 1.9% at G=160, 1.1% at G=240 for the Ebola incidence grid).
    g = build_disease_grids(e, f, m, G=160)
    for k, gr in g.items():
        spread = float(np.max(gr.traj.std(axis=0) / (np.abs(gr.traj).mean(axis=0) + 1e-12)))
        assert spread > 0.01, f"{k} grid is flat in R0 (spread {spread:.2e})"
        err = gr.interpolation_error(n_test=8)
        assert err["max_rel_error"] < 0.03, f"{k} interpolation error too large: {err}"
        print(f"  {k} grid: R0 spread {spread:.2f}, interp err {err['max_rel_error']:.1e}")


def test_r0_identifiable():
    """THE critical test. Every R0 must have a non-zero likelihood gradient."""
    from core.bhmm.bhmm_model import build_disease_grids, build_full_bhmm
    from core.diagnostics import identifiability_gate
    e, n, f, m = _data()
    mod = build_full_bhmm(e, n, f, m, grids=build_disease_grids(e, f, m, G=160))
    g = identifiability_gate(mod, ["e_log_R0_raw", "mea_log_R0_raw",
                                   "flu_log_R0_raw", "n_log_R0_raw"])
    for _, r in g.iterrows():
        assert r.status == "PASS", (
            f"{r.parameter} has zero likelihood gradient -- it is NOT identified. "
            "This is the v5 defect; see core/bhmm/ode_surrogate.py.")
        print(f"  {r.parameter}: |grad| = {r.max_abs_grad:.4g}  identified")


def test_measles_grid_covers_literature():
    """Measles R0 is 12-18 classically; the grid ceiling must exceed it."""
    from core.bhmm.priors import MEASLES
    assert MEASLES.grid_R0_hi >= 25.0, (
        f"measles grid ceiling {MEASLES.grid_R0_hi} is too low; the posterior "
        "would clip against it and R0 would silently lose its gradient")
    assert MEASLES.grid_R0_lo <= 1.0
    print(f"  measles grid spans R0 [{MEASLES.grid_R0_lo}, {MEASLES.grid_R0_hi}]")


def test_measles_provenance_intact():
    """The measles files must be the extracted originals, not regenerated."""
    ll = pd.read_csv(ROOT/"data"/"raw"/"measles"/"measles_hagelloch_1861_linelist.csv")
    assert len(ll) == 188, "linelist must contain exactly 188 children"
    assert int(ll["DEAD"].notna().sum()) == 12, "12 deaths expected (CFR 6.38%)"
    assert set(ll["CL"].unique()) == {"preschool", "1st class", "2nd class"}
    assert (ROOT/"data"/"raw"/"measles"/"PROVENANCE.md").exists()
    print("  measles provenance intact (188 children, 12 deaths, 3 classes)")


def test_no_stale_arms():
    src = (ROOT/"core"/"bhmm"/"bhmm_model.py").read_text().lower()
    assert "rabies" not in src.replace("rabies car", ""), "stale Rabies references in model"
    assert "sezr_person_day" not in src, "model must never read synthetic zombie data"
    free = None
    print("  no stale arms, no synthetic data read by the model")


def test_prior_labels_complete():
    from core.bhmm.priors import prior_table
    rows = prior_table()
    valid = {"LITERATURE", "STRUCTURAL", "WEAK", "DISCLOSURE"}
    for r in rows:
        assert r["label"] in valid, f"unlabelled prior: {r}"
        assert r["source"], f"prior without a source: {r}"
    print(f"  all {len(rows)} priors labelled and sourced")


if __name__ == "__main__":
    print("Integrity tests:")
    test_data_shapes()
    test_km_attack_rate()
    test_measles_grid_covers_literature()
    test_measles_provenance_intact()
    test_grid_is_not_flat_in_r0()
    test_r0_identifiable()
    test_no_stale_arms()
    test_prior_labels_complete()
    print("All tests passed.")
