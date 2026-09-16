"""
scripts/01_prepare_data.py
==========================
Build analysis-ready canonical datasets from the real raw data files.

No synthetic data is created anywhere. Every canonical row derives from a
published raw file in data/raw/. The zombie CSVs in data/raw/zombie/ are
synthetic SEZR simulation output, are labelled as such in that directory's
README, and are never read by this script or by any model script.

Run:  python scripts/01_prepare_data.py
"""
import sys, hashlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, CAN = ROOT / "data" / "raw", ROOT / "data" / "canonical"
CAN.mkdir(parents=True, exist_ok=True)

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def prepare_ebola():
    df = pd.read_csv(RAW / "ebola" / "ebola_kikwit_1995.csv", parse_dates=["date"])
    df["day"] = (df.date - df.date.min()).dt.days
    # analysis_eligible: active surveillance only. reporting=FALSE rows are
    # structural zeros and a perfect separator; retained but excluded downstream.
    df["analysis_eligible"] = df.reporting.astype(bool)
    # 14-day lagged cumulative CFR: DESCRIPTIVE ONLY, never a model input.
    df["cfr_cumulative_lag14"] = (
        df.death.cumsum() / df.onset.shift(14).cumsum().replace(0, np.nan)
    ).clip(0, 1)
    out = CAN / "ebola_kikwit_1995_analysis_ready.csv"
    df.to_csv(out, index=False)
    print(f"  Ebola      : {len(df)} rows, {int(df.analysis_eligible.sum())} eligible, "
          f"{int(df.onset.sum())} onsets | {sha(out)[:12]}")


def prepare_norovirus():
    df = pd.read_csv(RAW / "norovirus" / "norovirus_derbyshire_2001_school.csv")
    # OUTCOME IS ill, NOT day_absent. 126 of 417 well students were absent;
    # using absence as the outcome misclassifies them as ill.
    df["ill"] = (df.start_illness > 0).astype(int)
    df["class10_outlier_flag"] = (df["class"] == 10).astype(int)
    df["class3_zero_flag"] = (df["class"] == 3).astype(int)
    out = CAN / "norovirus_derbyshire_2001_analysis_ready.csv"
    df.to_csv(out, index=False)
    print(f"  Norovirus  : {len(df)} students, {int(df.ill.sum())} ill "
          f"(AR {df.ill.mean():.3f}) | {sha(out)[:12]}")


def prepare_influenza():
    df = pd.read_csv(RAW / "influenza" / "influenza_england_1978.csv", parse_dates=["date"])
    # in_bed = boys CONFINED TO BED. This is a confinement compartment, not the
    # infectious compartment -- see docs/INFLUENZA_1978_LIMITATION.md.
    df["analysis_eligible"] = True
    df["day_of_week"] = df.date.dt.day_name()
    # Reported attack rate from the source publication, for the final-size check.
    df["reported_total_ill"] = 512
    out = CAN / "influenza_england_1978_analysis_ready.csv"
    df.to_csv(out, index=False)
    print(f"  Influenza  : {len(df)} days, N={int(df.N.iloc[0])}, "
          f"peak in_bed={int(df.in_bed.max())} on day {int(df.day[df.in_bed.idxmax()])} "
          f"| {sha(out)[:12]}")


def prepare_measles():
    """
    Measles Hagelloch 1861. rash_onset is an INCIDENCE series (new symptomatic
    cases per day), compared to the flux sigma*E(t) -- not to prevalence.

    All 86 days are retained, including the 51 with zero onsets. Unlike Ebola,
    these are genuine observed zeros: Pfeilsticker observed the whole village
    continuously, so a zero means no child developed a rash that day. There is
    no surveillance gap and therefore no eligibility flag.
    """
    df = pd.read_csv(RAW / "measles" / "measles_hagelloch_1861_daily.csv",
                     parse_dates=["date"])
    df["analysis_eligible"] = True
    ll = pd.read_csv(RAW / "measles" / "measles_hagelloch_1861_linelist.csv")
    df["reported_total_cases"] = int(len(ll))          # 188
    df["reported_deaths"] = int(ll["DEAD"].notna().sum())  # 12
    out = CAN / "measles_hagelloch_1861_analysis_ready.csv"
    df.to_csv(out, index=False)
    assert int(df.rash_onset.sum()) == len(ll), (
        "daily rash onsets must sum to the linelist size")
    print(f"  Measles    : {len(df)} days, N={int(df.N.iloc[0])}, "
          f"{int(df.rash_onset.sum())} onsets, peak {int(df.rash_onset.max())} "
          f"on day {int(df.loc[df.rash_onset.idxmax(),'day'])}, "
          f"{int((df.rash_onset==0).sum())} zero days | {sha(out)[:12]}")


def write_sums():
    rows = [f"{sha(f)}  {f.relative_to(ROOT)}"
            for f in sorted(CAN.rglob('*')) if f.is_file()]
    rows += [f"{sha(f)}  {f.relative_to(ROOT)}"
             for f in sorted((RAW).rglob('*')) if f.is_file() and f.suffix in {'.csv', '.json'}]
    (ROOT / "data" / "SHA256SUMS.txt").write_text("\n".join(rows) + "\n")
    print(f"  SHA256SUMS : {len(rows)} files")


if __name__ == "__main__":
    print("Preparing canonical datasets (real published data only)...")
    prepare_ebola(); prepare_norovirus(); prepare_influenza()
    prepare_measles(); write_sums()
    print("Done.")
