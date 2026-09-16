"""
scripts/00_extract_measles.py
=============================
Regenerate the Measles Hagelloch raw files from the original R data object.

This script exists so the measles files in data/raw/measles/ are auditable: a
reviewer can delete them, run this, and get byte-identical output. It is NOT
part of the normal pipeline (run_all.py skips it) because the files are already
committed.

    python scripts/00_extract_measles.py --verify     # check committed files match
    python scripts/00_extract_measles.py --refetch    # re-download and rewrite

Requires: pip install rdata
Source: https://raw.githubusercontent.com/cran/surveillance/master/data/hagelloch.RData
See data/raw/measles/PROVENANCE.md for full citation chain.
"""
import argparse, hashlib, sys, urllib.request, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "measles"
URL = "https://raw.githubusercontent.com/cran/surveillance/master/data/hagelloch.RData"
LINELIST = RAW / "measles_hagelloch_1861_linelist.csv"
DAILY = RAW / "measles_hagelloch_1861_daily.csv"

KEEP = ["PN", "NAME", "FN", "HN", "AGE", "SEX", "PRO", "ERU", "CL", "DEAD",
        "IFTO", "tPRO", "tERU", "tDEAD", "tR", "tI", "x.loc", "y.loc"]


def extract(rdata_bytes: bytes):
    import rdata, pandas as pd
    tmp = ROOT / "data" / "generated" / "_hagelloch.RData"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(rdata_bytes)
    conv = rdata.conversion.convert(rdata.parser.parse_file(tmp))
    tmp.unlink()
    df = conv["hagelloch.df"].copy()
    df.columns = [str(c) for c in df.columns]

    # PRO / ERU / DEAD are R Date values: days since 1970-01-01.
    for c in ["PRO", "ERU", "DEAD"]:
        df[c + "_date"] = pd.to_datetime(df[c], unit="D", origin="unix")

    d0 = df["ERU_date"].min()
    df["eru_day"] = (df["ERU_date"] - d0).dt.days
    df["pro_day"] = (df["PRO_date"] - d0).dt.days
    T = int(df["eru_day"].max()) + 1
    idx = pd.RangeIndex(0, T, name="day")

    daily = pd.DataFrame({
        "day": idx,
        "rash_onset": df.groupby("eru_day").size().reindex(idx, fill_value=0).values,
        "prodrome_onset": df.groupby("pro_day").size().reindex(idx, fill_value=0).values,
    })
    daily["date"] = d0 + pd.to_timedelta(daily["day"], unit="D")
    daily["N"] = 188

    linelist = df[[c for c in KEEP if c in df.columns]].copy()
    return linelist, daily


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--refetch", action="store_true")
    a = ap.parse_args()

    if not (a.verify or a.refetch):
        a.verify = True

    print(f"Fetching {URL}")
    try:
        raw = urllib.request.urlopen(URL, timeout=30).read()
    except Exception as e:
        sys.exit(f"download failed: {e}\nThe committed files in {RAW} remain valid; "
                 "this script only re-derives them.")
    print(f"  {len(raw)} bytes")

    ll, daily = extract(raw)
    print(f"  linelist {ll.shape}, daily {daily.shape}")
    print(f"  N=188? {len(ll) == 188}   deaths={int(ll.DEAD.notna().sum())}   "
          f"total rash onsets={int(daily.rash_onset.sum())}")

    if a.refetch:
        RAW.mkdir(parents=True, exist_ok=True)
        ll.to_csv(LINELIST, index=False)
        daily.to_csv(DAILY, index=False)
        print(f"  written: {LINELIST.name} {sha(LINELIST)[:16]}")
        print(f"  written: {DAILY.name} {sha(DAILY)[:16]}")
        return

    ok = True
    for path, fresh in [(LINELIST, ll), (DAILY, daily)]:
        if not path.exists():
            print(f"  MISSING {path.name}"); ok = False; continue
        tmp = path.with_suffix(".tmp")
        fresh.to_csv(tmp, index=False)
        same = sha(tmp) == sha(path)
        tmp.unlink()
        print(f"  {'MATCH   ' if same else 'MISMATCH'} {path.name}  {sha(path)[:16]}")
        ok &= same
    print("verification " + ("passed" if ok else "FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
