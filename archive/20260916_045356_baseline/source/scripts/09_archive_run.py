"""
scripts/09_archive_run.py
=========================
Snapshot everything specific to one run into archive/<run_id>[_<label>]/.

Each archive is self-contained and reproducible: it holds the trace, every
table and figure, every generated report, the exact model source that produced
them, the resolved run metadata, and a SHA-256 manifest of all of it.

Run:
    python scripts/09_archive_run.py                       # auto run_id from clock
    python scripts/09_archive_run.py --label baseline      # named experiment
    python scripts/09_archive_run.py --list                # list existing archives
    python scripts/09_archive_run.py --prune 10            # keep newest 10

Called automatically at the end of scripts/04_run_bhmm.py.
"""
import sys, json, shutil, hashlib, argparse, platform
from datetime import datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
ARCH = ROOT/"archive"; ARCH.mkdir(exist_ok=True)

# (source dir, destination subdir, glob patterns)
COPY = [
    ("tables",           "tables",        ["*.csv", "*.json"]),
    ("figures/prefit",   "figures/prefit",["*.png"]),
    ("figures/bhmm",     "figures/bhmm",  ["*.png"]),
    ("figures/debug",    "figures/debug", ["*.png"]),
    ("reports",          "reports",       ["*.pdf"]),
    ("data/generated",   "trace",         ["*.nc", "*.npz"]),
    ("core/bhmm",        "source/core/bhmm", ["*.py"]),
    ("core",             "source/core",   ["*.py"]),
    ("scripts",          "source/scripts",["*.py"]),
    ("docs",             "docs",          ["*.md"]),
    ("data/canonical",   "data/canonical",["*.csv", "*.json"]),
]

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def archive(run_id: str | None, label: str | None) -> Path:
    run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{run_id}_{label}" if label else run_id
    dest = ARCH/name
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    n = 0
    for src_rel, dst_rel, pats in COPY:
        src = ROOT/src_rel
        if not src.exists():
            continue
        out = dest/dst_rel; out.mkdir(parents=True, exist_ok=True)
        for pat in pats:
            for f in sorted(src.glob(pat)):
                if f.is_file():
                    shutil.copy2(f, out/f.name); n += 1

    # resolved run metadata
    meta = {}
    rm = ROOT/"tables"/"run_metadata.json"
    if rm.exists():
        meta = json.loads(rm.read_text())
    meta.update(archive_name=name, archived_at=datetime.now().isoformat(timespec="seconds"),
                python=platform.python_version(), platform=platform.platform(),
                n_files=n)
    try:
        import pymc, arviz, numpy, scipy, pytensor
        meta["versions"] = dict(pymc=pymc.__version__, arviz=arviz.__version__,
                                numpy=numpy.__version__, scipy=scipy.__version__,
                                pytensor=pytensor.__version__)
    except Exception:
        pass
    (dest/"RUN_METADATA.json").write_text(json.dumps(meta, indent=2))

    # manifest
    lines = []
    for f in sorted(dest.rglob("*")):
        if f.is_file() and f.name != "MANIFEST.sha256":
            lines.append(f"{sha(f)}  {f.relative_to(dest)}")
    (dest/"MANIFEST.sha256").write_text("\n".join(lines)+"\n")

    # human-readable index
    idx = [f"# Run archive: {name}", ""]
    if meta:
        idx += ["| key | value |", "|---|---|"]
        for k in ("run_id","label","seed","draws","chains","tune","target_accept",
                  "grid_nodes","use_grw_ebola","use_grw_influenza","divergences",
                  "n_bad_rhat","n_low_ess","elapsed_sec","timestamp"):
            if k in meta:
                idx.append(f"| {k} | {meta[k]} |")
        idx.append("")
    idx += ["## Contents", ""]
    for sub in sorted({p.parent.relative_to(dest) for p in dest.rglob("*") if p.is_file()}):
        files = sorted(p.name for p in (dest/sub).glob("*") if p.is_file())
        idx.append(f"- `{sub}/` ({len(files)} files)")
    idx += ["", "Verify integrity:", "", "```bash",
            f"cd archive/{name} && sha256sum -c MANIFEST.sha256", "```"]
    (dest/"README.md").write_text("\n".join(idx)+"\n")

    print(f"  archived {n} files -> archive/{name}")
    print(f"  manifest : archive/{name}/MANIFEST.sha256")
    return dest


def list_archives():
    rows = sorted(p for p in ARCH.iterdir() if p.is_dir())
    if not rows:
        print("  (no archives yet)"); return
    print(f"  {'archive':34s} {'files':>6s}  {'divs':>5s} {'seed':>10s}  label")
    for p in rows:
        m = {}
        f = p/"RUN_METADATA.json"
        if f.exists():
            try: m = json.loads(f.read_text())
            except Exception: pass
        nf = sum(1 for _ in p.rglob("*") if _.is_file())
        print(f"  {p.name:34s} {nf:6d}  {str(m.get('divergences','-')):>5s} "
              f"{str(m.get('seed','-')):>10s}  {m.get('label') or ''}")


def prune(keep: int):
    rows = sorted((p for p in ARCH.iterdir() if p.is_dir()), key=lambda p: p.name)
    for p in rows[:-keep] if keep < len(rows) else []:
        shutil.rmtree(p); print(f"  pruned {p.name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--label", default=None)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--prune", type=int, default=None)
    a = ap.parse_args()
    if a.list: list_archives()
    elif a.prune is not None: prune(a.prune)
    else: archive(a.run_id, a.label)
