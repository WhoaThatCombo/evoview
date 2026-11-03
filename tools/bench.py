"""Run evoview --bench over a fixed set of cars and tracks; write JSON.

    python tools/bench.py                          # full set at 1080p + 4K -> L:/evoview_work/bench/<rev>.json
    python tools/bench.py --compare tools/bench_baseline.json
    python tools/bench.py --only ks_bmw_m4_gt3 --seconds 5
    python tools/bench.py --out tools/bench_baseline.json   # (re)record the baseline

One run at a time, never in parallel: the numbers are only comparable when
nothing else is loading the GPU or the disk. Each run is killed if its
working set passes --ram-limit-gb, so a track that blows up cannot take the
PC down with it.

The PC is shared (builds, other viewers, Wallpaper Engine), so one run of a
sub-millisecond car frame can swing +-50%. Each item runs --repeat times
(default 3) and every timing is the MINIMUM over the runs - interference only
ever adds time, so the minimum is the stable estimate. All runs are kept
under "runs" in the JSON.

Load times depend on the OS file cache. Every run is preceded by an untimed
warm-up load of the same item (--no-warm skips it), so numbers are warm-cache
numbers, which are the stable ones.
"""
import argparse
import datetime
import json
import os
import subprocess
import sys
import threading
import time

import psutil

PKG = r"C:\Program Files (x86)\Steam\steamapps\common\Assetto Corsa EVO\content.kspkg"
EXE = os.environ.get("EVOVIEW_EXE", r"L:\build\evoview-perf\release\evoview.exe")

# (name, kind) - a GT3, a road car, an open-wheeler, two circuits.
SET = [
    ("ks_bmw_m4_gt3", "car"),
    ("ks_toyota_gr86", "car"),
    ("ks_ferrari_sf_25", "car"),
    ("brands_hatch", "track"),
    ("laguna_seca", "track"),
]


# extra viewer flags for every run (--demo: cars run the scripted drive, so
# the cost of the animation rig and the per-frame hero shadow is measured)
EXTRA = []


def run_one(exe, pkg, name, kind, seconds, size, ram_limit, timeout):
    cmd = [exe, pkg, "--" + kind, name, "--bench", str(seconds), "--size", size] + EXTRA
    # evoview caps itself at 60 fps unless told otherwise
    env = dict(os.environ, EVOVIEW_MAX_FPS="0")
    p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding="utf-8", errors="replace")
    out_lines = []
    reader = threading.Thread(target=lambda: out_lines.extend(p.stdout), daemon=True)
    reader.start()
    proc = psutil.Process(p.pid)
    peak = 0
    killed = None
    t0 = time.time()
    while p.poll() is None:
        try:
            rss = proc.memory_info().rss
            peak = max(peak, rss)
            if rss > ram_limit:
                killed = f"working set {rss / 2**30:.2f} GB > limit"
                p.kill()
        except psutil.NoSuchProcess:
            break
        if time.time() - t0 > timeout:
            killed = "timeout"
            p.kill()
        time.sleep(0.1)
    reader.join(5)
    text = "".join(out_lines)
    for line in text.splitlines():
        if line.startswith("BENCH_JSON "):
            r = json.loads(line[len("BENCH_JSON "):])
            r["kind"] = kind
            return r, text
    return {"name": name, "kind": kind,
            "error": killed or f"exit {p.returncode}, no BENCH_JSON",
            "peak_ws_mb": peak / 2**20}, text


def combine(runs):
    """Per-metric minimum over successful runs; memory is the maximum."""
    ok = [r for r in runs if "error" not in r]
    if not ok:
        return runs[-1]
    out = dict(min(ok, key=lambda r: pick(r, ("gpu_ms", "p50")) or pick(r, ("cpu_ms", "p50")) or 0))
    out["load_s"] = min(r["load_s"] for r in ok)
    for key in ("cpu_ms", "gpu_ms"):
        if all(r.get(key) for r in ok):
            out[key] = {k: min(r[key][k] for r in ok) for k in ok[0][key]}
    out["phases_s"] = {k: min(r["phases_s"].get(k, 0) for r in ok) for k in ok[0]["phases_s"]}
    for key in ("vram_mb", "peak_ws_mb", "peak_commit_mb"):
        out[key] = max(r.get(key) or 0 for r in ok)
    out["runs"] = runs
    return out


def fmt(v, unit=""):
    return "-" if v is None else f"{v:.2f}{unit}"


def pick(r, path):
    for k in path:
        if not isinstance(r, dict) or r.get(k) is None:
            return None
        r = r[k]
    return r


COLS = [
    ("load s", ("load_s",)),
    ("cpu p50", ("cpu_ms", "p50")),
    ("cpu p99", ("cpu_ms", "p99")),
    ("gpu mean", ("gpu_ms", "mean")),
    ("gpu p99", ("gpu_ms", "p99")),
    ("vram MB", ("vram_mb",)),
    ("RAM MB", ("peak_ws_mb",)),
]


def table(results, base=None):
    head = f"{'item':38}" + "".join(f"{c[0]:>18}" for c in COLS)
    print(head)
    for name, r in results.items():
        if "error" in r:
            print(f"{name:38}  ERROR: {r['error']}")
            continue
        row = f"{name:38}"
        b = (base or {}).get(name)
        for _, path in COLS:
            v = pick(r, path)
            cell = fmt(v)
            bv = pick(b, path) if b else None
            if v is not None and bv:
                cell += f" ({(v - bv) / bv * 100:+.0f}%)"
            row += f"{cell:>18}"
        print(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default=EXE)
    ap.add_argument("--pkg", default=PKG)
    ap.add_argument("--seconds", type=float, default=10)
    ap.add_argument("--sizes", nargs="+", default=["1920x1080", "3840x2160"],
                    help="official comparison is 1920x1080 and 3840x2160")
    ap.add_argument("--only", nargs="*", help="subset of item names")
    ap.add_argument("--out", help="JSON output path")
    ap.add_argument("--compare", help="baseline JSON to diff against")
    ap.add_argument("--label", default="", help="free text stored in the JSON")
    ap.add_argument("--ram-limit-gb", type=float, default=3.5)
    ap.add_argument("--no-warm", action="store_true")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--demo", action="store_true", help="cars run the scripted drive demo (drive.rs)")
    a = ap.parse_args()
    if a.demo:
        EXTRA.append("--demo")

    rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                         text=True, cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()
    items = [s for s in SET if not a.only or s[0] in a.only]
    results = {}
    for name, kind in items:
        if not a.no_warm:
            # untimed load to warm the OS file cache; 1 s of frames is enough
            run_one(a.exe, a.pkg, name, kind, 1, a.sizes[0], a.ram_limit_gb * 2**30, 600)
        for size in a.sizes:
            print(f"bench {name} @ {size} ...", flush=True)
            runs = []
            for _ in range(max(1, a.repeat)):
                r, log = run_one(a.exe, a.pkg, name, kind, a.seconds, size,
                                 a.ram_limit_gb * 2**30, 600 + a.seconds)
                if "error" in r:
                    print(log[-3000:])
                runs.append(r)
            results[f"{name}@{size}"] = combine(runs)

    doc = {
        "rev": rev,
        "label": a.label,
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "sizes": a.sizes,
        "seconds": a.seconds,
        "repeat": a.repeat,
        "results": results,
    }
    out = a.out or os.path.join(r"L:\evoview_work\bench", f"{rev or 'bench'}.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1)
    base = None
    if a.compare:
        with open(a.compare, encoding="utf-8") as f:
            base = json.load(f)["results"]
        print(f"\nvs {a.compare} (percent change; lower is better):")
    table(results, base)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    sys.exit(main())
