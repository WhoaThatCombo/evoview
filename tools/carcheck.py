"""Check many cars with `evoview --check`: animation / suspension (check_anim.rs)
and effects - lights, LEDs, dash, glow, exhaust, textures (check_fx.rs) - in one
report per car.

    python tools/carcheck.py --stock                     # every car in content.kspkg
    python tools/carcheck.py --mods "C:/Users/joshu/Saved Games/ACE/mods"
    python tools/carcheck.py ks_toyota_gr86 path/to/mod.kspkg ...
        [--sheet] [--out L:/evoview_work/check] [--exe ...] [--tag stock]

One evoview at a time at idle priority (the PC is shared). Each car writes
<out>/<car>.json: {car, records: [{car, check, severity, detail}], counts,
anim: {...}}; --sheet also writes <out>/<car>_sheet.png (animation key poses
+ effect renders). The run writes <out>/<tag>.json (every record, flat) and
<out>/<tag>.txt (findings per check, then every warning / error).

A stock run is the baseline: the thresholds in check_anim.rs / check_fx.rs are
set so the stock fleet comes out (nearly) clean, which makes a finding on a
converted car mean something.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

PKG = r"C:\Program Files (x86)\Steam\steamapps\common\Assetto Corsa EVO\content.kspkg"
EXE = os.environ.get("EVOVIEW_EXE", r"L:\build\evoview-fx\release\evoview.exe")
OUT = r"L:\evoview_work\check"
IDLE = 0x00000040  # IDLE_PRIORITY_CLASS


def stock_cars(exe):
    p = subprocess.run([exe, PKG], capture_output=True, text=True, encoding="utf-8", errors="replace")
    cars = [l.split()[1] for l in p.stdout.splitlines() if l.strip().startswith("car ")]
    return [c for c in cars if c != "common_assets" and not c.startswith("common")]


def mod_car_id(kspkg):
    # a converted mod's package holds one car; its folder name is the id
    p = subprocess.run([EXE, kspkg], capture_output=True, text=True, encoding="utf-8", errors="replace")
    for l in p.stdout.splitlines():
        if l.strip().startswith("car "):
            return l.split()[1]
    return os.path.splitext(os.path.basename(kspkg))[0]


def sheet(tiles_dir, out_png, title):
    from PIL import Image, ImageDraw
    names = sorted(f for f in os.listdir(tiles_dir) if f.endswith(".png"))
    if not names:
        return
    ims = [Image.open(os.path.join(tiles_dir, n)).convert("RGB") for n in names]
    w, h = ims[0].size
    cols = 3
    rows = (len(ims) + cols - 1) // cols
    S = Image.new("RGB", (w * cols, h * rows + 24), (24, 24, 24))
    d = ImageDraw.Draw(S)
    d.text((6, 4), title, fill=(230, 230, 230))
    for i, (n, im) in enumerate(zip(names, ims)):
        x, y = (i % cols) * w, (i // cols) * h + 24
        S.paste(im, (x, y))
        d.text((x + 6, y + 4), n[:-4].split("_", 1)[-1].replace("_", " "), fill=(255, 220, 120))
    S.save(out_png)


def label_sheet(png, records, title):
    """Write each tile's name (the report's sheet_cells record) onto the sheet."""
    cells = next((x["detail"] for x in records if x["check"] == "sheet_cells"), None)
    if not cells or not os.path.exists(png):
        return
    from PIL import Image, ImageDraw
    size, names = cells.split(": ", 1)
    w, h = (int(v) for v in size.split(" ")[0].split("x"))
    names = names.split(" | ")
    im = Image.open(png).convert("RGB")
    S = Image.new("RGB", (im.width, im.height + 24), (24, 24, 24))
    S.paste(im, (0, 24))
    d = ImageDraw.Draw(S)
    d.text((6, 4), title, fill=(230, 230, 230))
    for i, n in enumerate(names):
        d.text(((i % 4) * w + 6, (i // 4) * h + 24 + 4), n, fill=(255, 220, 120))
    S.save(png)


def run(exe, item, out, want_sheet):
    is_pkg = item.lower().endswith(".kspkg")
    car = mod_car_id(item) if is_pkg else item
    pkg = item if is_pkg else PKG
    rep = os.path.join(out, f"{car}.json")
    cmd = [exe, pkg, "--car", car, "--check", rep]
    if is_pkg:
        cmd += ["--base", PKG]
    if want_sheet:
        cmd += ["--sheet", os.path.join(out, f"{car}_sheet.png")]
    env = dict(os.environ, EVOVIEW_MAX_FPS="30")
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                       env=env, creationflags=IDLE if os.name == "nt" else 0, timeout=900)
    dt = time.time() - t0
    if not os.path.exists(rep) or p.returncode != 0 and "CHECK " not in p.stdout:
        tail = (p.stdout + p.stderr).strip().splitlines()[-5:]
        return {"car": car, "source": item, "failed": True, "error": " | ".join(tail), "seconds": dt,
                "records": [{"car": car, "check": "run", "severity": "error",
                             "detail": "evoview failed: " + " | ".join(tail)}]}
    r = json.load(open(rep, encoding="utf-8"))
    if want_sheet:
        label_sheet(os.path.join(out, f"{car}_sheet.png"), r.get("records", []), f"{car}  ({os.path.basename(item)})")
    return {"car": car, "source": item, "counts": r.get("counts", {}), "seconds": dt,
            "records": r.get("records", [])}


def main():
    global EXE
    ap = argparse.ArgumentParser()
    ap.add_argument("items", nargs="*")
    ap.add_argument("--stock", action="store_true")
    ap.add_argument("--mods", help="folder of mod .kspkg files")
    ap.add_argument("--sheet", action="store_true")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--exe", default=EXE)
    ap.add_argument("--tag", default="summary")
    a = ap.parse_args()
    EXE = a.exe
    items = list(a.items)
    if a.stock:
        items += stock_cars(a.exe)
    if a.mods:
        items += sorted(os.path.join(a.mods, f) for f in os.listdir(a.mods) if f.lower().endswith(".kspkg"))
    os.makedirs(a.out, exist_ok=True)
    results = []
    for i, it in enumerate(items):
        try:
            r = run(a.exe, it, a.out, a.sheet)
        except subprocess.TimeoutExpired:
            r = {"car": it, "failed": True, "error": "timeout"}
        results.append(r)
        c = r.get("counts", {})
        print(f"[{i + 1}/{len(items)}] {r['car']:40s} "
              + ("FAILED " + r.get("error", "") if r.get("failed") else
                 f"E{c.get('error', 0):3d} W{c.get('warn', 0):3d} I{c.get('info', 0):3d}  {r['seconds']:.1f}s"),
              flush=True)
    # which checks fire, over the whole run
    recs = [x for r in results for x in r.get("records", [])]
    by = {}
    for x in recs:
        by.setdefault(x["check"], {}).setdefault(x["severity"], set()).add(x["car"])
    lines = [f"{a.tag}: {len(results)} car(s), {len(recs)} record(s)", ""]
    print("\nfindings by check:")
    for chk in sorted(by):
        row = "  ".join(f"{sev}:{len(by[chk][sev])}" for sev in ("ok", "info", "warn", "error") if sev in by[chk])
        lines.append(f"{chk:<32} {row}")
        if any(sev in by[chk] for sev in ("warn", "error")):
            print(f"  {chk:<32} {row}")
    lines += ["", "warnings / errors:"]
    lines += [f"  [{x['severity']}] {x['car']}: {x['check']}: {x['detail']}"
              for x in recs if x["severity"] in ("warn", "error")]
    open(os.path.join(a.out, a.tag + ".txt"), "w", encoding="utf-8").write("\n".join(lines))
    json.dump(recs, open(os.path.join(a.out, a.tag + ".json"), "w"), indent=1)
    print("wrote", os.path.join(a.out, a.tag + ".txt"))


if __name__ == "__main__":
    main()
