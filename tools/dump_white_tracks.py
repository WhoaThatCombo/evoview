"""Largest meshes + albedo on COTA / Laguna / Watkins."""
import os, sys, struct
sys.path.insert(0, r"C:\Users\joshu\Downloads\acecm")
from acecm import kspkg
src = open(os.path.join(os.path.dirname(__file__), "dump_track.py"), encoding="utf-8").read()
ns = {}
exec(src[:src.find("\nroot = ")], ns)
fields, strings_under = ns["fields"], ns["strings_under"]
pkg = r"C:\Program Files (x86)\Steam\steamapps\common\Assetto Corsa EVO\content.kspkg"
tmp = r"C:\Users\joshu\Downloads\evoview\tools\_spa_mats"
os.makedirs(tmp, exist_ok=True)

def placeholder(p):
    low = (p or "").replace("\\", "/").lower()
    return (not low) or "editor/" in low or "default_material" in low

want_tracks = ("cota", "laguna_seca", "watkins_glen")
meshes = {t: [] for t in want_tracks}
mats = {t: {} for t in want_tracks}
for path, size, off in kspkg.iter_entries(pkg):
    low = path.lower().replace("/", "\\")
    for t in want_tracks:
        if not low.startswith(f"content\\tracks\\{t}\\"):
            continue
        if low.endswith(".mesh") and size > 80_000:
            meshes[t].append((size, path, off))
        if low.endswith(".material"):
            mats[t][low.rsplit("\\", 1)[-1]] = (path, size, off)

with open(pkg, "rb") as f:
    for t in want_tracks:
        print(f"\n==== {t}")
        used = {}
        for size, path, off in sorted(meshes[t], reverse=True)[:12]:
            d = kspkg.read_entry(f, size, off, path)
            refs = []
            strings_under(d, 0, len(d), 0, refs)
            ms = [s.rsplit("\\", 1)[-1].rsplit("/", 1)[-1].lower() for s in refs if ".material" in s.lower()]
            print(f"  {size:8} {path.rsplit(chr(92),1)[-1][:40]:40} {ms[:3]}")
            for m in ms:
                used[m] = True
        for name in sorted(used):
            rec = mats[t].get(name)
            if not rec:
                print(f"    MAT {name} MISSING")
                continue
            raw = kspkg.read_entry(f, rec[1], rec[2], rec[0])
            dest = os.path.join(tmp, "_w.material")
            open(dest, "wb").write(raw)
            shader, slots, params, colors = ns["read_material"](dest)
            alb = slots.get("Base_BaseColorMap") or slots.get("txDiffuse") or ""
            red = slots.get("Red_BaseColorMap") or ""
            spat = slots.get("SpatMap") or slots.get("SplatMap") or ""
            print(f"    MAT {name} {shader} alb={os.path.basename(alb) or '-'} ph={placeholder(alb)} spat={bool(spat)} red={os.path.basename(red) or '-'} re={params.get('Red_Enable')}")
