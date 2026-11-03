"""splat-maps.material on Laguna (the inner terrain chunks)."""
import os, sys
sys.path.insert(0, r"C:\Users\joshu\Downloads\acecm")
from acecm import kspkg
src = open(os.path.join(os.path.dirname(__file__), "dump_track.py"), encoding="utf-8").read()
ns = {}
exec(src[:src.find("\nroot = ")], ns)
pkg = r"C:\Program Files (x86)\Steam\steamapps\common\Assetto Corsa EVO\content.kspkg"
tmp = r"C:\Users\joshu\Downloads\evoview\tools\_spa_mats"
want = ("splat-maps.material", "splat_maps.material", "terrain.material",
        "terrain_far.material", "mi_watkins_terrain_grass_01.material")
found = {}
for path, size, off in kspkg.iter_entries(pkg):
    base = path.lower().replace("/", "\\").rsplit("\\", 1)[-1]
    if base in want:
        found[path] = (base, size, off)
with open(pkg, "rb") as f:
    for path, (base, size, off) in sorted(found.items()):
        d = kspkg.read_entry(f, size, off, path)
        dest = os.path.join(tmp, "_s.material")
        open(dest, "wb").write(d)
        shader, slots, params, colors = ns["read_material"](dest)
        print(f"\n==== {path}")
        print(f"  shader={shader}")
        for k, v in sorted(slots.items()):
            if any(x in k.lower() for x in ("color", "diffus", "spat", "splat", "opac")):
                print(f"  {k} = {v}")
        for k, v in sorted(params.items()):
            if "nable" in k or "riplan" in k or "blend" in k.lower():
                print(f"  PAR {k}={v}")
        bc = colors.get("Base_Basecolor") or colors.get("Basecolor")
        print("  bc", None if not bc else [round(x,2) for x in bc[:3]])
