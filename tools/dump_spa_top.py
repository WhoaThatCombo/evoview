"""Slots on Spa top-ext / new_top / sbanc ground materials."""
import os, sys
sys.path.insert(0, r"C:\Users\joshu\Downloads\acecm")
from acecm import kspkg
src = open(os.path.join(os.path.dirname(__file__), "dump_track.py"), encoding="utf-8").read()
ns = {}
exec(src[:src.find("\nroot = ")], ns)
pkg = r"C:\Program Files (x86)\Steam\steamapps\common\Assetto Corsa EVO\content.kspkg"
tmp = r"C:\Users\joshu\Downloads\evoview\tools\_spa_mats"
want = {
    "top-ext.material", "top-ext_simple.material", "new_top_1.material",
    "sbanc_new.material", "sbanc_new_b.material", "concrete_strip.material",
    "concrete_b.material",
}
found = {}
for path, size, off in kspkg.iter_entries(pkg):
    base = path.lower().replace("/", "\\").rsplit("\\", 1)[-1]
    if base in want:
        found[base] = (path, size, off)
with open(pkg, "rb") as f:
    for base, rec in sorted(found.items()):
        d = kspkg.read_entry(f, rec[1], rec[2], rec[0])
        dest = os.path.join(tmp, "_x.material")
        open(dest, "wb").write(d)
        shader, slots, params, colors = ns["read_material"](dest)
        print(f"\n==== {base} shader={shader}")
        for k, v in sorted(slots.items()):
            print(f"  {k} = {v}")
        for k, v in sorted(params.items()):
            if any(x in k.lower() for x in ("enable", "triplan", "blend", "uv")):
                print(f"  PAR {k}={v}")
        for k, v in sorted(colors.items()):
            if any(x in k.lower() for x in ("basecolor", "base_base", "uvscale", "triplan", "splat")):
                print(f"  COL {k}={[round(x,3) for x in v]}")
