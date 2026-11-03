import os, sys
sys.path.insert(0, r"C:\Users\joshu\Downloads\acecm")
from acecm import kspkg
src = open(os.path.join(os.path.dirname(__file__), "dump_track.py"), encoding="utf-8").read()
ns = {}
exec(src[:src.find("\nroot = ")], ns)
pkg = r"C:\Program Files (x86)\Steam\steamapps\common\Assetto Corsa EVO\content.kspkg"
tmp = r"C:\Users\joshu\Downloads\evoview\tools\_spa_mats"
for path, size, off in kspkg.iter_entries(pkg):
    base = path.lower().replace("/", "\\").rsplit("\\", 1)[-1]
    if base != "top_near.material":
        continue
    with open(pkg, "rb") as f:
        d = kspkg.read_entry(f, size, off, path)
    dest = os.path.join(tmp, "_tn.material")
    open(dest, "wb").write(d)
    shader, slots, params, colors = ns["read_material"](dest)
    print(path, "shader", shader)
    for k, v in sorted(slots.items()):
        print(f"  {k} = {v}")
    print("bc", colors.get("Base_Basecolor"), "trip", params.get("Base_Triplanar"),
          "re", params.get("Red_Enable"), "ge", params.get("Green_Enable"))
