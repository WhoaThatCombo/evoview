"""Extract Spa wall/grass/asphalt/stand materials and list mesh prefixes."""
import os, sys, collections
sys.path.insert(0, r"C:\Users\joshu\Downloads\acecm")
from acecm import kspkg

pkg = r"C:\Program Files (x86)\Steam\steamapps\common\Assetto Corsa EVO\content.kspkg"
src = open(r"C:\Users\joshu\Downloads\evoview\tools\dump_track.py", encoding="utf-8").read()
ns = {}
exec(src[:src.find("\nroot = ")], ns)
read_material = ns["read_material"]

want_mats = (
    "grass.material", "grass_brd.material", "asph_spa.material",
    "asph_spa_dark.material", "asph_ext.material", "walls.material",
    "walls_b.material", "grand_stand.material", "transp-alpha_test.material",
    "sand_new.material", "road-ext-tile.material", "01wall",
)
prefixes = collections.Counter()
to_extract = []
for path, size, off in kspkg.iter_entries(pkg):
    low = path.lower().replace("/", "\\")
    if not low.startswith("content\\tracks\\spa\\"):
        continue
    if low.endswith(".mesh") and "\\static_mesh\\" in low:
        name = low.rsplit("\\", 1)[-1]
        pref = name.split("_")[0].split(".")[0]
        # strip trailing digits for 01wall001
        import re
        pref = re.sub(r"\d+$", "", pref) or pref
        prefixes[pref] += 1
    if low.endswith(".material"):
        base = low.rsplit("\\", 1)[-1]
        if any(w in base for w in (
            "grass", "asph", "wall", "grand", "transp", "sand", "road-ext",
            "physics", "dummy", "seat",
        )):
            to_extract.append(path)

print("MESH PREFIXES")
for p, n in prefixes.most_common(40):
    print(f"  {n:4} {p}")

# extract a handful of materials
tmp = r"C:\Users\joshu\Downloads\evoview\tools\_spa_mats"
os.makedirs(tmp, exist_ok=True)
wanted = {
    "grass.material", "grass_brd.material", "asph_spa.material",
    "asph_spa_dark.material", "asph_ext.material", "walls.material",
    "walls_b.material", "grand_stand.material", "transp-alpha_test.material",
    "sand_new.material", "road-ext-tile.material", "asph_spa_old.material",
}
# second pass extract
found = {w: None for w in wanted}
for path, size, off in kspkg.iter_entries(pkg):
    low = path.lower().replace("/", "\\")
    base = low.rsplit("\\", 1)[-1]
    if base in wanted:
        found[base] = (path, size, off)

with open(pkg, "rb") as f:
    for base, rec in found.items():
        if not rec:
            print("MISSING", base)
            continue
        path, size, off = rec
        data = kspkg.read_entry(f, size, off, path)
        dest = os.path.join(tmp, base)
        open(dest, "wb").write(data)
        shader, slots, params, colors = read_material(dest)
        alb = slots.get("txDiffuse") or slots.get("Base_BaseColorMap") or "?"
        print(f"\n{base} shader={shader}")
        print(f"  albedo={alb}")
        for k, v in sorted(slots.items()):
            if any(x in k.lower() for x in ("color", "diffuse", "detail", "opac")):
                print(f"  {k} = {os.path.basename(v)}")
        print("  blend", params.get("blendMode"), "rough", params.get("Roughness") or params.get("Base_Roughness") or params.get("ksRoughness"))
