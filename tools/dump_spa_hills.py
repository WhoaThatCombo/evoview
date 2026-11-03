"""Find every Spa mesh that uses ground materials; list terrain-named files."""
import os, sys, collections
sys.path.insert(0, r"C:\Users\joshu\Downloads\acecm")
from acecm import kspkg

src = open(os.path.join(os.path.dirname(__file__), "dump_track.py"), encoding="utf-8").read()
ns = {}
exec(src[:src.find("\nroot = ")], ns)
fields, strings_under = ns["fields"], ns["strings_under"]

pkg = r"C:\Program Files (x86)\Steam\steamapps\common\Assetto Corsa EVO\content.kspkg"
GROUND = (
    "grass.material", "grass_brd.material", "grass-side1.material",
    "sand_new.material", "terrain_paint_inst.material", "gravel_mask.material",
    "stones_detail1.material", "outerland", "dirt", "soil", "erba",
)
NAME_HINT = ("grass", "terrain", "ground", "outer", "land", "dirt", "sand",
             "hill", "bank", "far", "map", "mount", "gravel", "cut", "lawn",
             "field", "side1", "side2", "erba")

by_mat = collections.defaultdict(list)
named = []
folders = collections.Counter()
print("scanning…")
for path, size, off in kspkg.iter_entries(pkg):
    low = path.lower().replace("/", "\\")
    if not low.startswith("content\\tracks\\spa\\"):
        continue
    if low.endswith(".mesh"):
        folders[low.rsplit("\\", 1)[0]] += 1
        base = low.rsplit("\\", 1)[-1]
        if any(h in base for h in NAME_HINT):
            named.append((size, path, off, base))

print("spa mesh folders:")
for k, n in folders.most_common(20):
    print(f"  {n:4} {k}")

print(f"\nnamed ground-like meshes {len(named)}")
for size, path, off, base in sorted(named, reverse=True)[:50]:
    print(f"  {size:8} {base}")

# read materials on named meshes + any static_mesh that mentions grass.material
print("\n==== material refs on named meshes")
with open(pkg, "rb") as f:
    for size, path, off, base in sorted(named, reverse=True)[:40]:
        d = kspkg.read_entry(f, size, off, path)
        refs = []
        strings_under(d, 0, len(d), 0, refs)
        mats = [s for s in refs if ".material" in s.lower()]
        print(f"  {base:40} {mats[:5]}")
        for m in mats:
            by_mat[m.rsplit("\\", 1)[-1].rsplit("/", 1)[-1].lower()].append(base)

print("\n==== who uses grass/terrain mats (named set)")
for m, names in sorted(by_mat.items()):
    if any(g in m for g in ("grass", "terrain", "sand", "gravel", "stone", "dirt", "land")):
        print(f"  {m}: {len(names)} e.g. {names[:4]}")
