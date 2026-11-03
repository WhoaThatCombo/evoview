"""Spa: numbered helper meshes, material slots, asphalt textures, stand overlap."""
import os, re, sys, collections
sys.path.insert(0, r"C:\Users\joshu\Downloads\acecm")
from acecm import kspkg

src = open(os.path.join(os.path.dirname(__file__), "dump_track.py"), encoding="utf-8").read()
ns = {}
exec(src[:src.find("\nroot = ")], ns)
fields, text, strings_under = ns["fields"], ns["text"], ns["strings_under"]

pkg = r"C:\Program Files (x86)\Steam\steamapps\common\Assetto Corsa EVO\content.kspkg"
tmp = r"C:\Users\joshu\Downloads\evoview\tools\_spa_mats"
os.makedirs(tmp, exist_ok=True)

def read_mat_bytes(d, file="x.material"):
    dest = os.path.join(tmp, os.path.basename(file))
    open(dest, "wb").write(d)
    return ns["read_material"](dest)

num_wall = []
num_asph = []
stand = []
mats = []
tex = []
asph_common = []
scene = None
want_mat = (
    "asph_spa.material", "asph_spa_dark.material", "asph_spa_old.material",
    "asph_ext.material", "asph_ext_dark.material", "asph_paint_black.material",
    "road-ext-tile.material", "walls.material", "walls_b.material",
    "grand_stand.material",
)

print("indexing…")
entries = []
for path, size, off in kspkg.iter_entries(pkg):
    low = path.lower().replace("/", "\\")
    base = low.rsplit("\\", 1)[-1]
    if low.startswith("content\\tracks\\spa\\static_mesh\\") and low.endswith(".mesh"):
        if re.match(r"^\d+wall", base):
            num_wall.append((base, size, path, off))
        if re.match(r"^\d+asph", base) or "asph" in base:
            num_asph.append((base, size, path, off))
        if any(x in base for x in ("stand", "seat", "gstand", "bleacher")):
            stand.append((base, size, path, off))
    if low.startswith("content\\tracks\\spa\\") and low.endswith(".material"):
        if base in want_mat:
            mats.append((base, size, path, off))
    if low.endswith(".texture") and any(x in base for x in ("asph", "tarmac", "road", "track_base")):
        if "common_assets" in low or low.startswith("content\\tracks\\spa\\"):
            asph_common.append((low, size))
    if low.endswith("\\spa.scene"):
        scene = (path, size, off)

print(f"\nnumbered walls {len(num_wall)} sizes", sorted({s for _, s, _, _ in num_wall})[:12])
print("sample walls", [n for n, *_ in num_wall[:8]])
print(f"asph-named meshes {len(num_asph)}")
for n, s, *_ in sorted(num_asph)[:20]:
    print(f"  {s:8} {n}")
print("stands", stand)

print("\n==== materials")
with open(pkg, "rb") as f:
    for base, size, path, off in mats:
        d = kspkg.read_entry(f, size, off, path)
        shader, slots, params, colors = read_mat_bytes(d, base)
        print(f"\n{base} shader={shader}")
        for k, v in sorted(slots.items()):
            print(f"  {k} = {v}")
        bc = colors.get("Base_Basecolor") or colors.get("Basecolor") or colors.get("ksDiffuse")
        print("  Basecolor", bc, "rough", params.get("Roughness") or params.get("Base_Roughness"))

    print("\n==== mesh batches")
    for group, label in ((num_wall[:6], "WALL"), (num_asph[:8], "ASPH"), (stand, "STAND")):
        for base, size, path, off in group:
            d = kspkg.read_entry(f, size, off, path)
            refs = []
            strings_under(d, 0, len(d), 0, refs)
            mats_r = [s for s in refs if ".material" in s.lower()]
            print(f"  {label} {base:32} {size:8} {mats_r[:6]}")

print("\n==== asphalt-like textures")
for p, s in sorted(asph_common)[:40]:
    print(f"  {s:8} {p}")

if scene:
    print("\n==== spa.scene stand-like + seating paths")
    with open(pkg, "rb") as f:
        d = kspkg.read_entry(f, scene[1], scene[2], scene[0])
    paths = collections.Counter()
    for fnum, wt, v in fields(d, 0, len(d)):
        if fnum != 2 or wt != 2:
            continue
        blob = None
        for sf, swt, sv in fields(d, v[0], v[1]):
            if sf == 50 and swt == 2:
                blob = sv
        if not blob:
            continue
        for sf, swt, sv in fields(d, blob[0], blob[1]):
            if sf != 105 or swt != 2:
                continue
            for kf, kwt, kv in fields(d, sv[0], sv[1]):
                if kf == 1 and kwt == 2:
                    p = text(d, kv).replace("\\", "/").lower()
                    paths[p] += 1
    for p, n in sorted(paths.items()):
        if any(x in p for x in ("stand", "seat", "grand", "tribun", "bleacher", "wall")):
            print(f"  {n:3} {p}")
