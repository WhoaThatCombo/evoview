"""Which Spa static meshes are the white hills, and what do they sample?"""
import os, re, struct, sys, collections
sys.path.insert(0, r"C:\Users\joshu\Downloads\acecm")
from acecm import kspkg

src = open(os.path.join(os.path.dirname(__file__), "dump_track.py"), encoding="utf-8").read()
ns = {}
exec(src[:src.find("\nroot = ")], ns)
fields, text, strings_under = ns["fields"], ns["text"], ns["strings_under"]

pkg = r"C:\Program Files (x86)\Steam\steamapps\common\Assetto Corsa EVO\content.kspkg"
tmp = r"C:\Users\joshu\Downloads\evoview\tools\_spa_mats"
os.makedirs(tmp, exist_ok=True)

def aabb(d):
    for fnum, wt, v in fields(d, 0, len(d)):
        if fnum != 5 or wt != 2:
            continue
        for lf, lwt, lv in fields(d, v[0], v[1]):
            if lf == 5 and lwt == 2:
                n = (lv[1] - lv[0]) // 12
                lo = [1e9] * 3
                hi = [-1e9] * 3
                step = max(1, n // 120)
                for i in range(0, n, step):
                    p = struct.unpack_from("<fff", d, lv[0] + i * 12)
                    for k in range(3):
                        lo[k] = min(lo[k], p[k])
                        hi[k] = max(hi[k], p[k])
                return n, lo, hi
    return 0, None, None

def mats_of(d):
    refs = []
    strings_under(d, 0, len(d), 0, refs)
    return [s.replace("\\", "/") for s in refs if ".material" in s.lower()]

HINT = ("grass", "terrain", "ground", "outer", "land", "dirt", "sand",
        "soil", "hill", "bank", "side", "far", "map", "mount", "gravel",
        "stone", "rock", "erba", "lawn", "field", "cut", "ext")

large = []
mat_files = {}
tex_files = {}
print("indexing…")
for path, size, off in kspkg.iter_entries(pkg):
    low = path.lower().replace("/", "\\")
    if low.startswith("content\\tracks\\spa\\static_mesh\\") and low.endswith(".mesh"):
        if size >= 80_000:
            large.append((size, path, off))
    if low.startswith("content\\tracks\\spa\\") and low.endswith(".material"):
        mat_files[low.rsplit("\\", 1)[-1]] = (path, size, off)
    if low.endswith(".texture") and (
        "\\spa\\" in low or "\\common_assets\\textures\\" in low
    ):
        tex_files[low.rsplit("\\", 1)[-1]] = (path, size, low)

print(f"large static meshes {len(large)}, mats {len(mat_files)}")

# biggest meshes + hinted names
large.sort(reverse=True)
print("\n==== largest static meshes")
seen_mats = collections.Counter()
with open(pkg, "rb") as f:
    for size, path, off in large[:40]:
        d = kspkg.read_entry(f, size, off, path)
        n, lo, hi = aabb(d)
        mats = mats_of(d)
        base = path.replace("\\", "/").rsplit("/", 1)[-1]
        hinted = any(h in base.lower() for h in HINT)
        span = 0
        if lo:
            span = ((hi[0]-lo[0])**2 + (hi[1]-lo[1])**2 + (hi[2]-lo[2])**2) ** 0.5
        if hinted or span > 80 or size > 400_000:
            print(f"  {size:8} v~{n:6} span={span:7.0f} {base}")
            print(f"           aabb={[round(x,0) for x in (lo or [0,0,0])]}..{[round(x,0) for x in (hi or [0,0,0])]}")
            print(f"           mats={mats[:4]}")
        for m in mats:
            seen_mats[m.rsplit("/", 1)[-1].lower()] += 1

print("\n==== materials used on large meshes")
want = set(seen_mats)
# plus known terrain names
for extra in ("grass.material", "grass_brd.material", "sand_new.material",
              "terrain_paint_inst.material", "grass-side1.material",
              "gravel_mask.material", "stones_detail1.material"):
    want.add(extra)

def placeholder(p):
    low = (p or "").replace("\\", "/").lower()
    return (not low) or "editor/" in low or "default_material" in low or "null.texture" in low

with open(pkg, "rb") as f:
    for name in sorted(want):
        rec = mat_files.get(name)
        if not rec:
            print(f"  MISSING {name}")
            continue
        d = kspkg.read_entry(f, rec[1], rec[2], rec[0])
        dest = os.path.join(tmp, "_t.material")
        open(dest, "wb").write(d)
        shader, slots, params, colors = ns["read_material"](dest)
        alb = slots.get("Base_BaseColorMap") or slots.get("txDiffuse") or ""
        spat = slots.get("SpatMap") or slots.get("SplatMap") or ""
        red = slots.get("Red_BaseColorMap") or ""
        grn = slots.get("Green_BaseColorMap") or ""
        bc = colors.get("Base_Basecolor") or colors.get("Basecolor")
        trip = params.get("Base_Triplanar")
        print(f"\n  {name} shader={shader} used={seen_mats.get(name,0)}")
        print(f"    albedo={alb} placeholder={placeholder(alb)}")
        if spat:
            print(f"    spat={spat}")
        if red:
            print(f"    red={red}")
        if grn:
            print(f"    green={grn}")
        print(f"    bc={None if not bc else [round(x,2) for x in bc[:3]]} triplanar={trip}")
        # texture sizes
        for label, p in (("alb", alb), ("spat", spat), ("red", red), ("grn", grn)):
            if not p or placeholder(p):
                continue
            b = p.replace("\\", "/").rsplit("/", 1)[-1].lower()
            rec_t = tex_files.get(b)
            if rec_t:
                print(f"    {label} tex {b} hdr={rec_t[1]} path=...{rec_t[2][-70:]}")
