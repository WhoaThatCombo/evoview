"""Physics-folder names, grass/sand UVs, and full grass.material params."""
import os, struct, sys, collections
sys.path.insert(0, r"C:\Users\joshu\Downloads\acecm")
from acecm import kspkg

src = open(os.path.join(os.path.dirname(__file__), "dump_track.py"), encoding="utf-8").read()
ns = {}
exec(src[:src.find("\nroot = ")], ns)
fields, text, strings_under = ns["fields"], ns["text"], ns["strings_under"]

pkg = r"C:\Program Files (x86)\Steam\steamapps\common\Assetto Corsa EVO\content.kspkg"
tmp = r"C:\Users\joshu\Downloads\evoview\tools\_spa_mats"

phys = collections.Counter()
phys_sample = []
want_mesh = {}
want_mat = {}
want_tex = {}

for path, size, off in kspkg.iter_entries(pkg):
    low = path.lower().replace("/", "\\")
    base = low.rsplit("\\", 1)[-1]
    if "\\spa\\static_mesh\\physics\\" in low and low.endswith(".mesh"):
        pref = base.split("_")[0].split(".")[0]
        import re
        pref = re.sub(r"\d+$", "", pref) or pref
        phys[pref] += 1
        if len(phys_sample) < 12:
            phys_sample.append((base, size, path, off))
    if base in ("grass_1.mesh", "grass_8.mesh", "sand_01.mesh", "gravel_mask__07.mesh",
                "grass_brd7.mesh"):
        want_mesh[base] = (path, size, off)
    if base in ("grass.material", "sand_new.material", "grass_brd.material"):
        want_mat[base] = (path, size, off)
    if base in ("grass_top_lv2d.texture", "ground_dirty_2.texture",
                "border_test3_albedo.texture", "grass_2_general_albedo.texture"):
        want_tex[base] = (path, size, off)

print("PHYSICS folder prefixes", sum(phys.values()), "meshes")
for k, n in phys.most_common(20):
    print(f"  {n:4} {k}")

def uv_aabb(d):
    pos = uv = None
    for fnum, wt, v in fields(d, 0, len(d)):
        if fnum != 5 or wt != 2:
            continue
        for lf, lwt, lv in fields(d, v[0], v[1]):
            if lf == 5 and lwt == 2:
                pos = lv
            if lf == 7 and lwt == 2:
                uv = lv
        break
    out = {}
    if pos:
        n = (pos[1]-pos[0])//12
        lo=[1e9]*3; hi=[-1e9]*3
        for i in range(0, n, max(1,n//80)):
            p = struct.unpack_from("<fff", d, pos[0]+i*12)
            for k in range(3):
                lo[k]=min(lo[k],p[k]); hi[k]=max(hi[k],p[k])
        out["n"]=n
        out["pos"]=(lo,hi)
    if uv:
        n = (uv[1]-uv[0])//8
        ulo=[1e9]*2; uhi=[-1e9]*2
        for i in range(0, n, max(1,n//80)):
            p = struct.unpack_from("<ff", d, uv[0]+i*8)
            for k in range(2):
                ulo[k]=min(ulo[k],p[k]); uhi[k]=max(uhi[k],p[k])
        out["uv"]=(ulo,uhi)
    return out

with open(pkg, "rb") as f:
    print("\n==== physics samples")
    for base, size, path, off in phys_sample:
        d = kspkg.read_entry(f, size, off, path)
        refs = []
        strings_under(d, 0, len(d), 0, refs)
        mats = [s for s in refs if ".material" in s.lower()]
        print(f"  {base:32} {size:7} {mats[:3]}")

    print("\n==== grass/sand mesh UV + world")
    for base, rec in sorted(want_mesh.items()):
        d = kspkg.read_entry(f, rec[1], rec[2], rec[0])
        info = uv_aabb(d)
        print(f"  {base} n={info.get('n')}")
        if "pos" in info:
            lo,hi=info["pos"]
            print(f"    world {[round(x,1) for x in lo]} .. {[round(x,1) for x in hi]}")
        if "uv" in info:
            lo,hi=info["uv"]
            print(f"    uv    {[round(x,3) for x in lo]} .. {[round(x,3) for x in hi]}")

    print("\n==== materials full")
    for base, rec in sorted(want_mat.items()):
        d = kspkg.read_entry(f, rec[1], rec[2], rec[0])
        dest = os.path.join(tmp, "_w.material")
        open(dest, "wb").write(d)
        shader, slots, params, colors = ns["read_material"](dest)
        print(f"\n  {base} shader={shader}")
        for k, v in sorted(slots.items()):
            print(f"    SLOT {k} = {v}")
        for k, v in sorted(params.items()):
            if any(x in k.lower() for x in ("enable", "triplan", "blend", "uv", "detail", "rough")):
                print(f"    PAR  {k} = {v}")
        for k, v in sorted(colors.items()):
            print(f"    COL  {k} = {[round(x,3) for x in v]}")

    print("\n==== texture headers")
    for base, rec in sorted(want_tex.items()):
        hdr = kspkg.read_entry(f, rec[1], rec[2], rec[0])
        w=h=mips=fmt=None
        for fn, wt, v in fields(hdr, 0, len(hdr)):
            if wt == 0:
                if fn==1: w=v
                if fn==2: h=v
                if fn==3: mips=v
                if fn==4: fmt=v
        print(f"  {base} {w}x{h} mips={mips} fmt={fmt} hdr={rec[1]}")
