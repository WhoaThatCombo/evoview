"""Laguna / COTA / Watkins far-ground materials and mesh UVs."""
import os, struct, sys
sys.path.insert(0, r"C:\Users\joshu\Downloads\acecm")
from acecm import kspkg
src = open(os.path.join(os.path.dirname(__file__), "dump_track.py"), encoding="utf-8").read()
ns = {}
exec(src[:src.find("\nroot = ")], ns)
fields, strings_under, text = ns["fields"], ns["strings_under"], ns["text"]
pkg = r"C:\Program Files (x86)\Steam\steamapps\common\Assetto Corsa EVO\content.kspkg"
tmp = r"C:\Users\joshu\Downloads\evoview\tools\_spa_mats"
os.makedirs(tmp, exist_ok=True)

HINT = ("terrain", "far", "sat", "background", "outer", "land", "ground", "sbanc")

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
        step=max(1,n//80)
        for i in range(0,n,step):
            p=struct.unpack_from("<fff", d, pos[0]+i*12)
            for k in range(3):
                lo[k]=min(lo[k],p[k]); hi[k]=max(hi[k],p[k])
        out["n"]=n; out["pos"]=(lo,hi)
        out["span"]=((hi[0]-lo[0])**2+(hi[1]-lo[1])**2+(hi[2]-lo[2])**2)**0.5
    if uv:
        n=(uv[1]-uv[0])//8
        ulo=[1e9]*2; uhi=[-1e9]*2
        step=max(1,n//80)
        for i in range(0,n,step):
            p=struct.unpack_from("<ff", d, uv[0]+i*8)
            for k in range(2):
                ulo[k]=min(ulo[k],p[k]); uhi[k]=max(uhi[k],p[k])
        out["uv"]=(ulo,uhi)
    return out

for track in ("laguna_seca", "cota", "watkins_glen"):
    print(f"\n######## {track}")
    mats = {}
    meshes = []
    for path, size, off in kspkg.iter_entries(pkg):
        low = path.lower().replace("/", "\\")
        if not low.startswith(f"content\\tracks\\{track}\\"):
            continue
        base = low.rsplit("\\", 1)[-1]
        if low.endswith(".material") and any(h in base for h in HINT):
            mats[base] = (path, size, off)
        if low.endswith(".mesh") and any(h in base for h in HINT) and size > 20000:
            meshes.append((size, path, off, base))

    with open(pkg, "rb") as f:
        print("-- materials --")
        for name, rec in sorted(mats.items()):
            d = kspkg.read_entry(f, rec[1], rec[2], rec[0])
            dest = os.path.join(tmp, "_l.material")
            open(dest, "wb").write(d)
            shader, slots, params, colors = ns["read_material"](dest)
            alb = slots.get("Base_BaseColorMap") or slots.get("txDiffuse") or ""
            red = slots.get("Red_BaseColorMap") or ""
            grn = slots.get("Green_BaseColorMap") or ""
            spat = slots.get("SpatMap") or slots.get("SplatMap") or ""
            bc = colors.get("Base_Basecolor") or colors.get("Basecolor")
            print(f"  {name} shader={shader}")
            print(f"    alb={alb}")
            print(f"    red={red} en={params.get('Red_Enable')} trip={params.get('Red_Triplanar')}")
            print(f"    grn={grn} en={params.get('Green_Enable')} trip={params.get('Green_Triplanar')}")
            print(f"    spat={spat} trip={params.get('Base_Triplanar')} bc={None if not bc else [round(x,2) for x in bc[:3]]}")

        print("-- meshes --")
        for size, path, off, base in sorted(meshes, reverse=True)[:12]:
            d = kspkg.read_entry(f, size, off, path)
            refs = []
            strings_under(d, 0, len(d), 0, refs)
            ms = [s.rsplit("\\",1)[-1] for s in refs if ".material" in s.lower()]
            info = uv_aabb(d)
            uv = ""
            if "uv" in info:
                lo,hi=info["uv"]
                uv = f" uv={[round(x,2) for x in lo]}..{[round(x,2) for x in hi]}"
            span = info.get("span", 0)
            print(f"  {size:8} span={span:7.0f} {base:40} {ms[:3]}{uv}")
