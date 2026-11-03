"""AABB of spa grass_2 cards and GS meshes."""
import os, struct, sys
sys.path.insert(0, r"C:\Users\joshu\Downloads\acecm")
from acecm import kspkg
src = open(os.path.join(os.path.dirname(__file__), "dump_track.py"), encoding="utf-8").read()
ns = {}
exec(src[:src.find("\nroot = ")], ns)
fields, text = ns["fields"], ns["text"]

pkg = r"C:\Program Files (x86)\Steam\steamapps\common\Assetto Corsa EVO\content.kspkg"
want = ("grass_2_b4_lod0.mesh", "grass_2_b3_lod0_ext.mesh", "cf1_b.mesh",
        "GS_04_000.mesh", "GS_05_000.mesh", "GS_06_000.mesh", "GS_08_000.mesh")

def aabb(d):
    for fnum, wt, v in fields(d, 0, len(d)):
        if fnum != 5 or wt != 2:
            continue
        for lf, lwt, lv in fields(d, v[0], v[1]):
            if lf == 5 and lwt == 2:
                n = (lv[1]-lv[0])//12
                lo = [1e9]*3; hi=[-1e9]*3
                for i in range(n):
                    p = struct.unpack_from("<fff", d, lv[0]+i*12)
                    for k in range(3):
                        lo[k]=min(lo[k],p[k]); hi[k]=max(hi[k],p[k])
                return n, lo, hi
    return 0, None, None

found = {}
for path, size, off in kspkg.iter_entries(pkg):
    base = path.lower().replace("/", "\\").rsplit("\\", 1)[-1]
    if base in {w.lower() for w in want}:
        found[base] = (path, size, off)

with open(pkg, "rb") as f:
    for base, rec in sorted(found.items()):
        d = kspkg.read_entry(f, rec[1], rec[2], rec[0])
        n, lo, hi = aabb(d)
        if not lo:
            print(base, "no pos")
            continue
        dx,dy,dz = hi[0]-lo[0], hi[1]-lo[1], hi[2]-lo[2]
        print(f"{base:28} n={n:5} dx={dx:7.2f} dy={dy:7.2f} dz={dz:7.2f} lo={[round(x,2) for x in lo]} hi={[round(x,2) for x in hi]}")
