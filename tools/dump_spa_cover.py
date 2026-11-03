"""Which Spa static meshes cover the white banks next to the stands?"""
import os, struct, sys
sys.path.insert(0, r"C:\Users\joshu\Downloads\acecm")
from acecm import kspkg

src = open(os.path.join(os.path.dirname(__file__), "dump_track.py"), encoding="utf-8").read()
ns = {}
exec(src[:src.find("\nroot = ")], ns)
fields, strings_under = ns["fields"], ns["strings_under"]

pkg = r"C:\Program Files (x86)\Steam\steamapps\common\Assetto Corsa EVO\content.kspkg"

# left terrace in spa_view9, and the big pale patch right of the circuit
REGIONS = {
    "left_bank": (-1320, 10, -2350, -1160, 45, -2050),
    "right_hill": (-800, 0, -2000, -200, 80, -1400),
    "paddock_ground": (-1200, 15, -2200, -1000, 30, -1850),
}

def aabb(d):
    for fnum, wt, v in fields(d, 0, len(d)):
        if fnum != 5 or wt != 2:
            continue
        for lf, lwt, lv in fields(d, v[0], v[1]):
            if lf == 5 and lwt == 2:
                n = (lv[1]-lv[0])//12
                lo=[1e9]*3; hi=[-1e9]*3
                step=max(1, n//100)
                for i in range(0, n, step):
                    p=struct.unpack_from("<fff", d, lv[0]+i*12)
                    for k in range(3):
                        lo[k]=min(lo[k],p[k]); hi[k]=max(hi[k],p[k])
                return n, lo, hi
    return 0, None, None

def overlap(lo, hi, r):
    return not (hi[0] < r[0] or lo[0] > r[3] or hi[1] < r[1] or lo[1] > r[4] or hi[2] < r[2] or lo[2] > r[5])

hits = {k: [] for k in REGIONS}
print("scanning static meshes…")
with open(pkg, "rb") as f:
    for path, size, off in kspkg.iter_entries(pkg):
        low = path.lower().replace("/", "\\")
        if not low.startswith("content\\tracks\\spa\\static_mesh\\") or not low.endswith(".mesh"):
            continue
        if "\\physics\\" in low:
            continue
        d = kspkg.read_entry(f, size, off, path)
        n, lo, hi = aabb(d)
        if not lo:
            continue
        refs = []
        strings_under(d, 0, len(d), 0, refs)
        mats = [s.rsplit("\\",1)[-1].rsplit("/",1)[-1] for s in refs if ".material" in s.lower()]
        base = low.rsplit("\\", 1)[-1]
        for name, r in REGIONS.items():
            if overlap(lo, hi, r):
                span = ((hi[0]-lo[0])**2+(hi[1]-lo[1])**2+(hi[2]-lo[2])**2)**0.5
                hits[name].append((span, size, base, mats[:3], [round(x,0) for x in lo], [round(x,0) for x in hi]))

for name, rows in hits.items():
    rows.sort(reverse=True)
    print(f"\n==== {name} ({len(rows)} meshes)")
    for span, size, base, mats, lo, hi in rows[:18]:
        print(f"  span={span:6.0f} {size:8} {base:32} {mats}")
        print(f"         {lo}..{hi}")
