"""Find the big white pad under Watkins Glen."""
import struct, sys
sys.path.insert(0, r"C:\Users\joshu\Downloads\acecm")
from acecm import kspkg
src = open(r"C:\Users\joshu\Downloads\evoview\tools\dump_track.py", encoding="utf-8").read()
ns = {}
exec(src[:src.find("\nroot = ")], ns)
fields, strings_under = ns["fields"], ns["strings_under"]
pkg = r"C:\Program Files (x86)\Steam\steamapps\common\Assetto Corsa EVO\content.kspkg"

def span_of(d):
    for fnum, wt, v in fields(d, 0, len(d)):
        if fnum != 5 or wt != 2:
            continue
        for lf, lwt, lv in fields(d, v[0], v[1]):
            if lf == 5 and lwt == 2:
                n = (lv[1]-lv[0])//12
                lo=[1e9]*3; hi=[-1e9]*3
                step=max(1,n//60)
                for i in range(0,n,step):
                    p=struct.unpack_from("<fff", d, lv[0]+i*12)
                    for k in range(3):
                        lo[k]=min(lo[k],p[k]); hi[k]=max(hi[k],p[k])
                dx,dy,dz=hi[0]-lo[0],hi[1]-lo[1],hi[2]-lo[2]
                return n, (dx*dx+dy*dy+dz*dz)**0.5, lo, hi
    return 0, 0, None, None

rows = []
with open(pkg, "rb") as f:
    for path, size, off in kspkg.iter_entries(pkg):
        low = path.lower().replace("/", "\\")
        if not low.startswith("content\\tracks\\watkins_glen\\") or not low.endswith(".mesh"):
            continue
        if "physics" in low:
            continue
        d = kspkg.read_entry(f, size, off, path)
        n, span, lo, hi = span_of(d)
        if span < 400:
            continue
        refs = []
        strings_under(d, 0, len(d), 0, refs)
        mats = [s.rsplit("\\",1)[-1] for s in refs if ".material" in s.lower()]
        rows.append((span, size, path.rsplit("\\",1)[-1], mats[:3], lo, hi))
for span, size, name, mats, lo, hi in sorted(rows, reverse=True)[:20]:
    print(f"  span={span:7.0f} {size:7} {name:40} {mats}")
