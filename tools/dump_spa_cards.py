"""Find large Spa instance meshes (forest cards) and sample eulers."""
import os, struct, sys, collections
sys.path.insert(0, r"C:\Users\joshu\Downloads\acecm")
from acecm import kspkg
src = open(os.path.join(os.path.dirname(__file__), "dump_track.py"), encoding="utf-8").read()
ns = {}
exec(src[:src.find("\nroot = ")], ns)
fields, text = ns["fields"], ns["text"]

pkg = r"C:\Program Files (x86)\Steam\steamapps\common\Assetto Corsa EVO\content.kspkg"
HINT = ("forest", "tree", "pine", "wood", "far", "map", "outer", "land",
        "hill", "veg", "bush", "card", "bb", "billboard", "terrain")

scene = None
meshes = {}
for path, size, off in kspkg.iter_entries(pkg):
    low = path.lower().replace("/", "\\")
    if low.endswith("\\spa.scene"):
        scene = (path, size, off)
    if low.startswith("content\\tracks\\spa\\") and low.endswith(".mesh"):
        meshes[low] = (path, size, off)

print("spa.scene", scene[1] if scene else None, "meshes", len(meshes))

def aabb(d):
    for fnum, wt, v in fields(d, 0, len(d)):
        if fnum != 5 or wt != 2:
            continue
        for lf, lwt, lv in fields(d, v[0], v[1]):
            if lf == 5 and lwt == 2:
                n = (lv[1]-lv[0])//12
                lo = [1e9]*3; hi=[-1e9]*3
                step = max(1, n//80)
                for i in range(0, n, step):
                    p = struct.unpack_from("<fff", d, lv[0]+i*12)
                    for k in range(3):
                        lo[k]=min(lo[k],p[k]); hi[k]=max(hi[k],p[k])
                return n, lo, hi
    return 0, None, None

with open(pkg, "rb") as f:
    d = kspkg.read_entry(f, scene[1], scene[2], scene[0])
    paths = collections.Counter()
    eulers = collections.defaultdict(list)
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
            pth = ""; rot_b = None; npos = 0
            for kf, kwt, kv in fields(d, sv[0], sv[1]):
                if kf == 1 and kwt == 2:
                    pth = text(d, kv)
                if kf == 7 and kwt == 2:
                    for tf, twt, tv in fields(d, kv[0], kv[1]):
                        if twt == 2 and tf == 2:
                            npos = (tv[1]-tv[0])//12
                        if twt == 2 and tf == 3:
                            rot_b = tv
            if pth.lower().endswith(".mesh") and npos:
                paths[pth.replace("\\","/").lower()] += npos
                if rot_b and len(eulers[pth]) < 3:
                    eulers[pth].append(struct.unpack_from("<fff", d, rot_b[0]))

print("\ninstance path groups", len(paths), "total inst", sum(paths.values()))
# print biggest by count
for p, n in paths.most_common(25):
    print(f"  {n:7} {p}")

print("\n---- hinted paths + aabb ----")
for p, n in sorted(paths.items(), key=lambda x: -x[1]):
    low = p.lower()
    if not any(h in low for h in HINT):
        continue
    key = low.replace("/", "\\")
    rec = meshes.get(key)
    span = "?"
    if rec:
        raw = kspkg.read_entry(f, rec[1], rec[2], rec[0])
        nv, lo, hi = aabb(raw)
        if lo:
            dx,dy,dz = hi[0]-lo[0], hi[1]-lo[1], hi[2]-lo[2]
            span = f"v={nv} dx={dx:.1f} dy={dy:.1f} dz={dz:.1f} lo={[round(x,1) for x in lo]}"
    eu = eulers.get(p.replace("/","\\")) or eulers.get(p) or []
    # keys in eulers are original path
    eu = []
    for k, vs in eulers.items():
        if k.replace("\\","/").lower() == p:
            eu = vs
            break
    print(f"  {n:6} {os.path.basename(p)} {span} e0={[tuple(round(x,1) for x in e) for e in eu[:2]]}")
