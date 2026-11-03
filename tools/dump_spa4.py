"""Spa textures + scene mesh paths that look like stands (not spectators)."""
import os, sys, collections
sys.path.insert(0, r"C:\Users\joshu\Downloads\acecm")
from acecm import kspkg

pkg = r"C:\Program Files (x86)\Steam\steamapps\common\Assetto Corsa EVO\content.kspkg"
tex = []
meshes = []
scene_paths = []
src = open(r"C:\Users\joshu\Downloads\evoview\tools\dump_track.py", encoding="utf-8").read()
ns = {}
exec(src[:src.find("\nroot = ")], ns)
fields, text = ns["fields"], ns["text"]

scene_raw = None
for path, size, off in kspkg.iter_entries(pkg):
    low = path.lower().replace("/", "\\")
    if low.startswith("content\\tracks\\spa\\") and low.endswith(".texture"):
        tex.append(low.rsplit("\\", 1)[-1])
    if low.startswith("content\\tracks\\spa\\") and low.endswith(".mesh") and "static_mesh" in low:
        name = low.rsplit("\\", 1)[-1]
        if any(x in name for x in ("stand", "gstand", "seat", "tribun", "bleacher")):
            meshes.append(name)
    if low.endswith("\\spa.scene"):
        scene_raw = (path, size, off)

print("SPA textures with asph/track/road/grass/wall:")
for t in sorted(tex):
    if any(x in t for x in ("asph", "track", "road", "tarmac", "wall", "grass")):
        print(" ", t)

print("\nSPA static stand meshes:")
for m in sorted(meshes):
    print(" ", m)

if scene_raw:
    with open(pkg, "rb") as f:
        d = kspkg.read_entry(f, scene_raw[1], scene_raw[2], scene_raw[0])
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
    print("\nscene unique meshes", len(paths))
    for p, n in paths.most_common(30):
        print(f"  {n:3} {p}")
    print("\nstand-like scene paths:")
    for p, n in sorted(paths.items()):
        if any(x in p for x in ("stand", "gstand", "seat", "tribun", "bleacher", "grand")):
            print(f"  {n:3} {p}")
