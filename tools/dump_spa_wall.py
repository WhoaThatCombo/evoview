"""What material do Spa 01wall / asph meshes use?"""
import os, sys
sys.path.insert(0, r"C:\Users\joshu\Downloads\acecm")
from acecm import kspkg
src = open(r"C:\Users\joshu\Downloads\evoview\tools\dump_track.py", encoding="utf-8").read()
ns = {}
exec(src[:src.find("\nroot = ")], ns)
fields, text = ns["fields"], ns["text"]

def mats_in_mesh(d):
    out = []
    ns["strings_under"](d, 0, len(d), 0, out) if "strings_under" in ns else None
    # fallback: collect .material strings
    s = []
    def rec(a, b, depth=0):
        for f, wt, v in fields(d, a, b):
            if wt == 2:
                t = d[v[0]:v[1]]
                if b".material" in t.lower() or (32 <= (t[0] if t else 0) < 127 and b"material" in t.lower()):
                    try:
                        s.append(t.decode("utf-8", "replace"))
                    except Exception:
                        pass
                if depth < 5:
                    rec(v[0], v[1], depth+1)
    rec(0, len(d))
    return [x for x in s if ".material" in x.lower()]

pkg = r"C:\Program Files (x86)\Steam\steamapps\common\Assetto Corsa EVO\content.kspkg"
want = ("01wall.mesh", "02wall.mesh", "01asph-spa_black_.mesh", "walls.mesh",
        "maingstand1.mesh", "multi_seating_01.mesh")
found = {}
for path, size, off in kspkg.iter_entries(pkg):
    low = path.lower().replace("/", "\\")
    if not low.startswith("content\\tracks\\spa\\static_mesh\\"):
        continue
    base = low.rsplit("\\", 1)[-1]
    if base in want or base.startswith("01wall") or base.startswith("02wall"):
        found[base] = (path, size, off)

with open(pkg, "rb") as f:
    for base, rec in sorted(found.items()):
        d = kspkg.read_entry(f, rec[1], rec[2], rec[0])
        print(base, "size", rec[1], "mats", mats_in_mesh(d)[:6])
