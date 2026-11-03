"""Terrain-named meshes on COTA / Laguna / Watkins."""
import sys
sys.path.insert(0, r"C:\Users\joshu\Downloads\acecm")
from acecm import kspkg
src = open(r"C:\Users\joshu\Downloads\evoview\tools\dump_track.py", encoding="utf-8").read()
ns = {}
exec(src[:src.find("\nroot = ")], ns)
HINT = ("terrain", "ground", "grass", "land", "map", "ext", "sand", "dirt",
        "sbanc", "outer", "far", "hill", "soil")
pkg = r"C:\Program Files (x86)\Steam\steamapps\common\Assetto Corsa EVO\content.kspkg"
for track in ("cota", "laguna_seca", "watkins_glen"):
    print(f"\n==== {track}")
    hits = []
    for path, size, off in kspkg.iter_entries(pkg):
        low = path.lower().replace("/", "\\")
        if not low.startswith(f"content\\tracks\\{track}\\"):
            continue
        base = low.rsplit("\\", 1)[-1]
        if low.endswith(".mesh") and any(h in base for h in HINT):
            hits.append((size, base))
        if low.endswith(".material") and any(h in base for h in HINT):
            hits.append((size, "MAT "+base))
    for size, name in sorted(hits, reverse=True)[:25]:
        print(f"  {size:8} {name}")
