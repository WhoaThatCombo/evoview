"""List Spa meshes and materials from content.kspkg (index only)."""
import os, sys
sys.path.insert(0, r"C:\Users\joshu\Downloads\acecm")
from acecm import kspkg

pkg = r"C:\Program Files (x86)\Steam\steamapps\common\Assetto Corsa EVO\content.kspkg"
HINT = ("wall", "phys", "collid", "invis", "transp", "dummy", "barrier",
        "grass", "asph", "road", "tarmac", "sand", "gravel", "stand",
        "bleacher", "grand", "seat", "people", "surface")
meshes = []
mats = []
for path, size, off in kspkg.iter_entries(pkg):
    low = path.lower().replace("/", "\\")
    if not low.startswith("content\\tracks\\spa\\"):
        continue
    if low.endswith(".mesh") and "\\static_mesh\\" in low:
        name = low.rsplit("\\", 1)[-1]
        if any(h in name for h in HINT):
            meshes.append((name, size))
    if low.endswith(".material"):
        name = low.rsplit("\\", 1)[-1]
        if any(h in name for h in HINT) or "physics" in name or "null" in name:
            mats.append(name)

print("SPA helper/terrain MESHES", len(meshes))
for n, s in sorted(meshes)[:80]:
    print(f"  {s:9} {n}")
print("... total", len(meshes))
print("\nSPA MATS")
for n in sorted(mats):
    print(" ", n)
