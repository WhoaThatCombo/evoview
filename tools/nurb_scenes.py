import sys
sys.path.insert(0, r"C:\Users\joshu\Downloads\acecm")
from acecm import kspkg
pkg = r"C:\Program Files (x86)\Steam\steamapps\common\Assetto Corsa EVO\content.kspkg"
scenes = []
nulls = 0
for path, size, off in kspkg.iter_entries(pkg):
    low = path.lower().replace("/", "\\")
    if not low.startswith("content\\tracks\\nurburgring\\"):
        continue
    base = low.rsplit("\\", 1)[-1]
    if low.endswith(".scene"):
        scenes.append((size, path))
    if base.startswith("null") or "\\physics\\" in low:
        nulls += 1
print("scenes", len(scenes), "null/phys files", nulls)
for s, p in sorted(scenes, reverse=True)[:25]:
    print(f"  {s/1e6:6.1f} MB  {p}")
