import collections, sys
sys.path.insert(0, r"C:\Users\joshu\Downloads\acecm")
from acecm import kspkg
pkg = r"C:\Program Files (x86)\Steam\steamapps\common\Assetto Corsa EVO\content.kspkg"
pref = collections.Counter()
n = 0
for path, size, off in kspkg.iter_entries(pkg):
    low = path.lower().replace("/", "\\")
    if not low.startswith("content\\tracks\\nurburgring\\"):
        continue
    if not low.endswith(".mesh"):
        continue
    n += 1
    base = low.rsplit("\\", 1)[-1]
    if "physics" in low or base.startswith("null") or "_wall" in base:
        p = base.split("_")[0]
        pref[p] += 1
print("meshes", n)
# folder sample
print("example paths already counted")
print("wall/null/phys prefixes")
for k, v in pref.most_common(25):
    print(f"  {v:4} {k}")
