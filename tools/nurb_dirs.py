import collections, sys
sys.path.insert(0, r"C:\Users\joshu\Downloads\acecm")
from acecm import kspkg
pkg = r"C:\Program Files (x86)\Steam\steamapps\common\Assetto Corsa EVO\content.kspkg"
dirs = collections.Counter()
samples = {}
for path, size, off in kspkg.iter_entries(pkg):
    low = path.lower().replace("/", "\\")
    if not low.startswith("content\\tracks\\nurburgring\\"):
        continue
    if not low.endswith(".mesh"):
        continue
    parts = low.split("\\")
    folder = "\\".join(parts[:5]) if len(parts) >= 5 else low
    dirs[folder] += 1
    samples.setdefault(folder, path)
print("mesh folders")
for k, n in dirs.most_common(20):
    print(f"  {n:5} {k}  e.g. {samples[k]}")
