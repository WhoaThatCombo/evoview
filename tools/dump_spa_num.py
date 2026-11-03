"""Numbered Spa static meshes that are not wall/asph."""
import os, re, sys, collections
sys.path.insert(0, r"C:\Users\joshu\Downloads\acecm")
from acecm import kspkg
src = open(os.path.join(os.path.dirname(__file__), "dump_track.py"), encoding="utf-8").read()
ns = {}
exec(src[:src.find("\nroot = ")], ns)

pkg = r"C:\Program Files (x86)\Steam\steamapps\common\Assetto Corsa EVO\content.kspkg"
pref = collections.Counter()
samples = collections.defaultdict(list)
with open(pkg, "rb") as f:
    for path, size, off in kspkg.iter_entries(pkg):
        low = path.lower().replace("/", "\\")
        if not low.startswith("content\\tracks\\spa\\static_mesh\\") or not low.endswith(".mesh"):
            continue
        base = low.rsplit("\\", 1)[-1]
        m = re.match(r"^(\d+)([a-z_]+)", base)
        if not m:
            continue
        kind = m.group(2)
        pref[kind] += 1
        if len(samples[kind]) < 3:
            d = kspkg.read_entry(f, size, off, path)
            refs = []
            ns["strings_under"](d, 0, len(d), 0, refs)
            mats = [s for s in refs if ".material" in s.lower()]
            samples[kind].append((base, size, mats[:3]))

print("numbered prefixes")
for k, n in pref.most_common(40):
    print(f"  {n:4} {k}")
    for base, size, mats in samples[k]:
        print(f"       {size:8} {base} {mats}")
