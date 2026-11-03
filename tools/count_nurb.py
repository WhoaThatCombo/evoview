"""How big is stock nurburgring in content.kspkg?"""
import sys
sys.path.insert(0, r"C:\Users\joshu\Downloads\acecm")
from acecm import kspkg
pkg = r"C:\Program Files (x86)\Steam\steamapps\common\Assetto Corsa EVO\content.kspkg"
tracks = {}
for path, size, off in kspkg.iter_entries(pkg):
    low = path.lower().replace("/", "\\")
    if not low.startswith("content\\tracks\\"):
        continue
    parts = low.split("\\")
    if len(parts) < 4:
        continue
    tid = parts[2]
    if tid == "common_assets":
        continue
    rec = tracks.setdefault(tid, {"mesh": 0, "msz": 0, "tex": 0, "tsz": 0, "mat": 0, "scene": 0, "ssz": 0})
    if low.endswith(".mesh"):
        rec["mesh"] += 1
        rec["msz"] += size
    elif low.endswith(".texture"):
        rec["tex"] += 1
        rec["tsz"] += size
    elif low.endswith(".material"):
        rec["mat"] += 1
    elif low.endswith(".scene"):
        rec["scene"] += 1
        rec["ssz"] += size

print(f"{'track':22} {'meshes':>6} {'meshMB':>7} {'tex':>5} {'texMB':>6} {'mat':>4} {'scn':>3} {'scnMB':>6}")
for tid, r in sorted(tracks.items(), key=lambda kv: -kv[1]["msz"]):
    print(f"{tid:22} {r['mesh']:6} {r['msz']/1e6:7.1f} {r['tex']:5} {r['tsz']/1e6:6.1f} {r['mat']:4} {r['scene']:3} {r['ssz']/1e6:6.1f}")
