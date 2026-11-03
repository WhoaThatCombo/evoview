"""Wall-like meshes on loose tracks that skip_track_mesh would keep."""
import os, re, sys
sys.path.insert(0, os.path.dirname(__file__))
src = open(os.path.join(os.path.dirname(__file__), "dump_track.py"), encoding="utf-8").read()
ns = {}
exec(src[:src.find("\nroot = ")], ns)

ROOT = r"C:\Users\joshu\Saved Games\ACE\mods\content\tracks"
HINT = ("wall", "dummy", "phys", "collid", "invis", "transp", "barrier", "limit")

def skip_name(n):
    n = n.lower()
    if n.startswith(("physics", "dummy", "lowres", "shadow", "transparent",
                     "collision", "wall_non_csp")):
        return True
    if "collider" in n or "collision" in n or "shadow_cache" in n:
        return True
    i = 0
    b = n.encode()
    while i < len(b) and 48 <= b[i] <= 57:
        i += 1
    if i and (n[i:].startswith("wall") or n[i:].startswith("asph")
              or n[i:].startswith("phys") or n[i:].startswith("out")):
        return True
    return False

for track in sorted(os.listdir(ROOT)):
    md = os.path.join(ROOT, track, "static_mesh")
    if not os.path.isdir(md):
        continue
    kept = []
    for fn in sorted(os.listdir(md)):
        if not fn.lower().endswith(".mesh"):
            continue
        low = fn.lower()
        if not any(h in low for h in HINT):
            continue
        if skip_name(low.replace(".mesh", "")):
            continue
        refs = ns["mesh_batches"](os.path.join(md, fn))
        mats = [r for r in refs if ".material" in r.lower()]
        kept.append((fn, os.path.getsize(os.path.join(md, fn)), mats[:3]))
    if not kept:
        continue
    print(f"\n==== {track} kept wall-like {len(kept)}")
    for fn, sz, mats in kept[:20]:
        print(f"  {sz:8} {fn} {mats}")
