"""Road-adjacent highlands meshes and materials."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
src = open(os.path.join(os.path.dirname(__file__), "dump_track.py"), encoding="utf-8").read()
ns = {}
exec(src[:src.find("\nroot = ")], ns)
read_material, mesh_batches = ns["read_material"], ns["mesh_batches"]

root = r"C:\Users\joshu\Saved Games\ACE\mods\content\tracks\highlands_kp"
md = os.path.join(root, "static_mesh")
want = ("road", "side", "line", "bord", "pave", "kerb", "curb", "wall", "barr", "white", "yellow", "dark", "crack")
for fn in sorted(os.listdir(md)):
    low = fn.lower()
    if not any(w in low for w in want):
        continue
    if low.startswith("physics"):
        continue
    refs = mesh_batches(os.path.join(md, fn))
    mats = [r for r in refs if ".material" in r.lower()]
    print(f"{fn:40} {mats[:4]}")
