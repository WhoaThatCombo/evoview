"""Dump grass/terrain materials and UV stats on the biggest grass-named meshes."""
import os, struct, sys
sys.path.insert(0, os.path.dirname(__file__))
from dump_track import read_material, fields, text, mesh_batches

root = r"C:\Users\joshu\Saved Games\ACE\mods\content\tracks\highlands_kp"
matdir = os.path.join(root, "materials")
for fn in sorted(os.listdir(matdir)):
    low = fn.lower()
    if not any(x in low for x in ("grass", "gravel", "lake", "pond", "mount", "dist", "shadow")):
        continue
    shader, slots, params, colors = read_material(os.path.join(matdir, fn))
    print(f"\n==== {fn}  shader={shader}")
    for k, v in sorted(slots.items()):
        print(f"  slot {k} = {v}")
    for k, v in sorted(params.items()):
        print(f"  param {k} = {v}")
    for k, v in sorted(colors.items()):
        print(f"  color {k} = {v}")

# UV occupancy for meshes whose name looks like terrain
mdir = os.path.join(root, "static_mesh")
print("\n==== MESH NAMES matching grass/terrain/ground/mount/dist/road")
for fn in sorted(os.listdir(mdir)):
    low = fn.lower()
    if any(x in low for x in ("grass", "terrain", "ground", "mount", "dist", "road", "gravel", "lake", "land")):
        p = os.path.join(mdir, fn)
        print(f"  {os.path.getsize(p)/1e6:.2f} MB  {fn}")
