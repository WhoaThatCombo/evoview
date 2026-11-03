"""Survey terrain materials, scenes, and grass-like textures on every local track."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from dump_track import read_material, scene_insts

ROOT = r"C:\Users\joshu\Saved Games\ACE\mods\content\tracks"

def is_ground_name(s):
    n = s.lower()
    keys = ("grass", "terrain", "ground", "land", "dirt", "sand", "soil",
            "earth", "field", "lawn", "erba", "outer", "ext", "far",
            "mount", "hill", "bank")
    return any(k in n for k in keys)

for track in sorted(os.listdir(ROOT)):
    tdir = os.path.join(ROOT, track)
    if not os.path.isdir(tdir):
        continue
    print(f"\n======== {track} ========")
    mats = os.path.join(tdir, "materials")
    tex = os.path.join(tdir, "textures")
    if os.path.isdir(mats):
        for fn in sorted(os.listdir(mats)):
            if not fn.endswith(".material"):
                continue
            if not is_ground_name(fn) and fn.lower() not in ("road.material", "asphalt.material"):
                continue
            shader, slots, params, colors = read_material(os.path.join(mats, fn))
            print(f"  MAT {fn} shader={shader}")
            for k, v in sorted(slots.items()):
                if any(x in k.lower() for x in ("color", "diffuse", "albedo", "detail", "opac", "spat")):
                    print(f"       {k} = {os.path.basename(v)}")
    if os.path.isdir(tex):
        grass_tex = [f for f in os.listdir(tex)
                     if f.lower().endswith(".texture")
                     and is_ground_name(f)
                     and not any(x in f.lower() for x in ("_nm", "_nrm", "normal", "_map", "mask", "alpha"))]
        print("  TEX", ", ".join(sorted(grass_tex)[:20]), f"({len(grass_tex)} ground-like)")
    scenes = []
    for dirpath, _, files in os.walk(tdir):
        for fn in files:
            if fn.endswith(".scene"):
                p = os.path.join(dirpath, fn)
                insts = scene_insts(p)
                tot = sum(n for _, n in insts)
                scenes.append((os.path.relpath(p, tdir), len(insts), tot))
    for rel, sets, tot in scenes:
        print(f"  SCN {rel}: {sets} sets, {tot} inst")
