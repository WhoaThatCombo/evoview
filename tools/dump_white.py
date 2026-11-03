"""List meshes/materials that likely render as white walls, borders, or ground."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
# pull helpers without running dump_track main
src = open(os.path.join(os.path.dirname(__file__), "dump_track.py"), encoding="utf-8").read()
ns = {}
exec(src[:src.find("\nroot = ")], ns)
read_material = ns["read_material"]

ROOT = r"C:\Users\joshu\Saved Games\ACE\mods\content\tracks"
HINTS = (
    "wall", "dummy", "phys", "collid", "invis", "barrier", "surface",
    "shadow", "lowres", "null", "default", "inv_", "bound", "limit",
    "wallcol", "col_", "_col", "no_render", "norender",
)

for track in sorted(os.listdir(ROOT)):
    tdir = os.path.join(ROOT, track)
    md = os.path.join(tdir, "static_mesh")
    matd = os.path.join(tdir, "materials")
    if not os.path.isdir(md):
        continue
    print(f"\n==== {track} ====")
    meshes = []
    for fn in sorted(os.listdir(md)):
        low = fn.lower()
        if any(h in low for h in HINTS):
            meshes.append(fn)
    print("  meshes:", len(meshes))
    for fn in meshes[:25]:
        print("   ", fn)
    if os.path.isdir(matd):
        for fn in sorted(os.listdir(matd)):
            if not fn.endswith(".material"):
                continue
            low = fn.lower()
            shader, slots, params, colors = read_material(os.path.join(matd, fn))
            alb = slots.get("txDiffuse") or slots.get("Base_BaseColorMap") or ""
            alb_b = os.path.basename(alb).lower()
            interesting = any(h in low for h in HINTS) or "null" in alb_b or alb_b == ""
            if not interesting:
                continue
            print(f"  MAT {fn} shader={shader} alb={alb_b or '(none)'} blend={params.get('blendMode')} cull={params.get('cullMode')}")
