"""wall.material / null.material slots on loose tracks."""
import os, sys
src = open(os.path.join(os.path.dirname(__file__), "dump_track.py"), encoding="utf-8").read()
ns = {}
exec(src[:src.find("\nroot = ")], ns)
ROOT = r"C:\Users\joshu\Saved Games\ACE\mods\content\tracks"
for track in sorted(os.listdir(ROOT)):
    md = os.path.join(ROOT, track, "materials")
    if not os.path.isdir(md):
        continue
    for fn in ("wall.material", "null.material", "dummy.material",
               "lowres.material", "glass_transp.material"):
        p = os.path.join(md, fn)
        if not os.path.isfile(p):
            continue
        shader, slots, params, colors = ns["read_material"](p)
        alb = slots.get("Base_BaseColorMap") or slots.get("txDiffuse") or ""
        print(f"{track:16} {fn:22} {shader} alb={os.path.basename(alb) or '-'} "
              f"blend={params.get('blendMode')} bc={colors.get('Base_Basecolor') or colors.get('Basecolor')}")
        for k, v in sorted(slots.items()):
            if any(x in k.lower() for x in ("color", "diffus", "opac")):
                print(f"    {k}={os.path.basename(v)}")
