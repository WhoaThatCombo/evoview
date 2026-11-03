"""Spa materials whose albedo is missing or editor-only."""
import os, sys
sys.path.insert(0, r"C:\Users\joshu\Downloads\acecm")
from acecm import kspkg
src = open(os.path.join(os.path.dirname(__file__), "dump_track.py"), encoding="utf-8").read()
ns = {}
exec(src[:src.find("\nroot = ")], ns)

pkg = r"C:\Program Files (x86)\Steam\steamapps\common\Assetto Corsa EVO\content.kspkg"
tmp = r"C:\Users\joshu\Downloads\evoview\tools\_spa_mats"
os.makedirs(tmp, exist_ok=True)

def placeholder(p):
    low = (p or "").replace("\\", "/").lower()
    return (not low) or "editor/" in low or "default_material" in low or low.endswith("null.texture")

n = 0
with open(pkg, "rb") as f:
    for path, size, off in kspkg.iter_entries(pkg):
        low = path.lower().replace("/", "\\")
        if not low.startswith("content\\tracks\\spa\\") or not low.endswith(".material"):
            continue
        d = kspkg.read_entry(f, size, off, path)
        dest = os.path.join(tmp, "_one.material")
        open(dest, "wb").write(d)
        shader, slots, params, colors = ns["read_material"](dest)
        alb = slots.get("Base_BaseColorMap") or slots.get("txDiffuse") or slots.get("AlbedoMap") or ""
        if not placeholder(alb):
            continue
        extra = {k: os.path.basename(v) for k, v in slots.items()
                 if not placeholder(v) and any(x in k.lower() for x in ("color", "diffuse", "detail", "spat"))}
        bc = colors.get("Base_Basecolor") or colors.get("Basecolor")
        print(f"{os.path.basename(path):32} {shader} alb={os.path.basename(alb) or '-'} extra={extra} bc={None if not bc else [round(x,2) for x in bc[:3]]}")
        n += 1
print("editor-only", n)
