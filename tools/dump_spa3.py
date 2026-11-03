"""All slots on Spa asphalt/wall materials; grandstand eulers."""
import os, sys, struct
sys.path.insert(0, r"C:\Users\joshu\Downloads\acecm")
from acecm import kspkg

src = open(r"C:\Users\joshu\Downloads\evoview\tools\dump_track.py", encoding="utf-8").read()
ns = {}
exec(src[:src.find("\nroot = ")], ns)
read_material, fields, text = ns["read_material"], ns["fields"], ns["text"]

pkg = r"C:\Program Files (x86)\Steam\steamapps\common\Assetto Corsa EVO\content.kspkg"
tmp = r"C:\Users\joshu\Downloads\evoview\tools\_spa_mats"
os.makedirs(tmp, exist_ok=True)

want = (
    "asph_spa.material", "asph_spa_dark.material", "asph_spa_old.material",
    "walls.material", "grass_brd.material", "grass.material",
    "asph_paint_black.material", "asph_ext_dark.material",
)

entries = {}
scene = None
for path, size, off in kspkg.iter_entries(pkg):
    low = path.lower().replace("/", "\\")
    if not low.startswith("content\\tracks\\spa\\"):
        continue
    base = low.rsplit("\\", 1)[-1]
    if base in want:
        entries[base] = (path, size, off)
    if low.endswith("\\spa.scene"):
        scene = (path, size, off)

with open(pkg, "rb") as f:
    for base, rec in entries.items():
        data = kspkg.read_entry(f, rec[1], rec[2], rec[0])
        dest = os.path.join(tmp, base)
        open(dest, "wb").write(data)
        shader, slots, params, colors = read_material(dest)
        print(f"\n==== {base} shader={shader}")
        for k, v in sorted(slots.items()):
            print(f"  {k} = {v}")

    if scene:
        raw = kspkg.read_entry(f, scene[1], scene[2], scene[0])
        print("\n==== spa.scene grandstand-like paths / eulers")
        # walk ISMesh
        def walk_105(d, hs, he):
            pth = ""
            pos_b = rot_b = None
            for kf, kwt, kv in fields(d, hs, he):
                if kf == 1 and kwt == 2:
                    pth = text(d, kv)
                if kf == 7 and kwt == 2:
                    for tf, twt, tv in fields(d, kv[0], kv[1]):
                        if twt == 2:
                            if tf == 2: pos_b = tv
                            if tf == 3: rot_b = tv
            return pth, pos_b, rot_b
        d = raw
        nprint = 0
        for fnum, wt, v in fields(d, 0, len(d)):
            if fnum != 2 or wt != 2:
                continue
            blob = None
            for sf, swt, sv in fields(d, v[0], v[1]):
                if sf == 50 and swt == 2:
                    blob = sv
            if not blob:
                continue
            for sf, swt, sv in fields(d, blob[0], blob[1]):
                if sf != 105 or swt != 2:
                    continue
                pth, pos_b, rot_b = walk_105(d, sv[0], sv[1])
                low = pth.lower()
                if not any(x in low for x in ("stand", "seat", "grand", "tribune", "bleacher")):
                    continue
                print(" PATH", pth)
                if rot_b:
                    n = (rot_b[1] - rot_b[0]) // 12
                    print("  inst", n)
                    for i in range(min(6, n)):
                        o = rot_b[0] + i * 12
                        e = struct.unpack_from("<fff", d, o)
                        print("   euler", tuple(round(x, 2) for x in e))
                nprint += 1
        print("grandstand sets", nprint)
