"""Instance transform stats + wild grass mesh material."""
import os, struct, sys
sys.path.insert(0, os.path.dirname(__file__))
from dump_track import fields, text, scene_insts, mesh_batches, read_material

root = r"C:\Users\joshu\Saved Games\ACE\mods\content\tracks\highlands_kp"
scene = os.path.join(root, "containers", "ground_grass.scene")
d = open(scene, "rb").read()

# reuse parse from dump_track by walking like parse_scene_instances
def all_insts(path):
    d = open(path, "rb").read()
    out = []
    def walk_105(hs, he):
        pth = ""
        pos_b = rot_b = scl_b = None
        for kf, kwt, kv in fields(d, hs, he):
            if kf == 1 and kwt == 2:
                pth = text(d, kv)
            if kf == 7 and kwt == 2:
                for tf, twt, tv in fields(d, kv[0], kv[1]):
                    if twt == 2:
                        if tf == 2: pos_b = tv
                        if tf == 3: rot_b = tv
                        if tf == 4: scl_b = tv
        if not pos_b or not pth.lower().endswith(".mesh"):
            return
        n = (pos_b[1] - pos_b[0]) // 12
        def f3(buf, i):
            o = i * 12
            return struct.unpack_from("<fff", buf, o)
        pb = d[pos_b[0]:pos_b[1]]
        rb = d[rot_b[0]:rot_b[1]] if rot_b else b""
        sb = d[scl_b[0]:scl_b[1]] if scl_b else b""
        for i in range(min(n, 20)):
            p = f3(pb, i)
            r = f3(rb, i) if len(rb) >= (i+1)*12 else (0,0,0)
            s = f3(sb, i) if len(sb) >= (i+1)*12 else (1,1,1)
            out.append((pth, p, r, s))
        out.append(("__count__", (n, 0, 0), (0,0,0), (0,0,0)))
    for f, wt, v in fields(d, 0, len(d)):
        if f != 2 or wt != 2:
            continue
        blob = None
        for sf, swt, sv in fields(d, v[0], v[1]):
            if sf == 50 and swt == 2:
                blob = sv
        if not blob:
            continue
        for sf, swt, sv in fields(d, blob[0], blob[1]):
            if sf == 105 and swt == 2:
                walk_105(sv[0], sv[1])
    return out

insts = all_insts(scene)
counts = [t for t in insts if t[0] == "__count__"]
samples = [t for t in insts if t[0] != "__count__"]
print("sets", len(counts), "total", sum(t[1][0] for t in counts))
print("sample path", samples[0][0] if samples else None)
for t in samples[:8]:
    print("  pos", tuple(round(x,2) for x in t[1]), "euler", tuple(round(x,1) for x in t[2]), "scale", tuple(round(x,3) for x in t[3]))

# compare a tree instance from main scene
insts2 = all_insts(os.path.join(root, "highlands_kp.scene"))
samples2 = [t for t in insts2 if t[0] != "__count__" and "spruce" in t[0].lower()]
print("spruce samples", len(samples2))
for t in samples2[:4]:
    print("  pos", tuple(round(x,2) for x in t[1]), "euler", tuple(round(x,1) for x in t[2]), "scale", tuple(round(x,3) for x in t[3]))
