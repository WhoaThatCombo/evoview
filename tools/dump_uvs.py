"""UV range for a few highlands meshes."""
import os, struct, sys
sys.path.insert(0, os.path.dirname(__file__))
from dump_track import fields

def uv_stats(path):
    d = open(path, "rb").read()
    uvs = None
    npos = 0
    for f, wt, v in fields(d, 0, len(d)):
        if f != 5 or wt != 2:
            continue
        for lf, lwt, lv in fields(d, v[0], v[1]):
            if lwt != 2:
                continue
            if lf == 5:
                npos = (lv[1] - lv[0]) // 12
            if lf == 7:
                uvs = d[lv[0]:lv[1]]
        break
    if not uvs or npos == 0:
        return f"no uv npos={npos} uvbytes={0 if not uvs else len(uvs)}"
    n = min(npos, len(uvs) // 8)
    us = [struct.unpack_from("<f", uvs, i * 8)[0] for i in range(0, n, max(1, n // 200))]
    vs = [struct.unpack_from("<f", uvs, i * 8 + 4)[0] for i in range(0, n, max(1, n // 200))]
    return f"n={n} u=[{min(us):.3f},{max(us):.3f}] v=[{min(vs):.3f},{max(vs):.3f}] sample=({us[0]:.3f},{vs[0]:.3f})"

root = r"C:\Users\joshu\Saved Games\ACE\mods\content\tracks\highlands_kp\static_mesh"
for fn in ["grass_ext_0_1.mesh", "grass_near_0_1.mesh", "Road_0_1.mesh",
           "grass_ext_mount_0_6.mesh", "dist_1_1.mesh", "lake_-3_2.mesh",
           "grass_BB1_0_1.mesh"]:
    p = os.path.join(root, fn)
    if os.path.isfile(p):
        print(fn, uv_stats(p))
