"""Dump node/bone local matrices + AABB for Spa stand / asph visual meshes."""
import os, struct, sys
sys.path.insert(0, r"C:\Users\joshu\Downloads\acecm")
from acecm import kspkg

src = open(os.path.join(os.path.dirname(__file__), "dump_track.py"), encoding="utf-8").read()
ns = {}
exec(src[:src.find("\nroot = ")], ns)
fields, text = ns["fields"], ns["text"]

pkg = r"C:\Program Files (x86)\Steam\steamapps\common\Assetto Corsa EVO\content.kspkg"
want = {
    "maingstand1.mesh", "maingstand2.mesh", "grand_stand_1.mesh",
    "grandstand_03.mesh", "multi_seating_01.mesh", "asph_spa_5.mesh",
    "new_wall_gstand.mesh", "walls.mesh",
}

def inspect(d, name, size):
    print(f"\n==== {name} size={size}")
    for fnum, wt, v in fields(d, 0, len(d)):
        if fnum != 5 or wt != 2:
            continue
        pos = None
        bones = []
        for lf, lwt, lv in fields(d, v[0], v[1]):
            if lf == 5 and lwt == 2:
                pos = lv
            if lf == 12 and lwt == 2:
                nm = par = ""
                mat = None
                for bf, bwt, bv in fields(d, lv[0], lv[1]):
                    if bf == 1 and bwt == 2:
                        nm = text(d, bv)
                    elif bf == 2 and bwt == 2:
                        par = text(d, bv)
                    elif bf == 3 and bwt == 2:
                        raw = d[bv[0]:bv[1]]
                        if len(raw) >= 64:
                            mat = struct.unpack_from("<16f", raw)
                bones.append((nm, par, mat))
        if pos:
            n = (pos[1] - pos[0]) // 12
            lo = [1e9]*3; hi = [-1e9]*3
            step = max(1, n // 200)
            for i in range(0, n, step):
                p = struct.unpack_from("<fff", d, pos[0] + i*12)
                for k in range(3):
                    lo[k] = min(lo[k], p[k]); hi[k] = max(hi[k], p[k])
            print(f"  verts~{n} aabb lo={[round(x,1) for x in lo]} hi={[round(x,1) for x in hi]}")
        print(f"  bones {len(bones)}")
        for nm, par, mat in bones[:8]:
            if mat:
                # translation typically last column or last row
                t_col = (round(mat[12],2), round(mat[13],2), round(mat[14],2))
                t_row = (round(mat[3],2), round(mat[7],2), round(mat[11],2))
                rot = [round(mat[i],3) for i in (0,1,2,4,5,6,8,9,10)]
                print(f"    {nm!r} parent={par!r} tcol={t_col} trow={t_row} r3={rot}")
            else:
                print(f"    {nm!r} parent={par!r} (no mat)")
        break

found = {}
for path, size, off in kspkg.iter_entries(pkg):
    low = path.lower().replace("/", "\\")
    if not low.startswith("content\\tracks\\spa\\static_mesh\\"):
        continue
    base = low.rsplit("\\", 1)[-1]
    if base in want:
        found[base] = (path, size, off)

with open(pkg, "rb") as f:
    for base in sorted(found):
        path, size, off = found[base]
        d = kspkg.read_entry(f, size, off, path)
        inspect(d, base, size)
