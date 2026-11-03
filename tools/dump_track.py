"""Dump track material slots, mesh batches, and scene instance paths."""
import os, sys, struct
from collections import Counter, defaultdict

def varint(d, i):
    r = 0
    s = 0
    while i < len(d):
        b = d[i]
        i += 1
        r |= (b & 0x7F) << s
        if b & 0x80 == 0:
            return r, i
        s += 7
        if s > 63:
            return None, i
    return None, i

def fields(d, start, end):
    i = start
    end = min(end, len(d))
    while i < end:
        key, i = varint(d, i)
        if key is None:
            return
        fnum, wt = key >> 3, key & 7
        if wt == 0:
            v, i = varint(d, i)
            if v is None:
                return
            yield fnum, 0, v
        elif wt == 1:
            if i + 8 > end:
                return
            yield fnum, 1, d[i:i+8]
            i += 8
        elif wt == 2:
            n, i = varint(d, i)
            if n is None or i + n > end:
                return
            yield fnum, 2, (i, i + n)
            i += n
        elif wt == 5:
            if i + 4 > end:
                return
            yield fnum, 5, struct.unpack_from("<f", d, i)[0]
            i += 4
        else:
            return

def text(d, r):
    a, b = r
    return d[a:b].decode("utf-8", "replace")

def strings_under(d, s, e, depth=0, out=None):
    out = out if out is not None else []
    for f, wt, v in fields(d, s, e):
        if wt == 2:
            a, b = v
            t = d[a:b]
            if t and all(32 <= x < 127 or x in (9, 10, 13) for x in t[:80]):
                try:
                    out.append(t.decode("utf-8"))
                except Exception:
                    pass
            if depth < 4:
                strings_under(d, a, b, depth + 1, out)
    return out

def floats_under(d, s, e, depth=0, out=None):
    out = out if out is not None else []
    for f, wt, v in fields(d, s, e):
        if wt == 5:
            out.append(v)
        elif wt == 2 and depth < 3:
            floats_under(d, v[0], v[1], depth + 1, out)
    return out

def read_material(path):
    d = open(path, "rb").read()
    shader = None
    slots = {}
    params = {}
    colors = {}
    for f, wt, v in fields(d, 0, len(d)):
        if f == 1 and wt == 2:
            shader = text(d, v)
        elif f == 4 and wt == 2:
            name = ""
            fl = []
            for pf, pwt, pv in fields(d, v[0], v[1]):
                if pf == 1 and pwt == 2:
                    name = text(d, pv)
                elif pf == 2 and pwt == 2:
                    floats_under(d, pv[0], pv[1], 0, fl)
            if name and fl:
                if len(fl) == 1:
                    params[name] = fl[0]
                else:
                    colors[name] = fl[:4]
        elif f == 5 and wt == 2:
            name = ""
            refs = []
            for sf, swt, sv in fields(d, v[0], v[1]):
                if sf == 1 and swt == 2:
                    name = text(d, sv)
                elif sf == 2 and swt == 2:
                    strings_under(d, sv[0], sv[1], 0, refs)
            if name and refs:
                slots[name] = refs[0]
    return shader, slots, params, colors

def mesh_batches(path):
    d = open(path, "rb").read()
    mats = []
    strings_under(d, 0, len(d), 0, mats)
    return [s for s in mats if ".material" in s.lower() or s.lower().endswith(".mesh")]

def scene_insts(path):
    d = open(path, "rb").read()
    out = []
    def walk_105(hs, he):
        pth = ""
        npos = 0
        for kf, kwt, kv in fields(d, hs, he):
            if kf == 1 and kwt == 2:
                pth = text(d, kv)
            if kf == 7 and kwt == 2:
                for tf, twt, tv in fields(d, kv[0], kv[1]):
                    if twt == 2 and tf == 2:
                        npos = (tv[1] - tv[0]) // 12
        if pth.lower().endswith(".mesh") and npos:
            out.append((pth, npos))
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

root = r"C:\Users\joshu\Saved Games\ACE\mods\content\tracks\highlands_kp"
print("=== MATERIALS ===")
want = [
    "grass_ext", "grass_near", "grass_ext_mount", "grass_BB1", "grass_BB2",
    "Road", "01_-_Default", "TREES", "bushes", "pines", "flowers_map",
    "hedges_D", "lake", "pond", "gravel", "mount_quinta", "dist",
]
matdir = os.path.join(root, "materials")
for fn in sorted(os.listdir(matdir)):
    if not fn.endswith(".material"):
        continue
    base = fn[:-9]
    if want and not any(w.lower() in base.lower() for w in want):
        continue
    shader, slots, params, colors = read_material(os.path.join(matdir, fn))
    print(f"\n{fn}")
    print(f"  shader={shader}")
    for k, v in sorted(slots.items()):
        print(f"  slot {k} = {v}")
    for k, v in sorted(params.items()):
        if any(x in k.lower() for x in ("rough", "metal", "blend", "opac", "scale", "uv", "tile", "detail", "enable")):
            print(f"  param {k} = {v}")
    for k, v in sorted(colors.items()):
        print(f"  color {k} = {v}")

print("\n=== ALL SLOT NAME COUNTS ===")
slot_names = Counter()
shader_c = Counter()
empty_albedo = []
for fn in sorted(os.listdir(matdir)):
    if not fn.endswith(".material"):
        continue
    shader, slots, params, colors = read_material(os.path.join(matdir, fn))
    shader_c[shader or "?"] += 1
    for k in slots:
        slot_names[k] += 1
    albedo_keys = [k for k in slots if any(x in k.lower() for x in ("color", "diffuse", "albedo", "base"))]
    if not albedo_keys:
        empty_albedo.append((fn, shader, list(slots)))
print("shaders:", shader_c)
print("slots:", slot_names.most_common(40))
print("no obvious albedo:", empty_albedo[:20])

print("\n=== LARGE MESHES + REFS ===")
mdir = os.path.join(root, "static_mesh")
sized = []
for fn in os.listdir(mdir):
    if fn.endswith(".mesh"):
        p = os.path.join(mdir, fn)
        sized.append((os.path.getsize(p), fn))
sized.sort(reverse=True)
for sz, fn in sized[:25]:
    refs = mesh_batches(os.path.join(mdir, fn))
    mats = [r for r in refs if ".material" in r.lower()]
    print(f"  {sz/1e6:.2f} MB  {fn}  mats={mats[:8]}")

print("\n=== SCENES ===")
for dirpath, _, files in os.walk(root):
    for fn in files:
        if not fn.endswith(".scene"):
            continue
        p = os.path.join(dirpath, fn)
        insts = scene_insts(p)
        if not insts:
            print(f"  {os.path.relpath(p, root)}: no ISMesh")
            continue
        tot = sum(n for _, n in insts)
        print(f"  {os.path.relpath(p, root)}: {len(insts)} sets, {tot} inst")
        for path, n in sorted(insts, key=lambda x: -x[1])[:12]:
            print(f"      {n:5d}  {path}")
