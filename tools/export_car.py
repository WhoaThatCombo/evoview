"""Export an AC EVO car to a flat binary the viewer can mmap and upload.

The reverse engineering lives here, in Python, where it was worked out; the
renderer stays a renderer. The .evocar file is deliberately dumb - little
endian, no compression, arrays already in GPU layout - so loading it is a read
and a memcpy rather than a parser.

    python export_car.py <car_dir> <out.evocar>

`car_dir` is an extracted car: meshes/*.mesh, materials/*.material,
textures/*.texture(+.texturemips).

FILE LAYOUT
    magic "EVOCAR01"
    u32 mesh_count, u32 texture_count, u32 material_count
    then a JSON manifest (u32 length + bytes) describing every section by
    offset/size, followed by the raw blobs.

Textures are DE-TILED here (see the 64 KB page rule) but NOT decompressed:
BC1/BC7 blocks go to the GPU as they are.
"""
import json
import os
import re
import struct
import sys

import paint as paintlib

PAGE = 65536

# format enum -> (gpu format, bits per pixel, tile w, tile h)
# One 64 KB page is always exactly one tile; the tile's pixel size follows
# from the bit depth. See acevo-texture-format-cracked.
FORMATS = {
    2:  ("rgba8", 32, 128, 128),
    3:  ("rgba8", 32, 128, 128),
    5:  ("rgba8", 32, 128, 128),
    6:  ("rgba8", 32, 128, 128),
    10: ("rgba8", 32, 128, 128),
    9:  ("rgba16f", 64, 128, 64),
    12: ("bc1", 4, 512, 256),
    13: ("bc1", 4, 512, 256),
    27: ("bc1", 4, 512, 256),
    # ⚠ 26 is BC6H, not BC7. Both are 16 bytes per 4x4 block, so every size
    # and page calculation matches either way and the mistake is invisible in
    # the layout - it only shows up as pixel-level shading noise wherever a
    # normal map is bound. In this content set enum 26 is used by normal maps
    # and by nothing else (18/18).
    26: ("bc6h", 8, 256, 256),
    33: ("bc7", 8, 256, 256),
    34: ("bc7", 8, 256, 256),
    35: ("bc7", 8, 256, 256),
}


# ----------------------------------------------------------------- protobuf --
def varint(d, i):
    r = s = 0
    while True:
        b = d[i]
        i += 1
        r |= (b & 0x7F) << s
        if not b & 0x80:
            return r, i
        s += 7


def fields(d, i, end):
    """Walk protobuf fields, stopping cleanly at anything malformed.

    ⚠ Every read is bounds-checked. These walkers get pointed at arbitrary
    bytes (a material's nested blobs may not be protobuf at all), and an
    unchecked struct.unpack_from happily reads past the end and raises - which
    took down a whole export over one stray sub-message.
    """
    end = min(end, len(d))
    while i < end:
        try:
            key, i = varint(d, i)
        except IndexError:
            return
        fn, wt = key >> 3, key & 7
        if wt == 0:
            try:
                v, i = varint(d, i)
            except IndexError:
                return
        elif wt == 2:
            try:
                ln, i = varint(d, i)
            except IndexError:
                return
            if ln < 0 or i + ln > end:
                return
            v = (i, i + ln)
            i += ln
        elif wt == 5:
            if i + 4 > end:
                return
            v = struct.unpack_from("<f", d, i)[0]
            i += 4
        elif wt == 1:
            if i + 8 > end:
                return
            v = struct.unpack_from("<d", d, i)[0]
            i += 8
        else:
            return
        yield fn, wt, v


def scalars(d):
    out = {}
    for fn, wt, v in fields(d, 0, len(d)):
        if wt in (0, 5):
            out.setdefault(fn, v)
    return out


# --------------------------------------------------------------------- mesh --
def read_mesh(path):
    """LOD0 only: positions, normals, uvs, tangents, indices, batches."""
    d = open(path, "rb").read()
    for fn, wt, v in fields(d, 0, len(d)):
        if fn != 5 or wt != 2:                 # first lods entry = LOD0
            continue
        lod = {"batches": []}
        for lf, lw, lv in fields(d, v[0], v[1]):
            if lw != 2:
                continue
            s, e = lv
            if lf == 5:
                lod["positions"] = d[s:e]
            elif lf == 6:
                lod["normals"] = d[s:e]
            elif lf == 7:
                lod["uvs"] = d[s:e]
            elif lf == 8:
                lod["tangents"] = d[s:e]
            elif lf == 11:
                idx, i = [], s
                while i < e:
                    x, i = varint(d, i)
                    idx.append(x)
                lod["indices"] = struct.pack(f"<{len(idx)}I", *idx)
            elif lf == 4:
                b = {}
                for bf, bw, bv in fields(d, s, e):
                    if bw == 2 and bf in (1, 4):
                        try:
                            t = d[bv[0]:bv[1]].decode("utf-8")
                        except UnicodeDecodeError:
                            t = ""
                        b["name" if bf == 1 else "material"] = t
                    elif bw == 0 and bf in (2, 3):
                        b["start" if bf == 2 else "count"] = bv
                if b.get("count"):
                    lod["batches"].append(b)
        return lod            # LOD0 is all the viewer needs
    return None


# ------------------------------------------------------------------ texture --
def read_texture(tex_path):
    """De-tile a texture into a linear, GPU-ready mip chain."""
    hdr = scalars(open(tex_path, "rb").read())
    w, h, mips, fmt = hdr.get(1), hdr.get(2), hdr.get(3, 1), hdr.get(4)
    mp = tex_path[:-len(".texture")] + ".texturemips"
    if fmt not in FORMATS or not os.path.isfile(mp):
        return None
    kind, bpp, tw, th = FORMATS[fmt]
    raw = open(mp, "rb").read()

    block = 4 if kind in ("bc1", "bc7", "bc6h") else 1
    out, off, levels = bytearray(), 0, []
    for m in range(mips):
        mw, mh = max(1, w >> m), max(1, h >> m)
        # ONE TILE IS ONE FULL 64 KB PAGE, padded in width AND height. The page
        # count therefore comes from the tile grid, never from the mip's own
        # byte size.
        #
        # ⚠ Deriving pages from the byte size agrees for most textures and is
        # wrong exactly where a mip spans several tiles across but is shorter
        # than one tile - e.g. 512x128 with 256x256 tiles, which really costs
        # two pages, not one. Get it wrong and that mip decodes half-black and
        # every mip after it is offset, so the texture is fine up close and
        # falls apart at distance. Verified against the exact byte size of
        # every .texturemips tested.
        cols = max(1, -(-mw // tw))
        rows = max(1, -(-mh // th))
        pages = cols * rows
        if off + pages * PAGE > len(raw):
            break

        row_bytes = _row_bytes(kind, mw, bpp)
        need = row_bytes * max(1, (mh + block - 1) // block)
        tile_row_bytes = _row_bytes(kind, tw, bpp)
        level = bytearray(need)
        for r in range(rows):
            for c in range(cols):
                page = off + (r * cols + c) * PAGE
                # a tile at the right or bottom edge is only partly meaningful
                keep_w = _row_bytes(kind, min(tw, mw - c * tw), bpp)
                keep_h = max(1, (min(th, mh - r * th) + block - 1) // block)
                for y in range(keep_h):
                    src = page + y * tile_row_bytes
                    dst = ((r * (th // block)) + y) * row_bytes + c * tile_row_bytes
                    if dst + keep_w > need:
                        break
                    level[dst:dst + keep_w] = raw[src:src + keep_w]
        out += level
        levels.append({"width": mw, "height": mh, "bytes": need})
        off += pages * PAGE
    return {"format": kind, "width": w, "height": h,
            "levels": levels, "data": bytes(out)}


def _row_bytes(kind, width, bpp):
    if kind == "bc1":
        return max(1, (width + 3) // 4) * 8
    if kind in ("bc7", "bc6h"):
        return max(1, (width + 3) // 4) * 16
    return width * bpp // 8


# ----------------------------------------------------------------- material --
def _strings_under(d, s, e, depth=0):
    """Every printable string anywhere under a submessage."""
    out = []
    for fn, wt, v in fields(d, s, e):
        if wt != 2:
            continue
        try:
            t = d[v[0]:v[1]].decode("utf-8")
            if t.isprintable() and len(t) > 2:
                out.append(t)
                continue
        except UnicodeDecodeError:
            pass
        if depth < 4:
            out += _strings_under(d, v[0], v[1], depth + 1)
    return out


def read_material(path):
    """Shader name, scalar params, and the texture bound to each slot.

    Layout (UberVehicleMaterial):
        f1 = shader name
        f4 = repeated { f1 = param name, f2 = value }
        f5 = repeated { f1 = slot name,  f2 = texture path }

    ⚠ Car paint often binds NO base-colour map - the body colour is a material
    parameter and the maps supply flakes, orange peel, AO and damage. Treating
    a missing albedo as an error would reject the most important material on
    the car.
    """
    d = open(path, "rb").read()
    out = {"file": os.path.basename(path), "shader": None,
           "slots": {}, "params": {}, "colors": {}}
    for fn, wt, v in fields(d, 0, len(d)):
        if fn == 1 and wt == 2:
            try:
                out["shader"] = d[v[0]:v[1]].decode("utf-8")
            except UnicodeDecodeError:
                pass
        elif fn == 5 and wt == 2:               # texture slot
            slot, paths = None, []
            for pf, pw, pv in fields(d, v[0], v[1]):
                if pf == 1 and pw == 2:
                    try:
                        slot = d[pv[0]:pv[1]].decode("utf-8")
                    except UnicodeDecodeError:
                        pass
                elif pf == 2 and pw == 2:
                    paths = _strings_under(d, pv[0], pv[1])
            if slot and paths:
                out["slots"][slot] = paths[0]
        elif fn == 4 and wt == 2:               # parameter
            name, floats = None, []

            def collect(s, e, depth=0):
                for xf, xw, xv in fields(d, s, e):
                    if xw == 5:
                        floats.append(round(xv, 4))
                    elif xw == 2 and depth < 3:
                        collect(xv[0], xv[1], depth + 1)

            for pf, pw, pv in fields(d, v[0], v[1]):
                if pf == 1 and pw == 2:
                    try:
                        name = d[pv[0]:pv[1]].decode("utf-8")
                    except UnicodeDecodeError:
                        pass
                elif pf == 2 and pw == 2:
                    collect(pv[0], pv[1])
            if not name or not floats:
                continue
            # ⚠ Keep VECTORS as well as scalars. Car paint carries its colour
            # in Base_Basecolor / ksBaseColor (rgb or rgba); reading only single
            # floats dropped every colour and left the whole body neutral grey.
            if len(floats) == 1:
                out["params"][name] = floats[0]
            else:
                out["colors"][name] = floats[:4]
    return out


# ------------------------------------------------------------------- export --
def _find_shared(ref, roots):
    """Locate a texture referenced by engine path, e.g.
    content/cars/common_assets/textures/carpaint/flakes_NM.texture

    Materials reference shared assets that live outside the car folder. Missing
    ones are not an error - the viewer falls back - but finding them is what
    makes carbon, flakes and clear-coat look right.
    """
    rel = ref.replace("\\", os.sep).replace("/", os.sep)
    base = os.path.basename(rel)
    for root in roots:
        cand = os.path.join(root, rel)
        if os.path.isfile(cand):
            return cand
        # fall back to a name match anywhere under the root
        for dirpath, _, files in os.walk(root):
            for f in files:
                if f.lower() == base.lower():
                    return os.path.join(dirpath, f)
    return None


def read_bones(path):
    """The skeleton stored alongside LOD0: name -> (parent, translation).

    ⚠ This is what actually places the wheels. Every rim mesh is authored at
    the ORIGIN and positioned at runtime from bone WHEEL_xx; exporting them as
    they sit stacks all four in the middle of the car.

    Field 3 is a 4x4 row-major matrix written as floats 1..16 with proto3
    defaults omitted, so the translation is fields 13/14/15 - and a missing
    field means zero rather than a parse failure.
    """
    d = open(path, "rb").read()
    bones = {}
    for fn, wt, v in fields(d, 0, len(d)):
        if fn != 5 or wt != 2:
            continue
        for lf, lw, lv in fields(d, v[0], v[1]):
            if lf != 12 or lw != 2:
                continue
            name = parent = None
            t = [0.0, 0.0, 0.0]
            for bf, bw, bv in fields(d, lv[0], lv[1]):
                if bw != 2:
                    continue
                if bf == 1:
                    name = d[bv[0]:bv[1]].decode("utf-8", "ignore")
                elif bf == 2:
                    parent = d[bv[0]:bv[1]].decode("utf-8", "ignore")
                elif bf == 3:
                    for tf, tw, tv in fields(d, bv[0], bv[1]):
                        if tw == 5 and tf in (13, 14, 15):
                            t[tf - 13] = tv
            if name:
                bones[name] = (parent, t)
        break                            # LOD0 carries the skeleton
    return bones


def bone_world(bones, name):
    """Translation of a bone in car space, walking up through its parents."""
    out = [0.0, 0.0, 0.0]
    seen = set()
    while name in bones and name not in seen:
        seen.add(name)
        parent, t = bones[name]
        for a in range(3):
            out[a] += t[a]
        name = parent
    return out


def _offset(positions, normals, delta):
    """Translate a packed position array; normals are unaffected."""
    pos = bytearray(positions)
    for i in range(len(pos) // 12):
        for a in range(3):
            o = i * 12 + a * 4
            v = struct.unpack_from("<f", pos, o)[0] + delta[a]
            struct.pack_into("<f", pos, o, v)
    return bytes(pos), normals


def read_tyretextures(path):
    """A compound's texture set: colour, normal and AO for the tread.

    The tyre material itself binds Tyre_Default_C, a flat grey placeholder -
    the game swaps in this set at runtime depending on which compound is
    fitted. Exporting the material as authored gives white tyres.
    """
    d = open(path, "rb").read()
    got = {}
    for fn, wt, v in fields(d, 0, len(d)):
        if wt == 2 and fn in (5, 6, 7):
            got[{5: "Base_BaseColorMap", 6: "Base_NormalMap",
                 7: "Base_AmbientOcclusionMap"}[fn]] = \
                d[v[0]:v[1]].decode("utf-8", "ignore")
    return got


def find_tyretextures(roots, wanted=None):
    """Pick a compound's texture set - the named one, else the first found."""
    hits = []
    for root in roots:
        for dirpath, _, files in os.walk(root):
            for f in files:
                if f.endswith(".tyretextures"):
                    hits.append(os.path.join(dirpath, f))
    if not hits:
        return None
    hits.sort()
    if wanted:
        for h in hits:
            if wanted.lower() in os.path.basename(h).lower():
                return h
    return hits[0]


def _corner_of(filename):
    """LF/RF/LR/RR for a wheel mesh, else None."""
    m = re.search(r"_rim(?:_blur)?_(lf|rf|lr|rr)\.mesh$", filename.lower())
    return m.group(1).upper() if m else None


def _find_skeleton(mesh_dir):
    """The skeleton lives on the body mesh, not on the wheels."""
    for f in sorted(os.listdir(mesh_dir)):
        if not f.endswith(".mesh") or "_rim" in f:
            continue
        b = read_bones(os.path.join(mesh_dir, f))
        if any(k.startswith("WHEEL_") for k in b):
            return b
    return {}


def _centre(positions):
    """Bounding-box centre of a packed float32 xyz array."""
    n = len(positions) // 12
    lo = [1e30] * 3
    hi = [-1e30] * 3
    for i in range(n):
        for a in range(3):
            v = struct.unpack_from("<f", positions, i * 12 + a * 4)[0]
            lo[a] = min(lo[a], v)
            hi[a] = max(hi[a], v)
    return [(lo[a] + hi[a]) * 0.5 for a in range(3)]


def _place(lod, centre, mirror):
    """Copy a tyre onto a wheel: translate to `centre`, mirror if it is a
    left-hand wheel.

    ⚠ Mirroring flips triangle winding. Without reversing the indices the
    mirrored tyres are backface-culled and the car appears to have two wheels.
    Tangents have to be mirrored as well, or the tread's normal map lights from
    the wrong side on half the car.
    """
    src_c = _centre(lod["positions"])
    pos = bytearray(lod["positions"])
    nrm = bytearray(lod.get("normals") or b"")
    tan = bytearray(lod.get("tangents") or b"")
    sx = -1.0 if mirror else 1.0
    for i in range(len(pos) // 12):
        for a, off in ((0, 0), (1, 4), (2, 8)):
            v = struct.unpack_from("<f", pos, i * 12 + off)[0] - src_c[a]
            if a == 0:
                v *= sx
            struct.pack_into("<f", pos, i * 12 + off, v + centre[a])
    if mirror:
        for i in range(len(nrm) // 12):
            v = struct.unpack_from("<f", nrm, i * 12)[0]
            struct.pack_into("<f", nrm, i * 12, -v)
        # tangent is xyz + handedness w; mirroring flips both x and handedness
        for i in range(len(tan) // 16):
            x = struct.unpack_from("<f", tan, i * 16)[0]
            w = struct.unpack_from("<f", tan, i * 16 + 12)[0]
            struct.pack_into("<f", tan, i * 16, -x)
            struct.pack_into("<f", tan, i * 16 + 12, -w)
    idx = bytearray(lod["indices"])
    if mirror:
        for t in range(len(idx) // 12):
            a = idx[t * 12:t * 12 + 4]
            c = idx[t * 12 + 8:t * 12 + 12]
            idx[t * 12:t * 12 + 4] = c
            idx[t * 12 + 8:t * 12 + 12] = a
    return bytes(pos), bytes(nrm), bytes(tan), bytes(idx)


def _tyre_sizes(car_dir, roots):
    """Front and rear tyre size strings from the car's .compatibletyres.

    The file lists physics tyres (road_235_45_17.tyre); the render meshes are
    named for the same size (tyre_235_45_17.mesh), which is the only link
    between the two.
    """
    sizes = []
    for root in roots:
        for dirpath, _, files in os.walk(root):
            for f in files:
                if not f.endswith(".compatibletyres"):
                    continue
                d = open(os.path.join(dirpath, f), "rb").read()
                # ⚠ Sizes are not all road-car shaped. A road tyre is
                # 235_45_17 but an F1 tyre is 355_660_13, so the middle group
                # has to allow three digits or every open-wheeler silently
                # comes out on bare rims.
                for m in re.finditer(rb"[A-Za-z0-9_]*?_(\d{2,3}_\d{2,3}_\d{2})\.tyre", d):
                    s = m.group(1).decode()
                    if s not in sizes:
                        sizes.append(s)
            break                       # top level only; presets/ is nested
    # front first, rear second - a car with one size uses it on all four
    return sizes[:2] if len(sizes) >= 2 else (sizes * 2)[:2]


def _tyre_family(car_dir, roots):
    """Which compound family this car runs, from its own .compatibletyres.

    ⚠ Do not just take the first .tyretextures found: once shared assets hold
    more than one set, "first" is alphabetical, and a road car ends up wearing
    F1 slicks. The physics tyre path names the family -
    common_phsx/tyres/<family>/... - so use that.
    """
    fams = []
    for root in [os.path.join(car_dir, "presets")] + list(roots):
        for dirpath, _, files in os.walk(root):
            for f in files:
                if not f.endswith(".compatibletyres"):
                    continue
                d = open(os.path.join(dirpath, f), "rb").read()
                for m in re.finditer(rb"tyres[\\/]([a-z0-9_]+)[\\/]", d):
                    fams.append(m.group(1).decode())
            break
    if not fams:
        return "road_"
    fam = fams[0]
    if fam.startswith("f1"):
        return "f1"
    if fam.startswith("racing") or "slick" in fam:
        return "racing"
    return "road_"


def _tyre_mesh(car_dir, roots, rear):
    """The tyre mesh file for this axle, chosen by the car's own tyre sizes."""
    sizes = _tyre_sizes(car_dir, [os.path.join(car_dir, "presets")] + list(roots))
    if not sizes:
        print("  tyres: no .compatibletyres found, wheels will be bare")
        return None
    size = sizes[1] if rear else sizes[0]
    path = _find_shared(f"tyre_{size}.mesh", roots)
    if not path:
        print(f"  tyres: no mesh for {size} - run tools/pull_shared.py")
    return path


def main(car_dir, out_path, extra_roots=()):
    manifest = {"meshes": [], "textures": [], "materials": []}
    blobs = bytearray()
    tex_index = {}                      # lowercase basename -> index

    def add(data):
        off = len(blobs)
        blobs.extend(data)
        while len(blobs) % 4:           # keep every section 4-byte aligned
            blobs.append(0)
        return off, len(data)

    def add_texture(path, name):
        # ⚠ Key on the resolved PATH, not the file name. A liveried car ships
        # skins/<livery>/ext_skin_d.texture AND textures/ext_skin_d.texture -
        # same name, different image. Keying by name silently binds whichever
        # was added first, so the livery loses and the car renders in its
        # default colours with only the decals showing.
        key = os.path.normcase(os.path.abspath(path))
        if key in tex_index:
            return tex_index[key]
        t = read_texture(path)
        if not t:
            return None
        o, n = add(t.pop("data"))
        t["name"] = name
        t["data"] = [o, n]
        manifest["textures"].append(t)
        tex_index[key] = len(manifest["textures"]) - 1
        return tex_index[key]

    # ---- meshes ----
    roots_early = [car_dir] + list(extra_roots)
    bones = {}
    wheel_pos = {}
    shared_materials = set()
    mesh_dir = os.path.join(car_dir, "meshes")
    for f in sorted(os.listdir(mesh_dir)) if os.path.isdir(mesh_dir) else []:
        if not f.endswith(".mesh") or "collider" in f:
            continue
        lod = read_mesh(os.path.join(mesh_dir, f))
        if not lod or "positions" not in lod or "indices" not in lod:
            continue
        # Wheels are the one part of the car not authored in place.
        corner = _corner_of(f)
        if corner:
            if not bones:
                bones.update(_find_skeleton(mesh_dir))
            where = bone_world(bones, "WHEEL_" + corner)
            if any(where):
                lod["positions"], lod["normals"] = _offset(
                    lod["positions"], lod.get("normals"), where)
                wheel_pos[corner] = where
        entry = {"name": f, "vertices": len(lod["positions"]) // 12,
                 "batches": lod["batches"]}
        for key in ("positions", "normals", "uvs", "tangents", "indices"):
            if key in lod:
                o, n = add(lod[key])
                entry[key] = [o, n]
        manifest["meshes"].append(entry)
        print(f"  mesh {f:44} {entry['vertices']:>7,} verts "
              f"{len(lod['indices'])//12:>7,} tris  {len(lod['batches'])} batches")

    # ---- tyres -----------------------------------------------------------
    # A car folder ships no tyre geometry: tyres are shared assets picked from
    # the car's .compatibletyres by size, modelled once at the origin, and put
    # on the same WHEEL_xx bones as the rims.
    for corner, where in sorted(wheel_pos.items()):
        rear = corner.endswith("R")
        tyre_path = _tyre_mesh(car_dir, roots_early, rear)
        if not tyre_path:
            continue
        lod = read_mesh(tyre_path)
        if not lod:
            print(f"  tyre {corner}: could not read {tyre_path}")
            continue
        mirror = where[0] < 0.0
        pos, nrm, tan, idx = _place(lod, where, mirror)
        entry = {"name": f"tyre_{corner}", "vertices": len(pos) // 12,
                 "batches": lod["batches"]}
        for key, data in (("positions", pos), ("normals", nrm),
                          ("uvs", lod.get("uvs")), ("tangents", tan),
                          ("indices", idx)):
            if data:
                o, n = add(data)
                entry[key] = [o, n]
        manifest["meshes"].append(entry)
        for b in lod["batches"]:
            if b.get("material"):
                shared_materials.add(b["material"])
        print(f"  tyre {entry['name']:44} {entry['vertices']:>7,} verts  at "
              f"({where[0]:.3f}, {where[1]:.3f}, {where[2]:.3f})"
              f"{'  mirrored' if mirror else ''}")

    # ---- the car's own textures ----
    tex_dir = os.path.join(car_dir, "textures")
    if os.path.isdir(tex_dir):
        for dirpath, _, files in os.walk(tex_dir):
            for f in sorted(files):
                if f.endswith(".texture"):
                    add_texture(os.path.join(dirpath, f), f[:-len(".texture")])
    own = len(manifest["textures"])

    # ---- materials, and any shared textures they reference ----
    roots = [car_dir] + list(extra_roots)
    mat_dir = os.path.join(car_dir, "materials")
    mat_paths = [os.path.join(mat_dir, f)
                 for f in (sorted(os.listdir(mat_dir))
                           if os.path.isdir(mat_dir) else [])
                 if f.endswith(".material")]
    # Materials referenced by shared geometry (tyres) sit outside the car.
    for ref in sorted(shared_materials):
        found = _find_shared(ref, roots)
        if found and found not in mat_paths:
            mat_paths.append(found)
        elif not found:
            print(f"  material not found: {ref}")

    tyre_set = find_tyretextures(
        roots, os.environ.get("EVOVIEW_TYRE") or _tyre_family(car_dir, roots))
    tyre_maps = read_tyretextures(tyre_set) if tyre_set else {}
    if tyre_maps:
        print(f"  tyres: {os.path.basename(tyre_set)[:-len('.tyretextures')]}")

    # Livery materials replace the car's own, so they must come last: the
    # viewer keys materials by name and the final entry wins.
    overrides = paintlib.design_overrides(car_dir, os.environ.get("EVOVIEW_DESIGN"))
    for slot_id, path in overrides:
        mat_paths.append(path)
        print(f"  livery {slot_id:12} <- {os.path.basename(path)}")
    override_names = {p: s for s, p in overrides}

    for path in mat_paths:
        f = os.path.basename(path)
        m = read_material(path)
        # Register an override under the SLOT name, which is what the mesh
        # batches actually reference.
        if path in override_names:
            m["file"] = override_names[path] + ".material"
        if (m.get("shader") or "").lower() == "vehicletyre":
            m["slots"].update(tyre_maps)
        resolved = {}
        for slot, ref in m["slots"].items():
            base = os.path.basename(ref.replace("\\", os.sep))
            name = base[:-len(".texture")] if base.endswith(".texture") else base
            # A material's own directory wins: that is what makes a livery
            # override its car's default texture of the same name.
            found = _find_shared(ref, [os.path.dirname(path)] + roots)
            if found:
                idx = add_texture(found, name)
                if idx is not None:
                    resolved[slot] = idx
        m["slots"] = resolved
        manifest["materials"].append(m)

    # ---- paint -----------------------------------------------------------
    # Colours are NOT in the materials: each design slot picks a
    # .oemmultilayercolor holding linear RGB and the finish scalars. Body and
    # wheels are separate slots in separate design files.
    picks = paintlib.choose_all(car_dir, roots, os.environ.get("EVOVIEW_PAINT"))
    if not picks:
        print("  paint: none found (surfaces use their material base colour)")
    for chosen, target in picks:
        if chosen["slot"] == "EXT_SKIN":
            manifest["paint"] = chosen
        tkey = target.lower().replace(".material", "")
        for m in manifest["materials"]:
            if m["file"].lower().replace(".material", "") != tkey:
                continue
            # ⚠ The paint belongs to the RED splat layer, not the base. A body
            # material is layered: red = paint, green = secondary, blue =
            # carbon/trim, base = whatever the splat map leaves black. Writing
            # the colour to Base_Basecolor paints the trim and carbon too, and
            # only looks right because the body dominates the panel area.
            layer = "Red" if "SpatMap" in m["slots"] else "Base"
            m["colors"][f"{layer}_Basecolor"] = chosen["rgb"] + [1.0]
            m["params"][f"{layer}_Metalness"] = chosen["metallic"]
            m["params"][f"{layer}_Roughness"] = chosen["roughness"]
            # Every factory finish is clear-coated, so a paint that does not
            # state a strength still gets the full lobe rather than rendering
            # as bare, chalky base coat.
            m["params"][f"{layer}_ClearCoat"] = chosen["clearcoat"] or 1.0
            m["params"]["ClearCoat"] = chosen["clearcoat"] or 1.0
            m["paint"] = chosen["name"]
        print(f"  paint {chosen['slot']:9} {chosen['name']:34} "
              f"rgb={[round(x, 3) for x in chosen['rgb']]} "
              f"metallic={chosen['metallic']} -> {target}"
              + (f"  ({len(chosen['options'])} options)"
                 if len(chosen["options"]) > 1 else ""))

    print(f"  {own} own textures, {len(manifest['textures']) - own} shared, "
          f"{len(manifest['materials'])} materials")

    js = json.dumps(manifest).encode("utf-8")
    with open(out_path, "wb") as fh:
        fh.write(b"EVOCAR01")
        fh.write(struct.pack("<III", len(manifest["meshes"]),
                             len(manifest["textures"]),
                             len(manifest["materials"])))
        fh.write(struct.pack("<I", len(js)))
        fh.write(js)
        fh.write(blobs)
    print(f"\n{out_path}  {os.path.getsize(out_path)/1e6:.1f} MB")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        raise SystemExit(2)
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
