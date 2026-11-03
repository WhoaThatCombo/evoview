"""Car paint: which colour a design offers, and what that colour actually is.

The chain is three files deep, and none of it lives in the material:

    skins/<name>.design           per-slot options; the EXT_SKIN slot lists
                                  every paint the car can wear
      -> *.oemmultilayercolor     the paint itself: linear RGB + a few scalars
      -> materials/EXT_SKIN.material   which the design names as the target

⚠ The paint files live under `content/cars/common_assets/customization/...`,
which is NOT inside a car folder - they usually have to come out of the game's
content.kspkg. A car with no paints found still exports; it just renders in the
material's own base colour.

Scalar meanings, inferred from which paints carry them: f4 appears only on
metallic and pearl finishes and never on solid colours, so it is the flake /
metallic amount. f5 tracks roughness (black 0.2, silver 0.45), f6 looks like
clear-coat strength, f8 only accompanies f4.
"""
import os
import struct


def _varint(d, i):
    r = s = 0
    while True:
        b = d[i]
        i += 1
        r |= (b & 0x7F) << s
        if not b & 0x80:
            return r, i
        s += 7


def _fields(d, i, end):
    end = min(end, len(d))
    while i < end:
        try:
            key, i = _varint(d, i)
        except IndexError:
            return
        fn, wt = key >> 3, key & 7
        if wt == 0:
            try:
                v, i = _varint(d, i)
            except IndexError:
                return
        elif wt == 2:
            try:
                ln, i = _varint(d, i)
            except IndexError:
                return
            if i + ln > end:
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
            v = 0.0
            i += 8
        else:
            return
        yield fn, wt, v


def _text(d, r):
    try:
        t = d[r[0]:r[1]].decode("utf-8")
        return t if t.isprintable() else None
    except UnicodeDecodeError:
        return None


def read_paint(path):
    """A .oemmultilayercolor: linear rgb plus finish scalars."""
    d = open(path, "rb").read()
    out = {"file": os.path.basename(path), "name": None,
           "rgb": None, "metallic": 0.0, "roughness": 0.45, "clearcoat": 0.0}
    for fn, wt, v in _fields(d, 0, len(d)):
        if fn == 2 and wt == 2:
            out["name"] = _text(d, v)
        elif fn == 3 and wt == 2:
            c = [x for f, w, x in _fields(d, v[0], v[1]) if w == 5]
            if len(c) >= 3:
                out["rgb"] = [round(x, 5) for x in c[:3]]
        elif wt == 5:
            if fn == 4:
                out["metallic"] = round(v, 3)
            elif fn == 5:
                out["roughness"] = round(v, 3)
            elif fn == 6:
                out["clearcoat"] = round(v, 3)
    return out if out["rgb"] else None


def read_design(path):
    """A .design: slot -> {material, paint options}."""
    d = open(path, "rb").read()
    slots = []
    for fn, wt, v in _fields(d, 0, len(d)):
        if fn != 3 or wt != 2:
            continue
        slot = {"id": None, "label": None, "material": None, "paints": []}
        for sf, sw, sv in _fields(d, v[0], v[1]):
            if sw != 2:
                continue
            t = _text(d, sv)
            if sf == 1:
                slot["id"] = t
            elif sf == 2:
                slot["label"] = t
            elif sf == 3:
                slot["material"] = t
            elif sf == 6:
                for of, ow, ov in _fields(d, sv[0], sv[1]):
                    if of == 4 and ow == 2:
                        p = _text(d, ov)
                        if p and p.endswith(".oemmultilayercolor"):
                            slot["paints"].append(p)
        if slot["id"]:
            slots.append(slot)
    return slots


def find_paint_file(ref, roots):
    """Locate a paint by its engine path, or by name anywhere under roots."""
    base = os.path.basename(ref.replace("\\", os.sep))
    for root in roots:
        direct = os.path.join(root, ref.replace("\\", os.sep))
        if os.path.isfile(direct):
            return direct
        for dirpath, _, files in os.walk(root):
            for f in files:
                if f.lower() == base.lower():
                    return os.path.join(dirpath, f)
    return None


def design_overrides(car_dir, wanted=None):
    """Livery materials a design substitutes for the car's own.

    A design slot's `material` field is "the material this slot uses". For a
    road car that is the car's own material, which a .oemmultilayercolor then
    recolours. For a liveried car - F1, GT3 - it instead points at a complete
    replacement material under `skins/<livery>/`, carrying the livery textures.
    Miss this and the car renders in its base material's colour: the F2004
    comes out black instead of Ferrari red.

    Returns [(slot_id, material_path)] for the substituting case only.
    """
    out = []
    skins = os.path.join(car_dir, "skins")
    if not os.path.isdir(skins):
        return out
    designs = sorted(f for f in os.listdir(skins) if f.endswith(".design"))
    if wanted:
        designs = [f for f in designs if wanted.lower() in f.lower()] or designs
    for f in designs[:1]:                      # one livery at a time
        for slot in read_design(os.path.join(skins, f)):
            ref = (slot.get("material") or "").replace("\\", os.sep)
            if not ref or os.sep + "skins" + os.sep not in ref.lower():
                continue                       # the car's own material: not an override
            found = find_paint_file(ref, [car_dir]) or _under(skins, os.path.basename(ref))
            if found:
                out.append((slot["id"], found))
    return out


def _under(root, base):
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower() == base.lower():
                return os.path.join(dirpath, f)
    return None


def choose_all(car_dir, roots, wanted=None):
    """Every painted slot the car's designs define, not just the body.

    ⚠ Returning on the first hit is wrong: the body lives in
    `design_*.design` and the wheels in `rimdesign_*.design`, so stopping at
    the first paint leaves the rims rendering in whatever colour their
    material happened to carry - black, for this car, instead of silver.

    Yields (paint dict, target material basename).
    """
    out = []
    skins = os.path.join(car_dir, "skins")
    if not os.path.isdir(skins):
        return out
    for f in sorted(os.listdir(skins)):
        if not f.endswith(".design"):
            continue
        for slot in read_design(os.path.join(skins, f)):
            if not slot["paints"]:
                continue
            refs = slot["paints"]
            pick = refs[0]
            # only the body slot honours the requested colour; a rim paint
            # name would never match it anyway
            if wanted and slot["id"] == "EXT_SKIN":
                for r in refs:
                    if wanted.lower() in os.path.basename(r).lower():
                        pick = r
                        break
            found = find_paint_file(pick, roots)
            if not found:
                continue
            p = read_paint(found)
            if not p:
                continue
            p["slot"] = slot["id"]
            p["options"] = [os.path.basename(r)[:-len(".oemmultilayercolor")]
                            for r in refs]
            target = os.path.basename(
                (slot["material"] or "").replace("\\", os.sep))
            if target:
                out.append((p, target))
    return out


def choose(car_dir, roots, wanted=None):
    """Pick a paint for this car: the named one, else the first on offer.

    Returns (paint dict, target material basename) or (None, None).
    """
    skins = os.path.join(car_dir, "skins")
    if not os.path.isdir(skins):
        return None, None
    for f in sorted(os.listdir(skins)):
        if not f.endswith(".design"):
            continue
        for slot in read_design(os.path.join(skins, f)):
            if not slot["paints"]:
                continue
            refs = slot["paints"]
            pick = refs[0]
            if wanted:
                for r in refs:
                    if wanted.lower() in os.path.basename(r).lower():
                        pick = r
                        break
            found = find_paint_file(pick, roots)
            if not found:
                continue
            paint = read_paint(found)
            if not paint:
                continue
            target = os.path.basename(
                (slot["material"] or "").replace("\\", os.sep))
            paint["options"] = [os.path.basename(r)[:-len(".oemmultilayercolor")]
                                for r in refs]
            return paint, target
    return None, None
