"""Decode an AC EVO .texture header + .texturemips payload (mip 0) to a PIL image.

Same layout rules as load.rs read_texture_lod: 64 KB pages, one tile per
page, tile size per format, tiles row-major. Needs texture2ddecoder for BC.
"""
import numpy as np
from PIL import Image

PAGE = 65536
# format enum -> (kind, bits per pixel, tile w, tile h)
FORMATS = {
    2: ("rgba8", 32, 128, 128), 3: ("rgba8", 32, 128, 128), 5: ("rgba8", 32, 128, 128),
    6: ("rgba8", 32, 128, 128), 10: ("rgba8", 32, 128, 128),
    12: ("bc1", 4, 512, 256), 13: ("bc1", 4, 512, 256), 27: ("bc1", 4, 512, 256),
    26: ("bc6h", 8, 256, 256),
    33: ("bc7", 8, 256, 256), 34: ("bc7", 8, 256, 256), 35: ("bc7", 8, 256, 256),
}


def _varint(d, i):
    r = s = 0
    while True:
        b = d[i]; i += 1
        r |= (b & 0x7F) << s
        if not b & 0x80:
            return r, i
        s += 7


def header(hdr):
    """Top-level varint fields of the protobuf header: {field: value}."""
    out, i = {}, 0
    while i < len(hdr):
        key, i = _varint(hdr, i)
        f, wt = key >> 3, key & 7
        if wt == 0:
            v, i = _varint(hdr, i); out.setdefault(f, v)
        elif wt == 2:
            n, i = _varint(hdr, i); i += n
        elif wt == 5:
            i += 4
        elif wt == 1:
            i += 8
        else:
            break
    return out


def _row_bytes(kind, w, bpp):
    if kind == "bc1":
        return max((w + 3) // 4, 1) * 8
    if kind in ("bc7", "bc6h"):
        return max((w + 3) // 4, 1) * 16
    return w * bpp // 8


def decode(hdr: bytes, raw: bytes) -> Image.Image:
    h = header(hdr)
    w, ht, fmt = h[1], h[2], h[4]
    kind, bpp, tw, th = FORMATS[fmt]
    block = 4 if kind.startswith("bc") else 1
    cols, rows = max(-(-w // tw), 1), max(-(-ht // th), 1)
    rb, trb = _row_bytes(kind, w, bpp), _row_bytes(kind, tw, bpp)
    nrows = -(-ht // block)
    level = bytearray(rb * nrows)
    for r in range(rows):
        for c in range(cols):
            page = (r * cols + c) * PAGE
            keep_w = _row_bytes(kind, min(tw, w - c * tw), bpp)
            keep_h = max(-(-min(th, ht - r * th) // block), 1)
            for y in range(keep_h):
                s = page + y * trb
                d = (r * (th // block) + y) * rb + c * trb
                level[d:d + keep_w] = raw[s:s + keep_w]
    data = bytes(level)
    if kind == "rgba8":
        # formats 5 / 6 are B8G8R8A8 (load.rs read_texture_lod)
        return Image.frombuffer("RGBA", (w, ht), data, "raw", "BGRA" if fmt in (5, 6) else "RGBA", 0, 1)
    import texture2ddecoder as t2d
    fn = {"bc1": t2d.decode_bc1, "bc7": t2d.decode_bc7, "bc6h": t2d.decode_bc6}[kind]
    out = fn(data, w, ht)
    return Image.frombuffer("RGBA", (w, ht), out, "raw", "BGRA", 0, 1)


def decode_files(texture_path, mips_path) -> Image.Image:
    with open(texture_path, "rb") as f:
        hdr = f.read()
    with open(mips_path, "rb") as f:
        raw = f.read()
    return decode(hdr, raw)
