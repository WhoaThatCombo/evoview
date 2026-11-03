"""Extract the game's own car thumbnails and UI images from content.kspkg as PNG.

The game renders a studio thumbnail per car per (mech preset, visual preset)
into content/cars/<id>/generated/thumbnails/*.texture(+.texturemips),
1024x1024 BC7. Track previews live in uiresources/images/tracks.

    python tools/ref_extract.py <content.kspkg> L:/evoview_work/ref

Reads the package in place (read-only) - nothing else is written to the game.
"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kstex

KEY = np.array([0xc1, 0x35, 0x11, 0x7d, 0xa9, 0x21, 0x97, 0x9f], np.uint8)
TABLE = 64 * 1024 * 1024


def index(pkg):
    """path(lowercase, forward slashes) -> (offset, size, xored)"""
    with open(pkg, "rb") as f:
        f.seek(0, 2); n = f.tell(); f.seek(n - TABLE)
        t = np.frombuffer(f.read(TABLE), np.uint8).copy()
    t ^= np.tile(KEY, TABLE // 8)
    out = {}
    for r in t.reshape(-1, 256):
        if r[0] == 0:
            continue
        kind = int(r[228]) | int(r[229]) << 8
        if kind == 1:
            continue
        b = bytes(r[:228]); end = b.find(b"\0")
        p = b[:end if end >= 0 else 228].decode("utf-8", "replace").replace("\\", "/").lower()
        out[p] = (int.from_bytes(bytes(r[248:256]), "little"),
                  int.from_bytes(bytes(r[240:244]), "little"), kind != 0)
    return out


def read(f, e):
    off, size, xored = e
    f.seek(off)
    d = np.frombuffer(f.read(size), np.uint8).copy()
    if xored:
        d ^= np.resize(KEY, size)
    return d.tobytes()


def main():
    pkg, out = sys.argv[1], sys.argv[2]
    idx = index(pkg)
    jobs = []
    for p in idx:
        if p.endswith(".texture") and (p + "mips") in idx:
            if "/generated/thumbnails/" in p:
                car = p.split("/")[2]
                name = os.path.basename(p)[len(car) + 1:-len(".texture")]
                jobs.append((p, os.path.join(out, car, name + ".png")))
            elif p.startswith("uiresources/images/tracks/"):
                jobs.append((p, os.path.join(out, "_tracks", os.path.basename(p)[:-8] + ".png")))
    with open(pkg, "rb") as f:
        for p, dst in sorted(jobs):
            if os.path.exists(dst):
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            try:
                kstex.decode(read(f, idx[p]), read(f, idx[p + "mips"])).save(dst)
            except Exception as e:  # unknown format etc.
                print("skip", p, e)
    print(len(jobs), "images")


if __name__ == "__main__":
    main()
