"""Extract a whole car folder from content.kspkg into a viewer-ready tree.

    python pull_car.py <content.kspkg> <car_id> <out_dir>

`pull_shared.py` follows references one at a time, which is right for shared
assets but wrong for a car: a car is thousands of entries and every per-file
lookup rescans the 64 MB index. This walks the index ONCE, keeps every entry
under `content\\cars\\<car_id>\\`, then reads each by offset.

⚠ `.texturemips` are stored UNENCRYPTED while everything else is XOR'd with the
engine key. kspkg_extract decrypts unconditionally, so mips come out scrambled
and have to be XOR'd back. See memory acevo-texturemips-not-encrypted.
"""
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "AC_EVO_Car_Conversion_Toolkit", "scripts"))
import kspkg_extract as K            # noqa: E402

KEY = bytes.fromhex("c135117da921979f")


def unxor(d):
    b = bytearray(d)
    for i in range(len(b)):
        b[i] ^= KEY[i % 8]
    return bytes(b)


def pull(kspkg, prefix, out_dir):
    prefix = prefix.lower().rstrip("\\") + "\\"
    wanted = []
    for path, size, offset in K.iter_entries(kspkg):
        if size and path.lower().startswith(prefix):
            wanted.append((path, size, offset))
    print(f"{len(wanted)} entries under {prefix}")
    if not wanted:
        return 0

    # Sequential reads: the entries are scattered, but sorting by offset keeps
    # the 74 GB file streaming forwards instead of seeking back and forth.
    wanted.sort(key=lambda x: x[2])
    total = 0
    with open(kspkg, "rb") as f:
        for path, size, offset in wanted:
            rel = path.split("\\", 3)[-1] if path.count("\\") >= 3 else path
            dst = os.path.join(out_dir, rel.replace("\\", os.sep))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            f.seek(offset)
            data = K.decrypt(f.read(size), 0)
            if path.lower().endswith(".texturemips"):
                data = unxor(data)
            with open(dst, "wb") as o:
                o.write(data)
            total += size
    print(f"{total / 1e6:.0f} MB written to {out_dir}")
    return len(wanted)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(__doc__)
        raise SystemExit(2)
    kspkg, car_id, out_dir = sys.argv[1:4]
    pull(kspkg, f"content\\cars\\{car_id}", out_dir)
