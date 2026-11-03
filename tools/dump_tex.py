"""Print texture header fields for grass vs known-good maps."""
import os, struct

def varint(d, i):
    r = s = 0
    while i < len(d):
        b = d[i]; i += 1
        r |= (b & 0x7F) << s
        if b & 0x80 == 0:
            return r, i
        s += 7
    return None, i

def fields(d):
    i = 0
    while i < len(d):
        key, i = varint(d, i)
        if key is None: return
        f, wt = key >> 3, key & 7
        if wt == 0:
            v, i = varint(d, i)
            yield f, v
        elif wt == 2:
            n, i = varint(d, i)
            if n is None: return
            i += n
        elif wt == 5:
            i += 4
        elif wt == 1:
            i += 8
        else:
            return

root = r"C:\Users\joshu\Saved Games\ACE\mods\content\tracks\highlands_kp\textures"
names = [
    "grass_far", "grass_D", "Grass19", "grass15", "mount_D",
    "asphalt_base4", "grass_BB3_D", "lake", "trees4",
]
fmt_name = {12:"bc1", 13:"bc1", 26:"bc6h", 27:"bc1", 33:"bc7", 34:"bc7", 35:"bc7", 2:"rgba8"}
for n in names:
    p = os.path.join(root, n + ".texture")
    m = os.path.join(root, n + ".texturemips")
    if not os.path.isfile(p):
        print(n, "MISSING")
        continue
    d = open(p, "rb").read()
    vals = {}
    for f, v in fields(d):
        vals[f] = v
    ms = os.path.getsize(m) if os.path.isfile(m) else 0
    kind = fmt_name.get(vals.get(4), f"fmt{vals.get(4)}")
    print(f"{n:16} {vals.get(1)}x{vals.get(2)} mips={vals.get(3)} {kind} hdr={len(d)} mipsfile={ms/1e6:.2f}MB")
