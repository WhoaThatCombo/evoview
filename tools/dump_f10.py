"""Dump ISMesh field 10 and Barber helper mesh names."""
import os, struct, collections
import importlib.util
spec = importlib.util.spec_from_file_location(
    "dt", r"C:\Users\joshu\Downloads\evoview\tools\dump_track.py")
# avoid running dump_track main by exec of functions only
ns = {}
src = open(r"C:\Users\joshu\Downloads\evoview\tools\dump_track.py", encoding="utf-8").read()
# cut at the first top-level print of MATERIALS
cut = src.find('\nroot = ')
exec(src[:cut], ns)
fields, text = ns["fields"], ns["text"]

def f10_stats(path):
    d = open(path, "rb").read()
    samples = []
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
            if sf != 105 or swt != 2:
                continue
            for kf, kwt, kv in fields(d, sv[0], sv[1]):
                if kf == 10:
                    if kwt == 2:
                        a, b = kv
                        chunk = d[a:b]
                        samples.append(("bytes", len(chunk), chunk[:32].hex()))
                        # try floats
                        if len(chunk) >= 16 and len(chunk) % 4 == 0:
                            fl = struct.unpack("<" + "f"*(min(8, len(chunk)//4)), chunk[:min(32,len(chunk))])
                            samples.append(("f32", fl))
                    elif kwt == 0:
                        samples.append(("varint", kv))
                    elif kwt == 5:
                        samples.append(("f32s", kv))
    return samples

p = r"C:\Users\joshu\Saved Games\ACE\mods\content\tracks\highlands_kp\highlands_kp.scene"
print("highlands f10", f10_stats(p)[:8])
p = r"C:\Users\joshu\Saved Games\ACE\mods\content\tracks\barber_mo_ra\barber_mo_ra.scene"
print("barber f10", f10_stats(p)[:8])

print("\nbarber static helpers:")
md = r"C:\Users\joshu\Saved Games\ACE\mods\content\tracks\barber_mo_ra\static_mesh"
for fn in sorted(os.listdir(md)):
    low = fn.lower()
    if any(x in low for x in ("dummy", "shadow", "lowres", "outer", "phys", "decal", "null")):
        print(" ", fn, os.path.getsize(os.path.join(md, fn)))
