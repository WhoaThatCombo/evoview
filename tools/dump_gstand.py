"""Dump Spa grandstands_*.scene instance paths and eulers."""
import os, struct, sys
sys.path.insert(0, r"C:\Users\joshu\Downloads\acecm")
from acecm import kspkg
src = open(os.path.join(os.path.dirname(__file__), "dump_track.py"), encoding="utf-8").read()
ns = {}
exec(src[:src.find("\nroot = ")], ns)
fields, text = ns["fields"], ns["text"]

pkg = r"C:\Program Files (x86)\Steam\steamapps\common\Assetto Corsa EVO\content.kspkg"
want = []
for path, size, off in kspkg.iter_entries(pkg):
    low = path.lower().replace("/", "\\")
    if "\\spa\\" in low and low.endswith(".scene") and "grand" in low:
        want.append((path, size, off))
        print("FOUND", path, size)

with open(pkg, "rb") as f:
    for path, size, off in want:
        d = kspkg.read_entry(f, size, off, path)
        print(f"\n==== {path}")
        for fnum, wt, v in fields(d, 0, len(d)):
            if fnum != 2 or wt != 2:
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
                pth = ""; pos_b = rot_b = scl_b = None
                for kf, kwt, kv in fields(d, sv[0], sv[1]):
                    if kf == 1 and kwt == 2:
                        pth = text(d, kv)
                    if kf == 7 and kwt == 2:
                        for tf, twt, tv in fields(d, kv[0], kv[1]):
                            if twt == 2:
                                if tf == 2: pos_b = tv
                                if tf == 3: rot_b = tv
                                if tf == 4: scl_b = tv
                if not pth.lower().endswith(".mesh"):
                    continue
                n = (pos_b[1]-pos_b[0])//12 if pos_b else 0
                print(f"  {n:4} {pth}")
                for i in range(min(4, n)):
                    p = struct.unpack_from("<fff", d, pos_b[0]+i*12) if pos_b else (0,0,0)
                    e = struct.unpack_from("<fff", d, rot_b[0]+i*12) if rot_b else (0,0,0)
                    s = struct.unpack_from("<fff", d, scl_b[0]+i*12) if scl_b else (1,1,1)
                    print(f"      pos={[round(x,1) for x in p]} euler={[round(x,1) for x in e]} scl={[round(x,3) for x in s]}")
