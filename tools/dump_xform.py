"""Inspect ISMesh transform fields and which instance paths overlap static meshes."""
import os, struct, collections
from dump_track import fields, text

def inspect_105(d, hs, he, out):
    keys = collections.Counter()
    path = ""
    inner = {}
    for kf, kwt, kv in fields(d, hs, he):
        keys[kf] += 1
        if kf == 1 and kwt == 2:
            path = text(d, kv)
        if kf == 7 and kwt == 2:
            for tf, twt, tv in fields(d, kv[0], kv[1]):
                inner[tf] = (twt, tv if twt != 2 else (tv[1]-tv[0]))
    out.append((path, dict(keys), inner))

def walk(path):
    d = open(path, "rb").read()
    hits = []
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
                inspect_105(d, sv[0], sv[1], hits)
    return hits

root = r"C:\Users\joshu\Saved Games\ACE\mods\content\tracks"
for track in ["highlands_kp", "barber_mo_ra", "miami_int_dv"]:
    tdir = os.path.join(root, track)
    print(f"\n==== {track} scenes ====")
    for dirpath, _, files in os.walk(tdir):
        for fn in files:
            if not fn.endswith(".scene"):
                continue
            p = os.path.join(dirpath, fn)
            hits = walk(p)
            if not hits:
                continue
            print(f"  {os.path.relpath(p, tdir)}: {len(hits)} ISMesh")
            for path, keys, inner in hits[:2]:
                print(f"    path={path}")
                print(f"    f105 keys={keys}")
                print(f"    f7 children={inner}")

    # instance path prefixes
    paths = collections.Counter()
    for dirpath, _, files in os.walk(tdir):
        for fn in files:
            if fn.endswith(".scene"):
                for path, _, _ in walk(os.path.join(dirpath, fn)):
                    if path.lower().endswith(".mesh"):
                        parts = path.replace("\\", "/").lower().split("/")
                        if "meshes" in parts:
                            i = parts.index("meshes")
                            paths["/".join(parts[i:i+3])] += 1
                        else:
                            paths["/".join(parts[-3:])] += 1
    print("  path groups:", paths.most_common(8))
