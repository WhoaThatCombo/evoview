"""Pull shared car assets out of content.kspkg into evoview/assets.

A car folder is not self-contained. Tyres, paints and some textures live under
`content/cars/common_assets/...`, and without them the car renders on bare rims
in whatever colour its material happened to carry. This copies the pieces the
viewer needs, following each reference one hop at a time:

    <name>.tyremesh  ->  <name>.mesh  ->  *.material  ->  *.texture(+mips)

Run once per car; the exporter then finds everything under `assets/`.

    python pull_shared.py <content.kspkg> <asset_root> <engine_path> [...]
"""
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXTRACT = os.path.join(
    os.path.dirname(HERE), "..", "AC_EVO_Car_Conversion_Toolkit",
    "scripts", "kspkg_extract.py")

# Only follow references to things the viewer can actually consume. Chasing
# every string in a blob pulls in physics and audio for no benefit.
FOLLOW = (".mesh", ".material", ".texture", ".texturemips",
          ".oemmultilayercolor")

REF = re.compile(rb"content[\\/][ -~]{4,240}?\.[a-z]{3,14}")


def get(kspkg, engine_path, root):
    """Extract one entry, preserving its engine-relative path under root."""
    out = os.path.join(root, engine_path.replace("\\", os.sep))
    if os.path.exists(out):
        return out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    # ⚠ Lookup is by exact path, so case matters. References inside meshes are
    # written in whatever case the artist typed (EXT_TYRE.material) while the
    # package stores ext_tyre.material - try the lowercase form too.
    r = None
    for cand in (engine_path, engine_path.lower()):
        r = subprocess.run(
            [sys.executable, EXTRACT, "get", kspkg, cand, "--out=" + out],
            capture_output=True, text=True)
        if os.path.exists(out) and os.path.getsize(out) > 0:
            break
    # ⚠ .texturemips are the one entry type stored UNENCRYPTED - the engine
    # streams them straight off disk. kspkg_extract XORs everything, so its
    # output for a mips file is plaintext run through the key once, i.e.
    # scrambled. Undo it. (Symptom: tyres and other shared textures decode to
    # rainbow confetti while every mesh and material is perfect.)
    if os.path.exists(out) and engine_path.lower().endswith(".texturemips"):
        d = bytearray(open(out, "rb").read())
        key = bytes.fromhex("c135117da921979f")
        for i in range(len(d)):
            d[i] ^= key[i % 8]
        open(out, "wb").write(bytes(d))
    if not os.path.exists(out) or os.path.getsize(out) == 0:
        # Missing is normal - a reference may point at content this build does
        # not ship. Say so once and carry on rather than aborting the pull.
        print(f"  MISS {engine_path}  {r.stderr.strip()[:60]}")
        if os.path.exists(out):
            os.remove(out)
        return None
    print(f"  ok   {engine_path}  {os.path.getsize(out)} B")
    return out


def refs(path):
    d = open(path, "rb").read()
    out = []
    for m in REF.findall(d):
        t = m.decode("ascii", "ignore").replace("/", "\\")
        if t.lower().endswith(FOLLOW):
            out.append(t)
    return out


def pull(kspkg, root, seeds, depth=4):
    seen, queue = set(), list(seeds)
    while queue and depth > 0:
        nxt = []
        for p in queue:
            key = p.lower()
            if key in seen:
                continue
            seen.add(key)
            f = get(kspkg, p, root)
            if not f:
                continue
            # A .texture is useless without its mip payload, and the mips are a
            # sibling file rather than a reference inside the header.
            if p.lower().endswith(".texture"):
                nxt.append(p + "mips")
            nxt.extend(refs(f))
        queue = nxt
        depth -= 1
    return seen


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(__doc__)
        raise SystemExit(2)
    n = pull(sys.argv[1], sys.argv[2], sys.argv[3:])
    print(f"\n{len(n)} reference(s) walked into {sys.argv[2]}")
