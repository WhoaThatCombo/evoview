"""Placement audit: every placed track mesh copy (EVOVIEW_PLACE_DUMP) against
the collision ground (EVOVIEW_GROUND_DUMP). A copy whose bounding box floats
above the ground, or sits wholly under it, is a placement suspect; grouped
by mesh, a whole set off by the same amount points at a transform bug.

  python tools/place_audit.py place.tsv ground.bin [float_m] [max_size_m]
"""
import sys, struct, collections
import numpy as np

place, ground = sys.argv[1], sys.argv[2]
FLOAT = float(sys.argv[3]) if len(sys.argv) > 3 else 0.75
MAXSZ = float(sys.argv[4]) if len(sys.argv) > 4 else 40.0
CELL = 4.0

b = open(ground, 'rb').read()
nv, nt = struct.unpack_from('<II', b, 0)
V = np.frombuffer(b, '<f4', nv * 3, 8).reshape(-1, 3)
T = np.frombuffer(b, '<u4', nt * 3, 8 + nv * 12).reshape(-1, 3)
A, B, C = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
lo = np.minimum(np.minimum(A, B), C)
hi = np.maximum(np.maximum(A, B), C)
grid = collections.defaultdict(list)
for i in range(nt):
    x0, z0 = int(lo[i, 0] // CELL), int(lo[i, 2] // CELL)
    x1, z1 = int(hi[i, 0] // CELL), int(hi[i, 2] // CELL)
    if (x1 - x0 + 1) * (z1 - z0 + 1) > 64:
        continue
    for gx in range(x0, x1 + 1):
        for gz in range(z0, z1 + 1):
            grid[(gx, gz)].append(i)

def heights(x, z):
    """every ground height under (x, z)"""
    out = []
    for i in grid.get((int(x // CELL), int(z // CELL)), ()):
        a, b_, c = A[i], B[i], C[i]
        d = (b_[2] - c[2]) * (a[0] - c[0]) + (c[0] - b_[0]) * (a[2] - c[2])
        if abs(d) < 1e-9:
            continue
        l1 = ((b_[2] - c[2]) * (x - c[0]) + (c[0] - b_[0]) * (z - c[2])) / d
        l2 = ((c[2] - a[2]) * (x - c[0]) + (a[0] - c[0]) * (z - c[2])) / d
        l3 = 1 - l1 - l2
        if l1 >= -1e-4 and l2 >= -1e-4 and l3 >= -1e-4:
            out.append(l1 * a[1] + l2 * b_[1] + l3 * c[1])
    return out

rows = []
for line in open(place):
    p = line.rstrip('\n').split('\t')
    name = p[0]
    x0, y0, z0, x1, y1, z1, tx, ty, tz = map(float, p[1:10])
    if max(x1 - x0, z1 - z0) > MAXSZ:
        continue
    cx, cz = (x0 + x1) / 2, (z0 + z1) / 2
    hs = heights(cx, cz)
    if not hs:
        rows.append((name, 'noground', 0.0, cx, y0, cz))
        continue
    # the ground surface nearest the copy's base (bridges: the deck it is on)
    g = min(hs, key=lambda h: abs(h - y0))
    if y0 - g > FLOAT:
        rows.append((name, 'float', y0 - g, cx, y0, cz))
    elif y1 < g - 0.2:
        rows.append((name, 'buried', g - y1, cx, y0, cz))
    else:
        rows.append((name, 'ok', y0 - g, cx, y0, cz))

by = collections.defaultdict(list)
for r in rows:
    by[r[0]].append(r)
stat = collections.Counter(r[1] for r in rows)
print('copies checked', len(rows), dict(stat))
print('\nmeshes with floating / buried copies (copies bad/total, median offset m, example position):')
out = []
for name, rs in by.items():
    bad = [r for r in rs if r[1] in ('float', 'buried')]
    if not bad:
        continue
    med = float(np.median([r[2] for r in bad]))
    out.append((len(bad), len(rs), name, bad[0][1], med, bad[0][3:]))
out.sort(key=lambda o: -o[0])
for n, tot, name, kind, med, pos in out[:60]:
    short = name.replace('\\', '/').split('/')
    print(f'{n:5d}/{tot:<5d} {kind:6s} {med:7.2f}  ({pos[0]:7.1f} {pos[1]:6.1f} {pos[2]:7.1f})  {"/".join(short[-3:])}')
