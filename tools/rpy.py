"""Reader for AC EVO `.rpy` replay files.

Layout, as reverse-engineered from Saved Games\\ACE\\Replay\\Saved:

    u32   version            always 16 so far
    f64   duration seconds   at offset 2 (NOT 4 - the double straddles the
                             version field's tail, so the struct is packed
                             oddly; every file checked has its MSB at byte 9)
    ...   fixed preamble, u32 123 at offset 22
    u32   len + bytes        the car model of the RECORDING driver
    N x   [u32 len][protobuf] session metadata chunks (track, entry list,
                             session structure, steam ids). The first chunk
                             has no `123` prefix; later ones do.
    then the car array:
    u32   123
    u16   car count
    per car:
        16B guid, then the SAME guid again
        u32, u8   (purpose unknown, both 0 in every file seen)
        u32   nMatrixFrames
        nMatrixFrames x 104-byte MATRIX FRAME (see below)
        u32   nBaseFrames
        variable-length base-frame block (telemetry; encoding not decoded)

Matrix frame, 104 bytes, all little-endian f32 unless noted:

      0  vec3   body position, world space, Y-up (matches evoview's axes)
     12  8B     packed rotation - NOT a plain 4xi16 quaternion; decode
                unknown, and unnecessary (see below)
     20  vec3   wheel 0 position   front-RIGHT
     32  vec3   wheel 1 position   front-left
     44  vec3   wheel 2 position   rear-RIGHT
     56  vec3   wheel 3 position   rear-left
     68  vec3   ?  (near-zero x/z, y ~3.2)
     80  vec3   ?  (same shape)
     92  f32    -1..1
     96  4B     not a float (flags?)
    100  f32    0.9..3.0

The wheels are stored in WORLD space, so body orientation is recoverable
from them directly and the packed rotation at +12 can be ignored:

    forward = normalize(mid(w0,w1) - mid(w2,w3))
    right   = normalize(w0 - w1)     # w0 is the RIGHT wheel, not the left
    up      = normalize(cross(forward, right))   # re-orthogonalise

Verified against direction of travel across many frames: dot(forward,
velocity) lands between 0.98 and 1.000. That also gives real body roll and
pitch for free, which a stored quaternion would have given no better.
"""
import re
import struct

FRAME = 104
_HDR_TAGS = (0x08, 0x12, 0x1a, 0x22, 0x2a, 0x32)

# Car models are not named anywhere in the header - not in the session
# metadata and not in the entry list. They live in the customization blocks
# AFTER the car array, laid out as:
#     field 100 (tag a2 06), length 36 : the guid as a dashed string
#     field 101 (tag aa 06), length 21 : the same guid in binary
#     field  20 (tag a2 01)            : customization, holding paths like
#         content\cars\ks_ford_mustang_gt3\skins\rimdesign_mstggt3_1.design
# so the model is the first `content\cars\<model>\` path inside the block.
#
# Requiring field 101 to follow field 100 is what makes this safe: a bare
# guid string turns up elsewhere in the tail, and matching those produced
# wrong models. Anchored this way, 965 of 967 cars across all 73 local
# replays resolve.
_CAR_PATH = re.compile(rb"content\\cars\\([a-z0-9_]+)\\")
_GUID_FIELD = b"\xa2\x06\x24"          # field 100, wiretype 2, len 36
_PGUID_TAG = b"\xaa\x06"               # field 101, wiretype 2
_LOOKAHEAD = 4096


class Car:
    __slots__ = ("guid", "n_matrix", "n_base", "model", "_buf", "_off")

    def __init__(self, guid, n_matrix, n_base, buf, off):
        self.guid = guid
        self.n_matrix = n_matrix
        self.n_base = n_base
        self.model = None          # filled in by Replay._resolve_models
        self._buf = buf
        self._off = off

    @property
    def dashed(self):
        """The guid as the game writes it in the customization blocks."""
        a = struct.unpack("<Q", bytes.fromhex(self.guid)[:8])[0]
        b = struct.unpack("<Q", bytes.fromhex(self.guid)[8:])[0]
        s = "%016x%016x" % (a, b)
        return "%s-%s-%s-%s-%s" % (s[:8], s[8:12], s[12:16], s[16:20], s[20:])

    def frame(self, i):
        """(position, [4 wheel positions]) for matrix frame `i`."""
        o = self._off + i * FRAME
        pos = struct.unpack_from("<3f", self._buf, o)
        wheels = [struct.unpack_from("<3f", self._buf, o + 20 + 12 * k)
                  for k in range(4)]
        return pos, wheels

    def basis(self, i):
        """(forward, right, up) unit vectors, derived from the wheels."""
        _, w = self.frame(i)
        fm = [(w[0][k] + w[1][k]) * 0.5 for k in range(3)]
        rm = [(w[2][k] + w[3][k]) * 0.5 for k in range(3)]
        fwd = _norm([fm[k] - rm[k] for k in range(3)])
        # w0 is the RIGHT wheel of its axle; reversing this inverts up.
        right = _norm([w[0][k] - w[1][k] for k in range(3)])
        up = _norm(_cross(fwd, right))
        # right may not be exactly perpendicular to fwd once the car is
        # rolling, so rebuild it from the other two.
        right = _cross(up, fwd)
        return fwd, right, up

    def positions(self):
        return [self.frame(i)[0] for i in range(self.n_matrix)]


def _norm(v):
    n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) ** 0.5
    return [c / n for c in v] if n else [0.0, 0.0, 0.0]


def _cross(a, b):
    return [a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]]


class Replay:
    def __init__(self, path):
        with open(path, "rb") as fh:
            self.data = d = fh.read()
        self.version = struct.unpack_from("<I", d, 0)[0]
        self.duration = struct.unpack_from("<d", d, 2)[0]
        n = struct.unpack_from("<I", d, 26)[0]
        self.model = d[30:30 + n].decode("utf8", "replace")
        self.meta = []
        o = self._read_meta(30 + n)
        self.cars = self._read_cars(o)
        self._resolve_models()

    def _resolve_models(self):
        """Attach a car model to each car from the trailing customization data.

        Leaves `model` as None rather than guessing when the block does not
        validate - a wrong car is worse than a missing one. `Replay.model`
        (the recording driver's car, straight from the header) is always
        available as a fallback.
        """
        if not self.cars:
            return
        last = self.cars[-1]
        tail = self.data[last._off + last.n_matrix * FRAME:]
        for car in self.cars:
            i = tail.find(_GUID_FIELD + car.dashed.encode())
            if i < 0:
                continue
            j = i + len(_GUID_FIELD) + 36
            if tail[j:j + 2] != _PGUID_TAG:
                continue                   # chance string hit, not a record
            m = _CAR_PATH.search(tail, j, j + _LOOKAHEAD)
            if m:
                car.model = m.group(1).decode()

    def _read_meta(self, o):
        d = self.data
        while o + 8 <= len(d):
            v = struct.unpack_from("<I", d, o)[0]
            if v == 123:
                L = struct.unpack_from("<I", d, o + 4)[0]
                body = o + 8
                # The car array also starts with 123; tell them apart by
                # whether a protobuf tag follows.
                if body + L > len(d) or d[body] not in _HDR_TAGS:
                    return o
            else:
                L, body = v, o + 4
                if body + L > len(d):
                    return o
            self.meta.append(d[body:body + L])
            o = body + L
        return o

    def _read_cars(self, o):
        d = self.data
        count = struct.unpack_from("<H", d, o + 4)[0]
        q = o + 6
        cars = []
        for _ in range(count):
            if d[q:q + 16] != d[q + 16:q + 32]:
                break                      # lost sync; stop rather than guess
            guid = d[q:q + 16].hex()
            q += 32 + 5
            nm = struct.unpack_from("<I", d, q)[0]
            q += 4
            cars.append(Car(guid, nm, 0, d, q))
            q += nm * FRAME
            nb = struct.unpack_from("<I", d, q)[0]
            q += 4
            cars[-1].n_base = nb
            # Base frames are variable-length and not decoded, so skip to the
            # next car by finding its doubled guid rather than by arithmetic.
            q = _next_car(d, q)
            if q is None:
                break
        return cars


def _next_car(d, x):
    """Find the next car header by its doubled guid.

    Base frames are variable-length, so this is a scan rather than a seek.
    A doubled 16-byte run happens by chance inside base-frame data often
    enough to matter (it derailed 3 of 73 replays), so a candidate only
    counts if the frame count and first position behind it are sane too.
    """
    end = len(d) - 32
    while x < end:
        if d[x:x + 16] == d[x + 16:x + 32] and d[x:x + 16] != b"\0" * 16:
            if _plausible(d, x):
                return x
        x += 1
    return None


def _plausible(d, x):
    q = x + 32 + 5
    if q + 4 > len(d):
        return False
    n = struct.unpack_from("<I", d, q)[0]
    if not 0 < n < 5_000_000 or q + 4 + n * FRAME > len(d):
        return False
    if n * FRAME < 12:
        return False
    for k in (0, n // 2, n - 1):                 # spot-check the stream
        px, py, pz = struct.unpack_from("<3f", d, q + 4 + k * FRAME)
        for c in (px, py, pz):
            if c != c or abs(c) > 1e6:           # NaN or absurd
                return False
    return True


if __name__ == "__main__":
    import sys
    r = Replay(sys.argv[1])
    print(f"v{r.version} {r.duration:.1f}s {r.model} - {len(r.cars)} cars")
    for i, c in enumerate(r.cars):
        p, _ = c.frame(0)
        f, _, _ = c.basis(0)
        print(f"  car{i:2d} {c.model or '?':32s} {c.n_matrix:5d} frames "
              f"start=({p[0]:8.1f},{p[1]:7.1f},{p[2]:8.1f}) "
              f"fwd=({f[0]:5.2f},{f[1]:5.2f},{f[2]:5.2f})")
