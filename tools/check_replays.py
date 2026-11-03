"""Sanity check every local replay: does it decode into physically sane cars?

Run with no arguments to sweep `Saved Games\\ACE\\Replay\\Saved`, or pass a
directory. Exits non-zero if anything fails, so it can gate a change to the
reader.

The useful checks are geometric, because a decode that has drifted out of
alignment produces cars that are the wrong SHAPE long before it produces
obviously wild numbers:

  * track width and wheelbase must be plausible for a car
  * up.y must be positive - a wheel order flipped left-for-right inverts it,
    and that is invisible to a direction-of-travel check (see rpy.py)

⚠ Cars really do go upside down: they crash, and mod physics launches them.
A flip is only reported when it lasts, not when it happens.

Known and already investigated: `ACEVO_260629-230448_T_ks_toyota_gr86_mod_mech_1
_Suzuka.rpy` car0 trips both the size and the inversion check. It is NOT a
decode fault - its positions run smooth and continuous frame to frame, which a
misaligned decode cannot produce. The mod's physics throws the car into the air
and collapses its wheels together, and the replay records that faithfully. Every
other car in all 73 local replays passes.
"""
import glob
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rpy import Replay

DEFAULT = os.path.expanduser(r"~\Saved Games\ACE\Replay\Saved")
# F1 cars reach 3.6 m; road cars start around 2.0.
WHEELBASE = (1.8, 3.9)
TRACK_WIDTH = (1.1, 2.4)


def check(path):
    """Yield one string per problem found."""
    name = os.path.basename(path)
    try:
        r = Replay(path)
    except Exception as e:
        yield f"{name}: parse failed: {e!r}"
        return
    for ci, c in enumerate(r.cars):
        if c.n_matrix < 60:
            continue
        flips = bad_shape = wild = checked = 0
        step = max(1, c.n_matrix // 200)
        for f in range(1, c.n_matrix - 1, step):
            pos, w = c.frame(f)
            if not all(abs(v) < 1e6 for v in pos):
                wild += 1
                continue
            if pos == (0.0, 0.0, 0.0):
                continue                      # parked or not spawned yet
            checked += 1
            width = math.dist(w[0], w[1])
            base = math.dist([(w[0][k] + w[1][k]) / 2 for k in range(3)],
                             [(w[2][k] + w[3][k]) / 2 for k in range(3)])
            if not (TRACK_WIDTH[0] < width < TRACK_WIDTH[1]
                    and WHEELBASE[0] < base < WHEELBASE[1]):
                bad_shape += 1
            if c.basis(f)[2][1] < 0.0:
                flips += 1
        if not checked:
            continue
        # ⚠ Thresholds, not first-hit. Isolated frames really are degenerate -
        # a despawning car collapses its wheels together - and failing on one
        # of those in 2600 makes the check useless as a gate. Only a pattern
        # means the decode has drifted.
        if wild:
            yield f"{name}: car{ci} {wild} frame(s) with out-of-range positions"
        if bad_shape > checked * 0.02:
            yield (f"{name}: car{ci} implausible size in "
                   f"{bad_shape}/{checked} samples")
        if flips > checked * 0.25:
            yield (f"{name}: car{ci} inverted in {flips}/{checked} samples "
                   f"- check whether it is really tumbling")


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    files = sorted(glob.glob(os.path.join(root, "*.rpy")))
    if not files:
        print(f"no replays in {root}")
        return 1
    problems = [p for f in files for p in check(f)]
    print(f"{len(files)} replay(s) checked, {len(problems)} problem(s)")
    for p in problems:
        print("  ", p)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
