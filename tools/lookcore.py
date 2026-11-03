"""Shared pieces of the look-comparison tools (ref_fit.py, lookcmp.py).

* rendering evoview headless at an absolute camera (`--cam`)
* camera parameterisation used by tools/ref_cameras.json
* separating a game thumbnail's car from its studio background
"""
import json, math, os, subprocess, tempfile
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
CAMS = os.path.join(HERE, "ref_cameras.json")
GAME_PKG = r"C:\Program Files (x86)\Steam\steamapps\common\Assetto Corsa EVO\content.kspkg"
REF_ROOT = r"L:\evoview_work\ref"


# ------------------------------------------------------------------ camera --

def cam_from_params(p):
    """Orbit parameters (as the game's .thumbnailsettings stores them) -> camera.

    p = {target:[x,y,z], distance, azimuth (deg, from +Z toward +X),
         elevation (deg above horizontal), fov (vertical deg)}
    """
    az, el = math.radians(p["azimuth"]), math.radians(p["elevation"])
    t, d = p["target"], p["distance"]
    eye = [t[0] + d * math.cos(el) * math.sin(az),
           t[1] + d * math.sin(el),
           t[2] + d * math.cos(el) * math.cos(az)]
    return {"eye": eye, "target": list(t), "fov": p["fov"]}


def load_cams(path=CAMS):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------------ render --

def render(exe, car, cam, size=(512, 512), visual=None, alpha=False, pkg=GAME_PKG,
           out=None, extra_env=None, timeout=300):
    """Render `car` from the package with evoview; returns an RGBA uint8 array.

    alpha=True: transparent background and no floor, so A is the silhouette.
    """
    env = dict(os.environ)
    # The game's thumbnails are shot with an EMPTY cockpit (checked on the
    # SF-25, the most open car), so the seated driver stays out of the
    # comparison. extra_env can still turn it back on.
    env["EVOVIEW_DRIVER"] = "0"
    if visual:
        env["EVOVIEW_VISUAL"] = visual
    if alpha:
        env["EVOVIEW_SHOT_ALPHA"] = "1"
        env["EVOVIEW_NO_GROUND"] = "1"
    if extra_env:
        env.update(extra_env)
    tmp = None
    if out is None:
        fd, tmp = tempfile.mkstemp(suffix=".png", prefix="lookcmp_")
        os.close(fd)
        out = tmp
    cmd = [exe, pkg, "--car", car, "--shot", out, "--size", f"{size[0]}x{size[1]}",
           "--cam", json.dumps(cam)]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout,
                       cwd=os.path.dirname(os.path.abspath(exe)))
    if r.returncode != 0 or not os.path.exists(out) or os.path.getsize(out) == 0:
        raise RuntimeError(f"evoview failed ({r.returncode}): {r.stderr[-800:] or r.stdout[-800:]}")
    a = np.asarray(Image.open(out).convert("RGBA")).copy()
    if tmp:
        os.remove(tmp)
    return a


# --------------------------------------------------------- reference masks --

def luminance(rgb):
    rgb = rgb.astype(np.float32)
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def background_model(rgb, iters=4):
    """Smooth studio backdrop: a robust 2D cubic fit to the luminance."""
    h, w = rgb.shape[:2]
    lum = luminance(rgb)
    ys, xs = np.mgrid[0:h, 0:w]
    u, v = xs.ravel() / w - 0.5, ys.ravel() / h - 0.5
    terms = [np.ones_like(u)]
    for i in range(4):
        for j in range(4 - i):
            if i + j:
                terms.append(u ** i * v ** j)
    A = np.stack(terms, 1)
    y = lum.ravel()
    keep = np.ones_like(y, bool)
    # the backdrop sits between ~25 and ~70 in every thumbnail
    keep &= (y > 20) & (y < 80)
    for _ in range(iters):
        coef, *_ = np.linalg.lstsq(A[keep], y[keep], rcond=None)
        fit = A @ coef
        keep = np.abs(y - fit) < 4.0
    return fit.reshape(h, w)


def ref_masks(rgb):
    """(car, ignore) boolean masks for a game thumbnail.

    The thumbnail is opaque: car on a smooth dark backdrop with a contact
    shadow that goes to pure black. Black tyres/trim and the shadow cannot be
    told apart by value, so anything DARKER than the backdrop is 'ignore'
    (neither car nor background) and scores nothing either way. Brighter or
    coloured pixels are car.
    """
    bg = background_model(rgb)
    lum = luminance(rgb)
    rgbf = rgb.astype(np.float32)
    sat = rgbf.max(-1) - rgbf.min(-1)
    diff = lum - bg
    car = (diff > 8) | (sat > 14)
    dark = diff < -8
    ignore = dark & ~car
    # close tiny holes / specks so a reflection never reads as background
    car = _open_close(car)
    return car, ignore & ~car


def _open_close(m, r=1):
    from scipy import ndimage
    m = ndimage.binary_opening(m, iterations=r)
    return ndimage.binary_closing(m, iterations=2 * r)


def iou(render_mask, car, ignore):
    valid = ~ignore
    a, b = render_mask & valid, car & valid
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def resize_rgb(rgb, size):
    return np.asarray(Image.fromarray(rgb).resize(size, Image.LANCZOS))


def ref_image(car, preset, root=REF_ROOT):
    p = os.path.join(root, car, preset + ".png")
    return np.asarray(Image.open(p).convert("RGB"))
