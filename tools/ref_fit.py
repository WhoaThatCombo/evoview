"""Recover the camera of the game's car thumbnails by silhouette fitting.

Renders evoview's silhouette (alpha) at candidate cameras and maximises the
IoU against the thumbnail's background-separated mask (see lookcore.ref_masks).
The game generates every thumbnail with ONE camera from
system/thumbnails/thumbnails_generation.thumbnailsettings, so the fit is joint
over several cars; per-car results are reported to show how well that holds.

    python tools/ref_fit.py --exe L:/build/evoview-ref/release/evoview.exe \
        [--cars a,b,c] [--fit-cars a,b] [--write]

--write stores the result in tools/ref_cameras.json.
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lookcore as L

SIZE = (256, 256)
KEYS = ["tx", "ty", "tz", "distance", "azimuth", "elevation", "fov"]
STEP = {"tx": 0.2, "ty": 0.2, "tz": 0.2, "distance": 0.6, "azimuth": 4.0,
        "elevation": 4.0, "fov": 4.0}
MIN_STEP = {"tx": 0.01, "ty": 0.01, "tz": 0.01, "distance": 0.03, "azimuth": 0.2,
            "elevation": 0.2, "fov": 0.2}


def to_params(x):
    return {"target": [x["tx"], x["ty"], x["tz"]], "distance": x["distance"],
            "azimuth": x["azimuth"], "elevation": x["elevation"], "fov": x["fov"]}


class Scorer:
    def __init__(self, exe, cars, cams):
        self.exe, self.cars, self.cams = exe, cars, cams
        self.refs = {}
        for c in cars:
            e = cams["cars"][c]
            rgb = L.resize_rgb(L.ref_image(c, e["reference"]), SIZE)
            self.refs[c] = L.ref_masks(rgb)
        self.cache, self.evals = {}, 0

    def per_car(self, x, car):
        key = (car,) + tuple(round(x[k], 4) for k in KEYS)
        if key not in self.cache:
            cam = L.cam_from_params(to_params(x))
            a = L.render(self.exe, car, cam, SIZE, visual=self.cams["cars"][car]["visual"],
                         alpha=True)
            m = a[..., 3] > 127
            self.cache[key] = L.iou(m, *self.refs[car])
            self.evals += 1
        return self.cache[key]

    def __call__(self, x):
        return sum(self.per_car(x, c) for c in self.cars) / len(self.cars)


def descend(score, x, keys, log=print):
    step = {k: STEP[k] for k in keys}
    best = score(x)
    log(f"start {best:.4f}")
    while any(step[k] >= MIN_STEP[k] for k in keys):
        improved = False
        for k in keys:
            if step[k] < MIN_STEP[k]:
                continue
            for s in (+1, -1):
                y = dict(x); y[k] += s * step[k]
                v = score(y)
                if v > best + 1e-5:
                    x, best, improved = y, v, True
                    # keep going the same way while it helps
                    while True:
                        z = dict(x); z[k] += s * step[k]
                        w = score(z)
                        if w > best + 1e-5:
                            x, best = z, w
                        else:
                            break
                    break
        if not improved:
            for k in keys:
                step[k] *= 0.5
        log(f"  {best:.4f}  " + " ".join(f"{k}={x[k]:.3f}" for k in keys))
    return x, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--fit-cars", help="cars for the joint fit (default: first 4)")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--keys", default=",".join(KEYS))
    ap.add_argument("--per-car", action="store_true",
                    help="keep the global camera, refine only each car's placement (target x,y,z)")
    ap.add_argument("--cars", help="restrict --per-car to these cars")
    a = ap.parse_args()
    cams = L.load_cams()
    allcars = list(cams["cars"])
    fit_cars = a.fit_cars.split(",") if a.fit_cars else allcars[:4]
    g = cams["global"]
    x = {"tx": g["target"][0], "ty": g["target"][1], "tz": g["target"][2],
         "distance": g["distance"], "azimuth": g["azimuth"],
         "elevation": g["elevation"], "fov": g["fov"]}
    t0 = time.time()
    if a.per_car:
        return per_car(a, cams, x)
    sc = Scorer(a.exe, fit_cars, cams)
    x, best = descend(sc, x, a.keys.split(","))
    print(f"joint fit on {fit_cars}: IoU {best:.4f} ({sc.evals} renders, {time.time()-t0:.0f}s)")
    allsc = Scorer(a.exe, allcars, cams)
    per = {c: allsc.per_car(x, c) for c in allcars}
    for c, v in per.items():
        print(f"  {c:40s} IoU {v:.4f}")
    if a.write:
        cams["global"] = to_params(x)
        cams["global"]["iou_mean"] = round(sum(per.values()) / len(per), 4)
        for c, v in per.items():
            cams["cars"][c]["iou"] = round(v, 4)
        with open(L.CAMS, "w", encoding="utf-8") as f:
            json.dump(cams, f, indent=2)
        print("wrote", L.CAMS)


def per_car(a, cams, g):
    """The game uses one camera; what differs per car is where the car's model
    origin sits relative to the thumbnail rig. Refine that translation."""
    cars = a.cars.split(",") if a.cars else list(cams["cars"])
    for c in cars:
        sc = Scorer(a.exe, [c], cams)
        before = sc(g)
        x, best = descend(sc, dict(g), ["tx", "ty", "tz"], log=lambda *_: None)
        e = cams["cars"][c]
        e["offset"] = [round(x[k] - g[k], 4) for k in ("tx", "ty", "tz")]
        e["iou_global"], e["iou"] = round(before, 4), round(best, 4)
        e["camera"] = {k: ([round(v, 4) for v in val] if isinstance(val, list) else round(val, 4))
                       for k, val in L.cam_from_params(to_params(x)).items()}
        print(f"  {c:40s} IoU {before:.4f} -> {best:.4f}  offset {e['offset']}  ({sc.evals} renders)")
        if a.write:
            with open(L.CAMS, "w", encoding="utf-8") as f:
                json.dump(cams, f, indent=2)


if __name__ == "__main__":
    main()
