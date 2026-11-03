"""Compare evoview against the game's own car thumbnails.

    python tools/lookcmp.py --exe <path to evoview.exe> --out <dir>
        [--cars a,b] [--size 1024] [--baseline tools/lookcmp_baseline.json]
        [--write-baseline [path]] [--label text] [--all-liveries]

--all-liveries scores EVERY livery that has a game thumbnail
(<ref root>/<car>/<mech>-<visual>.png), not just the default one in
ref_cameras.json; results are keyed  <car>/<mech>-<visual>  and the default
livery keeps its plain <car> key so baselines still compare.

For every car in tools/ref_cameras.json it renders evoview with the recovered
thumbnail camera, writes  reference | evoview | difference heatmap  to
<out>/<car>.png, measures both images over the car mask and writes
<out>/index.html + <out>/scores.json. Higher score = closer to the game
(100 = identical statistics). References come from tools/ref_extract.py
(default L:/evoview_work/ref).

Metrics (all over the car mask = evoview's silhouette, eroded 2 px, minus
pixels the reference shows as backdrop):
  lum      mean luminance (sRGB 0-255, Rec.709 weights)
  p2/p98   black / white point (2nd / 98th percentile luminance)
  sat      mean HSV saturation (0-1)
  emd      colour histogram distance: Earth Mover's per R,G,B (0-255 units), averaged
  hl       highlight area fraction (luminance > 200)
  edge     edge sharpness: mean Sobel gradient magnitude of luminance
  dE       mean CIE76 Delta E between the two images, pixel for pixel

Every metric is reported twice: raw, and exposure-normalised (evoview scaled
in linear light so its mean car luminance equals the game's; the game
auto-exposes from a histogram and tonemaps GT/Uchimura + a grading LUT).
"""
import argparse, datetime, html, json, os, sys, time
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lookcore as L
from scipy import ndimage

# error normalisers: an error of this size costs the full weight
NORM = {"lum": 40.0, "p2": 30.0, "p98": 40.0, "sat": 0.25, "emd": 30.0,
        "hl": 0.10, "edge": 1.0, "dE": 25.0}
WEIGHT = {"lum": 1.5, "p2": 1.0, "p98": 1.0, "sat": 1.0, "emd": 1.5,
          "hl": 1.0, "edge": 1.0, "dE": 2.0}


def srgb_to_lab(rgb):
    c = rgb.astype(np.float32) / 255.0
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722],
                  [0.0193, 0.1192, 0.9505]], np.float32)
    xyz = c @ M.T / np.array([0.95047, 1.0, 1.08883], np.float32)
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])], -1)


def stats(rgb, mask):
    lum = L.luminance(rgb)
    px = rgb[mask].astype(np.float32)
    lm = lum[mask]
    mx, mn = px.max(1), px.min(1)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
    gx, gy = ndimage.sobel(lum, 1), ndimage.sobel(lum, 0)
    grad = np.hypot(gx, gy)[mask] / 8.0
    hists = [np.bincount(rgb[..., i][mask], minlength=256).astype(np.float64) for i in range(3)]
    return {
        "lum": float(lm.mean()), "p2": float(np.percentile(lm, 2)),
        "p98": float(np.percentile(lm, 98)), "sat": float(sat.mean()),
        "hl": float((lm > 200).mean()), "edge": float(grad.mean()),
        "_hist": [h / h.sum() for h in hists],
    }


def emd(h1, h2):
    """1D Earth Mover's distance between two normalised 256-bin histograms."""
    return float(np.abs(np.cumsum(h1) - np.cumsum(h2)).sum())


def heat(de, mask):
    """Delta E -> colour (black-purple-orange-yellow), dimmed outside the mask."""
    t = np.clip(de / 40.0, 0, 1)
    stops = np.array([[0, 0, 0], [80, 20, 120], [220, 70, 40], [255, 220, 60], [255, 255, 255]], np.float32)
    x = t * (len(stops) - 1)
    i = np.minimum(x.astype(int), len(stops) - 2)
    f = (x - i)[..., None]
    img = stops[i] * (1 - f) + stops[i + 1] * f
    img[~mask] *= 0.35
    edge = mask ^ ndimage.binary_erosion(mask, iterations=2)
    img[edge] = (90, 200, 255)
    return img.astype(np.uint8)


def compare(exe, car, entry, size, out_dir, cams, tag=None):
    cam = entry.get("camera") or L.cam_from_params(cams["global"])
    ref = L.ref_image(car, entry["reference"])
    if ref.shape[0] != size:
        ref = L.resize_rgb(ref, (size, size))
    # the reference is <mech preset>-<visual preset>; the mech preset fits the wheels
    mech = {"EVOVIEW_MECH": entry["reference"].split("-", 1)[0]}
    evo = L.render(exe, car, cam, (size, size), visual=entry["visual"], extra_env=mech)[..., :3]
    sil = L.render(exe, car, cam, (size, size), visual=entry["visual"], alpha=True,
                   extra_env=mech)[..., 3] > 127
    ref_car, ref_ign = L.ref_masks(ref)
    backdrop = ~(ref_car | ref_ign)
    mask = ndimage.binary_erosion(sil, iterations=2) & ~backdrop
    if mask.sum() < 500:
        raise RuntimeError("car mask nearly empty - camera or render broken")
    m, score, de = measure(ref, evo, mask)
    # Exposure-normalised: scale evoview's LINEAR radiance so its mean car
    # luminance matches the game's, then measure again. The game auto-exposes
    # from a histogram and grades through a LUT; this separates material /
    # shape differences from plain exposure.
    k = linear_lum(ref)[mask].mean() / max(linear_lum(evo)[mask].mean(), 1e-6)
    evo_n = to_srgb8(np.clip(to_linear(evo) * k, 0, 1))
    mn, score_n, de_n = measure(ref, evo_n, mask, skip=("lum",))
    iou = L.iou(sil, ref_car, ref_ign)
    side = np.concatenate([ref, evo, heat(de, mask)], 1)
    tag = tag or car
    Image.fromarray(np.concatenate([ref, evo_n, heat(de_n, mask)], 1)).save(
        os.path.join(out_dir, tag + "_norm.png"))
    Image.fromarray(side).save(os.path.join(out_dir, tag + ".png"))
    Image.fromarray(evo).save(os.path.join(out_dir, tag + "_evo.png"))
    return {"score": round(score, 2), "score_norm": round(score_n, 2), "exposure_k": round(float(k), 4),
            "iou": round(iou, 4), "metrics": m, "metrics_norm": mn,
            "category": entry.get("category", "")}


def list_liveries(exe, car, pkg=L.GAME_PKG):
    """Visual preset stems of `car`, default first (evoview --list-liveries)."""
    import subprocess
    r = subprocess.run([exe, pkg, "--car", car, "--list-liveries"], capture_output=True, text=True,
                       timeout=120, cwd=os.path.dirname(os.path.abspath(exe)))
    return [l.split("\t")[0] for l in r.stdout.splitlines() if l.strip()]


def livery_refs(car, root=L.REF_ROOT):
    """Every '<mech>-<visual>' thumbnail extracted for `car`, sorted."""
    d = os.path.join(root, car)
    if not os.path.isdir(d):
        return []
    return sorted(f[:-4] for f in os.listdir(d) if f.endswith(".png") and "-" in f)


def to_linear(rgb):
    c = rgb.astype(np.float32) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def to_srgb8(lin):
    c = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * np.power(lin, 1 / 2.4) - 0.055)
    return np.clip(c * 255 + 0.5, 0, 255).astype(np.uint8)


def linear_lum(rgb):
    c = to_linear(rgb)
    return 0.2126 * c[..., 0] + 0.7152 * c[..., 1] + 0.0722 * c[..., 2]


def measure(ref, evo, mask, skip=()):
    """(metrics, score 0-100, per-pixel Delta E) of evo against ref over mask."""
    sr, se = stats(ref, mask), stats(evo, mask)
    de = np.linalg.norm(srgb_to_lab(ref) - srgb_to_lab(evo), axis=-1)
    m = {}
    for k in ("lum", "p2", "p98", "sat", "hl", "edge"):
        m[k] = {"ref": sr[k], "evo": se[k], "delta": se[k] - sr[k]}
    e = float(np.mean([emd(a, b) for a, b in zip(sr["_hist"], se["_hist"])]))
    m["emd"] = {"ref": 0.0, "evo": e, "delta": e}
    m["dE"] = {"ref": 0.0, "evo": float(de[mask].mean()), "delta": float(de[mask].mean())}
    # edge sharpness compared as a ratio: 1.0 = same crispness
    m["edge"]["ratio"] = se["edge"] / max(sr["edge"], 1e-6)
    err = wsum = 0.0
    for k, w in WEIGHT.items():
        if k in skip:
            continue
        e = abs(m[k]["ratio"] - 1.0) if k == "edge" else abs(m[k]["delta"])
        err += w * min(e / NORM[k], 1.0)
        wsum += w
    return m, 100.0 * (1.0 - err / wsum), de


def fmt(v, k):
    if v is None:
        return ""
    if k in ("sat", "hl"):
        return f"{v:.3f}"
    return f"{v:.1f}" if k != "edge" else f"{v:.2f}"


def _table(res, base, mkey, skey, bkey):
    keys = ["lum", "p2", "p98", "sat", "emd", "hl", "edge", "dE"]
    rows = []
    for car, r in res.items():
        if "error" in r:
            rows.append(f"<tr><td>{car}</td><td colspan=12>ERROR: {html.escape(r['error'])}</td></tr>")
            continue
        m = r[mkey]
        b = base.get("cars", {}).get(car, {}).get(bkey) if base else None
        bd = f" <small>({r[skey] - b:+.1f})</small>" if b is not None else ""
        cells = []
        for k in keys:
            if k in ("emd", "dE"):
                cells.append(f"<td>{fmt(m[k]['evo'], k)}</td>")
            else:
                cells.append(f"<td>{fmt(m[k]['ref'], k)} / {fmt(m[k]['evo'], k)}</td>")
        rows.append(f"<tr><td><a href='#{r.get('tag', car)}'>{car}</a><br><small>{html.escape(r['category'])}</small></td>"
                    f"<td><b>{r[skey]:.1f}</b>{bd}</td><td>{r['iou']:.3f}</td><td>{r['exposure_k']:.2f}</td>{''.join(cells)}</tr>")
    return ("<div class=wrap><table><tr><th>car</th><th>score</th><th>IoU</th><th>exp k</th><th>lum ref/evo</th>"
            "<th>p2</th><th>p98</th><th>sat</th><th>emd</th><th>hl&gt;200</th><th>edge</th><th>&Delta;E</th></tr>"
            + "".join(rows) + "</table></div>")


def write_html(out_dir, res, overall, overall_n, base, label):
    imgs = "".join(
        f"<h3 id='{r.get('tag', c)}'>{c} &mdash; {r['score']:.1f} raw / {r['score_norm']:.1f} exposure-normalised</h3>"
        f"<p class=cap>reference | evoview | &Delta;E heatmap (0..40, outline = car mask)</p><img src='{r.get('tag', c)}.png' loading=lazy>"
        f"<p class=cap>same, evoview scaled &times;{r['exposure_k']:.2f} in linear to the game's mean car luminance</p>"
        f"<img src='{r.get('tag', c)}_norm.png' loading=lazy>" for c, r in res.items() if "error" not in r)
    bline = ""
    if base:
        bline = (f"<p>Baseline ({html.escape(base.get('label', ''))}): raw {base.get('overall', 0):.2f} &rarr; "
                 f"{overall - base.get('overall', 0):+.2f}, normalised {base.get('overall_norm', 0):.2f} &rarr; "
                 f"{overall_n - base.get('overall_norm', 0):+.2f}</p>")
    doc = f"""<!doctype html><html><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>evoview look report</title><style>
:root{{--bg:#16181b;--fg:#e6e6e6;--mut:#9aa0a6;--line:#2c3035;--acc:#7cc4ff}}
body{{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif;margin:0;padding:16px;max-width:1700px}}
table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}td,th{{border-bottom:1px solid var(--line);padding:4px 8px;text-align:right}}
td:first-child,th:first-child{{text-align:left}}small,.cap{{color:var(--mut)}}a{{color:var(--acc)}}
img{{width:100%;height:auto;display:block;border:1px solid var(--line)}}.wrap{{overflow-x:auto}}
.big{{font-size:40px;font-weight:600;margin-right:32px;display:inline-block}}</style></head><body>
<h1>evoview vs game thumbnails</h1><p class=cap>{html.escape(label)} &middot; {datetime.datetime.now():%Y-%m-%d %H:%M}</p>
<div class=big>{overall:.2f}<br><small>raw</small></div><div class=big>{overall_n:.2f}<br><small>exposure-normalised</small></div>
<p class=cap>overall score = mean over cars, 100 = same statistics as the game. Normalised: evoview's linear radiance
scaled so mean car luminance matches (exp k), which separates material/shape from exposure and tonemap.</p>{bline}
<h2>Raw</h2>{_table(res, base, "metrics", "score", "score")}
<h2>Exposure-normalised</h2>{_table(res, base, "metrics_norm", "score_norm", "score_norm")}
{imgs}</body></html>"""
    with open(os.path.join(out_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(doc)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exe", required=True)
    ap.add_argument("--out", default=r"L:\evoview_work\report")
    ap.add_argument("--cars")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--baseline", default=os.path.join(L.HERE, "lookcmp_baseline.json"))
    ap.add_argument("--write-baseline", nargs="?", const=os.path.join(L.HERE, "lookcmp_baseline.json"))
    ap.add_argument("--label", default="")
    ap.add_argument("--list-exe", help="an evoview with --list-liveries, to skip thumbnails of "
                    "presets the car no longer ships (default: no check)")
    ap.add_argument("--all-liveries", action="store_true",
                    help="score every livery with a thumbnail, not just the default")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cams = L.load_cams()
    cars = a.cars.split(",") if a.cars else list(cams["cars"])
    presets = {}  # car -> set of real visual preset stems (None = unknown)
    def real_visuals(car):
        if car not in presets:
            presets[car] = list_liveries(a.list_exe, car) if a.list_exe else None
        return presets[car]
    for car in cars:
        if car not in cams["cars"]:
            # not catalogued: the global thumbnail camera, and the thumbnail
            # of the livery evoview wears by default (the first real preset)
            vis = real_visuals(car)
            refs = [r for r in livery_refs(car) if vis is None or r.split("-", 1)[1] in vis]
            if vis:
                refs.sort(key=lambda r: (vis.index(r.split("-", 1)[1]), r))
            if not refs:
                print(f"{car}: no reference thumbnail, skipped")
                continue
            ref = refs[0]
            cams["cars"][car] = {"reference": ref, "visual": ref.split("-", 1)[1], "category": "(global camera)"}
    cars = [c for c in cars if c in cams["cars"]]
    label = a.label or os.path.abspath(a.exe)
    res = {}
    t0 = time.time()
    jobs = []  # (result key, car, entry, file tag)
    for car in cars:
        entry = cams["cars"][car]
        jobs.append((car, car, entry, car))
        if a.all_liveries:
            vis = real_visuals(car)
            for ref in livery_refs(car):
                if ref == entry["reference"]:
                    continue
                if vis is not None and ref.split("-", 1)[1] not in vis:
                    print(f"{car}/{ref}: no such visual preset (stale thumbnail), skipped")
                    continue
                e = dict(entry, reference=ref, visual=ref.split("-", 1)[1])
                jobs.append((f"{car}/{ref}", car, e, f"{car}__{ref}"))
    for key, car, entry, tag in jobs:
        try:
            res[key] = compare(a.exe, car, entry, a.size, a.out, cams, tag=tag)
            res[key]["tag"] = tag
            res[key]["reference"] = entry["reference"]
            r = res[key]
            print(f"{key:70s} score {r['score']:6.2f}  norm {r['score_norm']:6.2f}  k {r['exposure_k']:.2f}  IoU {r['iou']:.3f}", flush=True)
        except Exception as e:
            res[key] = {"error": str(e)}
            print(f"{key:70s} ERROR {e}", flush=True)
    ok = [r["score"] for r in res.values() if "error" not in r]
    overall = round(float(np.mean(ok)), 2) if ok else 0.0
    okn = [r["score_norm"] for r in res.values() if "error" not in r]
    overall_n = round(float(np.mean(okn)), 2) if okn else 0.0
    base = None
    if a.baseline and os.path.exists(a.baseline) and not a.write_baseline:
        with open(a.baseline, encoding="utf-8") as f:
            base = json.load(f)
        # several baselines may be stored; compare with the newest
        if "runs" in base:
            base = base["runs"][-1]
    write_html(a.out, res, overall, overall_n, base, label)
    rnd = lambda mm: {k: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()}
                      for k, v in mm.items()}
    summary = {"label": label, "overall": overall, "overall_norm": overall_n,
               "date": datetime.datetime.now().isoformat(timespec="seconds"),
               "cars": {c: ({"score": r["score"], "score_norm": r["score_norm"], "exposure_k": r["exposure_k"],
                             "iou": r["iou"], "metrics": rnd(r["metrics"]), "metrics_norm": rnd(r["metrics_norm"])}
                            if "error" not in r else r) for c, r in res.items()}}
    with open(os.path.join(a.out, "scores.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    if a.all_liveries:
        dflt = [res[c]["score"] for c in cars if "error" not in res.get(c, {"error": 1})]
        print(f"default liveries only: {np.mean(dflt):.2f} over {len(dflt)} car(s); all liveries: {overall:.2f} over {len(ok)}")
    print(f"overall {overall:.2f}  normalised {overall_n:.2f}  ({len(ok)}/{len(jobs)} renders, {time.time()-t0:.0f}s) -> {os.path.join(a.out, 'index.html')}")
    if base:
        print(f"baseline {base.get('overall', 0):.2f} / {base.get('overall_norm', 0):.2f}  delta "
              f"{overall - base.get('overall', 0):+.2f} / {overall_n - base.get('overall_norm', 0):+.2f}")
    if a.write_baseline:
        runs = []
        if os.path.exists(a.write_baseline):
            with open(a.write_baseline, encoding="utf-8") as f:
                old = json.load(f)
            runs = [r for r in old.get("runs", []) if r.get("label") != label]
        runs.append(summary)
        with open(a.write_baseline, "w", encoding="utf-8") as f:
            json.dump({"runs": runs}, f, indent=1)
        print("baseline written to", a.write_baseline)


if __name__ == "__main__":
    main()
