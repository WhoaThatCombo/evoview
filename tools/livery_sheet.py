"""Contact sheet of every livery lookcmp scored: game | evoview per cell.

    python tools/livery_sheet.py <lookcmp --all-liveries out dir> [out.png]
        [--cols 4] [--cell 480] [--before <earlier out dir>]

Reads <dir>/scores.json and the <tag>.png strips lookcmp writes (reference |
evoview | heatmap) and lays out one row block per car, each cell labelled
with the livery stem and its score (and the change from --before).
"""
import argparse, json, os
from PIL import Image, ImageDraw, ImageFont


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--cell", type=int, default=480, help="width of one game|evoview pair")
    ap.add_argument("--before")
    a = ap.parse_args()
    res = json.load(open(os.path.join(a.dir, "scores.json"), encoding="utf-8"))["cars"]
    before = {}
    if a.before:
        before = json.load(open(os.path.join(a.before, "scores.json"), encoding="utf-8"))["cars"]
    cars = {}
    for key, r in res.items():
        if "error" in r:
            continue
        car, _, ref = key.partition("/")
        tag = key.replace("/", "__")
        cars.setdefault(car, []).append((ref or "default", tag, r, before.get(key)))
    W, H, LBL = a.cell, a.cell // 2, 34
    try:
        font = ImageFont.truetype("arial.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
    rows = []
    for car, items in cars.items():
        for i in range(0, len(items), a.cols):
            rows.append((car if i == 0 else "", items[i:i + a.cols]))
    sheet = Image.new("RGB", (W * a.cols, (H + LBL) * len(rows) + 20 * sum(1 for r in rows if r[0])), (18, 20, 23))
    d = ImageDraw.Draw(sheet)
    y = 0
    for car, items in rows:
        if car:
            d.text((6, y + 3), car, fill=(124, 196, 255), font=font)
            y += 20
        for c, (ref, tag, r, b) in enumerate(items):
            p = os.path.join(a.dir, tag + ".png")
            if not os.path.exists(p):
                continue
            im = Image.open(p).convert("RGB")
            h = im.height
            pair = im.crop((0, 0, 2 * h, h)).resize((W, H), Image.LANCZOS)
            x = c * W
            sheet.paste(pair, (x, y))
            vis = ref.split("-", 1)[-1]
            delta = f" ({r['score'] - b['score']:+.1f})" if b and "score" in b else ""
            d.text((x + 4, y + H + 2), vis[:46], fill=(230, 230, 230), font=font)
            d.text((x + 4, y + H + 17), f"score {r['score']:.1f}{delta}  IoU {r['iou']:.2f}",
                   fill=(154, 160, 166), font=font)
        y += H + LBL
    out = a.out or os.path.join(a.dir, "livery_sheet.png")
    sheet.save(out)
    print(out, sheet.size)


if __name__ == "__main__":
    main()
