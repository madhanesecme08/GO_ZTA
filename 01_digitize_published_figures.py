"""
Digitise box/error-bar plots (Figs 6-9) of Mohankumar et al., Phys. Scr. 101 (2026) 025007.
Input : embedded raster images extracted from the published PDF with `pdfimages -png`.
Output: digitized_mechanical_properties.csv (median / Q1 / Q3 / whisker ends per condition).
Method: colour segmentation of the four temperature series, x-clustering into five GO groups,
        detection of horizontal line segments inside each glyph, linear y-axis calibration
        from the detected y-axis tick marks.
"""
import sys, json
import numpy as np
from PIL import Image

COLOURS = {1200: "red", 1300: "blue", 1400: "purple", 1500: "green"}
GO_LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]

def mask(a, c):
    r, g, b = a[..., 0].astype(int), a[..., 1].astype(int), a[..., 2].astype(int)
    if c == "red":    return (r > 170) & (g < 110) & (b < 110)
    if c == "blue":   return (b > 160) & (r < 90) & (g < 110)
    if c == "purple": return (r > 110) & (b > 170) & (g < 100)
    if c == "green":  return (g > 140) & (r < 110) & (b < 110)

def axis_and_ticks(a):
    dark = a.sum(axis=2) < 200
    h, w = dark.shape
    # y-axis: longest vertical dark run in left 30 % of the image
    col_counts = dark[:, : int(0.3 * w)].sum(axis=0)
    ax_x = int(np.argmax(col_counts))
    rows = np.where(dark[:, ax_x])[0]
    y_top, y_bot = rows.min(), rows.max()
    # ticks: dark pixels immediately left of the axis line
    strip = dark[:, ax_x - 14: ax_x - 3].sum(axis=1)
    tick_rows = np.where(strip >= 8)[0]
    ticks = []
    for r in tick_rows:
        if not ticks or r - ticks[-1][-1] > 2: ticks.append([r])
        else: ticks[-1].append(r)
    ticks = [float(np.mean(t)) for t in ticks]
    return ax_x, y_top, y_bot, ticks

def hlines(sub):
    """rows of a glyph mask that are (nearly) full-width horizontal segments"""
    cols = np.where(sub.any(axis=0))[0]
    W = cols.max() - cols.min() + 1
    cnt = sub.sum(axis=1)
    full = np.where(cnt >= 0.6 * W)[0]
    groups = []
    for r in full:
        if not groups or r - groups[-1][-1] > 1: groups.append([r])
        else: groups[-1].append(r)
    return [(float(np.mean(g)), len(g)) for g in groups], W

def digitise(path, tick_values, kind, x_max_frac, legend=None):
    a = np.asarray(Image.open(path).convert("RGB"))
    ax_x, y_top, y_bot, ticks = axis_and_ticks(a)
    if len(ticks) != len(tick_values):
        raise RuntimeError(f"{path}: found {len(ticks)} ticks {ticks}, expected {len(tick_values)}")
    k, c0 = np.polyfit(ticks, tick_values, 1)         # value = k*row + c0
    resid = np.array(tick_values) - (k * np.array(ticks) + c0)
    to_val = lambda r: k * r + c0
    out = []
    xlim = int(x_max_frac * a.shape[1])
    for T, cname in COLOURS.items():
        m = mask(a, cname); m[:, : ax_x + 3] = False
        if legend is None: m[:, xlim:] = False
        else:
            lx, ly = legend; m[: int(ly * a.shape[0]), int(lx * a.shape[1]):] = False
        colsum = m.sum(axis=0)
        xs = np.where(colsum > 0)[0]
        groups = []
        for x in xs:
            if not groups or x - groups[-1][-1] > 6: groups.append([x])
            else: groups[-1].append(x)
        groups = [g for g in groups if len(g) >= 8]
        if len(groups) != 5:
            raise RuntimeError(f"{path} T={T}: {len(groups)} glyphs found")
        for go, g in zip(GO_LEVELS, groups):
            sub = m[:, g[0]: g[-1] + 1]
            rows = np.where(sub.any(axis=1))[0]
            lines, W = hlines(sub)
            ys = [l[0] for l in lines]
            rec = {"GO_wt_pct": go, "Sintering_T_C": T,
                   "whisker_upper": to_val(rows.min()), "whisker_lower": to_val(rows.max())}
            if kind == "box":
                # expected: box top, median, box bottom (caps narrower than box)
                ys = sorted(ys)
                if len(ys) >= 3:
                    rec.update(Q3=to_val(ys[0]), median=to_val(ys[len(ys)//2]), Q1=to_val(ys[-1]))
                elif len(ys) == 2:   # median coincides with a box edge -> the merged (thicker) line holds it
                    ls = sorted(lines)
                    thick = max(ls, key=lambda l: l[1])
                    rec.update(Q3=to_val(ls[0][0]), Q1=to_val(ls[1][0]), median=to_val(thick[0]),
                               median_flag="median merged with box edge; +/- half IQR")
                rec["n_lines"] = len(ys)
            else:  # mean marker = thickest horizontal bar
                thick = max(lines, key=lambda l: l[1])
                rec.update(mean=to_val(thick[0]), n_lines=len(lines))
            out.append(rec)
    px_res = abs(k)
    return out, {"axis_x": ax_x, "tick_rows": ticks, "value_per_pixel": px_res,
                 "calibration_max_residual": float(np.abs(resid).max())}

if __name__ == "__main__":
    import pandas as pd, os
    IMG = sys.argv[1] if len(sys.argv) > 1 else "."
    OUT = (sys.argv[2] if len(sys.argv) > 2 else ".") + "/"
    os.chdir(IMG)
    specs = {
        "Relative_Density_pct":         ("fig-000.png", [96, 97, 98, 99, 100], "mean", 0.78),
        "Vickers_Hardness_GPa":         ("fig-001.png", [15, 16, 17, 18, 19, 20], "box", 0.78),
        "Youngs_Modulus_GPa":           ("fig-002.png", [360, 370, 380, 390, 400, 410], "box", 0.75, (0.77, 0.40)),
        "Fracture_Toughness_MPa_m0p5":  ("fig-003.png", [3, 4, 5, 6, 7], "box", 0.74),
    }
    frames, meta = [], {}
    for prop, spec in specs.items():
        f, tv, kind, xf = spec[:4]; lg = spec[4] if len(spec) > 4 else None
        recs, info = digitise(f, sorted(tv, reverse=True), kind, xf, lg)
        meta[prop] = info
        d = pd.DataFrame(recs); d.insert(0, "property", prop); frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(OUT + "digitized_long.csv", index=False)
    json.dump(meta, open(OUT + "digitize_calibration.json", "w"), indent=2, default=float)
    print(json.dumps(meta, indent=1, default=float))
    pd.set_option("display.width", 200); print(df.round(3).to_string())
