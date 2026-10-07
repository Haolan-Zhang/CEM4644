"""Original practice drawings for MP7, drawn by code, with problem text and the exact ground truth.

    python build/generate_practice.py [--out DIR] [--n 5] [--seed 4644]

Three families: foundation plans (CMU and grout), gable roofs (underlayment and shingles), and door-schedule sheets
laid out like a construction-document sheet (small text, so the full sheet is hard to read and a crop is not).
Everything here is made for the course; nothing is copied or traced from another drawing.
"""
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Polygon, Rectangle  # noqa: E402

INK = "#111111"
FONT = {"family": "DejaVu Sans"}
CMU_FACE = 8 * 16 / 144                      # ft² of wall per CMU (8 in x 16 in nominal face)
GROUT_PER_CMU = {6: 0.17, 8: 0.26, 10: 0.33}  # ft³ of grout per CMU, all cells filled (values given in the problem)
ROLLS = {"#15": (144, 3), "#30": (72, 3)}     # roll length x width (ft)


def ft(v):
    """12.5 -> 12'-6"."""
    whole = int(math.floor(v + 1e-9)); inch = round((v - whole) * 12)
    if inch == 12:
        whole, inch = whole + 1, 0
    return f"{whole}'-{inch}\""


def page(w, h):
    """A figure whose axes fill the whole page, so drawing units are page inches (needed for exact crops)."""
    fig = plt.figure(figsize=(w, h)); ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")
    return fig, ax


def sheet_frame(ax, w, h, title, number):
    """A border and a small title block, so every drawing looks like a sheet."""
    ax.add_patch(Rectangle((0.15, 0.15), w - 0.3, h - 0.3, fill=False, lw=1.6, ec=INK))
    tb_w = min(3.2, w * 0.28)
    ax.add_patch(Rectangle((w - 0.15 - tb_w, 0.15), tb_w, 0.9, fill=False, lw=1.0, ec=INK))
    ax.text(w - 0.15 - tb_w + 0.1, 0.85, "CEM4644 · MP7 PRACTICE DRAWING", fontsize=6.5, weight="bold", va="center", **FONT)
    ax.text(w - 0.15 - tb_w + 0.1, 0.6, title, fontsize=6, va="center", **FONT)
    ax.text(w - 0.15 - tb_w + 0.1, 0.35, "NOT FOR CONSTRUCTION", fontsize=5.5, va="center", color="#555", **FONT)
    ax.text(w - 0.25, 0.48, number, fontsize=11, weight="bold", ha="right", va="center", **FONT)


def dim(ax, p, q, text, off, size=8.5):
    """A dimension line between p and q, offset perpendicular by off (drawing units), with ticks and a label."""
    p, q = np.array(p, float), np.array(q, float)
    d = q - p; n = np.array([-d[1], d[0]]) / (np.hypot(*d) + 1e-9)
    a, b = p + n * off, q + n * off
    ax.plot(*zip(p + n * off * 0.15, a + n * 0.08 * np.sign(off)), color=INK, lw=0.5)
    ax.plot(*zip(q + n * off * 0.15, b + n * 0.08 * np.sign(off)), color=INK, lw=0.5)
    ax.plot(*zip(a, b), color=INK, lw=0.6)
    for c in (a, b):
        t = np.array([d[0] + d[1], d[1] - d[0]]) / (np.hypot(*d) + 1e-9) * 0.09
        ax.plot([c[0] - t[0], c[0] + t[0]], [c[1] - t[1], c[1] + t[1]], color=INK, lw=1.0)
    m = (a + b) / 2 + n * 0.17 * np.sign(off)
    ang = math.degrees(math.atan2(d[1], d[0]))
    if ang > 90 or ang < -90:
        ang += 180
    ax.text(*m, text, fontsize=size, ha="center", va="center", rotation=ang, **FONT)


# ============================================================================ foundation plans
def foundation_shape(rng):
    """A rectilinear footprint (ft) as a counter-clockwise list of corners."""
    kind = rng.choice(["rectangle", "L", "bump", "notch"])
    W = int(rng.integers(15, 31)) * 2; H = int(rng.integers(12, 25)) * 2
    if kind == "rectangle":
        pts = [(0, 0), (W, 0), (W, H), (0, H)]
    elif kind == "L":
        a = int(rng.integers(4, W // 2 // 2 + 2)) * 2; b = int(rng.integers(3, H // 2 // 2 + 2)) * 2
        pts = [(0, 0), (W, 0), (W, H - b), (W - a, H - b), (W - a, H), (0, H)]
    elif kind == "bump":
        depth = int(rng.integers(4, 13)) * 2; width = int(rng.integers(4, H // 2 - 1)) * 2
        y0 = int(rng.integers(2, (H - width) // 2)) * 2 if H - width >= 6 else 2
        pts = [(0, 0), (W, 0), (W, y0), (W + depth, y0), (W + depth, y0 + width), (W, y0 + width), (W, H), (0, H)]
    else:
        depth = int(rng.integers(3, H // 4 + 1)) * 2; width = int(rng.integers(4, W // 2 - 1)) * 2
        x0 = int(rng.integers(2, (W - width) // 2)) * 2 if W - width >= 6 else 2
        pts = [(0, 0), (x0, 0), (x0, depth), (x0 + width, depth), (x0 + width, 0), (W, 0), (W, H), (0, H)]
    return kind, pts


def perimeter(pts):
    return sum(math.hypot(pts[(i + 1) % len(pts)][0] - pts[i][0], pts[(i + 1) % len(pts)][1] - pts[i][1]) for i in range(len(pts)))


def draw_foundation(pts, hide, thickness_in, path, number):
    xs, ys = zip(*pts); W, H = max(xs), max(ys)
    s = min(7.0 / W, 4.9 / H)                                   # drawing units (in) per ft, fitting the page with room for dimensions
    fig_w, fig_h = 11, 8.5
    fig, ax = page(fig_w, fig_h)
    ox, oy = (fig_w - W * s) / 2 - 0.4, 2.35
    P = [(ox + x * s, oy + y * s) for x, y in pts]
    t = max(thickness_in / 12 * s * 2.2, 0.06)                   # wall drawn a little thicker than scale, for legibility
    ax.add_patch(Polygon(P, closed=True, fill=False, lw=1.3, ec=INK))
    cx, cy = np.mean([p[0] for p in P]), np.mean([p[1] for p in P])
    inner = []
    for i in range(len(P)):                                     # inset polygon (offset each edge inward)
        a, b, c = np.array(P[i - 1]), np.array(P[i]), np.array(P[(i + 1) % len(P)])
        n1 = np.array([-(b - a)[1], (b - a)[0]]); n1 /= np.hypot(*n1)
        n2 = np.array([-(c - b)[1], (c - b)[0]]); n2 /= np.hypot(*n2)
        inner.append(b + (n1 + n2) / (1 + n1 @ n2) * t)
    ax.add_patch(Polygon(inner, closed=True, fill=False, lw=0.9, ec=INK))
    for i in range(len(P)):
        if i == hide:
            continue
        a, b = P[i], P[(i + 1) % len(P)]
        L = math.hypot(pts[(i + 1) % len(pts)][0] - pts[i][0], pts[(i + 1) % len(pts)][1] - pts[i][1])
        dim(ax, a, b, ft(L), -0.45)                              # outside (the polygon is counter-clockwise)
    bx = ox + W * s / 2; by = oy - 1.15
    ax.text(bx, by, "FOUNDATION PLAN", fontsize=11, weight="bold", ha="center", **FONT)
    ax.text(bx, by - 0.3, f"{thickness_in} in CMU FOUNDATION WALL", fontsize=8, ha="center", **FONT)
    ax.text(bx, by - 0.55, "DIMENSIONS ARE TO THE OUTSIDE FACE OF WALL · NOT TO SCALE", fontsize=6.5, ha="center", color="#333", **FONT)
    sheet_frame(ax, fig_w, fig_h, "FOUNDATION PLAN", number)
    fig.savefig(path, dpi=150, facecolor="white"); plt.close(fig)


def foundation_problem(rng, i, out):
    kind, pts = foundation_shape(rng)
    hide = int(rng.integers(0, len(pts))) if i % 2 == 1 else -1  # every other plan leaves one dimension to be derived
    thick = int(rng.choice([6, 8, 10])); height = int(rng.choice([4, 6, 8, 10])); waste = int(rng.choice([3, 4, 5, 6]))
    gwaste = int(rng.choice([5, 7, 10])); spacing = int(rng.choice([16, 24, 32, 48]))
    per = perimeter(pts); area = per * height
    cmu = area / CMU_FACE
    grout_all = cmu * GROUT_PER_CMU[thick] / 27 * (1 + gwaste / 100)
    grout_part = grout_all * 8 / spacing
    name = f"foundation_{i + 1}"
    draw_foundation(pts, hide, thick, out / f"{name}.png", f"S-10{i + 1}")
    text = (f"The attached foundation plan shows a {thick} in CMU foundation wall, {height} ft high, with no openings. "
            "Use 8 in x 16 in (nominal) CMUs.\n"
            f"Q1. How many CMUs must be purchased, including {waste} % waste?\n"
            f"Q2. If every cell is grouted, how many cubic yards of grout are needed, including {gwaste} % waste? "
            f"A fully grouted {thick} in CMU takes {GROUT_PER_CMU[thick]} ft³ of grout.\n"
            f"Q3. If only the cells with vertical reinforcement at {spacing} in on center are grouted, how many cubic yards of grout are needed, "
            f"including {gwaste} % waste?")
    key = [("Q1", "CMUs to purchase", math.ceil(cmu * (1 + waste / 100)), "CMU", 0.01),
           ("Q2", "grout, all cells", grout_all, "CY", 0.02), ("Q3", f"grout, {spacing} in O.C.", grout_part, "CY", 0.02)]
    rules = ("Estimating conventions: total wall length = sum of the outside dimensions (no deduction at corners); wall area = length x height; "
             f"one CMU covers 8 in x 16 in = 0.889 ft²; round CMUs up after adding waste. Grout = CMUs before waste x grout per CMU, "
             f"converted to CY (27 ft³ = 1 CY), then add grout waste. Cells are 8 in apart, so grouting at S in on center fills 8/S of the cells.")
    return {"id": name, "family": "foundation", "image": f"{name}.png", "shape": kind, "hidden_dimension": hide >= 0, "wall_length_ft": per,
            "text": text, "rules": rules, "key": [dict(q=qi, quantity=q, value=round(v, 3), unit=u, tolerance=t) for qi, q, v, u, t in key]}


# ============================================================================ gable roofs
def draw_roof(L, D, r, path, number):
    """A roof plan (eaves, ridge, rakes, eave length) and an end elevation (the gable, eave-to-eave distance, slope)."""
    fig_w, fig_h = 11, 8.5
    fig, ax = page(fig_w, fig_h)
    run, rise = D / 2, D / 2 * r / 12
    s = 6.4 / (L + D)
    x0, y0 = 1.3, 3.0
    w, h = L * s, D * s
    ax.add_patch(Rectangle((x0, y0), w, h, fill=False, ec=INK, lw=1.3))
    ax.plot([x0, x0 + w], [y0 + h / 2, y0 + h / 2], color=INK, lw=1.0)
    ax.text(x0 + w / 2, y0 + h / 2 + 0.1, "RIDGE", fontsize=8, ha="center", **FONT)
    ax.text(x0 + w / 2, y0 + 0.12, "EAVE", fontsize=8, ha="center", **FONT)
    ax.text(x0 + w / 2, y0 + h - 0.25, "EAVE", fontsize=8, ha="center", **FONT)
    ax.text(x0 + 0.12, y0 + h * 0.25, "RAKE", fontsize=8, rotation=90, va="center", **FONT)
    ax.text(x0 + w - 0.22, y0 + h * 0.25, "RAKE", fontsize=8, rotation=90, va="center", **FONT)
    for yy, d in ((y0 + h * 0.3, -1), (y0 + h * 0.7, 1)):        # slope arrows, down toward the eaves
        ax.annotate("", xy=(x0 + w * 0.65, yy + d * h * 0.14), xytext=(x0 + w * 0.65, yy - d * h * 0.06),
                    arrowprops=dict(arrowstyle="->", color=INK, lw=0.8))
    dim(ax, (x0, y0), (x0 + w, y0), f"{ft(L)} (EAVE LENGTH)", -0.45)
    dim(ax, (x0, y0 + h), (x0, y0), ft(D), -0.45)
    ax.text(x0 + w / 2, y0 - 1.1, "ROOF PLAN", fontsize=11, weight="bold", ha="center", **FONT)
    ex = x0 + w + 1.2; base = y0 + 0.9
    hw, hr = D * s, rise * s
    ax.add_patch(Polygon([(ex, base), (ex + hw, base), (ex + hw / 2, base + hr)], closed=True, fill=False, ec=INK, lw=1.3))
    ax.add_patch(Rectangle((ex + 0.15, base - 0.9), hw - 0.3, 0.9, fill=False, ec=INK, lw=0.8))
    dim(ax, (ex, base), (ex + hw, base), f"{ft(D)} (EAVE TO EAVE)", -1.25)
    g = (ex - 0.85, base + hr * 0.55)                           # slope triangle beside the gable, clear of the labels
    ax.plot([g[0], g[0] + 0.5], [g[1], g[1]], color=INK, lw=0.8); ax.plot([g[0] + 0.5, g[0] + 0.5], [g[1], g[1] + 0.5 * r / 12], color=INK, lw=0.8)
    ax.text(g[0] + 0.25, g[1] - 0.14, "12", fontsize=7.5, ha="center", **FONT); ax.text(g[0] + 0.58, g[1] + 0.25 * r / 12, str(r), fontsize=7.5, va="center", **FONT)
    ax.text(ex + hw / 2, base + hr + 0.12, "RIDGE", fontsize=8, ha="center", **FONT)
    ax.text(ex + hw / 2, y0 - 1.1, "END ELEVATION", fontsize=11, weight="bold", ha="center", **FONT)
    ax.text(fig_w / 2, fig_h - 0.75, "GABLE ROOF", fontsize=12, weight="bold", ha="center", **FONT)
    ax.text(fig_w / 2, fig_h - 1.05, f"ROOF SLOPE {r}:12 · NOT TO SCALE", fontsize=8, ha="center", **FONT)
    sheet_frame(ax, fig_w, fig_h, "ROOF PLAN AND END ELEVATION", number)
    fig.savefig(path, dpi=150, facecolor="white"); plt.close(fig)


def roof_problem(rng, i, out):
    L = int(rng.integers(14, 31)) * 2; D = int(rng.integers(10, 19)) * 2; r = int(rng.choice([4, 5, 6, 7, 8]))
    roll = str(rng.choice(["#15", "#30"])); expo = float(rng.choice([5, 5.5, 6]))
    rl, rw = ROLLS[roll]
    rake = math.hypot(D / 2, D / 2 * r / 12)
    net = rake * L * 2
    under = net + L * 1 + 4 / 12 * rake * 4
    starter, ridge, rakew = L * expo / 12 * 2, L * 1, 4 * rake * 0.25
    gross = net + starter + ridge + rakew
    name = f"roof_{i + 1}"
    draw_roof(L, D, r, out / f"{name}.png", f"A-20{i + 1}")
    text = (f"The attached diagram shows a gable roof with a {r}:12 slope.\n"
            "Q1. What is the net roof area (ft²)?\n"
            f"Q2. How many rolls of {roll} single-coverage underlayment ({rl} ft x {rw} ft per roll) are needed? The underlayment laps 4 in under each rake edge "
            "and overlaps 1 ft at the ridge.\n"
            f"Q3. The roof is covered with 3-tab asphalt strip shingles, 3 ft x 1 ft, with {expo:g} in exposure. Find (a) the starter-course area at the eaves (ft²), "
            "(b) the area for the 1 ft ridge coverage (ft²), (c) the cutting-waste area at the rakes (ft², 0.25 ft² per linear foot of rake), "
            "(d) the gross roof area (ft²), and (e) the number of shingles.")
    key = [("Q1", "net roof area", net, "ft2", 0.01), ("Q2", "underlayment rolls", math.ceil(under / (rl * rw)), "rolls", 0),
           ("Q3a", "starter course area (both eaves)", starter, "ft2", 0.02), ("Q3b", "ridge coverage area", ridge, "ft2", 0.02),
           ("Q3c", "rake cutting waste area", rakew, "ft2", 0.02), ("Q3d", "roof gross area", gross, "ft2", 0.01),
           ("Q3e", "number of shingles", math.ceil(gross / (3 * expo / 12)), "shingles", 0.01)]
    rules = ("Estimating conventions: run = half the eave-to-eave distance; rise = run x slope; rake length = sqrt(run² + rise²); "
             "net roof area = rake length x eave length x 2 roof planes. Underlayment area = net area + ridge lap (eave length x 1 ft) + "
             "rake laps (4/12 ft x rake length x 4 rake edges); rolls = area / (roll length x width), rounded up. Starter course = eave length x "
             "exposure x 2 eaves; ridge coverage = eave length x 1 ft; rake waste = 4 rake edges x rake length x 0.25 ft²/ft; gross area = net + "
             "starter + ridge + rake waste; shingles = gross area / (3 ft x exposure), rounded up.")
    return {"id": name, "family": "roof", "image": f"{name}.png", "text": text, "rules": rules, "dims": dict(eave_length=L, eave_to_eave=D, slope=r),
            "key": [dict(q=qi, quantity=q, value=round(v, 3), unit=u, tolerance=t) for qi, q, v, u, t in key]}


# ============================================================================ door-schedule sheets
LOCATIONS = ["ENTRY", "BEDROOM", "BATH", "WALK-IN CLOSET", "CLOSET", "LAUNDRY", "MECH", "PANTRY", "OFFICE", "STORAGE", "LINEN"]
DOOR_TYPES = {"DA": "FLUSH WOOD", "DB": "PANEL WOOD", "DC": "BIFOLD", "DD": "PAIR, PANEL WOOD", "DE": "POCKET", "DF": "HOLLOW METAL"}


def schedule_rows(rng, n):
    rows = [{"mark": "1", "type": "DF", "width": "3'-0\"", "height": "7'-0\"", "frame": "HM", "rating": "20 MIN", "hw": str(rng.choice([31, 32, 33])), "location": "ENTRY"}]
    for k in range(2, n + 1):
        loc = str(rng.choice(LOCATIONS[1:]))
        typ = {"CLOSET": "DC", "LAUNDRY": "DD", "PANTRY": "DE", "LINEN": "DC"}.get(loc, str(rng.choice(["DA", "DB"])))
        width = {"DC": "4'-0\"", "DD": "5'-0\""}.get(typ, str(rng.choice(["2'-6\"", "2'-8\"", "2'-10\"", "3'-0\""])))
        rows.append({"mark": str(k), "type": typ, "width": width, "height": str(rng.choice(["6'-8\"", "7'-0\""])), "frame": "WD",
                     "rating": "-", "hw": str(rng.choice([34, 35, 36, 37, 38])), "location": loc})
    return rows


COLS = [("MARK", "mark", 0.45), ("DOOR TYPE", "type", 0.7), ("DESCRIPTION", "desc", 1.3), ("WIDTH", "width", 0.6), ("HEIGHT", "height", 0.6),
        ("FRAME", "frame", 0.5), ("FIRE RATING", "rating", 0.75), ("HDWR SET", "hw", 0.65), ("LOCATION", "location", 1.25)]


def draw_schedule_sheet(units, path, number):
    """A 36 in x 24 in sheet: door elevations and notes on the left, the unit door schedules on the right."""
    fig_w, fig_h = 36, 24
    fig, ax = page(fig_w, fig_h)
    fs = 5.6
    # left: door type elevations and general notes (filler that makes the sheet busy, as real sheets are)
    ax.text(1.0, 22.6, "DOOR TYPES", fontsize=14, weight="bold", **FONT)
    for j, (code, desc) in enumerate(DOOR_TYPES.items()):
        x = 1.2 + (j % 3) * 4.2; y = 17.2 - (j // 3) * 6.0
        w = 2.4 if code != "DD" else 3.2
        ax.add_patch(Rectangle((x, y), w, 4.4, fill=False, ec=INK, lw=1.0))
        if code == "DB":
            for yy in (y + 0.4, y + 2.4):
                ax.add_patch(Rectangle((x + 0.35, yy), w - 0.7, 1.6, fill=False, ec=INK, lw=0.6))
        if code == "DD":
            ax.plot([x + w / 2, x + w / 2], [y, y + 4.4], color=INK, lw=0.8)
        if code == "DC":
            for q in (0.25, 0.5, 0.75):
                ax.plot([x + w * q, x + w * q], [y, y + 4.4], color=INK, lw=0.5)
        ax.text(x + w / 2, y - 0.5, code, fontsize=12, weight="bold", ha="center", **FONT)
        ax.text(x + w / 2, y - 0.9, desc, fontsize=7, ha="center", **FONT)
    ax.text(1.0, 4.9, "GENERAL DOOR NOTES", fontsize=10, weight="bold", **FONT)
    notes = ["1. VERIFY ALL ROUGH OPENINGS IN THE FIELD BEFORE ORDERING DOORS.", "2. UNDERCUT DOORS 3/4 IN AT BATHS AND LAUNDRY FOR AIR TRANSFER.",
             "3. PROVIDE DOOR STOPS AT ALL SWING DOORS.", "4. HARDWARE SETS ARE LISTED IN THE PROJECT MANUAL.", "5. 20 MIN DOORS: POSITIVE LATCHING, SELF-CLOSING."]
    for k, n in enumerate(notes):
        ax.text(1.0, 4.4 - k * 0.42, n, fontsize=6.5, **FONT)
    # right: the schedule, one block per unit type
    x0 = 15.2; widths = [c[2] for c in COLS]; xs = np.cumsum([x0] + widths)
    ax.text(x0, 22.9, "UNIT DOOR SCHEDULE", fontsize=14, weight="bold", **FONT)
    y = 22.3; rh = 0.27
    ax.add_patch(Rectangle((x0, y - rh), xs[-1] - x0, rh, fill=False, ec=INK, lw=0.8))
    for (h, _, _), xa in zip(COLS, xs):
        ax.text(xa + 0.05, y - rh / 2, h, fontsize=fs * 0.9, weight="bold", va="center", **FONT)
    y -= rh
    blocks = {}
    for unit, rows in units.items():
        y -= rh * 1.2
        ax.text(x0 + 0.05, y - rh / 2, f"UNIT TYPE {unit}", fontsize=fs, weight="bold", va="center", **FONT)
        top = y; y -= rh
        for r in rows:
            vals = {**r, "desc": DOOR_TYPES[r["type"]]}
            for (_, k, _), xa in zip(COLS, xs):
                ax.text(xa + 0.05, y - rh / 2, vals[k], fontsize=fs, va="center", **FONT)
            ax.plot([x0, xs[-1]], [y - rh, y - rh], color=INK, lw=0.3)
            y -= rh
        for xa in xs:
            ax.plot([xa, xa], [top, y], color=INK, lw=0.3)
        blocks[unit] = (top + rh * 0.15, y - 0.05)
    filler_schedules(ax, xs[-1] + 0.9, fs)
    sheet_frame(ax, fig_w, fig_h, "DOOR SCHEDULE AND DOOR TYPES", number)
    dpi = 100
    fig.savefig(path, dpi=dpi, facecolor="white"); plt.close(fig)
    return blocks, (x0 - 0.1, xs[-1] + 0.1), dpi, fig_h


def filler_schedules(ax, x0, fs):
    """A window schedule and the hardware sets: real sheets are busy, which is what makes the full sheet hard to read."""
    rng = np.random.default_rng(int(x0 * 1000))
    ax.text(x0, 22.9, "WINDOW SCHEDULE", fontsize=14, weight="bold", **FONT)
    cols = [("MARK", 0.5), ("TYPE", 0.9), ("WIDTH", 0.6), ("HEIGHT", 0.6), ("SILL HT", 0.6), ("GLAZING", 0.9), ("U-FACTOR", 0.6), ("REMARKS", 1.6)]
    xs = np.cumsum([x0] + [c[1] for c in cols]); y = 22.3; rh = 0.27
    for (h, _), xa in zip(cols, xs):
        ax.text(xa + 0.05, y - rh / 2, h, fontsize=fs * 0.9, weight="bold", va="center", **FONT)
    for k in range(int(rng.integers(14, 22))):
        y -= rh
        vals = [f"W{k + 1}", str(rng.choice(["SINGLE HUNG", "CASEMENT", "FIXED", "SLIDER"])), str(rng.choice(["2'-0\"", "2'-8\"", "3'-0\"", "4'-0\""])),
                str(rng.choice(["3'-0\"", "4'-0\"", "5'-0\""])), str(rng.choice(["2'-6\"", "3'-0\""])), "1 IN INSUL", str(rng.choice(["0.30", "0.32", "0.35"])),
                str(rng.choice(["EGRESS", "TEMPERED", "-", "OBSCURE GLASS", "-"]))]
        for v, xa in zip(vals, xs):
            ax.text(xa + 0.05, y - rh / 2, v, fontsize=fs, va="center", **FONT)
        ax.plot([x0, xs[-1]], [y - rh, y - rh], color=INK, lw=0.3)
    y -= 1.2
    ax.text(x0, y, "HARDWARE SETS", fontsize=14, weight="bold", **FONT)
    for hs in range(31, 39):
        y -= 0.42
        items = ["3 HINGES", str(rng.choice(["ENTRY LOCKSET", "PASSAGE SET", "PRIVACY SET", "DUMMY PULL"])), str(rng.choice(["CLOSER", "WALL STOP", "FLOOR STOP"])),
                 str(rng.choice(["WEATHERSTRIP", "SILENCERS", "-"]))]
        ax.text(x0, y, f"SET {hs}:  " + ", ".join(items), fontsize=fs, **FONT)


def schedule_problem(rng, i, out):
    n_units = int(rng.integers(4, 7))
    names = [f"{a}{b}" for a in "123" for b in "ABC"][: n_units] if i % 2 == 0 else [f"{a}.{b}" for a in "1234" for b in "12"][: n_units]
    units = {u: schedule_rows(rng, int(rng.integers(4, 10))) for u in names}
    name = f"schedule_{i + 1}"
    blocks, (xa, xb), dpi, fig_h = draw_schedule_sheet(units, out / f"{name}.png", f"A-60{i + 1}")
    from PIL import Image
    im = Image.open(out / f"{name}.png")
    asked = [str(rng.choice(names))]
    crops = {}
    for u in asked:
        top, bottom = blocks[u]
        box = (int(xa * dpi), int((fig_h - top) * dpi), int(xb * dpi), int((fig_h - bottom) * dpi))
        crop = im.crop(box); crop = crop.resize((crop.width * 2, crop.height * 2), Image.LANCZOS)
        crops[u] = f"{name}_unit_{u}_crop.png"; crop.save(out / crops[u])
    u = asked[0]
    key = [{k: r[k] for k in ("mark", "type", "width", "height", "hw", "location")} for r in units[u]]
    return {"id": name, "family": "schedule", "image": f"{name}.png", "crop": crops[u], "unit": u, "units_on_sheet": names,
            "text": f"List every door of UNIT TYPE {u} in the door schedule.", "key": key}


# ============================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "data" / "practice"))
    ap.add_argument("--n", type=int, default=5); ap.add_argument("--seed", type=int, default=4644)
    a = ap.parse_args()
    out = Path(a.out); (out / "drawings").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    problems = []
    for i in range(a.n):
        problems += [foundation_problem(rng, i, out / "drawings"), roof_problem(rng, i, out / "drawings"), schedule_problem(rng, i, out / "drawings")]
    (out / "problems.json").write_text(json.dumps(problems, indent=1))
    print(f"{len(problems)} problems -> {out}")
    for p in problems:
        print(" ", p["id"], p.get("shape", ""), "| key:", [(k["quantity"], k["value"]) for k in p["key"]][:4] if p["family"] != "schedule" else f"unit {p['unit']}, {len(p['key'])} doors")
