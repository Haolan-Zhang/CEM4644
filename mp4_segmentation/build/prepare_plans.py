"""Instructor-side: build the plan sets from the CubiCasa5K archive.

    python build/prepare_plans.py --zip /path/to/cubicasa5k.zip
    python build/prepare_plans.py --dir /path/to/cubicasa5k        (the extracted archive)

For every plan listed in aec_seg/config.py this script takes the plan drawing at 1 pixel = 1 cm, either
the dataset's scanned image (F1_scaled.png) or, for sets with render=True, a clean rendering of the
dataset's own vector drawing (model.svg: uniform walls, light-blue windows, door arcs, fixture symbols;
no dimension strings, no "UNDEFINED" labels), resamples it to a plan-specific scale so that the scale
is not a round number, draws a 5 m scale bar under the plan, and writes

    data/plans/<set>/<plan_id>.png      the drawing students see
    data/plans/<set>/<plan_id>.key.json  the answer key: rooms (type, name, real area, polygon), doors,
                                         windows, fixtures, walls, scale, scale-bar position
    data/plans/<set>/credits.json

Source: CubiCasa5K (Kalervo et al., 2019), https://zenodo.org/records/2613548, CC BY-NC-SA 4.0.
"""
import argparse
import io
import json
import re
import sys
import zipfile
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from aec_seg.config import SETS, CUBICASA  # noqa: E402

NS = "{http://www.w3.org/2000/svg}"
ROOM_TYPES = {  # CubiCasa "Space ..." classes -> the room type students see
    "LivingRoom": "living room", "Bedroom": "bedroom", "Kitchen": "kitchen", "Bath": "bathroom", "Sauna": "sauna",
    "Entry": "hall", "Entry Lobby": "hall", "DraughtLobby": "hall", "Storage": "storage", "Dining": "dining room",
    "Outdoor": "balcony / terrace", "Outdoor Balcony": "balcony / terrace", "Garage": "garage", "CarPort": "garage",
    "DressingRoom": "walk-in closet", "Undefined": "other room", "Alcove": "alcove", "Office": "office",
    "Utility": "utility room", "Laundry": "utility room", "Corridor": "corridor", "Hallway": "corridor",
    "Toilet": "toilet room", "Closet": "closet", "TechnicalRoom": "utility room", "Recreation": "recreation room",
    "Hall": "hall", "Bath Shower": "bathroom", "Bath Sauna": "sauna", "Closet WalkIn": "walk-in closet", "Utility Laundry": "utility room",
    "Hall Corridor": "corridor", "Outdoor Terrace": "balcony / terrace", "Outdoor Yard": "balcony / terrace", "Library": "office",
    "Room": "room", "Basement": "basement", "Attic": "attic", "Dining Room": "dining room", "Living Room": "living room",
}


def room_type(raw: str) -> str:
    """CubiCasa's class tokens after 'Space ' (e.g. 'Bath Shower', 'Closet WalkIn') -> the room type students see."""
    if raw in ROOM_TYPES:
        return ROOM_TYPES[raw]
    first = raw.split()[0] if raw.split() else raw
    return ROOM_TYPES.get(first, raw.lower())
INDOOR_EXCLUDE = ("balcony / terrace", "garage")     # not counted in the floor area
FIXTURES = {"Toilet": "toilet", "Sink": "sink", "Bathtub": "bathtub", "Shower": "shower", "Stairs": "stairs",
            "Stove": "stove", "Refrigerator": "refrigerator", "WashingMachine": "washing machine", "Fireplace": "fireplace"}


def _poly(points: str) -> np.ndarray:
    pairs = re.findall(r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)", points or "")
    return np.array([[float(x), float(y)] for x, y in pairs]) if pairs else np.zeros((0, 2))


def _area(p: np.ndarray) -> float:
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def _bbox(p: np.ndarray):
    if p.ndim != 2 or len(p) < 2:
        return None
    return [float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max())]


def _transform(t: Optional[str]) -> np.ndarray:
    """An SVG transform attribute as a 3x3 matrix (matrix / translate / scale / rotate, left to right)."""
    M = np.eye(3)
    for name, args in re.findall(r"(matrix|translate|scale|rotate)\(([^)]*)\)", t or ""):
        v = [float(x) for x in re.split(r"[\s,]+", args.strip()) if x]
        if name == "matrix" and len(v) == 6:
            A = np.array([[v[0], v[2], v[4]], [v[1], v[3], v[5]], [0, 0, 1]])
        elif name == "translate":
            A = np.array([[1, 0, v[0]], [0, 1, v[1] if len(v) > 1 else 0.0], [0, 0, 1]])
        elif name == "scale":
            sx = v[0]; sy = v[1] if len(v) > 1 else sx
            A = np.diag([sx, sy, 1.0])
        elif name == "rotate":
            a = np.radians(v[0]); c, s_ = np.cos(a), np.sin(a)
            A = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])
            if len(v) == 3:
                T = np.array([[1, 0, v[1]], [0, 1, v[2]], [0, 0, 1]]); Ti = np.array([[1, 0, -v[1]], [0, 1, -v[2]], [0, 0, 1]])
                A = T @ A @ Ti
        else:
            continue
        M = M @ A
    return M


def _apply(M: np.ndarray, p: np.ndarray) -> np.ndarray:
    if p.ndim != 2 or len(p) == 0:
        return p
    q = M @ np.vstack([p.T, np.ones(len(p))])
    return q[:2].T


def _first_polygon(el, M: np.ndarray):
    """The first <polygon> below el, in the global frame (transforms composed along the way)."""
    for child in el:
        M2 = M @ _transform(child.get("transform"))
        if child.tag == NS + "polygon":
            return _apply(M2, _poly(child.get("points")))
        found = _first_polygon(child, M2)
        if found is not None:
            return found
    return None


def _child_polygon(g, M: np.ndarray):
    poly = g.find(NS + "polygon")
    if poly is None:
        return None
    return _apply(M @ _transform(poly.get("transform")), _poly(poly.get("points")))


def parse_svg(svg_text: str) -> dict:
    root = ET.fromstring(svg_text)
    rooms, doors, windows, walls, fixtures = [], [], [], [], []

    def walk(el, M):
        for g in el:
            if g.tag != NS + "g":
                continue
            M2 = M @ _transform(g.get("transform"))
            cls = g.get("class", "")
            if cls.startswith("Space "):
                p = _child_polygon(g, M2)
                if p is not None and len(p) >= 3:
                    name, dims = None, None
                    for t in g.iter(NS + "text"):
                        s = "".join(t.itertext()).strip()
                        m = re.match(r"([\d.]+) m x ([\d.]+) m", s)
                        f = re.match(r"(\d+)'(\d+)\" x (\d+)'(\d+)\"", s)
                        if m:
                            dims = (float(m.group(1)), float(m.group(2)))
                        elif f and dims is None:
                            dims = ((int(f.group(1)) * 12 + int(f.group(2))) * 0.0254, (int(f.group(3)) * 12 + int(f.group(4))) * 0.0254)
                        elif name is None and s and not re.search(r"\d", s):
                            name = s
                    if name and name.upper() == "UNDEFINED":
                        name = ""
                    raw = cls.split(" ", 1)[1]
                    rooms.append({"raw_type": raw, "type": room_type(raw), "label": name or "", "dims_m": dims, "poly": p})
            elif cls == "Door" or cls.startswith("Door "):
                b = _bbox(_child_polygon(g, M2)) if _child_polygon(g, M2) is not None else None
                if b: doors.append(b)
            elif cls.startswith("Window"):
                b = _bbox(_child_polygon(g, M2)) if _child_polygon(g, M2) is not None else None
                if b: windows.append(b)
            elif cls in ("Wall", "Wall External"):
                w_ = _child_polygon(g, M2)
                if w_ is not None and w_.ndim == 2 and len(w_) >= 3: walls.append(w_)
            elif cls.startswith("FixedFurnitureSet"):
                walk(g, M2)                   # a kitchen or bathroom set: its parts are the fixtures
                continue
            elif cls.startswith("FixedFurniture") or cls == "Stairs":
                # the symbol's outline is the first polygon below the group (its BoundaryPolygon / Flight child), in local
                # coordinates: the group and its parents carry the rotation and the position
                tokens = cls.split()
                kind = next((v for k, v in FIXTURES.items() if any(k in tok for tok in tokens)), None)
                inner = _first_polygon(g, M2) if kind else None
                b = _bbox(inner) if inner is not None else None
                if b: fixtures.append({"type": kind, "box": b})
                continue                      # a set's children are its parts: do not descend
            walk(g, M2)

    walk(root, _transform(root.get("transform")))
    # scale of the SVG frame: metres per unit, from every labelled room (should be 0.01 exactly)
    ests = []
    for r in rooms:
        if r["dims_m"] and min(r["dims_m"]) > 0.5:
            b = _bbox(r["poly"]); w, h = b[2] - b[0], b[3] - b[1]
            if w > 5 and h > 5:
                ests += [r["dims_m"][0] / w, r["dims_m"][1] / h]
    m_per_unit = float(np.median(ests)) if ests else 0.01
    return {"rooms": rooms, "doors": doors, "windows": windows, "walls": walls, "fixtures": fixtures, "m_per_unit": m_per_unit}


class Source:
    """The CubiCasa5K files, from the Zenodo zip or from the extracted folder."""

    def __init__(self, zip_path: Optional[str], dir_path: Optional[str]):
        self.z = zipfile.ZipFile(zip_path) if zip_path else None
        self.d = Path(dir_path) if dir_path else None
        if self.z is None and self.d is None:
            raise SystemExit("give --zip cubicasa5k.zip or --dir <extracted folder>")

    def read(self, pid: str, name: str) -> bytes:
        rel = f"high_quality_architectural/{pid}/{name}"
        if self.z is not None:
            return self.z.read("cubicasa5k/" + rel)
        return (self.d / rel).read_bytes()


def render_svg(svg_bytes: bytes) -> Image.Image:
    """The vector drawing as an image at its native size (1 unit = 1 px = 1 cm), white background.
    Text elements that only say UNDEFINED (a room without a name) are dropped."""
    from cairosvg import svg2png
    txt = svg_bytes.decode("utf-8")
    txt = re.sub(r"<text\b(?:(?!</text>).)*?>\s*UNDEFINED\s*</text>", "", txt, flags=re.S)
    png = svg2png(bytestring=txt.encode("utf-8"), background_color="white")
    return Image.open(io.BytesIO(png)).convert("RGB")


def draw_scale_bar(img: Image.Image, px_per_m: float, metres: int = 5):
    """Adds a white strip under the drawing with a labelled scale bar. Returns (image, bar box)."""
    pad = 70
    canvas = Image.new("RGB", (img.width, img.height + pad), "white")
    canvas.paste(img, (0, 0))
    d = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    L = metres * px_per_m
    x0 = max(20, min(img.width - L - 20, 40)); y0 = img.height + 28
    for k in range(metres):
        fill = "black" if k % 2 == 0 else "white"
        d.rectangle([x0 + k * px_per_m, y0, x0 + (k + 1) * px_per_m, y0 + 10], fill=fill, outline="black")
        d.text((x0 + k * px_per_m - 4, y0 + 13), str(k), fill="black", font=font)
    d.text((x0 + L - 4, y0 + 13), f"{metres} m", fill="black", font=font)
    d.text((x0, y0 - 22), "scale", fill="black", font=font)
    return canvas, [x0, y0, x0 + L, y0 + 10]


def build_plan(src: Source, pid: str, factor: float, out_dir: Path, meta: dict, render: bool = False) -> dict:
    svg = src.read(pid, "model.svg")
    img = render_svg(svg) if render else Image.open(io.BytesIO(src.read(pid, "F1_scaled.png"))).convert("RGB")
    key = parse_svg(svg.decode("utf-8"))
    # resample: CubiCasa's scaled image is 1 unit = 1 px = 1 cm; after resampling 1 px = (1/factor) cm
    W, H = round(img.width * factor), round(img.height * factor)
    img = img.resize((W, H), Image.LANCZOS)
    m_per_px = key["m_per_unit"] / factor
    px_per_m = 1 / m_per_px
    img, bar = draw_scale_bar(img, px_per_m)

    def sc(p):
        return (np.asarray(p, dtype=float) * factor).round(1).tolist()
    rooms = []
    for r in key["rooms"]:
        poly = np.asarray(r["poly"]) * factor
        area = _area(poly) * m_per_px ** 2
        if area < 0.5:
            continue
        rooms.append({"type": r["type"], "raw_type": r["raw_type"], "label": r["label"], "dims_m": r["dims_m"],
                      "area_m2": round(float(area), 2), "poly": poly.round(1).tolist(), "box": [round(v, 1) for v in _bbox(poly)]})
    indoor = [r for r in rooms if r["type"] not in INDOOR_EXCLUDE]
    out = {
        "id": pid, "source": "CubiCasa5K high_quality_architectural/" + pid, "image": f"{pid}.png",
        "drawing": "vector rendering (model.svg)" if render else "scanned drawing (F1_scaled.png)",
        "size": [img.width, img.height], "drawing_size": [W, H], "scale_m_per_px": round(m_per_px, 6),
        "scale_bar": {"box": [round(v, 1) for v in bar], "metres": 5},
        "rooms": rooms, "doors": [sc(b) for b in key["doors"]], "windows": [sc(b) for b in key["windows"]],
        "fixtures": [{"type": f["type"], "box": sc(f["box"])} for f in key["fixtures"]],
        "walls": [sc(w) for w in key["walls"]],
        "floor_area_m2": round(sum(r["area_m2"] for r in indoor), 1), "n_rooms": len(indoor),
        "n_doors": len(key["doors"]), "n_windows": len(key["windows"]),
        "title": meta.get("title", f"plan {pid}"), "note": meta.get("note", ""),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    img.save(out_dir / f"{pid}.png", optimize=True)
    (out_dir / f"{pid}.key.json").write_text(json.dumps(out))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=None, help="cubicasa5k.zip from Zenodo")
    ap.add_argument("--dir", default=None, help="the extracted archive (folder containing high_quality_architectural/)")
    ap.add_argument("--sets", nargs="*", default=None)
    a = ap.parse_args()
    src = Source(a.zip, a.dir)
    for key, spec in SETS.items():
        if a.sets and key not in a.sets:
            continue
        out_dir = REPO / spec.folder
        credits = []
        for pid, meta in spec.plans.items():
            k = build_plan(src, pid, meta["factor"], out_dir, meta, render=spec.render)
            credits.append({"id": pid, "file": f"{pid}.png", "title": meta["title"], "author": CUBICASA["author"],
                            "license": CUBICASA["license"], "source": CUBICASA["source"], "note": f"CubiCasa5K sample {pid}; " + meta.get("note", ""),
                            "size": k["size"]})
            print(f"{key}/{pid}: {k['size']}, {k['n_rooms']} rooms, {k['floor_area_m2']} m², {k['n_doors']} doors, {k['n_windows']} windows, "
                  f"{len(k['fixtures'])} fixtures, 1 px = {k['scale_m_per_px'] * 100:.2f} cm")
        (out_dir / "credits.json").write_text(json.dumps(credits, indent=2, ensure_ascii=False))
    print("done")


if __name__ == "__main__":
    main()
