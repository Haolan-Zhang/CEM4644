"""Every step of the take-off lab. Each function displays something; nothing is returned to the notebook.

The logic of the interactive steps lives in plain functions (`takeoff_compute`, `intro_prompts_compute`) so
that it can be tested without driving a widget.
"""
import io
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

from . import viz
from .config import Thing
from .data import plural
from .engine import SegResult

GREEN, RED, BLUE, PURPLE = "#2a9d8f", "#e76f51", "#457b9d", "#7b2cbf"      # found / missed / not one / your example
LEGEND = "green = a real one the model found, red outline = a real one it missed, blue = a region that is not one"
AREA_FILL = "#f4a261"     # every measured area in one colour: a number, not a verdict
AREA_COLORS = ["#e63946", "#4cc9f0", "#ffd166", "#06d6a0", "#7b2cbf", "#f4a261", "#457b9d", "#2a9d8f",
               "#e9c46a", "#8d6e63", "#ff70a6", "#9aa5b1"]

# A drawn thing smaller than this (pixels across) is below what SAM 3 can trace: it works internally at about
# 1008 px on the long side, so a 60 px symbol on a 2400 px sheet is already only 25 px to the model.
MIN_SYMBOL_PX = 60


# --------------------------------------------------------------------------- box arithmetic
def box_iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def centre_in(p, t) -> bool:
    cx, cy = (p[0] + p[2]) / 2, (p[1] + p[3]) / 2
    return t[0] <= cx <= t[2] and t[1] <= cy <= t[3]


def box_hit(truth, pred, iou: float = 0.2) -> bool:
    """A predicted box counts for a key box when they overlap enough, or when its centre is inside it and it is at
    least a quarter of its size (small symbols: a box a few pixels off has a low IoU but is clearly the same thing;
    a small part drawn inside a large symbol - the circle inside a light fixture - is not the symbol)."""
    if box_iou(truth, pred) >= iou:
        return True
    area = lambda b: max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])   # noqa: E731
    return centre_in(pred, truth) and area(pred) >= 0.25 * area(truth)


def match_boxes(pred: Sequence[Sequence[float]], truth: Sequence[Sequence[float]], iou: float = 0.2):
    """(found, missed, extra, found_flags, extra_flags)."""
    found = [any(box_hit(t, p, iou) for p in pred) for t in truth]
    extra = [not any(box_hit(t, p, iou) for t in truth) for p in pred]
    return int(sum(found)), int(len(truth) - sum(found)), int(sum(extra)), found, extra


# --------------------------------------------------------------------------- pixels
def wall_pixels(img) -> np.ndarray:
    """Thick dark strokes of a drawing (the walls), not thin lines or text: dark pixels that survive a 7x7 opening."""
    from scipy import ndimage
    dark = np.asarray(img.convert("L")) < 100
    return ndimage.binary_opening(dark, structure=np.ones((7, 7)))


def _hull_points(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Convex hull (monotone chain) of a boolean mask, from the left-most and right-most pixel of every row."""
    rows = np.nonzero(mask.any(1))[0]
    if len(rows) == 0:
        return []
    pts = []
    for y in rows:
        xs = np.nonzero(mask[y])[0]
        pts.append((int(xs[0]), int(y)))
        if xs[-1] != xs[0]:
            pts.append((int(xs[-1]), int(y)))
    pts = sorted(set(pts))
    if len(pts) < 3:
        return pts

    def half(points):
        out = []
        for p in points:
            while len(out) >= 2 and ((out[-1][0] - out[-2][0]) * (p[1] - out[-2][1])
                                     - (out[-1][1] - out[-2][1]) * (p[0] - out[-2][0])) <= 0:
                out.pop()
            out.append(p)
        return out

    lower, upper = half(pts), half(pts[::-1])
    return lower[:-1] + upper[:-1]


def _bbox_mask(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    out = np.zeros_like(mask)
    out[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = True
    return out


def _hull_mask(mask: np.ndarray) -> np.ndarray:
    pts = _hull_points(mask)
    if len(pts) < 3:
        return mask
    im = Image.new("1", (mask.shape[1], mask.shape[0]), 0)
    ImageDraw.Draw(im).polygon(pts, fill=1)
    return np.asarray(im, dtype=bool)


def fill_bites(mask: np.ndarray, rule: str = "hull") -> np.ndarray:
    """Give back the bites that door swings and furniture symbols take out of a room mask.

    A door swing, a kitchen counter, a bathtub or a fireplace is cut out of the mask, so the raw pixel count
    of a room comes out 6-9 % low on average. The cure is to read the room's OUTLINE instead of its pixels.

    'hull' (what the lab ships) = the convex hull of the mask: it closes every bite the drawing takes out of
    the edge of a room, and where a room really is L-shaped it can add at most half of the missing corner.
    'bbox' = the bounding box: better on a plain rectangle, but it adds the WHOLE missing corner (up to +49 %
    on the L-shaped halls of these sheets). 'raw' = the mask as it comes. The three are compared per sheet in
    docs/Instructor_Guide.md, section 2.
    """
    if rule == "raw" or not mask.any():
        return mask
    return _bbox_mask(mask) if rule == "bbox" else _hull_mask(mask)


AREA_RULE = "hull"       # decided by measuring raw / hull / bbox against the real answer keys


def room_mask(engine, img, box, walls=None, rule: str = AREA_RULE):
    """What the take-off cell does for a room box: SAM 3 with the phrase 'empty room' AND the box, the result
    clipped to the box, holes filled, walls removed, then the bites given back.
    Returns (mask, raw_pixel_count, score, filled_pixel_count)."""
    from scipy import ndimage
    r = engine.segment_room(img, box)
    if len(r) == 0:
        return np.zeros((img.height, img.width), bool), 0, 0.0, 0
    m = r.masks[0].copy()
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    clip = np.zeros_like(m)
    clip[max(0, y1):y2 + 1, max(0, x1):x2 + 1] = True
    m = ndimage.binary_fill_holes(m & clip)
    if walls is None:
        walls = wall_pixels(img)
    m = m & ~walls
    raw = int(m.sum())
    out = fill_bites(m, rule) & ~walls & clip
    return out, raw, float(r.scores[0]), int(out.sum())


# --------------------------------------------------------------------------- answer-key lines for the named things
def best_area(sheet, mask: np.ndarray) -> Tuple[Optional[dict], float]:
    """The answer-key area that a mask overlaps most (as a fraction of the MASK)."""
    best, frac = None, 0.0
    total = mask.sum()
    if total == 0:
        return None, 0.0
    for a in sheet.areas():
        f = (sheet.area_mask(a) & mask).sum() / total
        if f > frac:
            best, frac = a, float(f)
    return best, frac


def words(n: int, cat: str) -> str:
    """The plural word alone: 'pile footings', 'rooms'."""
    return plural(2, cat).split(" ", 1)[1] if n != 1 else cat


def truth_line(sheet, thing: Thing, res: SegResult, threshold: float) -> str:
    """What SAM 3 returned against the real things on the drawing, in the student's words."""
    if thing.truth is None:
        return (f"Nothing on this drawing to compare '{thing.name}' regions with: look at the picture and judge "
                "them yourself.")
    what, sub = thing.truth
    keep = res.scores >= threshold
    pred = [list(map(float, b)) for b in res.boxes[keep]]
    if what == "areas":
        rooms = sheet.areas("room", sub)
        label = sub or "room"
        if not rooms:
            return f"There is no {label} on this drawing: every region above is not one."
        ok, missed, extra, _, _ = match_boxes(pred, [a["box"] for a in rooms], iou=0.3)
        return compare_text(len(rooms), label, ok, missed, extra)
    truth = sheet.truth_boxes("counts", sub)
    if not truth:
        return f"Nothing on this drawing to compare '{sub}' regions with: look at the picture and judge them yourself."
    ok, missed, extra, _, _ = match_boxes(pred, truth, iou=0.2)
    return compare_text(len(truth), sub, ok, missed, extra)


def compare_text(n_true: int, cat: str, ok: int, missed: int, extra: int) -> str:
    w = words(n_true, cat)
    return (f"Compared with the {n_true} actual {w}:\n  Found: {ok} of {n_true}\n  Missed: {missed}\n"
            f"  Extra: {extra} region{'' if extra == 1 else 's'} that {'was not a ' + cat if extra == 1 else 'were not ' + w}")


def result_text(res: SegResult, threshold: float) -> str:
    keep = res.scores >= threshold
    n = int(keep.sum())
    if n == 0:
        return (f"SAM 3 segmented nothing at the selected confidence threshold"
                + (f" (it proposed {len(res)} weak region{'' if len(res) == 1 else 's'} below it)." if len(res) else "."))
    return (f"SAM 3 segmented {n} region{'' if n == 1 else 's'} at the selected confidence threshold "
            "(confidence of each: " + ", ".join(f"{s * 100:.0f}%" for s in res.scores[keep][:8]) + (", ..." if n > 8 else "") + ").")


def print_result(sheet, thing_name: str, res: SegResult, threshold: float):
    print(f"TEXT PROMPT: '{res.prompt}'")
    print(result_text(res, threshold))


# --------------------------------------------------------------------------- Part 1: the photos
INTRO_PHRASES = ["person", "helmet", "safety vest", "boots", "hose", "rebar", "wet concrete", "hand", "truck", "wheel", "brick wall", "sky", "window"]
INTRO_COLORS = ["#e63946", "#2a9d8f", "#f4a261", "#7b2cbf", "#457b9d", "#e9c46a", "#06d6a0", "#ff70a6"]


def intro_phrase(lab, photo_id: str, phrase: str, threshold: float):
    """SAM 3 asked by name on an ordinary photo: original | mask | overlay, and how sure it is."""
    photo = lab.intro_photos[photo_id]
    img = photo.load()
    phrase = (phrase or "").strip()
    if not phrase:
        print("Type or pick a phrase first."); return
    if lab.engine is None:
        print("SAM 3 is not loaded (run Step 0 with load_model ticked)."); return
    res = lab.engine.segment(img, phrase, threshold=0.1)
    viz.show(viz.three_panel(img, res, "#e63946", f"'{phrase}' on {photo.id}", threshold, what="photo"))
    keep = res.scores >= threshold
    n = int(keep.sum())
    top = ", ".join(f"{v:.2f}" for v in sorted(res.scores[keep], reverse=True)[:8])
    if n:
        print(f"'{phrase}': {n} region(s) at confidence >= {threshold:.2f} (confidences {top}), together "
              f"{res.area_pct(threshold):.1f} % of the photo, in {res.seconds:.1f} s.")
    else:
        below = int((res.scores >= 0.1).sum())
        print(f"'{phrase}': nothing at confidence >= {threshold:.2f}"
              + (f"; {below} weak region(s) between 0.10 and {threshold:.2f}: lower the slider to see them." if below else ". Try other words.")
              + f" ({res.seconds:.1f} s)")
    print(f"Photo: {photo.title}. {photo.author}, {photo.license}.")


def intro_prompts_compute(lab, img, prompts):
    """prompts: [(label, [x1, y1, x2, y2])] in full-image pixels; 'point' uses the centre of the drawn box as the click."""
    layers, lines, marks = [], [], []
    for i, (label, b) in enumerate(prompts):
        color = INTRO_COLORS[i % len(INTRO_COLORS)]
        if label == "point":
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            res = lab.engine.segment_visual(img, point=(cx, cy))
            what = f"a click at ({cx:.0f}, {cy:.0f})"
            marks.append(("point", (cx, cy), color))
        else:
            res = lab.engine.segment_visual(img, box=b)
            what = f"a box {[int(v) for v in b]}"
            marks.append(("box", b, color))
        m = res.masks[0]
        layers.append((str(i + 1), m, color))
        lines.append(f"{i + 1}. {what}: one object covering {m.mean() * 100:.1f} % of the photo "
                     f"(the model's own confidence {res.scores[0]:.2f}).")
    over = viz.multi_overlay(img, layers, alpha=0.55) if layers else img.copy()
    d = ImageDraw.Draw(over)
    for kind, g, color in marks:
        if kind == "point":
            cx, cy = g
            r = max(6, img.width // 120)
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color, outline="white", width=2)
        else:
            d.rectangle(g, outline=color, width=3)
    return over, lines


def intro_draw(lab, photo_id: str):
    """Draw boxes (label 'box') or tap objects (label 'point') on the photo; SAM 3 cuts each one out."""
    import ipywidgets as w
    from IPython.display import display
    from jupyter_bbox_widget import BBoxWidget
    photo = lab.intro_photos[photo_id]
    img = photo.load()
    if lab.engine is None:
        print("SAM 3 is not loaded (run Step 0 with load_model ticked)."); return
    disp = img.copy(); disp.thumbnail((1000, 1000))
    scale = img.width / disp.width
    buf = io.BytesIO(); disp.save(buf, "PNG")
    widget = BBoxWidget(classes=["box", "point"])
    widget.image_bytes = buf.getvalue()
    msg = w.HTML("Draw a <b>box</b> around an object, or a tiny box on it labelled <b>point</b> (its centre is the click). "
                 "Several are fine. Then <b>Submit</b>.")
    out = w.Output()

    @widget.on_submit
    def _go():
        prompts = [(b.get("label", "box"), [b["x"] * scale, b["y"] * scale, (b["x"] + b["width"]) * scale, (b["y"] + b["height"]) * scale])
                   for b in widget.bboxes]
        if not prompts:
            msg.value = "Draw at least one box first."; return
        msg.value = "Running SAM 3..."
        over, lines = intro_prompts_compute(lab, img, prompts)
        with out:
            out.clear_output(wait=True)
            viz.show_image(over, 900)
            print("\n".join(lines))
            print("No words were used: the model was given a place, and returned the object that a person would mean by it.")
        msg.value = "Done. Draw more, or redraw, and submit again."

    display(w.VBox([msg, widget, out]))


# --------------------------------------------------------------------------- Part 1 / Part 1 homework: the drawings
def gallery(lab, ids: Optional[Sequence[str]] = None, ncols: int = 2, size: float = 5.0, title: Optional[str] = None,
            tasks: bool = False):
    sel = [lab.sheets[i] for i in ids] if ids else list(lab.sheets)
    ims = [s.load() for s in sel]
    for im in ims:
        im.thumbnail((1100, 1100))
    titles = [f"{s.id} - {s.title}\n{s.discipline}; {s.facts()}" for s in sel]
    viz.show(viz.image_grid(ims, titles, ncols=ncols, size=size, suptitle=title))
    if not getattr(lab.spec, "guided", True):
        return
    for s in sel:
        print(f"\n{s.id} - {s.title}")
        print(f"  {s.discipline}. {s.credit.get('author', '')}, {s.credit.get('license', '')}.")
        print(f"  Scale: {s.scale_how}")
        if tasks and s.tasks:
            print("  Your take-off on this sheet:")
            print(s.task_text())
        if s.key.get("legend"):
            print(f"  A legend of the symbols is shipped with this sheet: run the legend step to see it.")


def show_legend(lab, sheet_id: Optional[str] = None):
    """The symbol legend shipped next to a sheet (the MEP sheets have one)."""
    found = False
    for s in lab.sheets:
        if sheet_id and s.id != sheet_id:
            continue
        p = s.legend_path
        if p and p.exists():
            found = True
            print(f"{s.id} - {s.title}: legend of the symbols on this sheet")
            viz.show_image(Image.open(p).convert("RGB"), 900)
    if not found:
        print("None of these drawings ships a legend image. On the drawings without one the symbols are named "
              "on the sheet itself (a schedule, a mark such as F4.0, or a note).")


# --------------------------------------------------------------------------- Part 2 workshop: ask by name
def segment_view(lab, sheet_id: str, thing_name: str, threshold: float):
    """Original | mask | overlay for one thing on one drawing, with a live confidence slider and the answer key."""
    import ipywidgets as w
    from IPython.display import display
    sheet = lab.sheets[sheet_id]
    t = lab.spec.thing(thing_name)
    img = sheet.load()
    res = lab.get(sheet.id, t)
    sl = w.FloatSlider(value=threshold, min=0.1, max=0.9, step=0.05, description="confidence >=", continuous_update=False, readout_format=".2f")
    out = w.Output()

    def render(*_):
        with out:
            out.clear_output(wait=True)
            viz.show(viz.three_panel(img, res, t.color, f"{sheet.id}: {sheet.title} - '{t.prompt}'", sl.value, what="drawing"))
            print_result(sheet, t.name, res, sl.value)
            print(truth_line(sheet, t, res, sl.value))

    sl.observe(render, names="value")
    display(w.VBox([sl, out]))
    render()


def count_view(lab, sheet_id: str, thing_name: str, threshold: float):
    """Hits, misses and extras drawn on the drawing: green = a real one found, red = a real one missed, blue = extra."""
    sheet = lab.sheets[sheet_id]
    t = lab.spec.thing(thing_name)
    if t.truth is None:
        print(f"Nothing on the drawings to compare '{t.name}' regions with, so found and missed cannot be drawn. "
              "Pick a room type."); return
    what, sub = t.truth
    truth = [a["box"] for a in sheet.areas("room", sub)] if what == "areas" else sheet.truth_boxes("counts", sub)
    label = sub or "room"
    if not truth:
        print(f"There is no {label} on this drawing, so found and missed cannot be drawn. Pick a room type here, "
              "and look at the picture for the rest."); return
    res = lab.get(sheet.id, t)
    keep = [int(i) for i in np.where(res.scores >= threshold)[0]]
    pred = [list(map(float, res.boxes[i])) for i in keep]
    iou = 0.3 if what == "areas" else 0.2
    ok, missed, extra, found_flags, extra_flags = match_boxes(pred, truth, iou=iou)
    img = viz.judged_image(sheet.load(), [res.masks[i] for i in keep], [not e for e in extra_flags],
                           [tb for tb, f in zip(truth, found_flags) if not f])
    viz.show_image(img, 1000)
    print(f"TEXT PROMPT: '{t.prompt}'"); print(result_text(res, threshold)); print(compare_text(len(truth), label, ok, missed, extra))
    print(LEGEND)


# --------------------------------------------------------------------------- Part 4 workshop: where it goes wrong
def phrase_lab(lab, sheet_id: str, thing_name: str, threshold: float):
    """The same thing asked for with different words."""
    sheet = lab.sheets[sheet_id]
    t = lab.spec.thing(thing_name)
    img = sheet.load()
    ims, titles = [], []
    for phrase in [t.prompt] + list(t.alternatives):
        res = lab.get_phrase(sheet.id, phrase)
        n = int((res.scores >= threshold).sum())
        ims.append(viz.overlay(img, res.union(threshold), t.color))
        titles.append(f"'{phrase}'\n{n} region(s), {sheet.sqft(res.union(threshold).sum()):,.0f} sq ft")
    viz.show(viz.image_grid(ims, titles, ncols=2, size=5.0,
                            suptitle=f"{sheet.id}: the same thing, different words (confidence >= {threshold * 100:.0f}%)"))
    print("With the main wording: " + truth_line(sheet, t, lab.get(sheet.id, t), threshold))


def inspector(lab, sheet_id: str, thing_name: str):
    """Every region on its own, with its confidence; a slider removes the weak ones."""
    import ipywidgets as w
    from IPython.display import display
    sheet = lab.sheets[sheet_id]
    t = lab.spec.thing(thing_name)
    img = sheet.load()
    res = lab.get(sheet.id, t)
    sl = w.FloatSlider(value=0.4, min=0.1, max=0.9, step=0.05, description="confidence >=", continuous_update=False, readout_format=".2f")
    out = w.Output()

    def render(*_):
        with out:
            out.clear_output(wait=True)
            viz.show_image(viz.instances_image(img, res, sl.value), 1000)
            keep = res.scores >= sl.value
            print(f"'{t.prompt}': {int(keep.sum())} region(s) kept of {len(res)} proposed.")
            for k, i in enumerate(np.where(keep)[0]):
                m = res.masks[i]
                area, frac = best_area(sheet, m)
                where = (f"mostly on the {area['type']} '{area['label']}' ({area['true_sqft']:,.0f} sq ft)"
                         if area and frac > 0.5 else "not on any one room of the drawing")
                print(f"  #{k + 1}: confidence {res.scores[i] * 100:.0f}%, {where}")

    sl.observe(render, names="value")
    display(w.VBox([sl, out]))
    render()


def live_phrase(lab, sheet_id: str, phrase: str, threshold: float):
    if lab.engine is None:
        print("SAM 3 is not loaded. Re-run Step 0 with a GPU runtime for live phrases."); return
    sheet = lab.sheets[sheet_id]
    img = sheet.load()
    res = lab.engine.segment(img, phrase, threshold=0.1)
    viz.show(viz.three_panel(img, res, "#ff70a6", f"{sheet.id} - your phrase: '{phrase}'", threshold, what="drawing"))
    print_result(sheet, phrase, res, threshold)
    print(f"(SAM 3 took {res.seconds:.1f} s)")


def phrase_check(lab, sheet_id: str, phrase: str, threshold: float):
    """Homework: a phrase typed straight to SAM 3 on any drawing, scored against the sheet's own answer key.

    No box from the student: the words alone. The regions are scored against the category of the key they match
    best (the rooms of a floor plan, the footings of a foundation plan, a fixture type of the ceiling plan); when
    they match nothing, against the sheet's main category, so the miss is stated in the task's own terms. For an
    area category the area of every region found is measured too. Kept in lab.phrases for the report summary."""
    sheet = lab.sheets[sheet_id]
    phrase = (phrase or "").strip()
    if not phrase:
        print("Type a phrase first."); return
    if lab.engine is None and not (lab.store and lab.store.has(sheet.id, phrase)):
        print("SAM 3 is not loaded. Re-run Step 0 with load_model ticked (and a GPU runtime)."); return
    res = lab.get_phrase(sheet.id, phrase)
    img = sheet.load()
    idx = [int(i) for i in np.where(res.scores >= threshold)[0]]
    pred = [list(map(float, res.boxes[i])) for i in idx]
    cats = sheet.area_categories + [c for c in sheet.count_categories if c not in sheet.area_categories]
    cat = cats[0] if cats else None
    if cats and pred:                                   # the category the regions match best, if any
        best, best_hits = None, 0
        for c in cats:
            is_area = c in sheet.area_categories
            truth = [a["box"] for a in sheet.areas(c)] if is_area else sheet.counts[c]
            ok, *_ = match_boxes(pred, truth, iou=0.3 if is_area else 0.2)
            if ok > best_hits:
                best, best_hits = c, ok
        if best:
            cat = best
    rec = {"sheet": sheet.id, "phrase": phrase, "threshold": float(threshold), "regions": len(pred), "proposed": len(res),
           "compare": cat}
    from IPython.display import HTML, display
    if cat is None:
        viz.show(viz.three_panel(img, res, "#ff70a6", f"{sheet.id} - '{phrase}'", threshold, what="drawing"))
        display(HTML(phrase_html(phrase, res, threshold, None)))
    else:
        is_area = cat in sheet.area_categories
        truth_items = sheet.areas(cat) if is_area else [{"box": b} for b in sheet.counts[cat]]
        truth = [list(t["box"]) for t in truth_items]
        iou = 0.3 if is_area else 0.2
        ok, n_missed, n_extra, found_flags, extra_flags = match_boxes(pred, truth, iou=iou)
        over = viz.judged_image(img, [res.masks[i] for i in idx], [not e for e in extra_flags],
                                [tb for tb, f in zip(truth, found_flags) if not f])
        viz.show_image(over, 1000)
        display(HTML(phrase_html(phrase, res, threshold, (len(truth), cat, ok, n_missed, n_extra))))
        rec.update(truth=len(truth), found=ok, missed=n_missed, extra=n_extra)
        if is_area and ok:
            rows = []
            for t in truth_items:
                best, bi = 0.0, -1
                for j, i in enumerate(idx):
                    v = box_iou(t["box"], pred[j])
                    if v > best:
                        best, bi = v, i
                if bi >= 0 and box_hit(t["box"], list(map(float, res.boxes[bi])), iou):
                    mine = sheet.sqft(int(res.masks[bi].sum()))
                    tru = float(t["true_sqft"])
                    rows.append((t.get("label") or t.get("type") or cat, mine, tru, (mine - tru) / tru * 100))
            if rows:
                print(f"\n  Each {cat} the phrase found, measured as the region comes (no box of yours, no clean-up):")
                print(f"  {'':24} {'phrase sq ft':>12} {'drawing sq ft':>13} {'error':>7}")
                for name, mine, tru, e in rows[:15]:
                    print(f"  {str(name)[:24]:24} {mine:12,.1f} {tru:13,.1f} {e:+6.0f} %")
                if len(rows) > 15:
                    print(f"  ... and {len(rows) - 15} more")
                errs = [abs(e) for *_, e in rows]
                print(f"  Median error {np.median(errs):.0f} %, worst {max(errs):.0f} %, on {len(rows)} {cat}(s).")
                rec.update(median_err=float(np.median(errs)), worst_err=float(max(errs)), measured=len(rows))
    print(f"(SAM 3 took {res.seconds:.1f} s)")
    lab.phrases[(sheet.id, phrase.lower())] = rec
    return rec


def phrase_html(phrase: str, res: SegResult, threshold: float, cmp) -> str:
    """The phrase cell's result: the prompt, what was segmented, and (when the drawing has such things) the comparison."""
    import html
    keep = res.scores >= threshold
    n = int(keep.sum())
    lines = [f"<b>TEXT PROMPT</b>: \u201c{html.escape(phrase)}\u201d"]
    if n == 0:
        lines.append("SAM 3 segmented <b>no region</b> at the selected confidence threshold"
                     + (f" (it proposed {len(res)} weak region{'' if len(res) == 1 else 's'} below it)." if len(res) else "."))
    else:
        lines.append(f"SAM 3 segmented <b>{n} region{'' if n == 1 else 's'}</b> at the selected confidence threshold.")
    if cmp is None:
        lines.append("Nothing on this drawing to compare them with: look at the picture and judge the regions yourself.")
    else:
        n_true, cat, ok, missed, extra = cmp
        w = words(n_true, cat)
        lines += [f"<b>Compared with the {n_true} actual {html.escape(w)}:</b>",
                  f"<b>Found:</b> {ok} of {n_true}", f"<b>Missed:</b> {missed}",
                  f"<b>Extra:</b> {extra} region{'' if extra == 1 else 's'} that {'was not a ' + html.escape(cat) if extra == 1 else 'were not ' + html.escape(w)}",
                  f"<span style='color:#666'>{LEGEND}</span>"]
    return "<div style='font-family:monospace;white-space:pre-wrap;line-height:1.5'>" + "\n".join(lines) + "</div>"


# --------------------------------------------------------------------------- Part 3 / Part 2: the take-off cell
def _span(box, axis: str) -> float:
    return abs(box[2] - box[0]) if axis == "x" else abs(box[3] - box[1])


def takeoff_compute(lab, sheet_id: str, boxes, threshold: Optional[float] = None, count_categories=None):
    """The whole take-off of one drawing, without any widget.

    boxes:     [(label, [x1, y1, x2, y2])] in full-image pixels; the labels are `Sheet.box_labels()`.
    threshold: the confidence the counts use (None = the value each answer key suggests).
    count_categories: which of the sheet's counting categories to run (None = all of them).

    Nothing here is a tally of the student's own boxes. A COUNT always comes from the model: the student
    draws ONE box labelled 'example: <category>' and SAM 3 (`segment_like`) finds everything else like it,
    which is what the answer key is then compared with. An AREA always comes from the model as well.
    Returns (overlay image, report dict). The report is what `print_takeoff` turns into words.
    """
    sheet = lab.sheets[sheet_id]
    img = sheet.load()
    drawn: Dict[str, List[List[float]]] = {}
    for label, b in boxes:
        drawn.setdefault(label, []).append([float(v) for v in b])
    rep = {"sheet": sheet.id, "title": sheet.title, "scale": {"refs": []}, "counts": {}, "counts_todo": [],
           "areas": {}, "rooms": None}

    # ---------------------------------------------------------------- 1. the scale
    px_per_ft, source = sheet.px_per_ft, "the drawing's known scale"
    student_values = []
    for ref in sheet.scale_refs:
        label = sheet.scale_label(ref)
        key_px_per_ft = _span(ref["box"], ref["axis"]) / float(ref["feet"])
        row = {"label": ref["label"], "feet": float(ref["feet"]), "axis": ref["axis"], "use": bool(ref.get("use")),
               "note": ref.get("note", ""), "drawn": False, "px": None, "px_per_ft": None,
               "key_px_per_ft": key_px_per_ft, "err_pct": None, "box": ref["box"], "via": None}
        b = drawn[label][0] if label in drawn else None
        if b is None and ref.get("from_category") and drawn.get(ref["from_category"]):
            b = drawn[ref["from_category"]][0]                 # the first footing box is the scale reference too
            row["via"] = ref["from_category"]
        if b is not None:
            px = max(1.0, _span(b, ref["axis"]))
            row.update(drawn=True, px=px, px_per_ft=px / float(ref["feet"]), box=b,
                       err_pct=(px / float(ref["feet"]) - key_px_per_ft) / key_px_per_ft * 100)
            student_values.append((ref, row["px_per_ft"]))
            if ref.get("use"):
                px_per_ft, source = row["px_per_ft"], "your box"
        rep["scale"]["refs"].append(row)
    rep["scale"]["used_px_per_ft"] = px_per_ft
    rep["scale"]["key_px_per_ft"] = sheet.px_per_ft
    rep["scale"]["from"] = source
    rep["scale"]["how"] = sheet.scale_how
    rep["scale"]["disagree_pct"] = None
    if len(student_values) >= 2:
        vals = [v for _r, v in student_values]
        lo, hi = min(vals), max(vals)
        rep["scale"]["disagree_pct"] = (hi - lo) / lo * 100
        trust = next((r for r, _v in student_values if r.get("use")), None)
        rep["scale"]["trust"] = trust["label"] if trust else None
        rep["scale"]["why"] = next((r.get("note", "") for r, _v in student_values if not r.get("use") and r.get("note")), "")

    def sqft(px):
        return float(px) / (px_per_ft * px_per_ft)

    # ---------------------------------------------------------------- 2. counts: one example box, SAM 3 finds the rest
    rep["todo_labels"] = {}
    count_masks = {}
    for cat in (sheet.count_categories if count_categories is None else count_categories):
        mine = drawn.get(sheet.example_label(cat), [])
        via = None
        if not mine and cat in sheet.area_categories and drawn.get(sheet.area_label(cat)):
            mine, via = drawn[sheet.area_label(cat)][:1], sheet.area_label(cat)   # the first footing box is the example
        if not mine:
            rep["counts_todo"].append(cat)
            rep["todo_labels"][cat] = sheet.area_label(cat) if cat in sheet.area_categories else sheet.example_label(cat)
            continue
        truth = sheet.counts.get(cat, [])
        hint = sheet.hint(cat)
        thr = float(hint["threshold"] if threshold is None else threshold)
        info = {"category": cat, "threshold": thr, "tip": hint.get("tip", ""), "size_range": hint["size_range"],
                "example": mine[0], "extra_examples": len(mine) - 1, "example_from": via, "truth": len(truth), "seconds": 0.0,
                "found": 0, "matched": 0, "missed": [], "extra": [], "pred": [], "extra_flags": []}
        if lab.engine is None:
            info["error"] = ("SAM 3 is not loaded, so your example box cannot be used to count. Re-run Step 0 "
                             "with load_model ticked.")
        else:
            res = lab.engine.segment_like(img, mine[0], threshold=thr, size_range=hint["size_range"])
            pred = [list(map(float, b)) for b in res.boxes]
            ok, _n_missed, _n_extra, found_flags, extra_flags = match_boxes(pred, truth, iou=0.2)
            info.update(found=len(pred), matched=ok, pred=pred, extra_flags=extra_flags, seconds=res.seconds,
                        missed=[list(t) for t, f in zip(truth, found_flags) if not f],
                        extra=[list(p) for p, e in zip(pred, extra_flags) if e])
            count_masks[cat] = res.masks
        rep["counts"][cat] = info

    # ---------------------------------------------------------------- 3. areas
    walls = None
    layers = []
    for cat in sheet.area_categories:
        mine = drawn.get(sheet.area_label(cat), []) or drawn.get(cat, [])
        if not mine:
            continue
        if lab.engine is None:
            rep["areas"][cat] = [{"n": i + 1, "box": b, "error": "SAM 3 is not loaded: areas need the live model."}
                                 for i, b in enumerate(mine)]
            continue
        key_areas = sheet.areas(cat)
        used = set()
        rows = []
        for i, b in enumerate(mine, 1):
            note = ""
            box_px = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
            if cat == "room":
                if walls is None:
                    walls = wall_pixels(img)
                mask, raw_px, score, filled_px = room_mask(lab.engine, img, b, walls)
                if not mask.any():
                    note = "SAM 3 found nothing in this box."
                elif filled_px > 1.15 * raw_px:
                    note = (f"{(filled_px - raw_px) / filled_px * 100:.0f} % of this area is drawn over (furniture, a stair, "
                            "door swings) and was filled back in from the outline of the mask - look at the picture and judge "
                            "whether that was right.")
            else:
                r = lab.engine.segment_visual(img, box=b)
                mask = r.masks[0] if len(r) else np.zeros((img.height, img.width), bool)
                short = min(b[2] - b[0], b[3] - b[1])
                fill = mask.sum() / box_px
                if short < MIN_SYMBOL_PX:
                    note = (f"your box is only {short:.0f} px across, and under about {MIN_SYMBOL_PX} px a symbol is too small "
                            f"for the model on this sheet: the number is your box, not the {cat}.")
                elif fill > 1.05:
                    note = (f"the mask is {fill * 100:.0f} % of your box: it has spilled outside the box and taken in something "
                            f"around the {cat}. Draw the box tighter, or read the size off the schedule instead.")
                elif fill > 0.9:
                    note = (f"the mask fills {fill * 100:.0f} % of your box - look at the outline: if it is a rounded copy of "
                            "your box rather than the thing inside it, this number is your box.")
            px = int(mask.sum())
            row = {"n": i, "box": b, "px": px, "your_sqft": sqft(px), "note": note,
                   "label": None, "true_sqft": None, "err_pct": None, "printed": None}
            best, iou = None, 0.0
            for j, a in enumerate(key_areas):
                v = box_iou(a["box"], b)
                if v > iou:
                    best, iou = j, v
            if best is not None and iou > 0.1:
                a = key_areas[best]
                used.add(best)
                row.update(label=a["label"], true_sqft=float(a["true_sqft"]), printed=a.get("printed"),
                           indoor=bool(a.get("indoor")),
                           err_pct=(row["your_sqft"] - float(a["true_sqft"])) / float(a["true_sqft"]) * 100)
                if "rounded copy" in (row["note"] or "") and abs(row["err_pct"]) < 15:
                    row["note"] = ""                     # the reading agrees with the drawing: no need to doubt it
            rows.append(row)
            layers.append((str(i), mask, GREEN if row["label"] else BLUE))
        rep["areas"][cat] = rows
        # the real ones nobody boxed get a red outline, except where the category is also counted from one example:
        # there the count's green masks already cover them, and measuring more of them is optional
        rep.setdefault("areas_missed", {})[cat] = ([] if cat in sheet.count_categories else
                                                   [a["box"] for j, a in enumerate(key_areas) if j not in used and a.get("takeoff", True)])
        if cat == "room":
            indoor = sheet.areas("room", indoor_only=True, takeoff_only=True)
            boxed = {key_areas[j]["label"] for j in used}
            rep["rooms"] = {"total": sum(r["your_sqft"] for r in rows if r.get("indoor")),
                            "n": sum(1 for r in rows if r.get("indoor")),
                            "outdoor": sum(r["your_sqft"] for r in rows if r["label"] and not r.get("indoor")),
                            "key_total": sum(a["true_sqft"] for a in indoor),
                            "key_n": len(indoor),
                            "not_boxed": [a["label"] for a in indoor if a["label"] not in boxed]}
        else:
            rep.setdefault("totals", {})[cat] = {"yours": sum(r["your_sqft"] for r in rows),
                                                 "key": sum(r["true_sqft"] for r in rows if r["true_sqft"])}

    # ---------------------------------------------------------------- the picture
    for cat, c in rep["counts"].items():
        for m, e in zip(count_masks.get(cat, []), c["extra_flags"]):
            layers.append(("", m, BLUE if e else GREEN))
    over = viz.multi_overlay(img, layers, alpha=0.45) if layers else img.copy()
    d = ImageDraw.Draw(over)
    w_thin = max(3, img.width // 500)
    for cat, c in rep["counts"].items():
        for tb in c["missed"]:
            d.rectangle(tb, outline=RED, width=w_thin)
        d.rectangle(c["example"], outline=PURPLE, width=w_thin + 2)
    for cat, rows in rep["areas"].items():
        for r in rows:
            d.rectangle(r["box"], outline=GREEN if r.get("label") else BLUE, width=w_thin)
            d.text((r["box"][0] + 4, r["box"][1] + 4), str(r["n"]), fill="black", font=viz._font(max(14, img.width // 90)))
        for tb in rep.get("areas_missed", {}).get(cat, []):
            d.rectangle(tb, outline=RED, width=w_thin)
    for row in rep["scale"]["refs"]:
        if row["drawn"]:
            d.rectangle(row["box"], outline=PURPLE, width=w_thin + 2)
    return over, rep


def print_takeoff(sheet, rep, guided: bool = True):
    """The report of `takeoff_compute` in plain words. `guided` adds the answer key's notes on the scale references
    (the homework's wrong scale bar is explained there); the workshop prints the numbers only."""
    sc = rep["scale"]
    drawn = [r for r in sc["refs"] if r["drawn"]]
    if not sc["refs"]:                       # a counting-only sheet: nothing is measured, so no scale is set
        print_counts(rep, tips=guided); print_areas(sheet, rep); return
    print("SCALE CHECK")
    for r in drawn:
        print(f"  {r['label']}: {r['px']:.0f} px \u00f7 {r['feet']:g} ft = {r['px_per_ft']:.2f} px/ft")
    if not drawn:
        print("  No scale box drawn: the drawing's known scale is used.")
    if sc.get("disagree_pct") is not None:
        print(f"  Your two scale estimates differ by {sc['disagree_pct']:.0f}%. For more accurate results, draw the scale box precisely.")
    if guided:
        for r in drawn:
            if r["note"]:
                print(f"  {r['label']}: {r['note']}")
    trusted = next((r for r in drawn if r["use"]), None)
    if trusted and trusted.get("via"):
        print(f"  Scale used for takeoff: {sc['used_px_per_ft']:.2f} px/ft (from your first {trusted['via']} box: {trusted['label']})")
    elif trusted:
        print(f"  Scale used for takeoff: {sc['used_px_per_ft']:.2f} px/ft (from the first scale box: {trusted['label']})")
    else:
        print(f"  Scale used for takeoff: {sc['used_px_per_ft']:.2f} px/ft (the drawing's known scale)")
    print_counts(rep, tips=guided)
    print_areas(sheet, rep)


def count_lines(rep, tips: bool = True, bold=lambda s: s):
    """The counting part of a report as (heading, [lines]): what SAM 3 found from one example box, against the key.
    `bold` wraps the four numbers a reader looks for first (regions returned, real ones, found, missed)."""
    if not (rep["counts"] or rep.get("counts_todo")):
        return None, []
    lines = []
    for cat, c in rep["counts"].items():
        if "error" in c:
            lines.append(f"{cat}: {c['error']}"); continue
        n = c["found"]; ne = len(c["extra"]); nm = len(c["missed"])
        real = words(c["truth"], cat)                                        # "footings", "pile footings"
        source = f"your first {c['example_from']} box" if c.get("example_from") else "your example box"
        lines += [f"{bold('EXAMPLE BOX')}: {source} (purple), confidence >= {c['threshold']:.2f}",
                  f"SAM 3 segmented {bold(f'{n} region' + ('' if n == 1 else 's'))} at the selected confidence threshold.",
                  bold(f"Compared with the {c['truth']} actual {real}:"),
                  f"{bold('Found:')} {c['matched']} of {c['truth']}",
                  f"{bold('Missed:')} {nm}",
                  f"{bold('Extra:')} {ne} region{'' if ne == 1 else 's'} that {'was not a ' + cat if ne == 1 else 'were not ' + real}"]
        if c.get("extra_examples"):
            lines.append(f"   You drew {c['extra_examples'] + 1} boxes labelled 'example: {cat}'. "
                         "Only the first one was used - one example is all the model needs.")
        if tips and c.get("tip"):
            lines.append(f"   {c['tip']}")
    for cat in rep.get("counts_todo") or []:
        label = (rep.get("todo_labels") or {}).get(cat, f"example: {cat}")
        lines.append(f"{cat}: not counted. Draw ONE clean box labelled '{label}' and submit again; "
                     "the count comes from the model, not from your boxes.")
    if rep["counts"]:
        lines.append(LEGEND + "; purple = your example box")
    return "", lines


def print_counts(rep, tips: bool = True):
    """The counts as HTML when a notebook is listening (bold numbers), plain text otherwise."""
    head, lines = count_lines(rep, tips)
    if not lines:
        return
    try:
        from IPython import get_ipython
        from IPython.display import HTML, display
        if get_ipython() is not None:
            display(HTML(counts_html(rep, tips))); return
    except Exception:  # noqa: BLE001
        pass
    print("\n" + "\n".join(lines))


def counts_html(rep, tips: bool = True) -> str:
    import html
    head, lines = count_lines(rep, tips, bold=lambda s: f"<b>{html.escape(s)}</b>")
    if not lines:
        return ""
    return "<div style='font-family:monospace;white-space:pre-wrap;line-height:1.5'>" + "\n".join(lines) + "</div>"


def area_lines(sheet, rep, cat, bold=lambda s: s):
    """The summary of one measured category, in the same shape as the count summary."""
    rows = [r for r in rep["areas"][cat] if "error" not in r]
    key_areas = sheet.areas(cat, takeoff_only=True)
    matched = [r for r in rows if r.get("label")]
    on_key = len({r["label"] for r in matched})
    errs = [abs(r["err_pct"]) for r in matched if r.get("err_pct") is not None]
    w = words(len(key_areas), cat)
    lines = [f"{bold('MEASURED AREAS')}: {cat}",
             f"SAM 3 measured {bold(f'{len(rows)} region' + ('' if len(rows) == 1 else 's'))} from your boxes.",
             bold(f"Compared with the {len(key_areas)} actual {w}:"),
             f"{bold('Boxed and measured:')} {on_key} of {len(key_areas)}" + (" (box more to measure more)" if cat in sheet.count_categories else ""),
             *([] if cat in sheet.count_categories else [f"{bold('Not boxed:')} {len(key_areas) - on_key}"]),
             f"{bold('Boxes that were not a ' + cat + ':')} {len(rows) - len(matched)}"]
    if errs:
        lines.append(f"{bold('Error of the areas:')} median {np.median(errs):.0f} %, worst {max(errs):.0f} %")
    if cat == "room" and rep.get("rooms"):
        rm = rep["rooms"]
        gap = (rm["total"] - rm["key_total"]) / rm["key_total"] * 100 if rm["key_total"] else 0.0
        lines.append(f"{bold('Total:')} {rm['total']:,.0f} sq ft measured vs {rm['key_total']:,.0f} sq ft on the drawing ({gap:+.0f} %)"
                     + (f"; porches and other outdoor spaces you boxed, {rm['outdoor']:,.0f} sq ft, are not in that total" if rm["outdoor"] else ""))
    elif (rep.get("totals") or {}).get(cat, {}).get("key"):
        t = rep["totals"][cat]
        lines.append(f"{bold('Total:')} {t['yours']:,.1f} sq ft measured vs {t['key']:,.1f} sq ft on the drawing ({(t['yours'] - t['key']) / t['key'] * 100:+.0f} %)")
    if cat not in sheet.count_categories:
        lines.append(LEGEND.replace("a real one the model found", "a box on a real one").replace("a real one it missed", "a real one you did not box").replace("a region that is not one", "a box on nothing"))
    return lines


def areas_html(sheet, rep) -> str:
    import html
    blocks = []
    for cat in rep["areas"]:
        blocks.append("\n".join(area_lines(sheet, rep, cat, bold=lambda s: f"<b>{html.escape(s)}</b>")))
    return "<div style='font-family:monospace;white-space:pre-wrap;line-height:1.5'>" + "\n\n".join(blocks) + "</div>"


def print_areas(sheet, rep):
    """The measuring part of a report: the table of every box, then the summary."""
    for cat, rows in rep["areas"].items():
        print(f"\nAREAS - {cat.upper()}")
        print(f"  {'#':>2} {'your sq ft':>11s} {'drawing':>9s} {'error':>7s}  what the sheet prints / note")
        for r in rows:
            if "error" in r:
                print(f"  {r['n']:>2} {r['error']}"); continue
            true = f"{r['true_sqft']:9,.0f}" if r["true_sqft"] else f"{'-':>9s}"
            err = f"{r['err_pct']:+6.0f} %" if r["err_pct"] is not None else f"{'':>7s}"
            tail = r["label"] or f"does not sit on any {cat} of the drawing"
            if r.get("printed"):
                tail += f" (the sheet prints {r['printed']})"
            if r["note"]:
                tail += f" -- {r['note']}"
            print(f"  {r['n']:>2} {r['your_sqft']:11,.1f} {true} {err}  {tail}")
    if not rep["areas"]:
        return
    try:
        from IPython import get_ipython
        from IPython.display import HTML, display
        if get_ipython() is not None:
            display(HTML(areas_html(sheet, rep))); return
    except Exception:  # noqa: BLE001
        pass
    for cat in rep["areas"]:
        print("\n" + "\n".join(area_lines(sheet, rep, cat)))


def takeoff(lab, sheet_id: str):
    """One cell for the whole take-off of one drawing: the scale, the areas, and every count the sheet asks
    for, each one made by SAM 3 from a single box labelled 'example: <category>'."""
    import ipywidgets as w
    from IPython.display import display
    from jupyter_bbox_widget import BBoxWidget
    sheet = lab.sheets[sheet_id]
    img = sheet.load()
    disp = img.copy(); disp.thumbnail((1400, 1400)); scale = img.width / disp.width
    buf = io.BytesIO(); disp.save(buf, "PNG")
    cats = sheet.count_categories if getattr(lab.spec, "takeoff_counts", True) else []
    labels = [l for l in sheet.box_labels() if cats or not l.startswith("example: ")]
    widget = BBoxWidget(classes=labels)
    widget.image_bytes = buf.getvalue()
    controls, sl, sl0 = [], None, None
    if cats:                                  # only a sheet with something to count needs the confidence
        sl0 = float(sheet.hint(cats[0])["threshold"])
        sl = w.FloatSlider(value=sl0, min=0.1, max=0.9, step=0.05, description="confidence >=",
                           continuous_update=False, readout_format=".2f")
        controls.append(w.HBox([sl, w.HTML("&nbsp;how sure SAM 3 must be to keep something it found from your "
                                           "<b>example:</b> box. Leave it alone and each thing is counted at the "
                                           "confidence suggested for it.")]))

    guided = getattr(lab.spec, "guided", True)
    todo = "<br>".join(f"<b>{i + 1}.</b> {t}" for i, t in enumerate(sheet.tasks))   # shown only when guided
    msg = w.HTML("" if not guided else f"<b>{sheet.id} - {sheet.title}</b><br>{todo}<br>"
                 "Pick the label above the picture before each box. Zoom with the mouse wheel. Then <b>Submit</b>."
                 + ("<br>A count is never a tally of your boxes: draw <b>one</b> box labelled "
                    "<b>example: ...</b> and SAM 3 finds all the others like it." if cats else ""))
    out = w.Output()

    @widget.on_submit
    def _go():
        boxes = [(b.get("label", labels[0]),
                  [b["x"] * scale, b["y"] * scale, (b["x"] + b["width"]) * scale, (b["y"] + b["height"]) * scale])
                 for b in widget.bboxes]
        if not boxes:
            msg.value = "Draw the boxes first."; return
        needs_model = any(l in sheet.area_categories or l.startswith("example: ") for l, _ in boxes)
        if lab.engine is None and needs_model:
            msg.value = ("SAM 3 is not loaded: areas cannot be measured and nothing can be counted. "
                         "Run Step 0 with load_model ticked."); return
        msg.value = "Running SAM 3 on your boxes..."
        thr = sl.value if (sl is not None and abs(sl.value - sl0) > 1e-9) else None   # moved = the student decides
        over, rep = takeoff_compute(lab, sheet.id, boxes, threshold=thr, count_categories=cats)
        with out:
            out.clear_output(wait=True)
            viz.show_image(over, 1100)
            print_takeoff(sheet, rep, guided=guided)
        lab.takeoffs[sheet.id] = rep
        msg.value = "" if not guided else ("Done. Adjust the boxes and submit again; a tight box gives a cleaner mask. "
                     "Copy the numbers into your report, then go to the next drawing.")

    display(w.VBox([msg, widget, *controls, out]))


def find_like(lab, sheet_id: str):
    """Workshop Step 3d: ONE box around one door and SAM 3 finds every other door on the sheet. This is the model's
    own exemplar prompt (a box as a visual example, no words); the answer key marks what it found and missed."""
    import ipywidgets as w
    from IPython.display import HTML, display
    from jupyter_bbox_widget import BBoxWidget
    sheet = lab.sheets[sheet_id]
    cats = sheet.count_categories
    if not cats:
        print(f"{sheet.id} has nothing to count."); return
    img = sheet.load()
    disp = img.copy(); disp.thumbnail((1400, 1400)); scale = img.width / disp.width
    buf = io.BytesIO(); disp.save(buf, "PNG")
    labels = [sheet.example_label(c) for c in cats]
    widget = BBoxWidget(classes=labels)
    widget.image_bytes = buf.getvalue()
    sl0 = float(sheet.hint(cats[0])["threshold"])
    sl = w.FloatSlider(value=sl0, min=0.1, max=0.9, step=0.05, description="confidence >=",
                       continuous_update=False, readout_format=".2f")
    guided = getattr(lab.spec, "guided", True)
    what = " and ".join(plural(len(sheet.counts[c]), c) for c in cats)
    msg = w.HTML("" if not guided else f"<b>{sheet.id} - {sheet.title}</b>: the drawing has {what}.<br>"
                 "Draw <b>ONE</b> box around one of them - a door is the opening <i>and</i> its swing arc - with the "
                 "<b>example:</b> label above the picture, then <b>Submit</b>. SAM 3 looks for everything like it; the "
                 "picture marks the real ones found (green), missed (red outline) and the regions that are not one (blue). "
                 "Zoom with the mouse wheel.")
    out = w.Output()

    @widget.on_submit
    def _go():
        boxes = [(b.get("label", labels[0]),
                  [b["x"] * scale, b["y"] * scale, (b["x"] + b["width"]) * scale, (b["y"] + b["height"]) * scale])
                 for b in widget.bboxes if str(b.get("label", "")).startswith("example: ")]
        if not boxes:
            msg.value = "Draw one box first, with an <b>example:</b> label."; return
        if lab.engine is None:
            msg.value = "SAM 3 is not loaded. Run Step 0 with load_model ticked."; return
        msg.value = "SAM 3 is looking for everything like your example..."
        thr = sl.value if abs(sl.value - sl0) > 1e-9 else None
        drawn = {l for l, _ in boxes}
        over, rep = takeoff_compute(lab, sheet.id, boxes, threshold=thr,
                                    count_categories=[c for c in cats if sheet.example_label(c) in drawn])
        with out:
            out.clear_output(wait=True)
            viz.show_image(over, 1100)
            display(HTML(counts_html(rep, tips=guided)))
        lab.found.setdefault(sheet.id, {}).update(rep["counts"])
        msg.value = "" if not guided else ("Done. Try another example (one in a cluttered corner, one drawn the other way round), move the "
                     "confidence, then do the next drawing. Every run is kept for the summary.")

    display(w.VBox([msg, widget, w.HBox([sl]), out]))
