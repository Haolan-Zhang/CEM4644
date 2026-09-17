"""Widgets for the floor-plan take-off lab. Every function displays something; nothing returns to the notebook."""
import io
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

from . import viz
from .config import LEGEND, Thing
from .engine import SegResult

GREEN, RED, BLUE, GREY = "#2a9d8f", "#e76f51", "#457b9d", "#8d99ae"


# --------------------------------------------------------------------------- helpers
def box_iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def box_near(p, t) -> bool:
    """The centre of box p lies within one door-width of the truth box t (a door SWING sits next to the door OPENING)."""
    w, h = t[2] - t[0], t[3] - t[1]
    r = max(w, h)
    cx, cy = (p[0] + p[2]) / 2, (p[1] + p[3]) / 2
    return (t[0] - r <= cx <= t[2] + r) and (t[1] - r <= cy <= t[3] + r) and max(p[2] - p[0], p[3] - p[1]) <= 3 * r


def box_hit(t, p, iou: float, near: bool) -> bool:
    return box_iou(t, p) >= iou or (near and box_near(p, t))


def match_boxes(pred: Sequence[Sequence[float]], truth: Sequence[Sequence[float]], iou: float = 0.3, near: bool = False) -> Tuple[int, int, int]:
    """(correct, missed, extra): a truth box is found if some prediction overlaps it (IoU >= iou, or for doors: a swing next to the
    opening); a prediction is extra if it matches nothing."""
    found = [any(box_hit(t, p, iou, near) for p in pred) for t in truth]
    extra = [not any(box_hit(t, p, iou, near) for t in truth) for p in pred]
    return int(sum(found)), int(len(truth) - sum(found)), int(sum(extra))


def best_room(plan, mask: np.ndarray) -> Tuple[Optional[dict], float]:
    """The answer-key room that a mask overlaps most (fraction of the MASK inside the room)."""
    best, frac = None, 0.0
    area = mask.sum()
    if area == 0:
        return None, 0.0
    for r in plan.rooms():
        rm = plan.room_mask(r)
        f = (rm & mask).sum() / area
        if f > frac:
            best, frac = r, f
    return best, float(frac)


def truth_line(plan, thing: Thing, res: SegResult, threshold: float) -> str:
    """One sentence comparing SAM 3's answer with the answer key for this thing on this plan."""
    if thing.truth is None:
        return ""
    what, sub = thing.truth
    keep = res.scores >= threshold
    pred_boxes = [list(map(float, b)) for b in res.boxes[keep]]
    if what == "rooms":
        rooms = plan.rooms(sub)
        true_area = sum(r["area_m2"] for r in rooms)
        ok, missed, extra = match_boxes(pred_boxes, [r["box"] for r in rooms], iou=0.3)
        label = sub or "room"
        return (f"The drawing says: {len(rooms)} {label}{'s' if len(rooms) != 1 else ''}, {true_area:.1f} m² in total. "
                f"Regions that sit on a real {label}: {ok} of {len(rooms)} found, {missed} missed, {extra} extra.")
    truth = plan.truth_boxes(what, sub)
    ok, missed, extra = match_boxes(pred_boxes, truth, iou=0.2, near=(what == "doors"))
    label = sub or what.rstrip("s")
    if what == "walls":
        return (f"The drawing has {len(truth)} wall segments: {ok} found, {missed} missed, {extra} extra region(s). "
                "SAM 3 has no notion of a 'wall'; a shape word (thick black line) is what finds them.")
    return f"The drawing has {len(truth)} {label}{'s' if len(truth) != 1 else ''}: {ok} found, {missed} missed, {extra} extra region(s) that are not one."


def print_result(plan, thing_name: str, res: SegResult, threshold: float):
    keep = res.scores >= threshold
    n = int(keep.sum())
    if n == 0:
        print(f"'{res.prompt}': nothing found at confidence ≥ {threshold * 100:.0f}%"
              + (f" (the model proposed {len(res)} weak region(s) below that)." if len(res) else "."))
        return
    px = int(res.union(threshold).sum())
    print(f"'{res.prompt}' ({thing_name}): {n} region(s), {px:,} pixels = {plan.area_m2(px):.1f} m² "
          f"= {px / plan.drawing_pixels * 100:.1f} % of the drawing. Confidence of each region: "
          + ", ".join(f"{s * 100:.0f}%" for s in res.scores[keep][:8]) + (" ..." if n > 8 else ""))


# --------------------------------------------------------------------------- Part 1
def gallery(lab, ids: Optional[Sequence[str]] = None, ncols: int = 2, size: float = 5.0, title: Optional[str] = None):
    sel = [lab.plans[i] for i in ids] if ids else lab.plans.plans
    ims = [p.load() for p in sel]
    for im in ims:
        im.thumbnail((900, 900))
    titles = [f"{p.id}: {p.title}\n{p.facts()}" for p in sel]
    viz.show(viz.image_grid(ims, titles, ncols=ncols, size=size, suptitle=title))


def legend():
    print("Room labels on the plans (Finnish abbreviations; set B has one Swedish plan):")
    for k, v in LEGEND.items():
        print(f"  {k:26s} {v}")


# --------------------------------------------------------------------------- Part 1: SAM 3 on a photo
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
    viz.show(viz.three_panel(img, res, "#e63946", f"'{phrase}' on {photo.id}", threshold))
    keep = res.scores >= threshold
    n = int(keep.sum())
    top = ", ".join(f"{v:.2f}" for v in sorted(res.scores[keep], reverse=True)[:8])
    if n:
        print(f"'{phrase}': {n} region(s) at confidence ≥ {threshold:.2f} (confidences {top}), together {res.area_pct(threshold):.1f} % of the photo, "
              f"in {res.seconds:.1f} s.")
    else:
        below = int((res.scores >= 0.1).sum())
        print(f"'{phrase}': nothing at confidence ≥ {threshold:.2f}" + (f"; {below} weak region(s) between 0.10 and {threshold:.2f}: lower the slider to see them." if below else ". Try other words.")
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
        lines.append(f"{i + 1}. {what}: one object of {int(m.sum()):,} px = {m.mean() * 100:.1f} % of the photo (the model's own confidence {res.scores[0]:.2f}).")
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
    """Draw boxes (label 'box') or tap objects (label 'point': the centre of a small box) on the photo; SAM 3 cuts each one out."""
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
    msg = w.HTML("Draw a <b>box</b> around an object, or a tiny box on it labelled <b>point</b> (its centre is the click). Several are fine. Then <b>Submit</b>.")
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


# --------------------------------------------------------------------------- Part 2
def segment_view(lab, plan_id: str, thing_name: str, threshold: float):
    """Original | mask | overlay for one thing on one plan, with a live confidence slider and the answer key."""
    import ipywidgets as w
    from IPython.display import display
    spec, plan = lab.spec, lab.plans[plan_id]
    t = spec.thing(thing_name)
    img = plan.load()
    res = lab.get(plan_id, t)
    sl = w.FloatSlider(value=threshold, min=0.1, max=0.9, step=0.05, description="confidence ≥", continuous_update=False, readout_format=".2f")
    out = w.Output()

    def render(*_):
        with out:
            out.clear_output(wait=True)
            viz.show(viz.three_panel(img, res, t.color, f"{plan.id}: {plan.title} — '{t.prompt}'", sl.value))
            print_result(plan, t.name, res, sl.value)
            line = truth_line(plan, t, res, sl.value)
            if line:
                print(line)

    sl.observe(render, names="value")
    display(w.VBox([sl, out]))
    render()


def count_view(lab, plan_id: str, thing_name: str, threshold: float):
    """Hits, misses and extras drawn on the plan: green = real ones the model found, red = real ones it missed, blue = extra."""
    spec, plan = lab.spec, lab.plans[plan_id]
    t = spec.thing(thing_name)
    if t.truth is None:
        print(f"No answer key for '{t.name}'."); return
    what, sub = t.truth
    truth = [r["box"] for r in plan.rooms(sub)] if what == "rooms" else plan.truth_boxes(what, sub)
    res = lab.get(plan_id, t)
    keep = res.scores >= threshold
    pred = [list(map(float, b)) for b in res.boxes[keep]]
    iou = 0.3 if what == "rooms" else 0.2
    near = what == "doors"          # a door is drawn as an opening in the wall plus a swing: a region on the swing counts
    img = viz.instances_image(plan.load(), res, threshold, alpha=0.35)
    d = ImageDraw.Draw(img)
    hits = misses = 0
    for tb in truth:
        found = any(box_hit(tb, p, iou, near) for p in pred)
        hits += found; misses += (not found)
        d.rectangle(tb, outline=GREEN if found else RED, width=4)
    extra = 0
    for p in pred:
        if not any(box_hit(tb, p, iou, near) for tb in truth):
            extra += 1; d.rectangle(p, outline=BLUE, width=3)
    viz.show_image(img, 1000)
    label = sub or (what.rstrip("s") if what != "rooms" else "room")
    print(f"'{t.prompt}' at confidence ≥ {threshold * 100:.0f}%: {len(pred)} region(s). Answer key: {len(truth)} {label}(s). "
          f"Found {hits} (green), missed {misses} (red), {extra} extra region(s) that are not a {label} (blue).")


def mix(lab, plan_id: str, threshold: float):
    """Every room type at once: SAM 3's area next to the drawing's real area; fixture and opening counts next to the truth."""
    spec, plan = lab.spec, lab.plans[plan_id]
    img = plan.load()
    rooms = [t for t in spec.things if t.kind == "room" and t.truth and t.truth[1]]
    sam, truth, colors, layers = {}, {}, {}, []
    for t in rooms:
        res = lab.get(plan_id, t)
        sam[t.name] = plan.area_m2(res.union(threshold).sum())
        truth[t.name] = sum(r["area_m2"] for r in plan.rooms(t.truth[1]))
        colors[t.name] = t.color
        if sam[t.name] > 0:
            layers.append((t.name, res.union(threshold), t.color))
    layers.sort(key=lambda x: -x[1].mean())
    over = viz.multi_overlay(img, layers)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1.3, 1]})
    axes[0].imshow(over); axes[0].axis("off")
    axes[0].set_title(f"{plan.id}: every room type at confidence ≥ {threshold * 100:.0f}%", fontsize=10)
    axes[0].legend(handles=[Patch(color=t.color, label=t.name) for t in rooms if sam[t.name] > 0], fontsize=7, loc="lower left", ncol=2)
    names = [t.name for t in rooms]
    y = np.arange(len(names))
    axes[1].barh(y + 0.2, [sam[n] for n in names], height=0.4, color=[colors[n] for n in names], label="SAM 3")
    axes[1].barh(y - 0.2, [truth[n] for n in names], height=0.4, color="#cccccc", label="drawing (answer key)")
    axes[1].set_yticks(y); axes[1].set_yticklabels(names, fontsize=8); axes[1].invert_yaxis()
    axes[1].set_xlabel("m²"); axes[1].legend(fontsize=8)
    for s in ("top", "right"):
        axes[1].spines[s].set_visible(False)
    fig.tight_layout(); viz.show(fig)
    print(f"{'room type':18s} {'SAM 3 (m²)':>11s} {'drawing (m²)':>13s} {'difference':>11s}")
    for n in names:
        diff = sam[n] - truth[n]
        print(f"{n:18s} {sam[n]:11.1f} {truth[n]:13.1f} {diff:+10.1f}")
    print(f"{'all indoor rooms':18s} {'':>11s} {plan.floor_area_m2:13.1f}   (floor area of the plan)")
    counts = [t for t in spec.things if t.kind in ("fixture", "opening") and t.truth]
    print("\nCounts (regions at this confidence vs. the answer key):")
    for t in counts:
        res = lab.get(plan_id, t)
        what, sub = t.truth
        tr = plan.truth_boxes(what, sub)
        keep = res.scores >= threshold
        ok, missed, extra = match_boxes([list(map(float, b)) for b in res.boxes[keep]], tr, iou=0.2, near=(what == "doors"))
        print(f"  {t.name:10s} SAM 3 {int(keep.sum()):3d}   drawing {len(tr):3d}   (found {ok}, missed {missed}, extra {extra})")


def guess_game(lab, rounds: int = 4, seed: Optional[int] = None):
    """A room is outlined on a plan; the student guesses its area; the drawing's number and SAM 3's are revealed."""
    import ipywidgets as w
    from IPython.display import display
    spec = lab.spec
    rng = np.random.default_rng(seed if seed is not None else int(time.time()) % 100000)
    pool = [(p, r) for p in lab.plans.plans for r in p.rooms(indoor_only=True) if r["area_m2"] >= 2]
    idx = rng.permutation(len(pool))[:rounds]
    items = [pool[i] for i in idx]
    bins = [("under 5 m²", 0, 5), ("5–10 m²", 5, 10), ("10–15 m²", 10, 15), ("15–25 m²", 15, 25), ("over 25 m²", 25, 1e9)]
    state = {"i": 0, "score": 0, "answered": False}
    out, msg = w.Output(), w.HTML()
    buttons = [w.Button(description=b[0], layout=w.Layout(width="auto", min_width="90px")) for b in bins]
    nxt = w.Button(description="Next ▶", button_style="primary")
    room_thing = spec.thing("room")

    def show_q():
        p, r = items[state["i"]]
        im = p.load(); d = ImageDraw.Draw(im); d.polygon([tuple(x) for x in r["poly"]], outline="#e63946", width=6)
        im.thumbnail((800, 800))
        with out:
            out.clear_output(wait=True); viz.show_image(im, 800)
        msg.value = (f"<b>Room {state['i'] + 1} of {len(items)}</b> (plan {p.id}, label <b>{r['label'] or r['type']}</b>): "
                     f"how big is the outlined room? Use the 5 m scale bar.")
        state["answered"] = False
        for b in buttons:
            b.button_style = ""; b.disabled = False

    def on_guess(b):
        if state["answered"]:
            return
        state["answered"] = True
        p, r = items[state["i"]]
        truth = next(bb for bb in bins if bb[1] <= r["area_m2"] < bb[2])
        ok = b.description == truth[0]
        state["score"] += int(ok)
        for bb in buttons:
            bb.disabled = True
            if bb.description == truth[0]:
                bb.button_style = "success"
            elif bb is b:
                bb.button_style = "danger"
        res = lab.get(p.id, room_thing)
        rm = p.room_mask(r); measured = None
        for m, s in zip(res.masks, res.scores):
            if s >= 0.3 and (m & rm).sum() > 0.5 * rm.sum():
                measured = p.area_m2((m & rm).sum() if m.sum() > 2 * rm.sum() else m.sum()); break
        im = viz.overlay(p.load(), rm, "#e63946"); im.thumbnail((800, 800))
        with out:
            out.clear_output(wait=True); viz.show_image(im, 800)
        sam_txt = f"SAM 3 ('room') measures <b>{measured:.1f} m²</b> here." if measured else "SAM 3 ('room') did not find this room at confidence 0.3."
        msg.value = (f"<b style='color:{GREEN}'>Right!</b>" if ok else f"<b style='color:{RED}'>Not quite.</b>") + \
                    f" The drawing says <b>{r['area_m2']:.1f} m²</b> ({r['type']}). {sam_txt} Score {state['score']}/{state['i'] + 1}."

    def on_next(_):
        if not state["answered"]:
            msg.value = "<i>Pick an answer first.</i>"; return
        state["i"] += 1
        if state["i"] >= len(items):
            msg.value = f"<b>Done: {state['score']} of {len(items)}.</b> Note your score for the report."
            nxt.disabled = True
            for bb in buttons:
                bb.disabled = True
            return
        show_q()

    for b in buttons:
        b.on_click(on_guess)
    nxt.on_click(on_next)
    display(w.VBox([msg, out, w.HBox(buttons), nxt]))
    if items:
        show_q()
    else:
        print("No rooms available for the game.")


# --------------------------------------------------------------------------- Part 3: compare
def compare(lab, plan_a: str, plan_b: str, threshold: float):
    spec = lab.spec
    rooms = [t for t in spec.things if t.kind == "room" and t.truth and t.truth[1]]
    groups, truths, ims, titles = {}, {}, [], []
    for pid in (plan_a, plan_b):
        plan = lab.plans[pid]
        img = plan.load()
        vals, layers = {}, []
        for t in rooms:
            res = lab.get(pid, t)
            vals[t.name] = plan.area_m2(res.union(threshold).sum())
            if vals[t.name] > 0:
                layers.append((t.name, res.union(threshold), t.color))
        layers.sort(key=lambda x: -x[1].mean())
        groups[pid] = vals
        truths[pid] = {t.name: sum(r["area_m2"] for r in plan.rooms(t.truth[1])) for t in rooms}
        ims.append(viz.multi_overlay(img, layers)); titles.append(f"{pid}: {plan.title}\nfloor area {plan.floor_area_m2} m² (drawing)")
    viz.show(viz.image_grid(ims, titles, ncols=2, size=5.6))
    viz.show(viz.grouped_chart(groups, {t.name: t.color for t in rooms}, title=f"m² per room type, SAM 3 at confidence ≥ {threshold * 100:.0f}%"))
    print(f"{'room type':18s} | {plan_a:>8s} SAM {'drawing':>8s} | {plan_b:>8s} SAM {'drawing':>8s}")
    for t in rooms:
        n = t.name
        print(f"{n:18s} | {groups[plan_a][n]:12.1f} {truths[plan_a][n]:8.1f} | {groups[plan_b][n]:12.1f} {truths[plan_b][n]:8.1f}")
    pa, pb = lab.plans[plan_a], lab.plans[plan_b]
    print(f"Floor area (drawing): {plan_a} {pa.floor_area_m2} m², {plan_b} {pb.floor_area_m2} m²; rooms: {pa.key['n_rooms']} vs {pb.key['n_rooms']}; "
          f"doors {pa.key['n_doors']} vs {pb.key['n_doors']}; windows {pa.key['n_windows']} vs {pb.key['n_windows']}.")


# --------------------------------------------------------------------------- Part 4: errors
def phrase_lab(lab, plan_id: str, thing_name: str, threshold: float):
    """The same thing asked for with different words."""
    spec, plan = lab.spec, lab.plans[plan_id]
    t = spec.thing(thing_name)
    img = plan.load()
    ims, titles = [], []
    for phrase in [t.prompt] + list(t.alternatives):
        res = lab.get_phrase(plan_id, phrase)
        n = int((res.scores >= threshold).sum())
        ims.append(viz.overlay(img, res.union(threshold), t.color))
        titles.append(f"'{phrase}'\n{n} region(s), {plan.area_m2(res.union(threshold).sum()):.1f} m²")
    viz.show(viz.image_grid(ims, titles, ncols=2, size=5.0, suptitle=f"{plan.id}: same thing, different words (confidence ≥ {threshold * 100:.0f}%)"))
    line = truth_line(plan, t, lab.get(plan_id, t), threshold)
    if line:
        print("With the main wording: " + line)


def inspector(lab, plan_id: str, thing_name: str):
    """Every region on its own, with its confidence; a slider removes the weak ones."""
    import ipywidgets as w
    from IPython.display import display
    spec, plan = lab.spec, lab.plans[plan_id]
    t = spec.thing(thing_name)
    img = plan.load()
    res = lab.get(plan_id, t)
    sl = w.FloatSlider(value=0.4, min=0.1, max=0.9, step=0.05, description="confidence ≥", continuous_update=False, readout_format=".2f")
    out = w.Output()

    def render(*_):
        with out:
            out.clear_output(wait=True)
            viz.show_image(viz.instances_image(img, res, sl.value), 1000)
            keep = res.scores >= sl.value
            print(f"'{t.prompt}': {int(keep.sum())} region(s) kept of {len(res)} proposed; together {plan.area_m2(res.union(sl.value).sum()):.1f} m².")
            for k, i in enumerate(np.where(keep)[0]):
                m = res.masks[i]
                room, frac = best_room(plan, m)
                where = f"mostly on the {room['type']} '{room['label']}' ({room['area_m2']:.1f} m²)" if room and frac > 0.5 else "not on any one room"
                print(f"  #{k + 1}: confidence {res.scores[i] * 100:.0f}%, {plan.area_m2(m.sum()):.1f} m², {where}")

    sl.observe(render, names="value")
    display(w.VBox([sl, out]))
    render()


def fix_with_box(lab, plan_id: str, thing_name: str, threshold: float):
    """Draw boxes over wrong regions; SAM 3 re-runs with them as negative prompts."""
    import ipywidgets as w
    from IPython.display import display
    from jupyter_bbox_widget import BBoxWidget
    spec, plan = lab.spec, lab.plans[plan_id]
    t = spec.thing(thing_name)
    img = plan.load()
    res0 = lab.get(plan_id, t)
    before = viz.overlay(img, res0.union(threshold), t.color)
    disp = before.copy(); disp.thumbnail((1000, 1000))
    scale = img.width / disp.width
    buf = io.BytesIO(); disp.save(buf, "PNG")
    widget = BBoxWidget(classes=["exclude"])
    widget.image_bytes = buf.getvalue()
    a0 = plan.area_m2(res0.union(threshold).sum())
    msg = w.HTML(f"<b>Before:</b> '{t.prompt}' covers {a0:.1f} m². Draw a box over each region that is WRONG, then click <b>Submit</b>. "
                 "SAM 3 will run again and treat your boxes as 'not this'.")
    out = w.Output()

    @widget.on_submit
    def _go():
        boxes = [[b["x"] * scale, b["y"] * scale, (b["x"] + b["width"]) * scale, (b["y"] + b["height"]) * scale] for b in widget.bboxes]
        if not boxes:
            msg.value = "Draw at least one box first."; return
        if lab.engine is None:
            msg.value = "SAM 3 is not loaded (Step 0 without a model). This step needs the live model."; return
        res1 = lab.engine.segment(img, t.prompt, threshold=0.1, negative_boxes=boxes)
        a1 = plan.area_m2(res1.union(threshold).sum())
        after = viz.overlay(img, res1.union(threshold), t.color)
        with out:
            out.clear_output(wait=True)
            viz.show(viz.image_grid([before, after], [f"before: {a0:.1f} m²", f"after {len(boxes)} negative box(es): {a1:.1f} m²"], ncols=2, size=5.5))
            line = truth_line(plan, t, res1, threshold)
            if line:
                print("After: " + line)
        msg.value = f"<b>After:</b> {a1:.1f} m² (was {a0:.1f}). Draw more boxes and submit again if needed."

    display(w.VBox([msg, widget, out]))


def live_phrase(lab, plan_id: str, phrase: str, threshold: float):
    if lab.engine is None:
        print("SAM 3 is not loaded. Re-run Step 0 with a GPU runtime for live phrases."); return
    plan = lab.plans[plan_id]
    img = plan.load()
    res = lab.engine.segment(img, phrase, threshold=0.1)
    viz.show(viz.three_panel(img, res, "#ff70a6", f"{plan.id} — your phrase: '{phrase}'", threshold))
    print_result(plan, phrase, res, threshold)
    print(f"(SAM 3 took {res.seconds:.1f} s)")


# --------------------------------------------------------------------------- Part 3: boxes and quantities
def scale_check(lab, plan_id: str):
    """Draw a box along the 5 m scale bar: the box width gives the scale; compared with the drawing's own scale."""
    import ipywidgets as w
    from IPython.display import display
    from jupyter_bbox_widget import BBoxWidget
    plan = lab.plans[plan_id]
    img = plan.load()
    disp = img.copy(); disp.thumbnail((1000, 1000))
    scale = img.width / disp.width
    buf = io.BytesIO(); disp.save(buf, "PNG")
    widget = BBoxWidget(classes=["scale bar", "a room I can read the size of"])
    widget.image_bytes = buf.getvalue()
    metres = plan.key["scale_bar"]["metres"]
    msg = w.HTML(f"Draw a box exactly from the <b>0</b> tick to the <b>{metres} m</b> tick of the scale bar (bottom left), label it <b>scale bar</b>, click <b>Submit</b>. "
                 "Only the WIDTH of the box is used.")
    out = w.Output()

    @widget.on_submit
    def _go():
        bars = [b for b in widget.bboxes if b.get("label") == "scale bar"]
        if not bars:
            msg.value = "Draw a box labelled 'scale bar' first."; return
        b = bars[0]
        w_px = b["width"] * scale
        cm_per_px = metres * 100 / w_px
        true = plan.m_per_px * 100
        with out:
            out.clear_output(wait=True)
            print(f"Your box: {w_px:.0f} px wide = {metres} m  ->  1 px = {cm_per_px:.2f} cm  ({1 / (cm_per_px / 100):.1f} px per metre).")
            print(f"The drawing's own scale: 1 px = {true:.2f} cm. Your reading is {abs(cm_per_px - true) / true * 100:.1f} % off"
                  + (": good enough for a take-off." if abs(cm_per_px - true) / true < 0.03 else ": redraw the box more carefully (zoom in) and submit again."))
            print(f"A 1 % error in the scale becomes a 2 % error in every area, because area is scale squared.")
        msg.value = "Done. You can redraw the box and submit again."

    display(w.VBox([msg, widget, out]))


def wall_pixels(img) -> np.ndarray:
    """Thick dark strokes of a drawing (the walls), not thin lines or text: dark pixels that survive a 7x7 opening."""
    from scipy import ndimage
    dark = np.asarray(img.convert("L")) < 100
    return ndimage.binary_opening(dark, structure=np.ones((7, 7)))


def takeoff_compute(lab, plan_id: str, boxes):
    """boxes: list of (label, [x1, y1, x2, y2]) in full-image pixels, labels 'scale bar' / 'window' / 'room'.
    Returns (overlay, report dict). Rooms: SAM 3 inside each box (phrase 'empty room' + the box), holes filled, walls removed,
    area from the student's scale (from their scale-bar box; the drawing's scale when there is none). Windows: counted from the
    boxes against the answer key. Everything is checked against the drawing."""
    from scipy import ndimage
    plan = lab.plans[plan_id]
    img = plan.load()
    rep = {"rooms": [], "windows": {}, "scale": {}, "missed_rooms": []}
    # ---- the scale from the student's box on the 5 m bar
    bars = [b for l, b in boxes if l == "scale bar"]
    metres = plan.key["scale_bar"]["metres"]
    true_m_per_px = plan.m_per_px
    if bars:
        w_px = max(1.0, bars[0][2] - bars[0][0])
        m_per_px = metres / w_px
        rep["scale"] = {"drawn": True, "cm_per_px": m_per_px * 100, "true_cm_per_px": true_m_per_px * 100,
                        "err_pct": (m_per_px - true_m_per_px) / true_m_per_px * 100, "w_px": w_px}
    else:
        m_per_px = true_m_per_px
        rep["scale"] = {"drawn": False, "cm_per_px": m_per_px * 100, "true_cm_per_px": true_m_per_px * 100, "err_pct": 0.0}
    area_m2 = lambda px: float(px) * m_per_px ** 2   # noqa: E731
    # ---- windows: the student's boxes against the drawing's windows
    truth_w = plan.truth_boxes("windows")
    win_boxes = [b for l, b in boxes if l == "window"]
    found_w = [any(box_iou(tb, b) >= 0.2 for b in win_boxes) for tb in truth_w]
    extra_w = [not any(box_iou(tb, b) >= 0.2 for tb in truth_w) for b in win_boxes]
    rep["windows"] = {"drawn": len(win_boxes), "truth": len(truth_w), "found": int(sum(found_w)), "missed": int(len(truth_w) - sum(found_w)), "extra": int(sum(extra_w))}
    # ---- rooms: SAM 3 inside each box
    colors = ["#e63946", "#4cc9f0", "#ffd166", "#06d6a0", "#7b2cbf", "#f4a261", "#457b9d", "#2a9d8f", "#e9c46a", "#8d6e63", "#ff70a6", "#9aa5b1"]
    layers, walls, matched_ids = [], None, set()
    room_boxes = [b for l, b in boxes if l == "room"]
    for k, box in enumerate(room_boxes, 1):
        r = lab.engine.segment_room(img, box)
        mask = r.masks[0].copy() if len(r) else np.zeros((img.height, img.width), bool)
        box_px = max(1.0, (box[2] - box[0]) * (box[3] - box[1]))
        note = ""
        if mask.any():
            if walls is None:
                walls = wall_pixels(img)
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            clip = np.zeros_like(mask); clip[max(0, y1):y2 + 1, max(0, x1):x2 + 1] = True
            mask = ndimage.binary_fill_holes(mask & clip) & ~walls
            if mask.sum() < 0.6 * box_px:
                note = f"the mask covers only {mask.sum() / box_px * 100:.0f} % of your box: no wall on one side (open plan)? The box itself is {area_m2(box_px):.1f} m²."
        else:
            note = "SAM 3 found nothing in this box."
        px = int(mask.sum()); m2 = area_m2(px)
        row = {"n": k, "px": px, "m2": m2, "box": box, "truth": None, "truth_m2": None, "err_pct": None, "note": note}
        room, frac = best_room(plan, mask) if px else (None, 0.0)
        if room and frac > 0.5:
            row["truth"] = room["label"] or room["type"]; row["truth_m2"] = room["area_m2"]
            row["err_pct"] = (m2 - room["area_m2"]) / room["area_m2"] * 100
            matched_ids.add(id(room))
        rep["rooms"].append(row)
        layers.append((str(k), mask, colors[(k - 1) % len(colors)]))
    rep["missed_rooms"] = [r["label"] or r["type"] for r in plan.rooms(indoor_only=True) if id(r) not in matched_ids]
    rep["total_m2"] = sum(r["m2"] for r in rep["rooms"])
    rep["matched_truth_m2"] = sum(r["truth_m2"] for r in rep["rooms"] if r["truth_m2"])
    rep["floor_area_m2"] = plan.floor_area_m2
    # ---- the picture
    over = viz.multi_overlay(img, layers, alpha=0.55) if layers else img.copy()
    d = ImageDraw.Draw(over)
    for k, box in enumerate(room_boxes, 1):
        d.rectangle(box, outline="black", width=3); d.text((box[0] + 4, box[1] + 4), str(k), fill="black")
    for tb, f in zip(truth_w, found_w):
        if not f:
            d.rectangle(tb, outline=RED, width=3)
    for b, e in zip(win_boxes, extra_w):
        d.rectangle(b, outline=BLUE if e else GREEN, width=3)
    for b in bars[:1]:
        d.rectangle(b, outline="#7b2cbf", width=3)
    return over, rep


def print_takeoff(plan, rep):
    sc = rep["scale"]
    if sc["drawn"]:
        print(f"Scale: your box on the 5 m bar is {sc['w_px']:.0f} px wide -> 1 px = {sc['cm_per_px']:.2f} cm. The drawing's own scale: 1 px = {sc['true_cm_per_px']:.2f} cm "
              f"({sc['err_pct']:+.1f} % {'-> about ' + format(abs(sc['err_pct']) * 2, '.0f') + ' % on every area, because area is scale squared' if abs(sc['err_pct']) >= 0.5 else '-> good'}). "
              "Your scale is used for the areas below.")
    else:
        print(f"No scale-bar box drawn: the drawing's own scale (1 px = {sc['true_cm_per_px']:.2f} cm) is used. Draw one to see how your reading changes the areas.")
    w = rep["windows"]
    if w["drawn"] or w["truth"]:
        print(f"Windows: you boxed {w['drawn']}; the drawing has {w['truth']}: {w['found']} found (green), {w['missed']} missed (thin red), {w['extra']} of your boxes are not on a window (blue).")
    if rep["rooms"]:
        print(f"\n{'#':>2} {'your m²':>8s} {'drawing':>9s} {'error':>7s}  room / note")
        for r in rep["rooms"]:
            if r["truth"]:
                print(f"{r['n']:>2} {r['m2']:8.1f} {r['truth_m2']:9.1f} {r['err_pct']:+6.0f} %  {r['truth']}" + (f"   {r['note']}" if r["note"] else ""))
            else:
                print(f"{r['n']:>2} {r['m2']:8.1f} {'—':>9s} {'':>7s}  does not sit on a single room of the drawing" + (f"   {r['note']}" if r["note"] else ""))
        n_ok = sum(1 for r in rep["rooms"] if r["truth"])
        print(f"\nYour rooms: {rep['total_m2']:.1f} m² over {len(rep['rooms'])} boxes; the {n_ok} matched rooms measure {rep['matched_truth_m2']:.1f} m² on the drawing "
              f"({(rep['total_m2'] - rep['matched_truth_m2']) / rep['matched_truth_m2'] * 100:+.0f} %). " if rep["matched_truth_m2"] else
              f"\nYour rooms: {rep['total_m2']:.1f} m² over {len(rep['rooms'])} boxes. ")
        print(f"The plan's whole indoor floor area is {rep['floor_area_m2']:.1f} m² ({len(plan.rooms(indoor_only=True))} rooms)."
              + (f" Rooms you did not box: {', '.join(rep['missed_rooms'])}." if rep["missed_rooms"] else " You boxed every room."))


def takeoff(lab, plan_id: str):
    """One cell for the whole take-off: the scale from a box on the 5 m bar, every window boxed and counted, every room boxed and
    measured by SAM 3, all checked against the drawing."""
    import ipywidgets as w
    from IPython.display import display
    from jupyter_bbox_widget import BBoxWidget
    plan = lab.plans[plan_id]
    img = plan.load()
    disp = img.copy(); disp.thumbnail((1100, 1100)); scale = img.width / disp.width
    buf = io.BytesIO(); disp.save(buf, "PNG")
    widget = BBoxWidget(classes=["scale bar", "window", "room"])
    widget.image_bytes = buf.getvalue()
    msg = w.HTML("In this order: <b>1.</b> one box from the <b>0</b> tick to the <b>5 m</b> tick of the scale bar (label <b>scale bar</b>); "
                 "<b>2.</b> a box on <b>every window</b> (label <b>window</b>: the small things first, so the boxes do not overlap); "
                 "<b>3.</b> a tight box on <b>every room</b>, edges on the inside faces of the walls (label <b>room</b>). Zoom with the mouse wheel. Then <b>Submit</b>.")
    out = w.Output()

    @widget.on_submit
    def _go():
        boxes = [(b.get("label", "room"), [b["x"] * scale, b["y"] * scale, (b["x"] + b["width"]) * scale, (b["y"] + b["height"]) * scale]) for b in widget.bboxes]
        if not boxes:
            msg.value = "Draw the boxes first."; return
        if any(l == "room" for l, _ in boxes) and lab.engine is None:
            msg.value = "SAM 3 is not loaded: rooms cannot be measured. Run Step 0 with load_model ticked (GPU runtime recommended)."; return
        msg.value = "Running SAM 3 on your rooms..."
        over, rep = takeoff_compute(lab, plan_id, boxes)
        with out:
            out.clear_output(wait=True)
            viz.show_image(over, 1000)
            print_takeoff(plan, rep)
        lab.takeoffs[plan_id] = rep
        msg.value = "Done. Adjust the boxes and submit again; a tight box gives a cleaner mask. Copy the table into your report, then do the next plan."

    display(w.VBox([msg, widget, out]))
