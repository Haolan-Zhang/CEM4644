"""Widgets for the segmentation lab. Every function displays something; nothing returns to the notebook."""
import time
from typing import Dict, List, Optional, Sequence

import numpy as np
from PIL import Image

from . import viz
from .engine import SegResult

GREEN, RED, BLUE, GREY = "#2a9d8f", "#e76f51", "#457b9d", "#8d99ae"


def gallery(photos, ids: Optional[Sequence[str]] = None, ncols: int = 3, size: float = 4.0, title: Optional[str] = None):
    sel = [photos[i] for i in ids] if ids else photos.photos
    ims = [p.load() for p in sel]
    for im in ims:
        im.thumbnail((640, 640))
    titles = [f"{p.id}: {p.title}\n{p.author} · {p.license}" for p in sel]
    viz.show(viz.image_grid(ims, titles, ncols=ncols, size=size, suptitle=title))


def print_result(res: SegResult, material: str, min_score: float):
    keep = res.scores >= min_score
    n = int(keep.sum())
    if n == 0:
        print(f"'{res.prompt}': nothing found at confidence >= {min_score * 100:.0f}%" + (f" (the model proposed {len(res)} weak region(s) below that)" if len(res) else "."))
        return
    print(f"'{res.prompt}' ({material}): {n} region(s) covering {res.area_pct(min_score):.1f}% of the photo; "
          f"confidence of each region: {', '.join(f'{s * 100:.0f}%' for s in res.scores[keep][:8])}" + (" ..." if n > 8 else ""))


# --------------------------------------------------------------------------- Part 2
def segment_view(lab, photo_id: str, material: str, threshold: float):
    """Original | mask | overlay for one material on one photo, with a live confidence slider."""
    import ipywidgets as w
    from IPython.display import display
    spec, photo = lab.spec, lab.photos[photo_id]
    m = spec.material(material)
    img = photo.load()
    res = lab.get(photo_id, m)
    sl = w.FloatSlider(value=threshold, min=0.2, max=0.9, step=0.05, description="confidence ≥", continuous_update=False, readout_format=".2f")
    out = w.Output()

    def render(*_):
        with out:
            out.clear_output(wait=True)
            viz.show(viz.three_panel(img, res, m.color, f"{photo.title} — '{m.prompt}'", sl.value))
            print_result(res, m.name, sl.value)

    sl.observe(render, names="value")
    display(w.VBox([sl, out]))
    render()


def material_mix(lab, photo_id: str, threshold: float):
    spec, photo = lab.spec, lab.photos[photo_id]
    img = photo.load()
    values, layers, colors = {}, [], {}
    for m in spec.materials:
        res = lab.get(photo_id, m)
        pct = res.area_pct(threshold)
        values[m.name] = pct; colors[m.name] = m.color
        if pct > 0:
            layers.append((m.name, res.union(threshold), m.color))
    layers.sort(key=lambda t: -t[1].mean())   # big regions first so small ones stay visible
    over = viz.multi_overlay(img, layers)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.3, 1]})
    axes[0].imshow(over); axes[0].axis("off"); axes[0].set_title(f"{photo.id}: every material at confidence ≥ {threshold * 100:.0f}%", fontsize=10)
    axes[0].legend(handles=[Patch(color=m.color, label=m.name) for m in spec.materials if values[m.name] > 0], fontsize=7, loc="lower left", ncol=2)
    names = [n for n in values if values[n] > 0]
    axes[1].barh(names[::-1], [values[n] for n in names[::-1]], color=[colors[n] for n in names[::-1]])
    for i, n in enumerate(names[::-1]):
        axes[1].text(values[n] + 0.5, i, f"{values[n]:.1f}%", va="center", fontsize=8)
    axes[1].set_xlabel("% of the photo (regions can overlap, so the total can exceed 100)"); axes[1].set_xlim(0, max(10, max(values.values()) * 1.2))
    for s in ("top", "right"):
        axes[1].spines[s].set_visible(False)
    fig.tight_layout(); viz.show(fig)
    print("Material mix: " + ", ".join(f"{n} {values[n]:.1f}%" for n in names))


def guess_game(lab, rounds: int = 4, seed: Optional[int] = None):
    """Show a photo and a material; the student guesses the share of the photo; reveal the overlay."""
    import ipywidgets as w
    from IPython.display import display
    spec = lab.spec
    rng = np.random.default_rng(seed if seed is not None else int(time.time()) % 100000)
    pairs = []
    for pid in spec.guess_photos:
        for m in spec.materials:
            res = lab.get(pid, m)
            pct = res.area_pct(0.5)
            if pct >= 3:
                pairs.append((pid, m, pct))
    rng.shuffle(pairs)
    pairs = pairs[:rounds]
    bins = [("under 5 %", 0, 5), ("5–15 %", 5, 15), ("15–30 %", 15, 30), ("30–50 %", 30, 50), ("over 50 %", 50, 101)]
    state = {"i": 0, "score": 0, "answered": False}
    out, msg = w.Output(), w.HTML()
    buttons = [w.Button(description=b[0], layout=w.Layout(width="auto", min_width="90px")) for b in bins]
    nxt = w.Button(description="Next ▶", button_style="primary")

    def show_q():
        pid, m, pct = pairs[state["i"]]
        with out:
            out.clear_output(wait=True)
            im = lab.photos[pid].load(); im.thumbnail((700, 700)); viz.show_image(im, 700)
        msg.value = f"<b>Photo {state['i'] + 1} of {len(pairs)}:</b> how much of this photo is <b>{m.name}</b>?"
        state["answered"] = False
        for b in buttons:
            b.button_style = ""; b.disabled = False

    def on_guess(b):
        if state["answered"]:
            return
        state["answered"] = True
        pid, m, pct = pairs[state["i"]]
        truth = next(bb for bb in bins if bb[1] <= pct < bb[2])
        ok = b.description == truth[0]
        state["score"] += int(ok)
        for bb in buttons:
            bb.disabled = True
            if bb.description == truth[0]:
                bb.button_style = "success"
            elif bb is b:
                bb.button_style = "danger"
        res = lab.get(pid, m)
        with out:
            out.clear_output(wait=True)
            im = viz.overlay(lab.photos[pid].load(), res.union(0.5), m.color); im.thumbnail((700, 700)); viz.show_image(im, 700)
        msg.value = (f"<b style='color:{GREEN}'>Right!</b>" if ok else f"<b style='color:{RED}'>Not quite.</b>") + \
                    f" SAM 3 measures <b>{pct:.1f}%</b> {m.name} (phrase '{m.prompt}', confidence ≥ 50%). Score {state['score']}/{state['i'] + 1}."

    def on_next(_):
        if not state["answered"]:
            msg.value = "<i>Pick an answer first.</i>"; return
        state["i"] += 1
        if state["i"] >= len(pairs):
            msg.value = f"<b>Done: {state['score']} of {len(pairs)}.</b> Note your score for the report."
            nxt.disabled = True
            for bb in buttons:
                bb.disabled = True
            return
        show_q()

    for b in buttons:
        b.on_click(on_guess)
    nxt.on_click(on_next)
    display(w.VBox([msg, out, w.HBox(buttons), nxt]))
    if pairs:
        show_q()
    else:
        print("No photo/material pairs available for the game.")


# --------------------------------------------------------------------------- Part 3: compare
def compare(lab, photo_a: str, photo_b: str, threshold: float, materials: Optional[Sequence[str]] = None):
    spec = lab.spec
    mats = [spec.material(x) for x in materials] if materials else spec.materials
    groups, ims, titles = {}, [], []
    for pid in (photo_a, photo_b):
        photo = lab.photos[pid]
        img = photo.load()
        vals, layers = {}, []
        for m in mats:
            res = lab.get(pid, m)
            pct = res.area_pct(threshold); vals[m.name] = pct
            if pct > 0:
                layers.append((m.name, res.union(threshold), m.color))
        layers.sort(key=lambda t: -t[1].mean())
        groups[f"{pid}"] = vals
        ims.append(viz.multi_overlay(img, layers)); titles.append(f"{pid}: {photo.title}")
    viz.show(viz.image_grid(ims, titles, ncols=2, size=5.2))
    viz.show(viz.grouped_chart(groups, {m.name: m.color for m in mats}, title=f"material share at confidence ≥ {threshold * 100:.0f}%"))
    a, b = groups[photo_a], groups[photo_b]
    print("Change from left to right (percentage points): " + ", ".join(f"{n} {b[n] - a[n]:+.1f}" for n in a))


def series_view(lab, series_name: str, threshold: float, materials: Optional[Sequence[str]] = None, ncols: int = 5):
    spec = lab.spec
    s = spec.series[series_name]
    mats = [spec.material(x) for x in materials] if materials else spec.materials
    rows, ims, titles = [], [], []
    for pid, lab_ in zip(s["photos"], s["labels"]):
        photo = lab.photos[pid]
        img = photo.load(); img.thumbnail((640, 640))
        vals, layers = {}, []
        for m in mats:
            res = lab.get(pid, m)
            pct = res.area_pct(threshold); vals[m.name] = pct
        rows.append((lab_, vals))
        # overlay of the two most present materials keeps the picture readable
        top = sorted(mats, key=lambda m: -vals[m.name])[:3]
        full = photo.load()
        layers = [(m.name, lab.get(pid, m).union(threshold), m.color) for m in top if vals[m.name] > 0]
        layers.sort(key=lambda t: -t[1].mean())
        ov = viz.multi_overlay(full, layers); ov.thumbnail((640, 640))
        ims.append(ov); titles.append(f"{lab_}\n" + ", ".join(f"{m.name} {vals[m.name]:.0f}%" for m in top if vals[m.name] > 0))
    viz.show(viz.image_grid(ims, titles, ncols=ncols, size=3.3, suptitle=s["title"]))
    viz.show(viz.series_chart(rows, {m.name: m.color for m in mats}, title=f"{s['title']} — material share over time (confidence ≥ {threshold * 100:.0f}%)"))


# --------------------------------------------------------------------------- Part 4: errors
def phrase_lab(lab, photo_id: str, material: str, threshold: float):
    """The same material asked for with different words."""
    spec, photo = lab.spec, lab.photos[photo_id]
    m = spec.material(material)
    img = photo.load()
    ims, titles = [], []
    for phrase in [m.prompt] + list(m.alternatives):
        res = lab.get_phrase(photo_id, phrase)
        n = int((res.scores >= threshold).sum())
        ims.append(viz.overlay(img, res.union(threshold), m.color))
        titles.append(f"'{phrase}'\n{res.area_pct(threshold):.1f}% of the photo, {n} region(s)")
    viz.show(viz.image_grid(ims, titles, ncols=3, size=3.9, suptitle=f"{photo.id}: same material, different words (confidence ≥ {threshold * 100:.0f}%)"))


def inspector(lab, photo_id: str, material: str):
    """Every region on its own, with its confidence; a slider removes the weak ones."""
    import ipywidgets as w
    from IPython.display import display
    spec, photo = lab.spec, lab.photos[photo_id]
    m = spec.material(material)
    img = photo.load()
    res = lab.get(photo_id, m)
    sl = w.FloatSlider(value=0.5, min=0.2, max=0.9, step=0.05, description="confidence ≥", continuous_update=False, readout_format=".2f")
    out = w.Output()

    def render(*_):
        with out:
            out.clear_output(wait=True)
            viz.show_image(viz.instances_image(img, res, sl.value), 900)
            keep = res.scores >= sl.value
            print(f"'{m.prompt}': {int(keep.sum())} region(s) kept of {len(res)} proposed; together {res.area_pct(sl.value):.1f}% of the photo.")
            for k, i in enumerate(np.where(keep)[0]):
                print(f"  #{k + 1}: confidence {res.scores[i] * 100:.0f}%, {res.masks[i].mean() * 100:.1f}% of the photo")

    sl.observe(render, names="value")
    display(w.VBox([sl, out]))
    render()


def fix_with_box(lab, photo_id: str, material: str, threshold: float):
    """Draw boxes over wrong regions; SAM 3 re-runs with them as negative prompts."""
    import ipywidgets as w
    from IPython.display import display
    from jupyter_bbox_widget import BBoxWidget
    import io
    spec, photo = lab.spec, lab.photos[photo_id]
    m = spec.material(material)
    img = photo.load()
    res0 = lab.get(photo_id, m)
    before = viz.overlay(img, res0.union(threshold), m.color)
    disp = before.copy(); disp.thumbnail((900, 900))
    scale = img.width / disp.width
    buf = io.BytesIO(); disp.save(buf, "JPEG", quality=88)
    widget = BBoxWidget(classes=["exclude"])
    widget.image_bytes = buf.getvalue()
    msg = w.HTML(f"<b>Before:</b> '{m.prompt}' covers {res0.area_pct(threshold):.1f}%. Draw a box over each region that is WRONG, then click <b>Submit</b>. "
                 "SAM 3 will run again and treat your boxes as 'not this'.")
    out = w.Output()

    @widget.on_submit
    def _go():
        boxes = [[b["x"] * scale, b["y"] * scale, (b["x"] + b["width"]) * scale, (b["y"] + b["height"]) * scale] for b in widget.bboxes]
        if not boxes:
            msg.value = "Draw at least one box first."; return
        if lab.engine is None:
            msg.value = "SAM 3 is not loaded (Step 0 without a model). This step needs the live model."; return
        res1 = lab.engine.segment(img, m.prompt, threshold=0.2, negative_boxes=boxes)
        after = viz.overlay(img, res1.union(threshold), m.color)
        with out:
            out.clear_output(wait=True)
            viz.show(viz.image_grid([before, after], [f"before: {res0.area_pct(threshold):.1f}%", f"after {len(boxes)} negative box(es): {res1.area_pct(threshold):.1f}%"], ncols=2, size=5))
        msg.value = f"<b>After:</b> {res1.area_pct(threshold):.1f}% (was {res0.area_pct(threshold):.1f}%). Draw more boxes and submit again if needed."

    display(w.VBox([msg, widget, out]))


def live_phrase(lab, photo_id: str, phrase: str, threshold: float):
    if lab.engine is None:
        print("SAM 3 is not loaded. Re-run Step 0 with a GPU runtime for live phrases."); return
    photo = lab.photos[photo_id]
    img = photo.load()
    res = lab.engine.segment(img, phrase, threshold=0.2)
    viz.show(viz.three_panel(img, res, "#ff70a6", f"{photo.id} — your phrase: '{phrase}'", threshold))
    print_result(res, phrase, threshold)
    print(f"(SAM 3 took {res.seconds:.1f} s)")


# --------------------------------------------------------------------------- plans
def takeoff(lab, plan_id: str):
    """Box prompts on a plan drawing -> masks -> pixel areas -> real areas via a known footing width."""
    import ipywidgets as w
    from IPython.display import display
    from jupyter_bbox_widget import BBoxWidget
    import io
    photo = lab.plans[plan_id]
    plan = next(p for p in lab.plan_specs if p["id"] == plan_id)
    img = photo.load()
    disp = img.copy(); disp.thumbnail((1100, 1100)); scale = img.width / disp.width
    buf = io.BytesIO(); disp.save(buf, "PNG")
    widget = BBoxWidget(classes=["reference", "footing", "opening", "other"])
    widget.image_bytes = buf.getvalue()
    ref_w = w.FloatText(value=plan["reference_width_ft"], description="ref. width (ft)", style={"description_width": "110px"})
    msg = w.HTML(f"Draw a box around one <b>{plan['reference']}</b> footing and label it <b>reference</b> (its real width is {plan['reference_width_ft']} ft). "
                 "Then draw boxes around the members you want to measure (label them footing / opening / other) and click <b>Submit</b>.")
    out = w.Output()
    state = {"masks": []}

    @widget.on_submit
    def _go():
        if lab.engine is None:
            msg.value = "SAM 3 is not loaded. This step needs the live model (GPU runtime recommended)."; return
        boxes = [(b.get("label", "other"), [b["x"] * scale, b["y"] * scale, (b["x"] + b["width"]) * scale, (b["y"] + b["height"]) * scale]) for b in widget.bboxes]
        if not boxes:
            msg.value = "Draw at least one box."; return
        results = []
        for label, box in boxes:
            r = lab.engine.segment_box(img, box)
            mask = r.union() if len(r) else np.zeros((img.height, img.width), bool)
            results.append((label, box, mask, float(r.scores[0]) if len(r) else 0.0))
        ref = [r for r in results if r[0] == "reference"]
        with out:
            out.clear_output(wait=True)
            layers = [(f"{lab_} {k + 1}", mask, ["#e63946", "#4cc9f0", "#ffd166", "#06d6a0", "#7b2cbf", "#f4a261"][k % 6]) for k, (lab_, box, mask, sc) in enumerate(results)]
            over = viz.multi_overlay(img, layers, alpha=0.6)
            from PIL import ImageDraw
            d = ImageDraw.Draw(over)
            for k, (lab_, box, mask, sc) in enumerate(results):
                d.rectangle(box, outline="black", width=2); d.text((box[0] + 3, box[1] + 3), f"{k + 1}", fill="black")
            viz.show_image(over, 1000)
            if not ref:
                print("No box labelled 'reference': pixel areas only.")
                ft_per_px = None
            else:
                lab_, box, mask, sc = ref[0]
                ys, xs = np.where(mask)
                px_w = (xs.max() - xs.min() + 1) if len(xs) else (box[2] - box[0])
                ft_per_px = ref_w.value / px_w
                print(f"Scale from the reference footing: {px_w:.0f} px wide = {ref_w.value} ft  ->  {ft_per_px:.4f} ft per pixel "
                      f"(1 px² = {ft_per_px ** 2:.5f} sq ft)")
            print(f"{'#':>2} {'label':10s} {'mask pixels':>12s} {'confidence':>10s} {'area (sq ft)':>13s}")
            for k, (lab_, box, mask, sc) in enumerate(results):
                px = int(mask.sum())
                area = f"{px * ft_per_px ** 2:.1f}" if ft_per_px else "-"
                print(f"{k + 1:>2} {lab_:10s} {px:>12d} {sc * 100:>9.0f}% {area:>13s}")
            if ref:
                lab_, box, mask, sc = ref[0]
                print(f"Check: the reference footing should be about {ref_w.value ** 2:.0f} sq ft if it is square; SAM 3 measured {int(mask.sum()) * ft_per_px ** 2:.1f} sq ft.")
        msg.value = "Done. Adjust boxes and submit again to refine; a tight box gives a cleaner mask."

    display(w.VBox([msg, ref_w, widget, out]))
