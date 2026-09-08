"""Widgets and figures for the detection lab."""
import math
from typing import Dict, List, Optional, Sequence

import numpy as np
from PIL import Image

from . import draw
from .evaluate import EvalResult, match

GREEN, RED, ORANGE, BLUE, GREY = "#2a9d8f", "#e76f51", "#f4a261", "#457b9d", "#8d99ae"


def close(fig):
    import matplotlib.pyplot as plt
    plt.close(fig)


def show(fig, dpi: int = 100):
    import io
    from IPython.display import Image as IPImage, display
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    close(fig)
    display(IPImage(data=buf.getvalue()))


def show_image(img: Image.Image, max_width: int = 900):
    """Display a PIL image directly (no matplotlib), scaled down for the notebook."""
    import io
    from IPython.display import Image as IPImage, display
    im = img
    if im.width > max_width:
        im = im.resize((max_width, round(im.height * max_width / im.width)), Image.BILINEAR)
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=88)
    display(IPImage(data=buf.getvalue()))


def _wrap(text: str, width: int) -> str:
    import textwrap
    return "\n".join(textwrap.fill(line, width, break_long_words=False) if line else "" for line in text.split("\n"))


def image_grid(images: Sequence[Image.Image], titles: Optional[Sequence[str]] = None, ncols: int = 3,
               size: float = 3.6, colors: Optional[Sequence[str]] = None, suptitle: Optional[str] = None, fontsize: float = 8.0):
    import matplotlib.pyplot as plt
    n = len(images)
    ncols = max(1, min(ncols, n))
    nrows = math.ceil(n / ncols)
    width_chars = max(18, int(size * 15.5 * 8.0 / fontsize))
    wrapped = [_wrap(t, width_chars) for t in titles] if titles is not None else None
    max_lines = max((t.count("\n") + 1 for t in wrapped), default=0) if wrapped else 0
    row_h = size * 0.8 + 0.16 * max_lines + 0.25
    fig, axes = plt.subplots(nrows, ncols, figsize=(size * ncols + 0.3, row_h * nrows + (0.4 if suptitle else 0)))
    axes = np.array(axes).reshape(-1)
    for k, ax in enumerate(axes):
        ax.axis("off")
        if k < n:
            ax.imshow(images[k])
            if wrapped is not None:
                ax.set_title(wrapped[k], fontsize=fontsize, color=(colors[k] if colors is not None else "black"), linespacing=1.25)
    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout(h_pad=0.6, w_pad=0.4)
    return fig


def legend_text(spec) -> str:
    return "box colours: " + ", ".join(f"{spec.pretty(c)} = {spec.colors.get(c, '?')}" for c in spec.classes)


def det_summary(det, class_names: Sequence[str], conf: float) -> str:
    d = det.at(conf)
    if len(d) == 0:
        return "nothing detected"
    counts: Dict[str, int] = {}
    for c in d.cls:
        counts[class_names[int(c)]] = counts.get(class_names[int(c)], 0) + 1
    return ", ".join(f"{k} ×{v}" for k, v in counts.items())


def gt_summary(item, class_names: Sequence[str]) -> str:
    if len(item.cls) == 0:
        return "no objects labelled"
    counts: Dict[str, int] = {}
    for c in item.cls:
        counts[class_names[int(c)]] = counts.get(class_names[int(c)], 0) + 1
    return ", ".join(f"{k} ×{v}" for k, v in counts.items())


# --------------------------------------------------------------------------- Part 1
def gallery(det_set, spec, category: str = "all", n: int = 6, seed: Optional[int] = None, with_boxes: bool = True):
    src = det_set if category in (None, "all", "any") else det_set.with_class(category)
    if len(src) == 0:
        print(f"No photos contain '{category}'."); return
    sub = src.sample(n, seed)
    images, titles = [], []
    for it in sub.items:
        im = it.load()
        if with_boxes:
            im = draw.draw_ground_truth(im, it, sub.pretty_classes, {spec.pretty(c): v for c, v in spec.colors.items()})
        images.append(im); titles.append(f"{it.path.name[:28]}\nlabelled: {gt_summary(it, sub.pretty_classes)}")
    show(image_grid(images, titles, ncols=3 if n > 2 else n, size=3.8,
                    suptitle=f"{len(sub)} random {'labelled ' if with_boxes else ''}photos" + (f" containing '{category}'" if category not in ('all', 'any', None) else "")))


def count_game(det_set, spec, rounds: int = 5, seed: Optional[int] = None):
    """Show a photo without boxes, student picks a count, reveal the labelled boxes."""
    import ipywidgets as w
    from IPython.display import display
    rng = np.random.default_rng(seed)
    pool = [i for i in range(len(det_set)) if len(det_set[i].cls) > 0]
    order = rng.permutation(pool)[:rounds]
    state = {"i": 0, "score": 0, "answered": False}
    img_out, msg = w.Output(), w.HTML()
    options = [str(k) for k in range(7)] + ["7 or more"]
    buttons = [w.Button(description=o, layout=w.Layout(width="auto", min_width="60px")) for o in options]
    next_btn = w.Button(description="Next photo ▶", button_style="primary")
    pretty_colors = {spec.pretty(c): v for c, v in spec.colors.items()}

    def truth(k):
        return det_set.count(k, spec.count_classes)

    def show_current():
        k = int(order[state["i"]])
        with img_out:
            img_out.clear_output(wait=True)
            show(image_grid([det_set[k].load()], [f"photo {state['i'] + 1} of {rounds}: {spec.count_question}"], ncols=1, size=6.5))
        msg.value = "<i>Click the number you think is right.</i>"
        state["answered"] = False
        for b in buttons:
            b.button_style = ""; b.disabled = False

    def on_guess(b):
        if state["answered"]:
            return
        state["answered"] = True
        k = int(order[state["i"]])
        t = truth(k)
        t_label = str(t) if t < 7 else "7 or more"
        ok = b.description == t_label
        state["score"] += int(ok)
        for bb in buttons:
            bb.disabled = True
            if bb.description == t_label:
                bb.button_style = "success"
            elif bb is b and not ok:
                bb.button_style = "danger"
        it = det_set[k]
        im = draw.draw_ground_truth(it.load(), it, det_set.pretty_classes, pretty_colors)
        with img_out:
            img_out.clear_output(wait=True)
            show(image_grid([im], [f"labelled answer: {t}   ({gt_summary(it, det_set.pretty_classes)})"], ncols=1, size=6.5))
        msg.value = (f"<b style='color:{GREEN}'>Correct!</b>" if ok else f"<b style='color:{RED}'>Not quite.</b> The labels say <b>{t}</b>.") + \
                    f" &nbsp; Score: {state['score']} / {state['i'] + 1}"

    def on_next(_):
        if not state["answered"]:
            msg.value = "<i>Make a guess first!</i>"; return
        state["i"] += 1
        if state["i"] >= rounds:
            msg.value = f"<b>Game over: {state['score']} of {rounds} right.</b> Note your score for the report."
            next_btn.disabled = True
            for bb in buttons:
                bb.disabled = True
            return
        show_current()

    for b in buttons:
        b.on_click(on_guess)
    next_btn.on_click(on_next)
    display(w.VBox([img_out, w.HBox(buttons, layout=w.Layout(flex_flow="row wrap")), msg, next_btn]))
    show_current()


# --------------------------------------------------------------------------- Part 2
def pick_and_detect(detector, det_set, spec, seed: Optional[int] = None):
    import ipywidgets as w
    from IPython.display import display
    rng = np.random.default_rng(seed)
    conf = w.FloatSlider(value=0.5, min=0.05, max=0.95, step=0.05, description="confidence ≥", continuous_update=False,
                         readout_format=".2f", style={"description_width": "110px"})
    truth = w.Checkbox(value=False, description="also show the true boxes (dashed)")
    btn = w.Button(description="🎲 Another photo", button_style="primary")
    out = w.Output()
    state = {"k": None}
    pretty_colors = {spec.pretty(c): v for c, v in spec.colors.items()}

    def render(*_):
        k = state["k"]
        it = det_set[k]
        img = it.load()
        det = detector.predict(img, key=str(it.path))
        im = draw.draw_predictions(img, det, detector.classes, pretty_colors, conf=conf.value)
        if truth.value:
            im = draw.draw_ground_truth(im, it, det_set.pretty_classes, pretty_colors, label=False)
        with out:
            out.clear_output(wait=True)
            show_image(im, 820)
            print(f"{detector.name} (confidence ≥ {conf.value * 100:.0f}%): {det_summary(det, detector.classes, conf.value)}")
            print(f"labelled truth: {gt_summary(it, det_set.pretty_classes)}")
            hidden = int(((det.conf < conf.value) & (det.conf >= 0.05)).sum())
            if hidden:
                print(f"({hidden} weaker detection(s) between 5% and {conf.value * 100:.0f}% are hidden by the threshold)")

    def another(*_):
        state["k"] = int(rng.integers(len(det_set))); render()

    conf.observe(render, names="value"); truth.observe(render, names="value"); btn.on_click(another)
    display(w.VBox([w.HBox([btn, conf, truth]), out]))
    another()


def threshold_explorer(res: EvalResult, spec):
    import ipywidgets as w
    import pandas as pd
    import matplotlib.pyplot as plt
    from IPython.display import display
    ths = np.round(np.arange(0.05, 0.96, 0.05), 2)
    table = {t: res.counts_at(t) for t in ths}
    slider = w.FloatSlider(value=0.5, min=0.05, max=0.95, step=0.05, description="confidence ≥", continuous_update=False,
                           readout_format=".2f", style={"description_width": "110px"})
    out = w.Output()

    def render(*_):
        t = float(min(ths, key=lambda x: abs(x - slider.value)))
        c = table[t]
        with out:
            out.clear_output(wait=True)
            df = pd.DataFrame([{"class": k, "true objects": v["n_gt"], "found (correct)": v["TP"], "missed": v["FN"],
                                "false alarms": v["FP"], "recall %": round(v["recall"] * 100), "precision %": round(v["precision"] * 100)}
                               for k, v in c.items()])
            display(df.style.hide(axis="index") if hasattr(df, "style") else df)
            fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
            for name in res.classes:
                col = spec.colors.get(next((k for k in spec.classes if spec.pretty(k) == name), name), None)
                axes[0].plot(ths, [table[x][name]["recall"] * 100 for x in ths], label=name, color=col)
                axes[1].plot(ths, [table[x][name]["precision"] * 100 for x in ths], label=name, color=col)
            for ax, ttl in zip(axes, ("recall: % of true objects found", "precision: % of detections that are right")):
                ax.axvline(t, color="black", ls="--"); ax.set_ylim(0, 102); ax.set_xlabel("confidence threshold"); ax.set_title(ttl, fontsize=10)
                ax.grid(alpha=0.3)
            axes[0].legend(fontsize=8)
            fig.tight_layout(); show(fig)

    slider.observe(render, names="value")
    display(w.VBox([slider, out]))
    render()


def error_explorer(res: EvalResult, spec, max_show: int = 6):
    import ipywidgets as w
    from IPython.display import display
    kind = w.Dropdown(options=["missed objects", "false alarms", "any mistake"], value="missed objects", description="show:",
                      style={"description_width": "initial"})
    cls = w.Dropdown(options=["any class"] + list(res.classes), value="any class", description="class:",
                     style={"description_width": "initial"})
    conf = w.FloatSlider(value=0.5, min=0.05, max=0.95, step=0.05, description="confidence ≥", continuous_update=False,
                         readout_format=".2f", style={"description_width": "110px"})
    out = w.Output()

    def render(*_):
        t = conf.value
        rows = []
        for i, (it, det) in enumerate(zip(res.items, res.dets)):
            d = det.at(t)
            mp, mg = match(d, it)
            if cls.value != "any class":
                c = res.classes.index(cls.value)
                n_fn = int((~mg & (it.cls == c)).sum()); n_fp = int((~mp & (d.cls == c)).sum())
            else:
                n_fn = int((~mg).sum()); n_fp = int((~mp).sum())
            score = n_fn if kind.value == "missed objects" else n_fp if kind.value == "false alarms" else n_fn + n_fp
            if score > 0:
                rows.append((score, i, n_fn, n_fp, mp, mg))
        rows.sort(key=lambda r: -r[0])
        with out:
            out.clear_output(wait=True)
            if not rows:
                print("No photos with that kind of mistake at this threshold. Try another class or move the threshold."); return
            print(f"{len(rows)} photo(s) have {kind.value} at threshold {t * 100:.0f}%; showing the worst {min(max_show, len(rows))}. "
                  "Green = correct, orange = false alarm, red dashed = missed.")
            images, titles = [], []
            for score, i, n_fn, n_fp, mp, mg in rows[:max_show]:
                it, det = res.items[i], res.dets[i]
                images.append(draw.draw_errors(it.load(), det, mp, it, mg, res.classes, t))
                titles.append(f"{it.path.name[:26]}\nmissed {n_fn}, false alarms {n_fp}")
            show(image_grid(images, titles, ncols=2, size=5.2))

    for x in (kind, cls, conf):
        x.observe(render, names="value")
    display(w.VBox([w.HBox([kind, cls, conf]), out]))
    render()


def compare_detectors(detectors: Sequence, image: Image.Image, spec_colors: Dict[str, str], conf: float = 0.4, caption: str = ""):
    images, titles = [], []
    for d in detectors:
        det = d.predict(image)
        cols = {c: spec_colors.get(c, "#ffffff") for c in d.classes}
        images.append(draw.draw_predictions(image, det, d.classes, cols, conf=conf))
        titles.append(f"{d.name}\n{det_summary(det, d.classes, conf)}")
    show(image_grid(images, titles, ncols=len(detectors), size=5.0, suptitle=caption))


def table(df):
    from IPython.display import display
    display(df)
