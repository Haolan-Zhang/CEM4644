"""Small plotting / widget helpers. Everything a student sees is produced here."""
import math
from typing import Dict, List, Optional, Sequence

import numpy as np
from PIL import Image

GREEN, RED, BLUE, GREY = "#2a9d8f", "#e76f51", "#457b9d", "#8d99ae"


def close(fig):
    import matplotlib.pyplot as plt
    plt.close(fig)


def show(fig, dpi: int = 100):
    """Render a matplotlib figure to PNG and display it (works in Colab, Jupyter and headless runs)."""
    import io
    from IPython.display import Image, display
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    close(fig)
    display(Image(data=buf.getvalue()))


def _wrap(text: str, width: int) -> str:
    import textwrap
    return "\n".join(textwrap.fill(line, width, break_long_words=False) if line else "" for line in text.split("\n"))


def image_grid(images: Sequence[Image.Image], titles: Optional[Sequence[str]] = None, ncols: int = 4,
               size: float = 2.6, colors: Optional[Sequence[str]] = None, suptitle: Optional[str] = None,
               fontsize: float = 8.0):
    """Grid of photos with (wrapped) captions above each one. Row height grows with the caption length."""
    import matplotlib.pyplot as plt
    n = len(images)
    ncols = max(1, min(ncols, n))
    nrows = math.ceil(n / ncols)
    width_chars = max(18, int(size * 15.5 * 8.0 / fontsize))
    wrapped = [_wrap(t, width_chars) for t in titles] if titles is not None else None
    max_lines = max((t.count("\n") + 1 for t in wrapped), default=0) if wrapped else 0
    row_h = size + 0.16 * max_lines + 0.25
    fig, axes = plt.subplots(nrows, ncols, figsize=(size * ncols + 0.3, row_h * nrows + (0.4 if suptitle else 0)))
    axes = np.array(axes).reshape(-1)
    for k, ax in enumerate(axes):
        ax.axis("off")
        if k < n:
            ax.imshow(images[k])
            if wrapped is not None:
                ax.set_title(wrapped[k], fontsize=fontsize, color=(colors[k] if colors is not None else "black"),
                             linespacing=1.25)
    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout(h_pad=0.6, w_pad=0.4)
    return fig


def prob_bars(ax, probs: Dict[str, float], true_label: Optional[str] = None, max_classes: int = 8):
    items = sorted(probs.items(), key=lambda kv: -kv[1])[:max_classes][::-1]
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    top = names[-1]
    colors = []
    for k in names:
        if true_label is None:
            colors.append(BLUE if k == top else GREY)
        else:
            colors.append((GREEN if k == true_label else RED) if k == top else (BLUE if k == true_label else GREY))
    ax.barh(names, [v * 100 for v in vals], color=colors)
    ax.set_xlim(0, 100)
    ax.set_xlabel("confidence (%)")
    for i, v in enumerate(vals):
        ax.text(min(v * 100 + 1, 88), i, f"{v * 100:.0f}%", va="center", fontsize=9)
    ax.tick_params(axis="y", labelsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def prediction_figure(image: Image.Image, probs: Dict[str, float], true_label: Optional[str] = None,
                      title: Optional[str] = None, model_name: str = "model"):
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.2, 3.3), gridspec_kw={"width_ratios": [1, 1.4]})
    a1.imshow(image); a1.axis("off")
    if title:
        a1.set_title(title, fontsize=9)
    top = max(probs, key=probs.get)
    verdict = f"{model_name} says: {top} ({probs[top] * 100:.0f}% confident)"
    if true_label is not None:
        verdict += "  ✅" if top == true_label else f"  ❌ (really: {true_label})"
    a2.set_title(verdict, fontsize=9.5, color=(GREEN if true_label == top else RED) if true_label else "black")
    prob_bars(a2, probs, true_label)
    fig.tight_layout()
    return fig


def show_prediction(clf, image: Image.Image, true_label: Optional[str] = None, title: Optional[str] = None):
    show(prediction_figure(image, clf.predict_one(image), true_label, title, clf.name))


# --------------------------------------------------------------------------- gallery / game
def gallery(image_set, category: str = "all", n: int = 12, seed: Optional[int] = None):
    cat = None if category in (None, "all", "any") else category
    sub = image_set.sample(n, seed=seed, class_name=cat)
    images = [sub.load(i) for i in range(len(sub))]
    titles = [sub.pretty(sub[i][1]) for i in range(len(sub))]
    show(image_grid(images, titles, ncols=4 if n > 6 else n, suptitle=f"{len(sub)} random photos" + (f" of '{cat}'" if cat else "")))


def guess_game(image_set, rounds: int = 5, seed: Optional[int] = None):
    """Show a photo, student clicks the class, reveal, keep score."""
    import ipywidgets as w
    from IPython.display import display
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(image_set))[:rounds]
    state = {"i": 0, "score": 0, "answered": False}
    img_out, msg = w.Output(), w.HTML()
    buttons = [w.Button(description=c, layout=w.Layout(width="auto", min_width="90px")) for c in image_set.pretty_classes]
    next_btn = w.Button(description="Next photo ▶", button_style="primary")
    box = w.VBox([img_out, w.HBox(buttons, layout=w.Layout(flex_flow="row wrap")), msg, next_btn])

    def show_current():
        k = int(order[state["i"]])
        with img_out:
            img_out.clear_output(wait=True)
            show(image_grid([image_set.load(k)], [f"photo {state['i'] + 1} of {rounds}: what is this?"], ncols=1, size=3.4))
        msg.value = "<i>Click the class you think is right.</i>"
        state["answered"] = False
        for b in buttons:
            b.button_style = ""; b.disabled = False

    def on_guess(b):
        if state["answered"]:
            return
        state["answered"] = True
        k = int(order[state["i"]])
        truth = image_set.pretty(image_set[k][1])
        ok = b.description == truth
        state["score"] += int(ok)
        for bb in buttons:
            bb.disabled = True
            if bb.description == truth:
                bb.button_style = "success"
            elif bb is b and not ok:
                bb.button_style = "danger"
        msg.value = (f"<b style='color:{GREEN}'>Correct!</b>" if ok else f"<b style='color:{RED}'>Not quite.</b> It is labelled <b>{truth}</b>.") + \
                    f" &nbsp; Score: {state['score']} / {state['i'] + 1}"

    def on_next(_):
        if not state["answered"]:
            msg.value = "<i>Make a guess first!</i>"; return
        state["i"] += 1
        if state["i"] >= rounds:
            msg.value = f"<b>Game over: you got {state['score']} of {rounds} right.</b> Write your score in your report."
            next_btn.disabled = True
            for bb in buttons:
                bb.disabled = True
            return
        show_current()

    for b in buttons:
        b.on_click(on_guess)
    next_btn.on_click(on_next)
    display(box)
    show_current()


def pick_and_predict(clf, image_set, seed: Optional[int] = None):
    """Random test photo -> prediction with confidence bars; optional class filter."""
    import ipywidgets as w
    from IPython.display import display
    rng = np.random.default_rng(seed)
    dd = w.Dropdown(options=["any class"] + image_set.pretty_classes, value="any class", description="pick from:",
                    style={"description_width": "initial"})
    reveal = w.Checkbox(value=True, description="show the true label")
    btn = w.Button(description="🎲 Another photo", button_style="primary")
    out = w.Output()

    def go(*_):
        src = image_set if dd.value == "any class" else image_set.only(dd.value)
        k = int(rng.integers(len(src)))
        img = src.load(k)
        truth = src.pretty(src[k][1]) if reveal.value else None
        with out:
            out.clear_output(wait=True)
            show(prediction_figure(img, clf.predict_one(img), truth, title=src[k][0].name, model_name=clf.name))

    btn.on_click(go)
    display(w.VBox([w.HBox([dd, reveal, btn]), out]))
    go()


def compare_models(clfs: Sequence, images: Sequence[Image.Image], captions: Sequence[str],
                   expected: Optional[Sequence[Optional[str]]] = None, ncols: int = 4):
    """Grid of images; under each, one line per model with its verdict."""
    titles, colors = [], []
    preds = [c.predict_proba(list(images)) for c in clfs]
    for i, cap in enumerate(captions):
        lines = [cap]
        ok_all = True
        for c, p in zip(clfs, preds):
            j = int(p[i].argmax())
            lines.append(f"{c.name}: {c.classes[j]} ({p[i, j] * 100:.0f}%)")
            if expected is not None and expected[i] is not None and c.classes[j] != expected[i]:
                ok_all = False
        titles.append("\n".join(lines))
        colors.append("black" if expected is None or expected[i] is None else (GREEN if ok_all else RED))
    show(image_grid(list(images), titles, ncols=ncols, colors=colors, size=2.8))
    return preds


def table(df):
    from IPython.display import display
    display(df)
