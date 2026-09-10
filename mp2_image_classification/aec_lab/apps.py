"""Gradio versions of two interactive steps (used only by the *_gradio notebooks).

* playground_app  -> Step 3b "Break it yourself": live sliders + a drawing pad.
* zero_shot_app   -> Step 5a "Type your own classes": re-classify the SAME photos with new names.

Both render inline in the notebook and keep all code hidden from students. On Colab, Gradio serves
them through its temporary public link (see _launch); that link is what makes the app work there.
"""
import os
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
from PIL import Image

from .tricky import DEFAULTS, PERTURB_LABELS, perturb
from .ui import GREEN, RED

_OPEN: Dict[str, object] = {}          # one running server per app key; re-running a cell replaces it
NO_APP = "AEC_LAB_NO_APP"              # set in headless tests: build the app but do not start a server

CSS = """
.aec-verdict {font-size: 1.05em; margin: 0.2em 0 0.4em 0}
.aec-note {color: #666; font-size: 0.9em}
"""


def _major() -> int:
    import gradio as gr
    try:
        return int(gr.__version__.split(".")[0])
    except Exception:
        return 6


def _blocks(title: str):
    """gr.Blocks with our CSS (Gradio < 6 takes css here, Gradio >= 6 takes it in launch())."""
    import gradio as gr
    return gr.Blocks(title=title) if _major() >= 6 else gr.Blocks(title=title, css=CSS)


def _launch(demo, key: str, share: Optional[bool], height: int):
    """Start (or restart) the app inline. Returns the Blocks object.

    `share=None` (the default) lets Gradio decide: on Colab it creates the temporary
    public link, which is the ONLY way the app works there — with share=False Gradio falls
    back to the Colab kernel proxy and the app loses its connection to the server.
    """
    if os.environ.get(NO_APP):
        print(f"({key}: app built but not started because {NO_APP} is set)")
        return demo
    old = _OPEN.pop(key, None)
    if old is not None:
        try:
            import contextlib
            import io
            with contextlib.redirect_stdout(io.StringIO()):   # hide "Closing server running on port ..."
                old.close()
        except Exception:
            pass
    kw = dict(inline=True, share=share, quiet=True, show_error=True, prevent_thread_lock=True, height=height)
    if _major() >= 6:
        kw["css"] = CSS
    demo.launch(**kw)
    _OPEN[key] = demo
    return demo


def _verdict_md(probs: Dict[str, float], truth: Optional[str], model_name: str, extra: str = "") -> str:
    top = max(probs, key=probs.get)
    txt = f"<b>{model_name} says: {top}</b> ({probs[top] * 100:.0f} % confident)"
    if truth is not None:
        ok = top == truth
        color = GREEN if ok else RED
        txt += f' &nbsp; <span style="color:{color}">{"✅ correct" if ok else "❌ really: " + truth}</span>'
    if extra:
        txt += f'<br><span class="aec-note">{extra}</span>'
    return f'<div class="aec-verdict">{txt}</div>'


def _editor_value(img: Image.Image):
    """A fresh drawing-pad value with `img` as background and no strokes."""
    return {"background": img.convert("RGBA"), "layers": [], "composite": img.convert("RGBA")}


def _editor_image(value) -> Optional[Image.Image]:
    """The picture the student sees in the drawing pad (background + strokes), as RGB."""
    if value is None:
        return None
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, np.ndarray):
        return Image.fromarray(value).convert("RGB")
    comp = value.get("composite") if isinstance(value, dict) else None
    if comp is None:
        comp = value.get("background") if isinstance(value, dict) else None
    if comp is None:
        return None
    if isinstance(comp, np.ndarray):
        comp = Image.fromarray(comp)
    if isinstance(comp, str):
        comp = Image.open(comp)
    return comp.convert("RGB")


# --------------------------------------------------------------------------- Step 3b
def playground_app(clf, image_set, seed: Optional[int] = None, share: Optional[bool] = None, height: int = 820,
                   key: str = "playground"):
    """Sliders that modify a test photo; the classifier re-runs on every change.
    Tab 2 is a drawing pad: paint a crack, a stain or a shadow yourself."""
    import gradio as gr

    rng = np.random.default_rng(seed)
    classes = ["any class"] + list(image_set.pretty_classes)
    L = PERTURB_LABELS
    slider_spec = [  # name, min, max, step, default
        ("rotate", -180, 180, 5, 0), ("zoom", 1.0, 6.0, 0.25, 1.0), ("blur", 0.0, 12.0, 0.5, 0.0),
        ("brightness", 0.1, 2.5, 0.1, 1.0), ("shadow", 0.0, 1.0, 0.1, 0.0), ("noise", 0.0, 0.6, 0.05, 0.0),
        ("line", 0, 20, 1, 0), ("line_angle", 0, 180, 15, 45),
    ]
    check_spec = [("flip", False), ("grayscale", False)]
    names = [s[0] for s in slider_spec] + [c[0] for c in check_spec]

    def pick(class_name):
        src = image_set if class_name == "any class" else image_set.only(class_name)
        k = int(rng.integers(len(src)))
        return {"img": src.load(k), "truth": src.pretty(src[k][1]), "name": src[k][0].name}

    def render(state, show_truth, *vals):
        kw = dict(zip(names, vals))
        im = perturb(state["img"], **kw)
        probs = clf.predict_one(im)
        changed = [L[k] for k in names if kw[k] != DEFAULTS[k] and not (k == "line_angle" and kw["line"] == 0)]
        extra = f"photo {state['name']} · changes: {', '.join(changed) if changed else 'none'}"
        truth = state["truth"] if show_truth else None
        return im, probs, _verdict_md(probs, truth, clf.name, extra)

    def another(class_name, show_truth, *vals):
        state = pick(class_name)
        im, probs, md = render(state, show_truth, *vals)
        return (state, im, probs, md, gr.update(value=_editor_value(state["img"])),
                "<div class='aec-note'>Fresh photo loaded into the drawing pad.</div>", {})

    def reset(state, show_truth):
        defaults = [s[4] for s in slider_spec] + [c[1] for c in check_spec]
        im, probs, md = render(state, show_truth, *defaults)
        return [*defaults, im, probs, md]

    def classify_drawing(value, state, show_truth):
        im = _editor_image(value)
        if im is None:
            return {}, "<div class='aec-note'>Draw something first (or load a photo).</div>"
        probs = clf.predict_one(im)
        truth = state["truth"] if show_truth else None
        return probs, _verdict_md(probs, truth, clf.name, "your drawing on " + state["name"])

    def load_current(state, *vals):
        kw = dict(zip(names, vals))
        im = perturb(state["img"], **kw)
        return gr.update(value=_editor_value(im)), "<div class='aec-note'>Loaded the photo with your slider changes. Now draw on it.</div>", {}

    state0 = pick("any class")
    with _blocks("Break it yourself") as demo:
        st = gr.State(state0)
        with gr.Row():
            pick_dd = gr.Dropdown(classes, value="any class", label="pick a test photo from", scale=2)
            show_truth = gr.Checkbox(value=True, label="show the true label", scale=1)
            another_btn = gr.Button("🎲 Another photo", variant="primary", scale=1)
        with gr.Tabs():
            with gr.Tab("🎚️ Sliders"):
                with gr.Row():
                    with gr.Column(scale=5):
                        img_out = gr.Image(type="pil", label="what the model sees", interactive=False, height=300)
                        verdict = gr.HTML()
                        label = gr.Label(num_top_classes=7, label="confidence")
                    with gr.Column(scale=4):
                        sliders = [gr.Slider(mn, mx, value=d, step=stp, label=L[n]) for n, mn, mx, stp, d in slider_spec]
                        checks = [gr.Checkbox(value=d, label=L[n]) for n, d in check_spec]
                        reset_btn = gr.Button("↺ reset sliders")
                gr.HTML("<div class='aec-note'>Try to flip the verdict with the <b>smallest</b> change. Every slider move re-runs the model.</div>")
            with gr.Tab("✏️ Draw on it"):
                gr.HTML("<div class='aec-note'>Use the brush to paint a crack, a stain or a shadow on the photo, then watch the verdict. "
                        "You can also upload or photograph your own wall and draw on that.</div>")
                with gr.Row():
                    with gr.Column(scale=5):
                        pad = gr.ImageEditor(value=_editor_value(state0["img"]), type="pil", label="drawing pad",
                                             brush=gr.Brush(default_size=4, colors=["#1a1414", "#6b5a4a", "#ffffff", "#2f7f3f"],
                                                            default_color="#1a1414"),
                                             sources=("upload", "webcam"), height=380)
                        with gr.Row():
                            load_btn = gr.Button("📥 Load the photo with my slider changes")
                            draw_btn = gr.Button("🔎 Classify my drawing", variant="primary")
                    with gr.Column(scale=4):
                        draw_verdict = gr.HTML()
                        draw_label = gr.Label(num_top_classes=7, label="confidence")

        controls = sliders + checks
        render_in = [st, show_truth, *controls]
        render_out = [img_out, label, verdict]
        # .change (not .release) so that dragging, typing a number and keyboard arrows all re-run the model;
        # always_last keeps only the newest request while one is running
        gr.on(triggers=[s.change for s in sliders] + [c.input for c in checks] + [show_truth.input],
              fn=render, inputs=render_in, outputs=render_out, api_name="render", trigger_mode="always_last")
        another_btn.click(another, inputs=[pick_dd, show_truth, *controls],
                          outputs=[st, *render_out, pad, draw_verdict, draw_label], api_name="another")
        reset_btn.click(reset, inputs=[st, show_truth], outputs=[*controls, *render_out], api_name="reset")
        pad.change(classify_drawing, inputs=[pad, st, show_truth], outputs=[draw_label, draw_verdict],
                   trigger_mode="always_last", api_name="draw")
        draw_btn.click(classify_drawing, inputs=[pad, st, show_truth], outputs=[draw_label, draw_verdict], api_name="draw_click")
        load_btn.click(load_current, inputs=[st, *controls], outputs=[pad, draw_verdict, draw_label], api_name="load_current")
        demo.load(render, inputs=render_in, outputs=render_out, api_name="first_render")

    demo._aec = {"render": render, "another": another, "reset": reset, "classify_drawing": classify_drawing,
                 "load_current": load_current, "state0": state0, "names": names}  # for tests
    return _launch(demo, key, share, height)


# --------------------------------------------------------------------------- Step 5a
def zero_shot_app(get_zs: Callable, image_set, tricky_items: Sequence[dict], default_names: str,
                  how_many: int = 8, share: Optional[bool] = None, height: int = 820, key: str = "zero_shot"):
    """Type class names -> CLIP picks one for each photo. The photos stay the same until the
    student asks for new ones, so changing the wording shows its effect directly."""
    import gradio as gr
    from .zeroshot import parse_classes

    rng = np.random.default_rng()

    def sample(source, n):
        n = int(n)
        if str(source).startswith("tricky"):
            items = list(tricky_items)[:n]
            images = [Image.open(m["path"]).convert("RGB") for m in items]
            captions = [m["caption"] for m in items]
        else:
            s = image_set.sample(n, seed=int(rng.integers(1_000_000)))
            images = [s.load(i) for i in range(len(s))]
            captions = [f"label: {s.pretty(s[i][1])}" for i in range(len(s))]
        return {"images": images, "captions": captions, "source": source}

    def classify(state, class_text):
        names = parse_classes(class_text)
        if len(names) < 2:
            return [(im, cap) for im, cap in zip(state["images"], state["captions"])], \
                "<div class='aec-note'>Type at least two class names, separated by commas.</div>"
        zs = get_zs()
        probs = zs.predict(state["images"], names)
        gallery, tally = [], {}
        for im, cap, p in zip(state["images"], state["captions"], probs):
            j1, j2 = np.argsort(-p)[:2]
            tally[names[j1]] = tally.get(names[j1], 0) + 1
            gallery.append((im, f"{names[j1]} ({p[j1] * 100:.0f} %) · then {names[j2]} ({p[j2] * 100:.0f} %) · {cap}"))
        counts = " · ".join(f"<b>{k}</b>: {v}" for k, v in sorted(tally.items(), key=lambda kv: -kv[1]))
        return gallery, f"<div class='aec-verdict'>CLIP's picks for these {len(gallery)} photos — {counts}</div>"

    def new_photos(source, n, class_text):
        state = sample(source, n)
        gallery, md = classify(state, class_text)
        return state, gallery, md

    def classify_own(img, class_text):
        names = parse_classes(class_text)
        if img is None or len(names) < 2:
            return {}
        p = get_zs().predict([img], names)[0]
        return {n: float(v) for n, v in zip(names, p)}

    state0 = sample("test photos", how_many)
    with _blocks("Type your own classes") as demo:
        st = gr.State(state0)
        gr.HTML("<div class='aec-note'>Type class names, click <b>Classify</b>, then change the wording and click again: "
                "the photos stay the same, so you see exactly what your words changed. The first click loads CLIP (about a minute).</div>")
        with gr.Row():
            names_tb = gr.Textbox(value=default_names, label="your class names (separate with commas)", scale=4)
            go_btn = gr.Button("🔎 Classify", variant="primary", scale=1)
        with gr.Tabs():
            with gr.Tab("📷 Photos from the lab"):
                with gr.Row():
                    source = gr.Radio(["test photos", "tricky photos"], value="test photos", label="which photos", scale=2)
                    n_sl = gr.Slider(4, 16, value=how_many, step=4, label="how many", scale=2)
                    new_btn = gr.Button("🎲 New photos", scale=1)
                summary = gr.HTML()
                gallery = gr.Gallery(value=[(im, cap) for im, cap in zip(state0["images"], state0["captions"])],
                                     columns=4, height=460, label="click a photo to enlarge", object_fit="contain")
            with gr.Tab("🖼️ Your own photo"):
                with gr.Row():
                    own = gr.Image(type="pil", label="upload, paste or photograph", sources=["upload", "webcam", "clipboard"], height=320)
                    own_label = gr.Label(label="CLIP's confidence for each of your class names")
                own_btn = gr.Button("🔎 Classify my photo", variant="primary")

        go_btn.click(classify, inputs=[st, names_tb], outputs=[gallery, summary], api_name="classify")
        names_tb.submit(classify, inputs=[st, names_tb], outputs=[gallery, summary], api_name="classify_submit")
        new_btn.click(new_photos, inputs=[source, n_sl, names_tb], outputs=[st, gallery, summary], api_name="new_photos")
        own_btn.click(classify_own, inputs=[own, names_tb], outputs=own_label, api_name="classify_own")
        own.change(classify_own, inputs=[own, names_tb], outputs=own_label, api_name="own_change")

    demo._aec = {"classify": classify, "new_photos": new_photos, "classify_own": classify_own, "state0": state0}
    return _launch(demo, key, share, height)
