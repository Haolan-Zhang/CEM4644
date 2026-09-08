"""Image perturbations ("break the model" sliders) and the hand-picked tricky gallery."""
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps


def perturb(img: Image.Image, rotate: float = 0, blur: float = 0, brightness: float = 1.0, zoom: float = 1.0,
            noise: float = 0.0, line: int = 0, line_angle: float = 45, flip: bool = False, grayscale: bool = False,
            shadow: float = 0.0) -> Image.Image:
    """All arguments at their defaults return the image unchanged."""
    im = img.convert("RGB")
    w, h = im.size
    if flip:
        im = ImageOps.mirror(im)
    if zoom and zoom != 1.0:
        cw, ch = max(8, int(w / zoom)), max(8, int(h / zoom))
        l, t = (w - cw) // 2, (h - ch) // 2
        im = im.crop((l, t, l + cw, t + ch)).resize((w, h), Image.BICUBIC)
    if rotate:
        im = im.rotate(rotate, resample=Image.BICUBIC, expand=False, fillcolor=(128, 128, 128))
    if brightness != 1.0:
        im = ImageEnhance.Brightness(im).enhance(brightness)
    if shadow:
        # dark diagonal band, like a cast shadow
        mask = Image.new("L", (w, h), 0)
        d = ImageDraw.Draw(mask)
        d.polygon([(0, int(h * 0.15)), (w, int(h * 0.55)), (w, int(h * 0.95)), (0, int(h * 0.55))], fill=int(255 * min(1, shadow)))
        mask = mask.filter(ImageFilter.GaussianBlur(w * 0.02))
        dark = ImageEnhance.Brightness(im).enhance(0.35)
        im = Image.composite(dark, im, mask)
    if blur:
        im = im.filter(ImageFilter.GaussianBlur(blur))
    if grayscale:
        im = ImageOps.grayscale(im).convert("RGB")
    if noise:
        a = np.asarray(im).astype(np.float32)
        a += np.random.default_rng(0).normal(0, noise * 255, a.shape)
        im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if line:
        d = ImageDraw.Draw(im)
        ang = math.radians(line_angle)
        cx, cy = w / 2, h / 2
        L = max(w, h)
        p1 = (cx - L * math.cos(ang), cy - L * math.sin(ang))
        p2 = (cx + L * math.cos(ang), cy + L * math.sin(ang))
        d.line([p1, p2], fill=(25, 20, 20), width=int(line))
    return im


DEFAULTS = {"rotate": 0, "zoom": 1.0, "blur": 0, "brightness": 1.0, "shadow": 0, "noise": 0, "line": 0,
            "line_angle": 45, "flip": False, "grayscale": False}

PERTURB_LABELS = {
    "rotate": "rotate (°)", "blur": "blur", "brightness": "brightness", "zoom": "zoom in", "noise": "noise",
    "line": "draw a dark line (width)", "line_angle": "line angle (°)", "shadow": "cast a shadow", "flip": "mirror", "grayscale": "black & white",
}


def playground(clf, image_set, seed: Optional[int] = None, extra_classifier=None):
    """Sliders that modify a photo live; the classifier re-runs on every change."""
    import ipywidgets as w
    from IPython.display import display
    from . import ui
    rng = np.random.default_rng(seed)
    state = {"img": None, "truth": None, "name": ""}

    def new_photo(*_):
        k = int(rng.integers(len(image_set)))
        state["img"] = image_set.load(k)
        state["truth"] = image_set.pretty(image_set[k][1])
        state["name"] = image_set[k][0].name
        render()

    L = PERTURB_LABELS
    sl = dict(
        rotate=w.IntSlider(value=0, min=-180, max=180, step=5, description=L["rotate"], continuous_update=False),
        zoom=w.FloatSlider(value=1.0, min=1.0, max=6.0, step=0.25, description=L["zoom"], continuous_update=False),
        blur=w.FloatSlider(value=0.0, min=0.0, max=12.0, step=0.5, description=L["blur"], continuous_update=False),
        brightness=w.FloatSlider(value=1.0, min=0.1, max=2.5, step=0.1, description=L["brightness"], continuous_update=False),
        shadow=w.FloatSlider(value=0.0, min=0.0, max=1.0, step=0.1, description=L["shadow"], continuous_update=False),
        noise=w.FloatSlider(value=0.0, min=0.0, max=0.6, step=0.05, description=L["noise"], continuous_update=False),
        line=w.IntSlider(value=0, min=0, max=20, step=1, description=L["line"], continuous_update=False),
        line_angle=w.IntSlider(value=45, min=0, max=180, step=15, description=L["line_angle"], continuous_update=False),
        flip=w.Checkbox(value=False, description=L["flip"]),
        grayscale=w.Checkbox(value=False, description=L["grayscale"]),
    )
    for s in sl.values():
        s.style = {"description_width": "150px"}
    reset = w.Button(description="↺ reset sliders")
    another = w.Button(description="🎲 another photo", button_style="primary")
    out = w.Output()

    def render(*_):
        if state["img"] is None:
            return
        kw = {k: v.value for k, v in sl.items()}
        im = perturb(state["img"], **kw)
        changed = [PERTURB_LABELS[k] for k, v in sl.items()
                   if v.value != DEFAULTS[k] and not (k == "line_angle" and sl["line"].value == 0)]
        with out:
            out.clear_output(wait=True)
            ui.show(ui.prediction_figure(im, clf.predict_one(im), state["truth"],
                                         title=f"{state['name']}\nchanges: {', '.join(changed) if changed else 'none'}",
                                         model_name=clf.name))
            if extra_classifier is not None:
                ui.show(ui.prediction_figure(im, extra_classifier.predict_one(im), None, title="same photo, other model",
                                             model_name=extra_classifier.name))

    def do_reset(_):
        for k, s in sl.items():
            s.unobserve(render, names="value")
        for k, s in sl.items():
            s.value = DEFAULTS[k]
        for k, s in sl.items():
            s.observe(render, names="value")
        render()

    for s in sl.values():
        s.observe(render, names="value")
    reset.on_click(do_reset); another.on_click(new_photo)
    left = w.VBox(list(sl.values()) + [w.HBox([another, reset])])
    display(w.HBox([left, out]))
    new_photo()


# --------------------------------------------------------------------------- tricky gallery
def load_tricky(folder: Path) -> List[Dict]:
    folder = Path(folder)
    meta = json.loads((folder / "tricky.json").read_text())
    for m in meta:
        m["path"] = folder / m["file"]
    return meta


def show_tricky(clfs, folder: Path, task: str, group: Optional[str] = None, ncols: int = 4):
    """task = 'binary' or 'multiclass': picks the matching 'expected' label from tricky.json."""
    from . import ui
    items = [m for m in load_tricky(folder) if group in (None, "all") or m.get("group") == group]
    images = [Image.open(m["path"]).convert("RGB") for m in items]
    expected = [m.get("expected", {}).get(task) for m in items]
    captions = []
    for m, e in zip(items, expected):
        if e == "?":
            hint = "(you decide what it should be)"
        elif e:
            hint = f"(should be: {e})"
        else:
            hint = "(no right answer: not a valid photo for this task)"
        captions.append(f"#{m['id']} {m['caption']}\n{hint}")
    expected = [None if e == "?" else e for e in expected]
    clfs = clfs if isinstance(clfs, (list, tuple)) else [clfs]
    return ui.compare_models(clfs, images, captions, expected, ncols=ncols)
