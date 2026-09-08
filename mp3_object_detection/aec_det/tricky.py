"""Image perturbations, the 'break it yourself' playground and the tricky gallery for detection."""
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps

DEFAULTS = {"rotate": 0, "zoom": 1.0, "shrink": 1.0, "blur": 0, "brightness": 1.0, "shadow": 0, "noise": 0,
            "flip": False, "grayscale": False}
LABELS = {"rotate": "rotate (°)", "zoom": "zoom in", "shrink": "move away (shrink)", "blur": "blur", "brightness": "brightness",
          "shadow": "cast a shadow", "noise": "noise", "flip": "mirror", "grayscale": "black & white"}


def perturb(img: Image.Image, rotate: float = 0, zoom: float = 1.0, shrink: float = 1.0, blur: float = 0,
            brightness: float = 1.0, shadow: float = 0.0, noise: float = 0.0, flip: bool = False, grayscale: bool = False) -> Image.Image:
    im = img.convert("RGB")
    w, h = im.size
    if flip:
        im = ImageOps.mirror(im)
    if zoom and zoom != 1.0:
        cw, ch = max(8, int(w / zoom)), max(8, int(h / zoom))
        l, t = (w - cw) // 2, (h - ch) // 2
        im = im.crop((l, t, l + cw, t + ch)).resize((w, h), Image.BICUBIC)
    if shrink and shrink != 1.0:
        sw, sh = max(8, int(w / shrink)), max(8, int(h / shrink))
        small = im.resize((sw, sh), Image.LANCZOS)
        canvas = Image.new("RGB", (w, h), (120, 125, 130))
        canvas.paste(small, ((w - sw) // 2, (h - sh) // 2))
        im = canvas
    if rotate:
        im = im.rotate(rotate, resample=Image.BICUBIC, expand=False, fillcolor=(128, 128, 128))
    if brightness != 1.0:
        im = ImageEnhance.Brightness(im).enhance(brightness)
    if shadow:
        mask = Image.new("L", (w, h), 0)
        d = ImageDraw.Draw(mask)
        d.polygon([(0, int(h * 0.15)), (w, int(h * 0.55)), (w, int(h * 0.95)), (0, int(h * 0.55))], fill=int(255 * min(1, shadow)))
        mask = mask.filter(ImageFilter.GaussianBlur(w * 0.02))
        im = Image.composite(ImageEnhance.Brightness(im).enhance(0.35), im, mask)
    if blur:
        im = im.filter(ImageFilter.GaussianBlur(blur))
    if grayscale:
        im = ImageOps.grayscale(im).convert("RGB")
    if noise:
        a = np.asarray(im).astype(np.float32)
        a += np.random.default_rng(0).normal(0, noise * 255, a.shape)
        im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    return im


def playground(detector, det_set, spec, seed: Optional[int] = None):
    import ipywidgets as w
    from IPython.display import display
    from . import draw, ui
    rng = np.random.default_rng(seed)
    state = {"img": None, "name": "", "truth": ""}
    pretty_colors = {spec.pretty(c): v for c, v in spec.colors.items()}
    sl = dict(
        rotate=w.IntSlider(value=0, min=-180, max=180, step=5, description=LABELS["rotate"], continuous_update=False),
        zoom=w.FloatSlider(value=1.0, min=1.0, max=4.0, step=0.25, description=LABELS["zoom"], continuous_update=False),
        shrink=w.FloatSlider(value=1.0, min=1.0, max=8.0, step=0.5, description=LABELS["shrink"], continuous_update=False),
        blur=w.FloatSlider(value=0.0, min=0.0, max=10.0, step=0.5, description=LABELS["blur"], continuous_update=False),
        brightness=w.FloatSlider(value=1.0, min=0.1, max=2.5, step=0.1, description=LABELS["brightness"], continuous_update=False),
        shadow=w.FloatSlider(value=0.0, min=0.0, max=1.0, step=0.1, description=LABELS["shadow"], continuous_update=False),
        noise=w.FloatSlider(value=0.0, min=0.0, max=0.5, step=0.05, description=LABELS["noise"], continuous_update=False),
        flip=w.Checkbox(value=False, description=LABELS["flip"]),
        grayscale=w.Checkbox(value=False, description=LABELS["grayscale"]),
    )
    conf = w.FloatSlider(value=0.5, min=0.05, max=0.95, step=0.05, description="confidence ≥", continuous_update=False, readout_format=".2f")
    for s in list(sl.values()) + [conf]:
        s.style = {"description_width": "150px"}
    reset = w.Button(description="↺ reset sliders"); another = w.Button(description="🎲 another photo", button_style="primary")
    out = w.Output()

    def render(*_):
        if state["img"] is None:
            return
        kw = {k: v.value for k, v in sl.items()}
        im = perturb(state["img"], **kw)
        det = detector.predict(im)
        changed = [LABELS[k] for k, v in sl.items() if v.value != DEFAULTS[k]]
        with out:
            out.clear_output(wait=True)
            ui.show_image(draw.draw_predictions(im, det, detector.classes, pretty_colors, conf=conf.value), 760)
            print(f"changes: {', '.join(changed) if changed else 'none'}")
            print(f"{detector.name} (≥ {conf.value * 100:.0f}%): {ui.det_summary(det, detector.classes, conf.value)}")
            print(f"labelled truth for the original photo: {state['truth']}")

    def new_photo(*_):
        k = int(rng.integers(len(det_set)))
        it = det_set[k]
        state["img"] = it.load(); state["name"] = it.path.name; state["truth"] = ui.gt_summary(it, det_set.pretty_classes)
        render()

    def do_reset(_):
        for k, s in sl.items():
            s.unobserve(render, names="value"); s.value = DEFAULTS[k]; s.observe(render, names="value")
        render()

    for s in list(sl.values()) + [conf]:
        s.observe(render, names="value")
    reset.on_click(do_reset); another.on_click(new_photo)
    display(w.HBox([w.VBox(list(sl.values()) + [conf, w.HBox([another, reset])]), out]))
    new_photo()


def load_tricky(folder: Path) -> List[Dict]:
    folder = Path(folder)
    meta = json.loads((folder / "tricky.json").read_text())
    for m in meta:
        m["path"] = folder / m["file"]
    return meta


def show_tricky(detectors, folder: Path, spec, group: Optional[str] = None, conf: float = 0.5, ncols: int = 2):
    from . import draw, ui
    items = [m for m in load_tricky(folder) if group in (None, "all") or m.get("group") == group]
    detectors = detectors if isinstance(detectors, (list, tuple)) else [detectors]
    pretty_colors = {spec.pretty(c): v for c, v in spec.colors.items()}
    images, titles = [], []
    for m in items:
        img = Image.open(m["path"]).convert("RGB")
        for d in detectors:
            det = d.predict(img, key=f"tricky:{m['file']}:{d.name}")
            cols = {c: pretty_colors.get(c, spec.colors.get(c, "#ffffff")) for c in d.classes}
            images.append(draw.draw_predictions(img, det, d.classes, cols, conf=conf))
            exp = m.get("expected", "")
            titles.append(f"#{m['id']} {m['caption']}\n{('truth: ' + exp) if exp else '(no labels: judge it yourself)'}\n"
                          f"{d.name}: {ui.det_summary(det, d.classes, conf)}")
    ui.show(ui.image_grid(images, titles, ncols=ncols if len(detectors) == 1 else len(detectors), size=5.0,
                          suptitle=f"confidence threshold {conf * 100:.0f}%"))
