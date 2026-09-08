"""Draw boxes on PIL images (no matplotlib needed for the image itself)."""
from typing import Dict, Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_FONT_CACHE = {}


def _font(size: int):
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    f = None
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "Arial.ttf", "arial.ttf", "LiberationSans-Bold.ttf"):
        try:
            f = ImageFont.truetype(name, size)
            break
        except Exception:
            continue
    if f is None:
        f = ImageFont.load_default()
    _FONT_CACHE[size] = f
    return f


def _dashed_rect(d: ImageDraw.ImageDraw, box, color, width, dash=10):
    x1, y1, x2, y2 = box
    segs = []
    for x in np.arange(x1, x2, dash * 2):
        segs.append([(x, y1), (min(x + dash, x2), y1)]); segs.append([(x, y2), (min(x + dash, x2), y2)])
    for y in np.arange(y1, y2, dash * 2):
        segs.append([(x1, y), (x1, min(y + dash, y2))]); segs.append([(x2, y), (x2, min(y + dash, y2))])
    for s in segs:
        d.line(s, fill=color, width=width)


def draw_boxes(img: Image.Image, boxes: np.ndarray, labels: Sequence[str], colors: Sequence[str],
               dashed: Optional[Sequence[bool]] = None, thickness: Optional[int] = None) -> Image.Image:
    im = img.convert("RGB").copy()
    d = ImageDraw.Draw(im)
    w, h = im.size
    t = thickness or max(2, round(min(w, h) / 220))
    fs = max(11, round(min(w, h) / 34))
    font = _font(fs)
    for k, (box, lab, col) in enumerate(zip(boxes, labels, colors)):
        x1, y1, x2, y2 = [float(v) for v in box]
        if dashed is not None and dashed[k]:
            _dashed_rect(d, (x1, y1, x2, y2), col, t)
        else:
            d.rectangle([x1, y1, x2, y2], outline=col, width=t)
        if lab:
            tw = d.textlength(lab, font=font)
            th = fs + 4
            ty = y1 - th if y1 - th >= 0 else y1
            d.rectangle([x1, ty, x1 + tw + 6, ty + th], fill=col)
            d.text((x1 + 3, ty + 1), lab, fill="black", font=font)
    return im


def draw_predictions(img: Image.Image, det, class_names: Sequence[str], colors: Dict[str, str], conf: float = 0.5,
                     show_conf: bool = True) -> Image.Image:
    d = det.at(conf)
    labels = [f"{class_names[int(c)]} {p * 100:.0f}%" if show_conf else class_names[int(c)] for c, p in zip(d.cls, d.conf)]
    cols = [colors.get(class_names[int(c)], "#ffffff") for c in d.cls]
    return draw_boxes(img, d.xyxy, labels, cols)


def draw_ground_truth(img: Image.Image, item, class_names: Sequence[str], colors: Dict[str, str], label: bool = True) -> Image.Image:
    labels = [class_names[int(c)] if label else "" for c in item.cls]
    cols = [colors.get(class_names[int(c)], "#ffffff") for c in item.cls]
    return draw_boxes(img, item.xyxy(), labels, cols, dashed=[True] * len(labels))


def draw_errors(img: Image.Image, det, matched_pred: np.ndarray, item, matched_gt: np.ndarray, class_names, conf: float):
    """Correct detections: plain green box. False alarms: orange with label. Missed ground truth: red dashed with label."""
    boxes, labels, cols, dashed = [], [], [], []
    d = det.at(conf)
    for k in range(len(d)):
        ok = bool(matched_pred[k])
        boxes.append(d.xyxy[k])
        labels.append("" if ok else f"FALSE ALARM: {class_names[int(d.cls[k])]}")
        cols.append("#2dc653" if ok else "#f77f00"); dashed.append(False)
    gt = item.xyxy()
    for k in range(len(gt)):
        if not matched_gt[k]:
            boxes.append(gt[k]); labels.append(f"MISSED: {class_names[int(item.cls[k])]}"); cols.append("#e63946"); dashed.append(True)
    return draw_boxes(img, np.array(boxes) if boxes else np.zeros((0, 4)), labels, cols, dashed)
