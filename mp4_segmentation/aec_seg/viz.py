"""Overlays, panels and charts for the segmentation lab."""
import io
import math
from typing import Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .engine import SegResult


def hex_rgb(h: str) -> Tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def show(fig, dpi: int = 100):
    import matplotlib.pyplot as plt
    from IPython.display import Image as IPImage, display
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    display(IPImage(data=buf.getvalue()))


def show_image(img: Image.Image, max_width: int = 900):
    from IPython.display import Image as IPImage, display
    im = img
    if im.width > max_width:
        im = im.resize((max_width, round(im.height * max_width / im.width)), Image.BILINEAR)
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=88)
    display(IPImage(data=buf.getvalue()))


def _font(size: int):
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "Arial.ttf", "LiberationSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _outline(mask: np.ndarray) -> np.ndarray:
    """1-pixel-ish boundary of a boolean mask (no scipy needed)."""
    m = mask.astype(bool)
    er = m.copy()
    er[1:, :] &= m[:-1, :]; er[:-1, :] &= m[1:, :]; er[:, 1:] &= m[:, :-1]; er[:, :-1] &= m[:, 1:]
    return m & ~er


def overlay(image: Image.Image, mask: np.ndarray, color: str, alpha: float = 0.55, outline: bool = True) -> Image.Image:
    im = np.asarray(image.convert("RGB")).astype(np.float32)
    rgb = np.array(hex_rgb(color), dtype=np.float32)
    m = mask.astype(bool)
    im[m] = im[m] * (1 - alpha) + rgb * alpha
    if outline and m.any():
        edge = _outline(m)
        for _ in range(2):
            edge = edge | np.roll(edge, 1, 0) | np.roll(edge, 1, 1)
        im[edge] = rgb * 0.6
    return Image.fromarray(np.clip(im, 0, 255).astype(np.uint8))


def multi_overlay(image: Image.Image, layers: Sequence[Tuple[str, np.ndarray, str]], alpha: float = 0.5) -> Image.Image:
    """layers: (name, mask, color) drawn in order; later layers win where they overlap."""
    im = np.asarray(image.convert("RGB")).astype(np.float32)
    for name, mask, color in layers:
        m = mask.astype(bool)
        if not m.any():
            continue
        rgb = np.array(hex_rgb(color), dtype=np.float32)
        im[m] = im[m] * (1 - alpha) + rgb * alpha
    return Image.fromarray(np.clip(im, 0, 255).astype(np.uint8))


def mask_image(mask: np.ndarray, color: str = "#ffffff") -> Image.Image:
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[mask.astype(bool)] = hex_rgb(color)
    return Image.fromarray(out)


def instances_image(image: Image.Image, res: SegResult, min_score: float = 0.0, alpha: float = 0.55) -> Image.Image:
    """Each instance in its own colour with its number and confidence."""
    palette = ["#e63946", "#f4a261", "#ffd166", "#06d6a0", "#4cc9f0", "#7b2cbf", "#ff70a6", "#c9a227", "#2a9d8f", "#bde0fe"]
    im = np.asarray(image.convert("RGB")).astype(np.float32)
    labels = []
    k = 0
    for i in range(len(res)):
        if res.scores[i] < min_score:
            continue
        col = np.array(hex_rgb(palette[k % len(palette)]), dtype=np.float32)
        m = res.masks[i]
        im[m] = im[m] * (1 - alpha) + col * alpha
        labels.append((res.boxes[i], f"#{k + 1} {res.scores[i] * 100:.0f}%", palette[k % len(palette)]))
        k += 1
    out = Image.fromarray(np.clip(im, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(out)
    fs = max(12, out.width // 60)
    font = _font(fs)
    for box, text, col in labels:
        x1, y1 = float(box[0]), float(box[1])
        tw = d.textlength(text, font=font)
        d.rectangle([x1, y1, x1 + tw + 8, y1 + fs + 6], fill=col)
        d.text((x1 + 4, y1 + 2), text, fill="black", font=font)
    return out


def three_panel(image: Image.Image, res: Optional[SegResult], color: str, title: str, min_score: float = 0.0,
                size: float = 4.2, what: str = "image"):
    import matplotlib.pyplot as plt
    mask = res.union(min_score) if res is not None else np.zeros((image.height, image.width), bool)
    n = int((res.scores >= min_score).sum()) if res is not None else 0
    fig, axes = plt.subplots(1, 3, figsize=(size * 3 + 0.4, size * image.height / image.width + 0.9))
    for ax, im, t in zip(axes, [image, mask_image(mask, color), overlay(image, mask, color)],
                         [f"the {what}", f"mask: {n} region(s) found", f"overlay: {mask.mean() * 100:.1f}% of the {what}"]):
        ax.imshow(im); ax.set_title(t, fontsize=10); ax.axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


def image_grid(images: Sequence[Image.Image], titles: Optional[Sequence[str]] = None, ncols: int = 3, size: float = 3.8,
               suptitle: Optional[str] = None, fontsize: float = 8.5):
    import matplotlib.pyplot as plt
    import textwrap
    n = len(images)
    ncols = max(1, min(ncols, n)); nrows = math.ceil(n / ncols)
    wrapped = [textwrap.fill(t, int(size * 15), break_long_words=False) for t in titles] if titles else None
    max_lines = max((t.count("\n") + 1 for t in wrapped), default=0) if wrapped else 0
    fig, axes = plt.subplots(nrows, ncols, figsize=(size * ncols + 0.3, (size * 0.78 + 0.16 * max_lines + 0.3) * nrows + (0.4 if suptitle else 0)))
    axes = np.array(axes).reshape(-1)
    for k, ax in enumerate(axes):
        ax.axis("off")
        if k < n:
            ax.imshow(images[k])
            if wrapped:
                ax.set_title(wrapped[k], fontsize=fontsize, linespacing=1.25)
    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout(h_pad=0.6, w_pad=0.4)
    return fig
