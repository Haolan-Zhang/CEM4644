"""Overlays, panels and charts for the segmentation lab."""
import io
import math
from typing import Dict, List, Optional, Sequence, Tuple

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


def three_panel(image: Image.Image, res: Optional[SegResult], color: str, title: str, min_score: float = 0.0, size: float = 4.2):
    import matplotlib.pyplot as plt
    mask = res.union(min_score) if res is not None else np.zeros((image.height, image.width), bool)
    n = int((res.scores >= min_score).sum()) if res is not None else 0
    fig, axes = plt.subplots(1, 3, figsize=(size * 3 + 0.4, size * image.height / image.width + 0.9))
    for ax, im, t in zip(axes, [image, mask_image(mask, color), overlay(image, mask, color)],
                         ["original photo", f"mask: {n} region(s) found", f"overlay: {mask.mean() * 100:.1f}% of the photo"]):
        ax.imshow(im); ax.set_title(t, fontsize=10); ax.axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


def bar_chart(values: Dict[str, float], colors: Dict[str, str], title: str = "", ylabel: str = "% of the photo", figsize=(7, 3)):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=figsize)
    names = list(values)
    ax.bar(names, [values[n] for n in names], color=[colors.get(n, "#8d99ae") for n in names])
    for i, n in enumerate(names):
        ax.text(i, values[n] + 0.5, f"{values[n]:.1f}", ha="center", fontsize=8)
    ax.set_ylabel(ylabel); ax.set_title(title, fontsize=10)
    ax.set_ylim(0, max(5.0, max(values.values()) * 1.15 if values else 5))
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right", fontsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


def grouped_chart(groups: Dict[str, Dict[str, float]], colors: Dict[str, str], title: str = "", figsize=(8, 3.4)):
    """groups: {photo label: {material: pct}} -> bars per material, one colour per photo? No: one bar group per material."""
    import matplotlib.pyplot as plt
    labels = list(groups)
    mats = list(next(iter(groups.values())).keys()) if groups else []
    x = np.arange(len(mats)); w = 0.8 / max(1, len(labels))
    fig, ax = plt.subplots(figsize=figsize)
    shades = ["#457b9d", "#e76f51", "#2a9d8f", "#f4a261", "#7b2cbf"]
    for j, lab in enumerate(labels):
        vals = [groups[lab].get(m, 0.0) for m in mats]
        ax.bar(x + (j - (len(labels) - 1) / 2) * w, vals, w, label=lab, color=shades[j % len(shades)])
    ax.set_xticks(x); ax.set_xticklabels(mats, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("% of the photo"); ax.set_title(title, fontsize=10); ax.legend(fontsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


def series_chart(rows: List[Tuple[str, Dict[str, float]]], colors: Dict[str, str], title: str = "", figsize=(9, 3.6)):
    """rows: [(time label, {material: pct})] -> one line per material over time."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=figsize)
    labels = [r[0] for r in rows]
    mats = list(rows[0][1].keys()) if rows else []
    for m in mats:
        ax.plot(range(len(rows)), [r[1].get(m, 0.0) for r in rows], marker="o", label=m, color=colors.get(m, None))
    ax.set_xticks(range(len(rows))); ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("% of the photo"); ax.set_title(title, fontsize=10); ax.grid(alpha=0.3); ax.legend(fontsize=8, ncol=2)
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
