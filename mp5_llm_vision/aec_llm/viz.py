"""Pictures and tables for the LLM vision lab: boxes and polygons drawn on photos, image grids, small charts."""
import io
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

GREEN, RED, BLUE, GREY = "#2dc653", "#e63946", "#0057ff", "#888888"


def hex_rgb(h: str) -> Tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _font(size: int):
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "Arial.ttf", "LiberationSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


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
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=88)
    display(IPImage(data=buf.getvalue()))


def draw_boxes(image: Image.Image, boxes: Sequence[Tuple[str, Sequence[float], str, int]], font_size: Optional[int] = None) -> Image.Image:
    """boxes: (text, [x1, y1, x2, y2], colour, width). Text is drawn on a filled tag above the box."""
    im = image.convert("RGB").copy()
    d = ImageDraw.Draw(im)
    fs = font_size or max(11, im.width // 60)
    font = _font(fs)
    for text, b, color, width in boxes:
        text = " ".join(str(text).split())        # labels off a drawing can carry line breaks, which cannot be measured
        x1, y1, x2, y2 = [float(v) for v in b]
        d.rectangle([x1, y1, x2, y2], outline=color, width=width)
        if text:
            tw = d.textlength(text, font=font)
            ty = y1 - fs - 4 if y1 - fs - 4 > 0 else y1 + 2
            d.rectangle([x1, ty, x1 + tw + 6, ty + fs + 4], fill=color)
            d.text((x1 + 3, ty + 1), text, fill="white", font=font)
    return im


def draw_polygons(image: Image.Image, polys: Sequence[Tuple[str, Sequence[Sequence[float]], str]], alpha: float = 0.45) -> Image.Image:
    """polys: (text, [[x, y], ...] in pixels, colour): filled translucent polygons with an outline and a label."""
    base = image.convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    font = _font(max(11, base.width // 60))
    for text, pts, color in polys:
        text = " ".join(str(text).split())        # see draw_boxes
        if len(pts) < 3:
            continue
        rgb = hex_rgb(color)
        d.polygon([tuple(p) for p in pts], fill=rgb + (int(255 * alpha),), outline=rgb + (255,))
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
        if text:
            d.text((cx - d.textlength(text, font=font) / 2, cy - 7), text, fill=(0, 0, 0, 255), font=font)
    return Image.alpha_composite(base, layer).convert("RGB")


def mask_overlay(image: Image.Image, mask: np.ndarray, color: str, alpha: float = 0.5) -> Image.Image:
    im = np.asarray(image.convert("RGB")).astype(np.float32)
    rgb = np.array(hex_rgb(color), dtype=np.float32)
    m = mask.astype(bool)
    im[m] = im[m] * (1 - alpha) + rgb * alpha
    return Image.fromarray(im.astype(np.uint8))


def image_grid(images: Sequence[Image.Image], titles: Optional[Sequence[str]] = None, ncols: int = 4, size: float = 3.2,
               suptitle: Optional[str] = None, title_size: int = 9):
    import matplotlib.pyplot as plt
    n = len(images)
    ncols = min(ncols, max(1, n))
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(size * ncols, size * nrows * 1.05))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, im, t in zip(axes, images, titles or [""] * n):
        ax.imshow(im)
        if t:
            ax.set_title(t, fontsize=title_size, wrap=True)
    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    return fig


def bar_compare(groups: Dict[str, float], title: str = "", ylabel: str = "%", colors: Optional[Sequence[str]] = None, figsize=(5.5, 3)):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=figsize)
    names = list(groups)
    vals = [groups[k] for k in names]
    ax.bar(names, vals, color=colors or ["#457b9d", "#f4a261", "#2a9d8f", "#e9c46a"][:len(names)])
    for i, v in enumerate(vals):
        ax.text(i, v + 1, f"{v:.0f}", ha="center", fontsize=9)
    ax.set_ylabel(ylabel); ax.set_title(title, fontsize=10)
    ax.set_ylim(0, max(100, max(vals) + 10) if vals else 100)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right", fontsize=8)
    fig.tight_layout()
    return fig


def confusion_text(truths: Sequence[str], preds: Sequence[str], classes: Sequence[str]) -> str:
    """A small confusion matrix as text (rows: truth, columns: prediction)."""
    short = {c: (c[:10] + "…" if len(c) > 11 else c) for c in classes}
    extra = sorted({p for p in preds if p not in classes})
    cols = list(classes) + extra
    w = max(11, max(len(short.get(c, c[:11])) for c in cols) + 1)
    head = "truth \\ predicted".ljust(24) + "".join(short.get(c, c[:11]).rjust(w) for c in cols)
    lines = [head]
    for t in classes:
        row = [sum(1 for tt, pp in zip(truths, preds) if tt == t and pp == c) for c in cols]
        lines.append(t[:23].ljust(24) + "".join((str(v) if v else "·").rjust(w) for v in row))
    return "\n".join(lines)


def table(rows: Sequence[Sequence], headers: Sequence[str], widths: Optional[Sequence[int]] = None) -> str:
    widths = list(widths) if widths else [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    def fmt(vals):
        return "  ".join(str(v)[:w].ljust(w) for v, w in zip(vals, widths))
    return "\n".join([fmt(headers), fmt(["-" * w for w in widths])] + [fmt(r) for r in rows])
