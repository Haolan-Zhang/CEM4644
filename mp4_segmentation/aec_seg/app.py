"""Gradio app for the students' own drawings: the same three ways of asking as the notebook - a phrase, a box (an
area), one example box (a count) - with the scale set from a box along a printed dimension. Opened from a link,
never embedded in the cell (the embedded frame is unreliable in Colab)."""
import contextlib
import io
from typing import Optional

import numpy as np
from PIL import ImageDraw

from . import ui, viz

MAX_SIDE = 1600
KINDS = {"scale": "scale: a box along a printed dimension", "room": "room: the floor area inside my box",
         "object": "object: the area of the thing in my box", "example": "example: find everything like my box"}
COLORS = {"scale": ui.BLUE, "room": ui.GREEN, "object": "#f4a261", "example": ui.PURPLE, "phrase": "#ff70a6"}
GUIDE = ("**Upload a drawing** (a photo of a sheet works). To draw a box, **click twice on the drawing**: the top-left "
         "corner, then the bottom-right corner; the box takes the label chosen on the left. **scale**: box a printed "
         "dimension from arrowhead to arrowhead and enter its length in feet. **room**, **object**: SAM 3 measures what is "
         "inside the box. **example**: one box, and SAM 3 finds everything on the sheet that looks like it. Or type a "
         "**phrase**. Then *Run SAM 3*.")


def new_state():
    return {"image": None, "boxes": [], "pending": None}


def prepare(img):
    im = img.convert("RGB")
    if max(im.size) > MAX_SIDE:
        s = MAX_SIDE / max(im.size)
        im = im.resize((round(im.width * s), round(im.height * s)))
    return im


def kind_of(choice: str) -> str:
    return choice.split(":", 1)[0].strip()


def canvas(state):
    """The drawing with the boxes drawn so far (and a cross where the first corner was clicked)."""
    im = state["image"]
    if im is None:
        return None
    out = im.copy()
    d = ImageDraw.Draw(out)
    w = max(2, round(max(out.size) / 400))
    for i, (kind, b) in enumerate(state["boxes"], 1):
        d.rectangle(b, outline=COLORS[kind], width=w)
        d.text((b[0] + w + 2, b[1] + w + 2), f"{i} {kind}", fill=COLORS[kind], font=viz._font(max(12, round(max(out.size) / 70))))
    if state["pending"]:
        x, y = state["pending"]
        r = 4 * w
        d.line([(x - r, y), (x + r, y)], fill="#000000", width=w)
        d.line([(x, y - r), (x, y + r)], fill="#000000", width=w)
    return out


def on_upload(img, state):
    state = new_state()
    if img is not None:
        state["image"] = prepare(img)
    return canvas(state), state, None, ""


def on_click(state, choice, xy):
    """Two clicks make a box: the first corner is remembered, the second completes it with the chosen label."""
    if state["image"] is None:
        return None, state
    x, y = float(xy[0]), float(xy[1])
    if state["pending"] is None:
        state["pending"] = (x, y)
    else:
        x0, y0 = state["pending"]
        box = [min(x0, x), min(y0, y), max(x0, x), max(y0, y)]
        state["pending"] = None
        if box[2] - box[0] >= 3 and box[3] - box[1] >= 3:
            state["boxes"].append((kind_of(choice), box))
    return canvas(state), state


def on_undo(state):
    if state["pending"] is not None:
        state["pending"] = None
    elif state["boxes"]:
        state["boxes"].pop()
    return canvas(state), state


def on_clear(state):
    state["boxes"], state["pending"] = [], None
    return canvas(state), state


def run(engine, state, phrase: str, conf: float, feet: float):
    """Everything asked for, in the words of the notebook. Returns (overlay image, text)."""
    im = state["image"]
    if im is None:
        return None, "Upload a drawing first."
    boxes = state["boxes"]
    phrase = (phrase or "").strip()
    if not boxes and not phrase:
        return canvas(state), "Draw a box (two clicks) or type a phrase, then run."
    lines, layers, outlines = [], [], []
    for i, (kind, b) in enumerate(boxes, 1):
        outlines.append((b, COLORS[kind], f"{i} {kind}"))

    # ------------------------------------------------------------- the scale
    scales = [b for k, b in boxes if k == "scale"]
    px_per_ft: Optional[float] = None
    lines.append("SCALE CHECK")
    if scales and feet and feet > 0:
        ests = []
        for b in scales:
            px = max(b[2] - b[0], b[3] - b[1])            # the dimension runs along the long side of the box
            ests.append(px / feet)
            lines.append(f"  {feet:g} ft dimension: {px:.0f} px ÷ {feet:g} ft = {px / feet:.2f} px/ft")
        if len(ests) >= 2:
            lo, hi = min(ests), max(ests)
            lines.append(f"  Your scale estimates differ by {(hi - lo) / lo * 100:.0f}%. For more accurate results, draw the scale box precisely.")
        px_per_ft = ests[0]
        lines.append(f"  Scale used for takeoff: {px_per_ft:.2f} px/ft (from the first scale box)")
    elif scales:
        lines.append("  Enter the length of your scale box in feet (the box is there, the length is not).")
    else:
        lines.append("  No scale box: draw one along a printed dimension (label: scale) and enter its length in feet. "
                     "Areas below are in pixels.")

    def area(px: int) -> str:
        return f"{px / (px_per_ft ** 2):,.1f} sq ft" if px_per_ft else f"{px:,} px"

    # ------------------------------------------------------------- a phrase
    if phrase:
        res = engine.segment(im, phrase, threshold=0.1)
        keep = res.scores >= conf
        n = int(keep.sum())
        lines.append(f"\nPHRASE '{phrase}' (confidence >= {conf:.2f})")
        if n == 0:
            lines.append(f"  nothing found" + (f" (the model proposed {len(res)} weak region(s) below that confidence)." if len(res) else "."))
        else:
            px = int(res.union(conf).sum())
            lines.append(f"  {n} region(s), {area(px)} in total. Confidence of each: "
                         + ", ".join(f"{s * 100:.0f}%" for s in res.scores[keep][:10]) + (" ..." if n > 10 else ""))
            for m in res.masks[keep]:
                layers.append(("", m, COLORS["phrase"]))
        lines.append(f"  (SAM 3 took {res.seconds:.1f} s)")

    # ------------------------------------------------------------- rooms and objects: the area inside a box
    walls = None
    rows = [(i, k, b) for i, (k, b) in enumerate(boxes, 1) if k in ("room", "object")]
    if rows:
        lines.append("\nAREAS")
    for i, kind, b in rows:
        box_px = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
        if kind == "room":
            if walls is None:
                walls = ui.wall_pixels(im)
            mask, raw_px, score, filled_px = ui.room_mask(engine, im, b, walls)
            note = "" if mask.any() else "SAM 3 found nothing in this box."
            if mask.any() and filled_px > 1.15 * raw_px:
                note = (f"{(filled_px - raw_px) / filled_px * 100:.0f} % of this area is drawn over (furniture, a stair, door "
                        "swings) and was filled back in from the outline of the mask.")
        else:
            r = engine.segment_visual(im, box=b)
            mask = r.masks[0] if len(r) else np.zeros((im.height, im.width), bool)
            short = min(b[2] - b[0], b[3] - b[1])
            note = ""
            if short < ui.MIN_SYMBOL_PX:
                note = (f"your box is only {short:.0f} px across, and under about {ui.MIN_SYMBOL_PX} px a symbol is too small "
                        "for the model: the number is your box, not the thing.")
            elif mask.any() and mask.sum() > 0.9 * box_px:
                note = "the mask fills your box: look at the outline - if it is a rounded copy of your box, this number is your box."
        px = int(mask.sum())
        lines.append(f"  box {i} ({kind}): {area(px)}" + (f" -- {note}" if note else ""))
        if mask.any():
            layers.append((str(i), mask, COLORS[kind]))

    # ------------------------------------------------------------- one example box: SAM 3 finds the rest
    examples = [b for k, b in boxes if k == "example"]
    if examples:
        res = engine.segment_like(im, examples[0], threshold=conf, size_range=(0.2, 5.0))
        n = len(res)
        lines.append("\nWHAT SAM 3 FOUND FROM YOUR EXAMPLE BOX")
        lines.append(f"  Using your example box (purple), SAM 3 returned {n} region{'' if n == 1 else 's'} in {res.seconds:.1f} s "
                     f"(confidence >= {conf:.2f})." + (f" Together they measure {area(int(res.union(0).sum()))}." if n else ""))
        if len(examples) > 1:
            lines.append(f"  You drew {len(examples)} example boxes. Only the first one was used - one example is all the model needs.")
        for m, bb in zip(res.masks, res.boxes):
            layers.append(("", m, COLORS["example"]))
            outlines.append(([float(v) for v in bb], COLORS["example"], ""))

    out = viz.multi_overlay(im, layers, alpha=0.45) if layers else im.copy()
    d = ImageDraw.Draw(out)
    w = max(2, round(max(out.size) / 400))
    font = viz._font(max(12, round(max(out.size) / 70)))
    for b, color, text in outlines:
        d.rectangle(b, outline=color, width=w)
        if text:
            d.text((b[0] + w + 2, b[1] + w + 2), text, fill=color, font=font)
    return out, "\n".join(lines)


def build(lab):
    import gradio as gr
    engine = lab.engine

    def _click(state, choice, evt: gr.SelectData):
        return on_click(state, choice, evt.index)

    def _run(state, phrase, conf, feet):
        return run(engine, state, phrase, conf, feet)

    with gr.Blocks(title="Your drawing, asked three ways") as demo:
        gr.Markdown("## Your drawing, asked three ways\n" + GUIDE)
        state = gr.State(new_state())
        with gr.Row():
            with gr.Column(scale=1, min_width=280):
                upload = gr.Image(type="pil", label="Upload a drawing", sources=["upload", "clipboard", "webcam"], height=200)
                choice = gr.Radio(list(KINDS.values()), value=KINDS["room"], label="the next box is")
                feet = gr.Number(value=0, label="length of the scale box, in feet", precision=2)
                phrase = gr.Textbox(value="", label="or a phrase (room, square, circle, curved line, thick black line...)")
                conf = gr.Slider(0.1, 0.9, value=0.3, step=0.05, label="confidence >=")
                with gr.Row():
                    run_btn = gr.Button("Run SAM 3", variant="primary")
                    undo_btn = gr.Button("Undo last box")
                    clear_btn = gr.Button("Clear boxes")
            with gr.Column(scale=2):
                canvas_img = gr.Image(type="pil", label="Your drawing: click twice to draw a box (top-left, then bottom-right)",
                                      interactive=False)
                out_img = gr.Image(type="pil", label="What SAM 3 found", interactive=False)
                out_txt = gr.Textbox(label="Result", lines=12)
        upload.change(on_upload, [upload, state], [canvas_img, state, out_img, out_txt])
        canvas_img.select(_click, [state, choice], [canvas_img, state])
        undo_btn.click(on_undo, [state], [canvas_img, state])
        clear_btn.click(on_clear, [state], [canvas_img, state])
        run_btn.click(_run, [state, phrase, conf, feet], [out_img, out_txt])
    return demo


def launch(lab, share=None):
    demo = build(lab)
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        demo.launch(share=True if share is None else share, inline=False, debug=False, quiet=True, show_error=True, prevent_thread_lock=True)
    url = getattr(demo, "share_url", None)
    if url:
        print(f"Open the app in a new tab (works on a phone too): {url}")
        print("The link stays alive while this notebook is running.")
    else:
        print("The public link could not be created (network hiccup). Run this cell again; "
              f"the app is running at {getattr(demo, 'local_url', 'the local address')}, which only works from this machine.")
    return demo
