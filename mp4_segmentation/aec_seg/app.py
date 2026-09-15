"""Gradio app: your own floor plan + your own phrase -> mask, overlay and a count / area."""


def launch(lab, share=None):
    import gradio as gr
    from . import viz

    def fn(img, phrase, conf, cm_per_px):
        if img is None or not phrase.strip():
            return None, None, ""
        im = img.convert("RGB")
        if max(im.size) > 1600:
            s = 1600 / max(im.size); im = im.resize((round(im.width * s), round(im.height * s)))
        res = lab.engine.segment(im, phrase.strip(), threshold=0.2)
        mask = res.union(conf)
        n = int((res.scores >= conf).sum())
        px = int(mask.sum())
        txt = f"'{phrase}': {n} region(s), {px:,} pixels = {mask.mean() * 100:.1f} % of the drawing"
        if cm_per_px and cm_per_px > 0:
            txt += f" = {px * (cm_per_px / 100) ** 2:.1f} m² at {cm_per_px:.2f} cm per pixel"
        else:
            txt += " (enter the scale, in cm per pixel, to get square metres)"
        return viz.mask_image(mask), viz.overlay(im, mask, "#ff70a6"), txt + f"  (SAM 3 in {res.seconds:.1f} s)"

    demo = gr.Interface(
        fn=fn,
        inputs=[gr.Image(type="pil", label="Your floor plan (upload, paste, or photograph a drawing)", sources=["upload", "webcam", "clipboard"]),
                gr.Textbox(value="room", label="What should SAM 3 find? (a short phrase)"),
                gr.Slider(0.2, 0.9, value=0.4, step=0.05, label="confidence threshold"),
                gr.Number(value=0, label="scale: centimetres per pixel (0 = unknown)")],
        outputs=[gr.Image(type="pil", label="mask"), gr.Image(type="pil", label="overlay"), gr.Textbox(label="measurement")],
        title="Segment what you name on your own plan",
        description="Type a room or an object (room, kitchen, toilet, stairs, window...), move the threshold, compare wordings. "
                    "If the plan has a scale bar, measure it (pixels per metre) to enter the scale.",
    )
    kw = dict(share=share, inline=True, debug=False, quiet=True, show_error=True, prevent_thread_lock=True)
    demo.launch(**kw)
    return demo
