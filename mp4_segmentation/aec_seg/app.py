"""Gradio app: your own drawing + your own phrase -> mask, overlay and a count / area in square feet."""


def launch(lab, share=None):
    import gradio as gr
    from . import viz

    def fn(img, phrase, conf, px_per_ft):
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
        if px_per_ft and px_per_ft > 0:
            txt += f" = {px / (px_per_ft ** 2):,.1f} sq ft at {px_per_ft:.2f} pixels per foot"
        else:
            txt += " (enter the scale, in pixels per foot, to get square feet)"
        return viz.mask_image(mask), viz.overlay(im, mask, "#ff70a6"), txt + f"  (SAM 3 in {res.seconds:.1f} s)"

    demo = gr.Interface(
        fn=fn,
        inputs=[gr.Image(type="pil", label="Your drawing (upload, paste, or photograph a sheet)", sources=["upload", "webcam", "clipboard"]),
                gr.Textbox(value="room", label="What should SAM 3 find? (a short phrase)"),
                gr.Slider(0.2, 0.9, value=0.4, step=0.05, label="confidence threshold"),
                gr.Number(value=0, label="scale: pixels per foot (0 = unknown)")],
        outputs=[gr.Image(type="pil", label="mask"), gr.Image(type="pil", label="overlay"), gr.Textbox(label="measurement")],
        title="Segment what you name on your own drawing",
        description="Type a room or an object (room, empty room, square, circle, curved line, thick black line...), move the "
                    "threshold, compare wordings. To get square feet, measure a printed dimension on your sheet first: count "
                    "the pixels along it, divide by its length in feet, and enter that number.",
    )
    # not embedded in the cell (the embedded frame is unreliable in Colab): the cell prints a link to open in a new tab
    import contextlib
    import io
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
