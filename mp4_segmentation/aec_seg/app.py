"""Gradio app: your own photo + your own phrase -> mask, overlay and area share."""


def launch(lab, share: bool = True):
    import gradio as gr
    from . import viz

    def fn(img, phrase, conf):
        if img is None or not phrase.strip():
            return None, None, ""
        im = img.convert("RGB")
        if max(im.size) > 1280:
            s = 1280 / max(im.size); im = im.resize((round(im.width * s), round(im.height * s)))
        res = lab.engine.segment(im, phrase.strip(), threshold=0.2)
        mask = res.union(conf)
        n = int((res.scores >= conf).sum())
        return viz.mask_image(mask), viz.overlay(im, mask, "#ff70a6"), f"'{phrase}': {n} region(s), {mask.mean() * 100:.1f}% of the photo (SAM 3 in {res.seconds:.1f} s)"

    demo = gr.Interface(
        fn=fn,
        inputs=[gr.Image(type="pil", label="Your photo (upload, paste, or camera)", sources=["upload", "webcam", "clipboard"]),
                gr.Textbox(value="concrete", label="What should SAM 3 find? (a short phrase)"),
                gr.Slider(0.2, 0.9, value=0.5, step=0.05, label="confidence threshold")],
        outputs=[gr.Image(type="pil", label="mask"), gr.Image(type="pil", label="overlay"), gr.Textbox(label="measurement")],
        title="Segment anything you name, measure its share of the photo",
        description="Type a material or object (concrete, rebar, brick wall, window...), move the threshold, compare wordings.",
    )
    demo.launch(share=share, inline=True, debug=False, quiet=True, show_error=True)
    return demo
