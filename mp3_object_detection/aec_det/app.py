"""Gradio app: upload / photograph an image, move the confidence slider, see the boxes."""
from typing import Dict


def launch(detectors: Dict[str, object], colors: Dict[str, str], title: str, description: str = "", share: bool = True):
    import gradio as gr
    from . import draw, ui
    names = list(detectors)

    def fn(img, which, conf):
        if img is None:
            return None, ""
        d = detectors[which]
        det = d.predict(img)
        cols = {c: colors.get(c, "#ffffff") for c in d.classes}
        return draw.draw_predictions(img, det, d.classes, cols, conf=conf), ui.det_summary(det, d.classes, conf)

    inputs = [gr.Image(type="pil", label="Your photo (upload, paste, or camera)", sources=["upload", "webcam", "clipboard"]),
              gr.Radio(names, value=names[0], label="Which model?"),
              gr.Slider(0.05, 0.95, value=0.5, step=0.05, label="confidence threshold")]
    outputs = [gr.Image(type="pil", label="What the model sees"), gr.Textbox(label="Counts")]
    kw = dict(fn=fn, inputs=inputs, outputs=outputs, title=title, description=description)
    try:
        demo = gr.Interface(**kw, flagging_mode="never")
    except TypeError:
        demo = gr.Interface(**kw, allow_flagging="never")
    demo.launch(share=share, inline=True, debug=False, quiet=True, show_error=True)
    return demo
