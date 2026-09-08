"""A tiny Gradio app: drag-and-drop or phone-camera photos -> model verdict."""
from typing import Dict


def launch(clfs: Dict[str, object], title: str = "Test the model on your own photo", share: bool = True,
           description: str = ""):
    import gradio as gr
    names = list(clfs)

    def fn(img, which):
        if img is None:
            return {}
        return clfs[which].predict_one(img)

    inputs = [gr.Image(type="pil", label="Your photo (upload, paste, or use the camera)",
                       sources=["upload", "webcam", "clipboard"]),
              gr.Radio(names, value=names[0], label="Which model?")]
    outputs = gr.Label(num_top_classes=7, label="Model verdict and confidence")
    kw = dict(fn=fn, inputs=inputs, outputs=outputs, title=title, description=description)
    try:
        demo = gr.Interface(**kw, flagging_mode="never")
    except TypeError:  # gradio < 5
        demo = gr.Interface(**kw, allow_flagging="never")
    demo.launch(share=share, inline=True, debug=False, quiet=True, show_error=True)
    return demo
