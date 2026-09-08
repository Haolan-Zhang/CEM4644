"""Gradio app for Step 3e: a Photo tab (upload / paste / camera snapshot) and a Live camera tab that
streams frames from the phone or laptop camera through the public link."""
from typing import Dict


def build(detectors: Dict[str, object], colors: Dict[str, str], title: str, description: str = ""):
    import gradio as gr
    from . import draw, ui
    names = list(detectors)

    def annotate(img, which, conf):
        d = detectors[which]
        det = d.predict(img)
        cols = {c: colors.get(c, "#ffffff") for c in d.classes}
        return draw.draw_predictions(img, det, d.classes, cols, conf=conf), ui.det_summary(det, d.classes, conf)

    def run_photo(img, which, conf):
        if img is None:
            return None, ""
        return annotate(img, which, conf)

    def run_live(frame, which, conf):
        if frame is None:
            return None
        return annotate(frame, which, conf)[0]

    # rear camera on phones when the browser allows choosing it
    cam_kwargs = {}
    try:
        cam_kwargs["webcam_options"] = gr.WebcamOptions(mirror=False, constraints={"video": {"facingMode": {"ideal": "environment"}}})
    except Exception:
        try:
            cam_kwargs["mirror_webcam"] = False
        except Exception:
            pass

    with gr.Blocks(title=title) as demo:
        gr.Markdown(f"## {title}\n{description}")
        with gr.Row():
            which = gr.Radio(names, value=names[0], label="Which model?")
            conf = gr.Slider(0.05, 0.95, value=0.5, step=0.05, label="confidence threshold")
        with gr.Tab("Photo"):
            with gr.Row():
                inp = gr.Image(type="pil", label="Your photo (upload, paste, or take one)", sources=["upload", "webcam", "clipboard"], **cam_kwargs)
                out = gr.Image(type="pil", label="What the model sees")
            txt = gr.Textbox(label="Counts", interactive=False)
            btn = gr.Button("Detect", variant="primary")
            btn.click(run_photo, [inp, which, conf], [out, txt])
            inp.change(run_photo, [inp, which, conf], [out, txt])
            conf.release(run_photo, [inp, which, conf], [out, txt])
            which.change(run_photo, [inp, which, conf], [out, txt])
        with gr.Tab("Live camera"):
            gr.Markdown("Allow camera access and point the camera at the room or the site. Frames travel to the notebook and back, "
                        "so expect one or two updates per second. Move the confidence slider above while it runs.")
            with gr.Row():
                cam = gr.Image(sources=["webcam"], streaming=True, type="pil", label="camera", **cam_kwargs)
                live = gr.Image(type="pil", label="live detections", streaming=True)
            cam.stream(run_live, [cam, which, conf], [live], time_limit=180, stream_every=0.5, concurrency_limit=8)
    return demo


def launch(detectors: Dict[str, object], colors: Dict[str, str], title: str = "Test the detector on your own photo",
           description: str = "", share: bool = True):
    demo = build(detectors, colors, title, description)
    demo.launch(share=share, inline=True, debug=False, quiet=True, show_error=True)
    return demo
