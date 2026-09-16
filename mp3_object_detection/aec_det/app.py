"""Gradio app for Step 3e: a Photo tab (upload / paste / camera snapshot) and a Live camera tab that
streams frames from the phone or laptop camera through the public link."""
from typing import Dict


def build(detector, colors: Dict[str, str], title: str, description: str = ""):
    """One detector (the course model), a confidence slider, a Photo tab and a Live camera tab."""
    import gradio as gr
    from . import draw, ui
    cols = {c: colors.get(c, "#ffffff") for c in detector.classes}

    def annotate(img, conf):
        det = detector.predict(img)
        return draw.draw_predictions(img, det, detector.classes, cols, conf=conf), ui.det_summary(det, detector.classes, conf)

    def run_photo(img, conf):
        if img is None:
            return None, ""
        return annotate(img, conf)

    def run_live(frame, conf):
        if frame is None:
            return None
        return annotate(frame, conf)[0]

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
        conf = gr.Slider(0.05, 0.95, value=0.5, step=0.05, label="confidence threshold")
        with gr.Tab("Photo"):
            with gr.Row():
                inp = gr.Image(type="pil", label="Your photo (upload, paste, or take one)", sources=["upload", "webcam", "clipboard"], **cam_kwargs)
                out = gr.Image(type="pil", label="What the model sees")
            txt = gr.Textbox(label="Counts", interactive=False)
            btn = gr.Button("Detect", variant="primary")
            btn.click(run_photo, [inp, conf], [out, txt])
            inp.change(run_photo, [inp, conf], [out, txt])
            conf.release(run_photo, [inp, conf], [out, txt])
        with gr.Tab("Live camera"):
            gr.Markdown("Allow camera access and point the camera at the room or the site. Frames travel to the notebook and back, "
                        "so expect one or two updates per second. Move the confidence slider above while it runs.")
            with gr.Row():
                cam = gr.Image(sources=["webcam"], streaming=True, type="pil", label="camera", **cam_kwargs)
                live = gr.Image(type="pil", label="live detections", streaming=True)
            cam.stream(run_live, [cam, conf], [live], time_limit=180, stream_every=0.5, concurrency_limit=8)
    return demo


def launch(detector, colors: Dict[str, str], title: str = "Test the detector on your own photo",
           description: str = "", share: bool = True):
    demo = build(detector, colors, title, description)
    import contextlib
    import io
    sink = io.StringIO()
    # the app is not embedded in the cell (the embedded frame is unreliable in Colab): the cell prints a link instead
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        demo.launch(share=share, inline=False, debug=False, quiet=True, show_error=True)
    url = getattr(demo, "share_url", None)
    if url:
        print(f"Open the app in a new tab (works on a phone too): {url}")
        print("The link stays alive while this notebook is running.")
    else:
        print("The public link could not be created (network hiccup). Run this cell again; "
              f"the app is running at {getattr(demo, 'local_url', 'the local address')}, which only works from this machine.")
    return demo
