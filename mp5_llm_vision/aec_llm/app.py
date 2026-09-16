"""The prompt lab: a small Gradio app in which students pick or upload an image, edit the prompt, switch the JSON
schema on or off, and see the raw reply next to what it draws. Always live (their own words need the model)."""
import json
import os
from typing import Optional

from PIL import Image

from . import config as C
from . import tasks, viz

TASKS = ["free text", "classify", "detect", "rooms: polygons", "rooms: boxes + SAM 3"]


def default_prompt(lab, task: str) -> str:
    sp = lab.spec
    if task == "free text":
        return C.fill("describe", sp, question=C.DESCRIBE_QUESTIONS[0])
    if task == "classify":
        return C.fill("classify_basic", sp)
    if task == "detect":
        return C.fill("detect", sp)
    if task == "rooms: polygons":
        return C.PROMPTS["rooms_masks"]
    return C.PROMPTS["rooms_boxes"]


def schema_for(lab, task: str) -> Optional[dict]:
    sp = lab.spec
    return {"free text": None, "classify": C.classify_schema(sp.photos.classes), "detect": C.detect_schema(sp.sites.classes),
            "rooms: polygons": C.rooms_schema(True), "rooms: boxes + SAM 3": C.rooms_schema(False)}[task]


def run(lab, image: Optional[Image.Image], task: str, prompt: str, use_schema: bool, model: str, thinking: str):
    if image is None:
        return None, "", "Pick or upload an image first."
    if not lab.client.live:
        return image, "", "No API key: the prompt lab needs a live model. Add your Gemini key in Step 0."
    schema = schema_for(lab, task) if use_schema else None
    r = lab.client.ask(image, prompt, schema=schema, model=model, thinking=thinking.lower(), use_cache=False)
    status = f"**{r.model}** · {r.seconds:.1f} s · tokens in {r.tokens_in}, out {r.tokens_out}, thinking {r.tokens_thought}"
    if r.error:
        return image, r.text, status + f"\n\n⚠️ {r.error}"
    out = image
    W, H = image.size
    try:
        if task == "detect" and isinstance(r.data, list):
            boxes, skipped = tasks.boxes_from_reply(r, image.size, lab.spec.sites.classes)
            cols = lab.spec.sites.colors
            out = viz.draw_boxes(image, [(l, b, cols.get(l, viz.BLUE), 3) for l, b in boxes])
            status += f"\n\n{len(boxes)} boxes drawn" + (f", {skipped} entries unusable" if skipped else "")
        elif task.startswith("rooms") and isinstance(r.data, list):
            polys, boxes = [], []
            for it in r.data:
                if not isinstance(it, dict):
                    continue
                b = tasks.box_2d_to_xyxy(it.get("box_2d"), image.size)
                poly = tasks.points_to_px(it.get("mask"), image.size) if it.get("mask") is not None else None
                if poly is not None:
                    polys.append((str(it.get("label", "")), poly, "#457b9d"))
                elif b is not None:
                    boxes.append((str(it.get("label", "")), b, "#457b9d", 3))
            out = viz.draw_polygons(image, polys) if polys else image
            if boxes:
                out = viz.draw_boxes(out, boxes)
            if task == "rooms: boxes + SAM 3" and lab.sam is not None and boxes:
                layer = out
                for i, (l, b, _, _) in enumerate(boxes):
                    res = lab.sam.segment_room(image, b)
                    if len(res):
                        layer = viz.mask_overlay(layer, res.masks[0], ["#e63946", "#2a9d8f", "#f4a261", "#7b2cbf", "#457b9d", "#e9c46a"][i % 6], 0.45)
                out = layer
            status += f"\n\n{len(polys)} polygons, {len(boxes)} boxes"
        elif task == "classify" and isinstance(r.data, dict):
            status += f"\n\n**{r.data.get('label')}** (confidence {r.data.get('confidence')}): {r.data.get('reason', '')}"
    except Exception as e:  # noqa: BLE001
        status += f"\n\n(could not draw the reply: {e})"
    return out, r.text, status


def build(lab):
    import gradio as gr
    ex = lab.examples
    choices = [f"photo: {p.label}" for p in ex.photos] + [f"site: {s.label}" for s in ex.sites] + [f"plan: {p.label}" for p in ex.plans]

    def pick(choice):
        if not choice:
            return None
        kind, label = choice.split(": ", 1)
        return {"photo": ex.photo, "site": ex.site, "plan": ex.plan}[kind](label).load()

    with gr.Blocks(title="Prompt lab") as demo:
        gr.Markdown("### Prompt lab · your image, your words, the raw reply\nPick a built-in image or upload one, choose a task (it fills in a starting prompt), edit the prompt, and run. "
                    "Tick *JSON schema* to make the API enforce the structure instead of merely asking for it.")
        with gr.Row():
            with gr.Column(scale=1):
                choice = gr.Dropdown(choices, label="built-in image", value=None)
                image = gr.Image(type="pil", label="image (or upload your own)")
                task = gr.Radio(TASKS, value="free text", label="task")
                prompt = gr.Textbox(default_prompt(lab, "free text"), label="prompt", lines=6)
                with gr.Row():
                    use_schema = gr.Checkbox(False, label="JSON schema (structured output)")
                    model = gr.Dropdown(C.MODELS, value=lab.client.model, label="model")
                    thinking = gr.Dropdown(["low", "medium", "high"], value="low", label="thinking")
                btn = gr.Button("Run", variant="primary")
            with gr.Column(scale=1):
                out_img = gr.Image(label="what the reply draws", type="pil")
                status = gr.Markdown()
                raw = gr.Textbox(label="raw reply", lines=12)
        choice.change(pick, choice, image)
        task.change(lambda t: default_prompt(lab, t), task, prompt)
        btn.click(lambda im, t, p, s, m, th: run(lab, im, t, p, s, m, th), [image, task, prompt, use_schema, model, thinking], [out_img, raw, status])
    return demo


def launch(lab):
    demo = build(lab)
    if os.environ.get("AEC_LAB_NO_APP"):
        return demo
    demo.launch(share=None, height=760, quiet=True, inline=True)
    return demo
