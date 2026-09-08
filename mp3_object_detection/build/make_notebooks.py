"""Generates the MP3 Workshop and Homework notebooks (+ report templates) from one template."""
import json
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from aec_det.config import DETSETS  # noqa: E402

GITHUB_URL = "https://github.com/Haolan-Zhang/CEM4644.git"

VARIANTS = {
    "workshop": dict(
        file="MP3_Workshop_Object_Detection.ipynb", label="Workshop (in class)", minutes=90, dataset="construction_safety",
        own_photos=1,
        intro_extra="Every worker is boxed as *person*, with a second box for the head (*helmet* or *NO helmet*) and a third for the torso (*vest* or *NO vest*).",
        own_photo_hint="a photo of people wearing (or not wearing) helmets and high-visibility vests, or any photo at all",
    ),
    "homework": dict(
        file="MP3_Homework_Object_Detection.ipynb", label="Homework (individual)", minutes=120, dataset="excavators",
        own_photos=5,
        intro_extra="Every machine is boxed as *excavator*, *dump truck* or *wheel loader*. Plant tracking, idle-time and site-logistics tools start from exactly this kind of detector.",
        own_photo_hint="photos of construction machinery (from a site you can access, from the street, or photos you find online)",
    ),
}


def n_train(spec):
    p = REPO / "models" / f"{spec.key}_training_log.json"
    if p.exists():
        return f"{json.loads(p.read_text())['n_train']:,}"
    return "hundreds of"


def form(title, body, notes=(), params=()):
    src = f'#@title {title} {{ display-mode: "form" }}\n'
    src += "".join(f"#@markdown {n}\n" for n in notes)
    src += "".join(p + "\n" for p in params)
    src += body.rstrip() + "\n"
    c = new_code_cell(src)
    c.metadata["cellView"] = "form"
    return c


def md(text):
    return new_markdown_cell(text.strip("\n"))


def q(n, text):
    return md(f"> ### 📝 Report question {n}\n> {text}")


def jlist(items):
    return json.dumps(list(items), ensure_ascii=False)


def build(variant):
    v = VARIANTS[variant]
    spec = DETSETS[v["dataset"]]
    classes = [spec.pretty(c) for c in spec.classes]
    other = DETSETS[spec.other_domain] if spec.other_domain else None
    questions, cells = [], []

    cells.append(md(f"""
# 🏗️ CEM4644 · MP3 — Object detection for construction
## {v['label']}: *{spec.title}*

**No coding needed.** Each grey box below is one *step*: click the ▶ (play) button at its left, wait until it finishes, look at the result, then answer the report question that follows. Run the steps **from top to bottom**.

**What you will do (about {v['minutes']} minutes)**
1. Look at labelled photos, count objects yourself, and label a few photos by hand.
2. Run a trained detector, read its confidence, and move the confidence threshold.
3. Measure how many objects it finds and how many false alarms it raises; look at its mistakes.
4. Try to break it: tricky photos, edited photos, a general-purpose detector, photos from another world, your own photos.
5. Turn boxes into a site decision: a PPE compliance or equipment-count dashboard.
6. Train your own detector and compare it with the course model.

**Before you start:** menu *Runtime → Change runtime type → T4 GPU → Save*. Detection works on CPU too, but the training step in Part 6 is much faster with a GPU.

**Dataset:** {spec.title}. Sources and licences are listed at the bottom.
"""))
    cells.append(form(
        "▶ Step 0 · Run me first (about 2 minutes)",
        f"""import os, sys, subprocess
if not os.path.isdir("CEM4644/mp3_object_detection"):
    subprocess.run(["git", "clone", "--depth", "1", "-q", "{GITHUB_URL}"], check=True)
sys.path.insert(0, os.path.abspath("CEM4644/mp3_object_detection"))
from aec_det import lab
lab.setup(dataset="{spec.key}")""",
        notes=["Click ▶ and wait for the green ✅ line. This downloads the photos and the course model and installs the detection library.",
               "If Colab asks whether to run a notebook that was not authored by Google, choose *Run anyway*."],
    ))

    cells.append(md("""
### ☕ While Step 0 runs: try a general-purpose detector in your browser

Open **https://www.ultralytics.com/yolo** in a new tab (no account needed). Use the *webcam* option, or upload a photo, and move the **confidence** slider.

- Point the camera at the room: it will find *person*, *chair*, *laptop*, *bottle*... These are the 80 everyday classes it was trained on.
- Now think about a construction site: it has no idea what a *helmet*, a *vest* or an *excavator* is. That is the gap this notebook closes with **fine-tuning** (Part 3, Step 3c).
- Notice how the number of boxes changes when you move the confidence slider. You will measure exactly that in Part 2.
"""))

    # ---------------------------------------------------------------- Part 1
    cells.append(md(f"""
## Part 1 · Meet the data

Classification (MP2) answers *what is in this photo?* **Object detection** answers *what is where?*: for every object it returns a **box**, a **class** and a **confidence**. Training data therefore needs a box drawn by a person around every object of interest. {v['intro_extra']}

Boxes drawn with a **dashed** line are labels made by people; boxes with a **solid** line (later) are the model's detections.
"""))
    cells.append(form(
        "▶ Step 1a · Browse the labelled photos",
        "lab.show_gallery(category, how_many, with_boxes)",
        notes=["Pick a class to see photos that contain it. Untick *with_boxes* to see the raw photos. Click ▶ again for a new selection."],
        params=[f'category = "all" #@param {jlist(["all"] + classes)}',
                'how_many = 6 #@param [3, 6, 9] {type:"raw"}',
                'with_boxes = True #@param {type:"boolean"}'],
    ))
    cells.append(form(
        "▶ Step 1b · Can *you* count them?",
        "lab.count_game(rounds)",
        notes=[f"A photo appears without boxes. Question: **{spec.count_question}** Click a number, then *Next photo*."],
        params=['rounds = 6 #@param {type:"slider", min:3, max:12, step:1}'],
    ))
    questions.append((1, f"What was your score in the counting game? Which photos were hard for **you** ({', '.join(['small or distant objects', 'objects partly hidden', 'unclear cases'])}), and would you have drawn the boxes the same way as the labellers?"))
    cells.append(q(*questions[-1]))
    cells.append(form(
        "▶ Step 1c · Label a few photos yourself",
        "lab.label_yourself(how_many)",
        notes=["Every box in the dataset was drawn by a person. Now it is your turn: draw a box around **every** object of every class, "
               "pick the class under the photo, then *Submit*. After each photo you see how your boxes compare with the dataset labels, "
               "and at the end how long the whole training set would take at your speed.",
               "If the drawing tool does not appear, run Step 0 again and then this cell."],
        params=['how_many = 3 #@param [2, 3, 5] {type:"raw"}'],
    ))
    questions.append((2, "Labelling: how long did you need per photo, and how many of your boxes agreed with the dataset labels? "
                         "Which objects or classes were hard to decide? At your speed, how many hours would the whole training set take, "
                         "and what does that mean for anyone who wants a detector for their own site?"))
    cells.append(q(*questions[-1]))

    # ---------------------------------------------------------------- Part 2
    cells.append(md(f"""
## Part 2 · Run a trained detector

The **course model** is a YOLO detector that was fine-tuned on {n_train(spec)} labelled photos. For every box it proposes it also gives a **confidence** (0–100 %). A **confidence threshold** decides which boxes you keep: everything below it is thrown away.

To score a detector we compare its boxes with the labelled boxes on unseen test photos. A detection is **correct** when it has the right class and overlaps the true box by at least half. From that we count, per class:
- **recall** = share of the true objects that were found (100 % = nothing missed);
- **precision** = share of the detections that were right (100 % = no false alarms);
- **mAP50** = one overall quality score (0–100) that summarises precision and recall over all thresholds.
"""))
    cells.append(form("▶ Step 2a · Detect, one photo at a time", "lab.pick_and_detect()",
                      notes=["Click *🎲 Another photo* and move the *confidence ≥* slider. Watch boxes appear and disappear. Tick the checkbox to overlay the true boxes (dashed)."]))
    cells.append(form("▶ Step 2b · Score it on all the unseen test photos", "lab.evaluate(how_many, threshold)",
                      notes=["Per class: how many true objects were found, how many were missed, how many detections were false alarms."],
                      params=['how_many = "all" #@param ["all", "100"]', 'threshold = 0.5 #@param {type:"slider", min:0.1, max:0.9, step:0.1}']))
    cells.append(form("▶ Step 2c · Move the threshold", "lab.threshold_explorer()",
                      notes=["The table and the two curves update as you move the slider. Recall and precision pull in opposite directions."]))
    cells.append(form("▶ Step 2d · Look at the mistakes", "lab.error_explorer()",
                      notes=["Green = correct, orange = false alarm, red dashed = missed object. Choose the kind of mistake and the class; the worst photos come first."]))
    questions.append((3, "Which class does the model find most reliably and which one does it miss most often (give the recall numbers)? Look at the missed objects in Step 2d: what do they have in common (size, distance, lighting, overlap, rarity in the training data)?"))
    cells.append(q(*questions[-1]))
    questions.append((4, "Set the threshold to 0.2 and to 0.8. What happens to the number of missed objects and to the number of false alarms? Which threshold would you choose for an automatic site alarm, and which for a weekly report? Explain the difference."))
    cells.append(q(*questions[-1]))

    # ---------------------------------------------------------------- Part 3
    cells.append(md("""
## Part 3 · Where does it break?

A detector only knows the kind of photos it was trained on. Let's find its limits.
"""))
    cells.append(form("▶ Step 3a · Tricky photos", "lab.tricky(group, threshold)",
                      notes=["*hard_real*: test photos with the most mistakes · *synthetic*: real photos we edited (far away, dark, blurred, rotated) · *other_domain*: photos from a different dataset · *out_of_scope*: not a site at all.",
                             "The caption gives the labelled truth where there is one; the line below it is what the model found."],
                      params=['group = "all" #@param ["all", "hard_real", "crowded", "synthetic", "other_domain", "out_of_scope"]',
                              'threshold = 0.5 #@param {type:"slider", min:0.1, max:0.9, step:0.1}']))
    cells.append(form("▶ Step 3b · Break it yourself", "lab.playground()",
                      notes=["Move the sliders: rotate, zoom, *move away* (everything becomes small), blur, darken, shadow, noise. The detector re-runs after every change. Find the smallest change that makes it lose an object."]))
    cells.append(form("▶ Step 3c · A general-purpose detector vs. the course model", "lab.compare_pretrained(how_many, threshold)",
                      notes=["Left: YOLO exactly as downloaded, trained on 80 everyday classes (person, car, truck...). Right: the same network after fine-tuning on our photos. Same photos, same threshold."],
                      params=['how_many = 3 #@param {type:"slider", min:1, max:6, step:1}', 'threshold = 0.4 #@param {type:"slider", min:0.1, max:0.9, step:0.1}']))
    if other is not None:
        cells.append(form("▶ Step 3d · Photos from a different world", "lab.domain_shift(how_many, threshold)",
                          notes=[f"Photos from **{other.title}** are given to the course model and to a model trained on that other dataset. Each model can only answer with its own classes."],
                          params=['how_many = 3 #@param {type:"slider", min:1, max:6, step:1}', 'threshold = 0.4 #@param {type:"slider", min:0.1, max:0.9, step:0.1}']))
    cells.append(form("▶ Step 3e · Your own photo", "lab.upload_app()",
                      notes=["A small app appears with two tabs. *Photo*: upload " + v["own_photo_hint"] + ". *Live camera*: open the public link on your phone, allow the camera and point it at the room or the site; boxes update about once or twice a second. The confidence slider works in both tabs.",
                             f"Test at least {v['own_photos']} photo(s) of your own and take screenshots for your report."]))
    questions.append((5, "List three photos (tricky gallery, sliders, or your own) where the detector missed something or invented something. For each, say what it found, what it should have found, and what you think confused it."))
    cells.append(q(*questions[-1]))
    questions.append((6, "In Step 3c, what does the general-purpose YOLO see in a site photo, and what does the fine-tuned course model add? In Step 3d, what happens when a model gets photos from the other dataset? What does this tell you about buying an 'AI camera' for your own site?"))
    cells.append(q(*questions[-1]))

    # ---------------------------------------------------------------- Part 4
    if spec.dashboard and spec.dashboard["kind"] == "ppe":
        p4 = "Boxes alone are not a decision. A safety officer wants **numbers**: how many heads without a helmet, which photos need a follow-up. The dashboard below counts the model's boxes over all test photos and compares them with the labelled truth."
        q6 = "From the dashboard: what compliance rate does the AI report and what do the labels say? How many photos would be flagged wrongly, and how many flags would be missed? Run it at threshold 0.3 and 0.7 and explain which one you would use for (a) an instant alarm on site and (b) a monthly safety statistic."
    else:
        p4 = "Boxes alone are not a decision. A site manager wants **numbers**: how many machines are on site in each photo, and of which type. The dashboard below counts the model's boxes over all test photos and compares them with the labelled truth."
        q6 = "From the dashboard: how many machines does the AI count in total and how many do the labels contain? On how many photos is the count exactly right? Run it at threshold 0.3 and 0.7 and explain which one you would use for (a) an automatic equipment log and (b) a quick check by a person."
    cells.append(md(f"## Part 4 · From boxes to decisions\n\n{p4}"))
    cells.append(form("▶ Step 4a · Dashboard", "lab.dashboard(threshold)",
                      notes=["Run it several times with different thresholds and compare the numbers."],
                      params=['threshold = 0.5 #@param {type:"slider", min:0.1, max:0.9, step:0.1}']))
    questions.append((7, q6))
    cells.append(q(*questions[-1]))

    # ---------------------------------------------------------------- Part 5
    cells.append(md("""
## Part 5 · Train your own detector

**Training** shows the network labelled photos, lets it predict boxes, and nudges it every time it is wrong. One pass over the training photos is one **epoch**. The course model started from a network **pretrained** on 120,000 everyday photos (COCO) and was then **fine-tuned** on our site photos. You can start from that pretrained network or from a **random** one.

A detector needs many more training steps than the classifier in MP2, so the runs here are longer. Suggested ladder (on a GPU each run takes one to two minutes): 60 photos · 10 passes → 120 · 15 → all · 12 → the best setting with a *random* start. Without a GPU stay with 60 photos and 10 passes (about five minutes).
"""))
    cells.append(form(
        "▶ Step 5a · Train",
        """n = 10**9 if training_photos == "all" else int(training_photos)
lab.train_my_model(n, passes, start, run_name)""",
        notes=["Choose the settings, name the run, click ▶. The quality score (mAP50) on the validation photos is printed after every pass; the final score on the unseen test photos is printed at the end."],
        params=['training_photos = "120" #@param ["60", "120", "all"]', 'passes = 10 #@param {type:"slider", min:3, max:15, step:1}',
                'start = "pretrained" #@param ["pretrained", "random"]', 'run_name = "run 1" #@param {type:"string"}'],
    ))
    cells.append(form("▶ Step 5b · Leaderboard", "lab.leaderboard()", notes=["All your runs, best first. Copy this table into your report."]))
    cells.append(form("▶ Step 5c · Your model vs. the course model on the tricky photos", "lab.compare_my_model(group, threshold)",
                      params=['group = "all" #@param ["all", "hard_real", "crowded", "synthetic", "other_domain", "out_of_scope"]',
                              'threshold = 0.5 #@param {type:"slider", min:0.1, max:0.9, step:0.1}']))
    questions.append((8, "Copy your leaderboard. How did the quality score change with more photos and more passes? What happened with a random start? Why does a detector need far more training than the classifier in MP2 to reach a useful score?"))
    cells.append(q(*questions[-1]))
    if v["own_photos"] > 1:
        questions.append((9, f"Test {v['own_photos']} photos of your own in Step 3e ({v['own_photo_hint']}). Include screenshots. Which detections were right, which were wrong, and what made the wrong ones hard?"))
        cells.append(q(*questions[-1]))
    questions.append((len(questions) + 1, "Imagine this detector running on a site camera. Where would you place the camera, what would you do with each alarm, and what could go wrong (technically and for the people being filmed)? What data would you need to collect to make it work on your own site?"))
    cells.append(q(*questions[-1]))

    cells.append(md("## Wrap-up"))
    cells.append(form("▶ Step 6 · Numbers for your report", "lab.report_summary()"))
    src = [f"- **{spec.title}** — {spec.description}"]
    if other is not None:
        src.append(f"- **{other.title}** (used in Step 3d) — {other.description}")
    cells.append(md("### Data and model sources\n" + "\n".join(src) + """
- Detector: YOLO11n by Ultralytics (AGPL-3.0), pretrained on COCO, fine-tuned for this course. The `ultralytics` library is installed in Step 0.
- Out-of-scope sample images: scikit-image data (public domain / CC0).
- Lab code: https://github.com/Haolan-Zhang/CEM4644 (folder `mp3_object_detection`).
"""))

    nb = new_notebook(cells=cells)
    nb.metadata.update({"colab": {"provenance": [], "gpuType": "T4", "toc_visible": True}, "accelerator": "GPU",
                        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                        "language_info": {"name": "python"}})
    out = REPO / v["file"]
    nbformat.write(nb, str(out))
    print("wrote", out)
    lines = [f"# CEM4644 · MP3 report — {v['label']}", "", "Name: ______________________    Date: ____________", "",
             f"Notebook: `{v['file']}` — dataset: *{spec.title}*.", "",
             "Answer every question in a few sentences. Paste screenshots where the question asks for photos or tables. "
             "Numbers must come from **your** run of the notebook (Step 6 prints them).", ""]
    for n, text in questions:
        lines += [f"## Question {n}", "", text, "", "*Your answer:*", "", "", ""]
    p = REPO / "docs" / f"MP3_{variant.capitalize()}_Report_Template.md"
    p.parent.mkdir(exist_ok=True)
    p.write_text("\n".join(lines))
    print("wrote", p)


if __name__ == "__main__":
    for variant in (sys.argv[1:] or VARIANTS):
        build(variant)
