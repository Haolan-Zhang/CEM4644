"""Generates the Workshop and Homework notebooks (and their report templates) from one
template. Students only ever see Colab forms; every cell body is a one-line call
into aec_lab. Run:  python build/make_notebooks.py
"""
import json
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from aec_lab.config import DATASETS  # noqa: E402

GITHUB_URL = "https://github.com/Haolan-Zhang/CEM4644.git"
GRADIO_VERSION = "6.26.0"   # the version the *_gradio copies were tested with (same pin as MP3/MP4)
GITHUB_BLOB = "https://github.com/Haolan-Zhang/CEM4644/blob/master/mp2_image_classification/"

VARIANTS = {
    "workshop": dict(
        file="MP2_Workshop_Image_Classification.ipynb",
        label="Workshop (in class)",
        minutes=80,
        binary="facade_defects",
        multiclass="facade_defects",
        task_names={"binary": "defect vs. no defect", "multiclass": "defect type"},
        own_photos=1,
        multiclass_intro=(
            "So far the question was *is there a defect?* Now we ask *which kind of defect?* The same photos "
            "are labelled with seven classes, and a second course model was trained on them."),
        zero_shot_default="a brick wall, a concrete wall, a painted wall, a stone wall, a window",
    ),
    "homework": dict(
        file="MP2_Homework_Image_Classification.ipynb",
        label="Homework (individual)",
        minutes=120,
        binary="concrete_cracks",
        multiclass="facade_styles",
        task_names={"binary": "crack vs. no crack", "multiclass": "architectural style"},
        own_photos=5,
        multiclass_intro=(
            "Classification is not only about defects. In this part the question is *which architectural style is "
            "this façade?* The images are computer-generated reference façades in ten styles, and a course model was "
            "trained on them. Keep in mind while you work: none of these images is a real building."),
        zero_shot_default="a glass office tower, a stone church, a brick warehouse, a concrete apartment block, a wooden house",
    ),
}


def n_train_full(spec, task):
    d = spec.binary_model if task == "binary" else spec.multiclass_model
    p = REPO / d / "training_log.json" if d else None
    if p and p.exists():
        return json.loads(p.read_text())["n_train"]
    return None


def form(title, body, notes=(), params=()):
    """A code cell shown as a Colab form: title + grey instructions + widgets, code hidden."""
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


def build(variant: str, gradio: bool = False):
    """gradio=True writes the *_gradio copy: Steps 3b and 5a become inline Gradio apps, everything else is identical."""
    v = VARIANTS[variant]
    sb, sm = DATASETS[v["binary"]], DATASETS[v["multiclass"]]
    B, M = v["task_names"]["binary"], v["task_names"]["multiclass"]
    bin_classes = sb.binary.classes
    multi_classes = [sm.pretty(c) for c in sm.classes]
    nb_train = n_train_full(sb, "binary")
    nb_train = f"{nb_train:,}" if nb_train else "thousands of"
    nm_train = n_train_full(sm, "multiclass")
    nm_train = f"{nm_train:,}" if nm_train else "hundreds of"
    questions = []
    cells = []

    # ------------------------------------------------------------------ title
    cells.append(md(f"""
# 🏗️ CEM4644 · MP2 — Image classification for construction
## {v['label']}: *{B}* and *{M}*

**No coding needed.** This notebook is a series of buttons. Each grey box below is one *step*: click the ▶ (play) button at its left, wait until it finishes, look at the result, then answer the report question that follows. Run the steps **from top to bottom**.

**What you will do (about {v['minutes']} minutes)**
1. Look at labelled photos and try to label some yourself.
2. Run a trained classifier, read its confidence, and measure how often it is right.
3. Try to break it: tricky photos, edited photos, photos from another world, your own photos.
4. Do the same with more than two classes.
5. Invent your own classes and classify with no training at all.
6. Train your own model and compare it with the course model.

**Before you start (optional, makes training faster):** menu *Runtime → Change runtime type → T4 GPU → Save*. Everything also works without a GPU.

**Datasets:** {sb.title} · {sm.title}. Sources and licences are listed at the bottom.
"""))

    cells.append(form(
        "▶ Step 0 · Run me first (1–2 minutes)",
        f"""import os, sys, subprocess
if not os.path.isdir("CEM4644/mp2_image_classification"):
    subprocess.run(["git", "clone", "--depth", "1", "-q", "{GITHUB_URL}"], check=True)
sys.path.insert(0, os.path.abspath("CEM4644/mp2_image_classification"))
from aec_lab import lab
lab.setup(binary="{v['binary']}", multiclass="{v['multiclass']}", task_names={json.dumps(v['task_names'], ensure_ascii=False)}{', gradio="' + GRADIO_VERSION + '"' if gradio else ''})""",
        notes=["Click ▶ and wait for the green ✅ line. This downloads the photos and the course models. Nothing else to do here.",
               "If Colab asks whether to run a notebook that was not authored by Google, choose *Run anyway*."],
    ))

    # ------------------------------------------------------------------ part 1
    cells.append(md(f"""
## Part 1 · Meet the data

A classifier learns from **labelled examples**: photos for which a person has already written down the answer. The answer is called the **label**, and each possible answer is a **class** (for example *{bin_classes[1]}* / *{bin_classes[0]}*).

The photos are kept in two separate piles:
- **training photos** — the model learns from these;
- **test photos** — hidden from the model during training, so that we can measure how it does on photos it has never seen. Every accuracy number in this notebook is computed on test photos only.
"""))
    cells.append(form(
        "▶ Step 1a · Browse the photos",
        "lab.show_gallery(task, category, how_many)",
        notes=["Choose which task, a class (or *all*), and how many photos. Click ▶ again for a new random selection."],
        params=[f'task = "{B}" #@param {jlist([B, M])}',
                f'category = "all" #@param {jlist(["all"] + bin_classes + multi_classes)}',
                'how_many = 12 #@param {type:"slider", min:4, max:24, step:4}'],
    ))
    cells.append(form(
        "▶ Step 1b · Can *you* tell them apart?",
        "lab.guess_game(task, rounds)",
        notes=["A photo appears: click the class you think is right, then *Next photo*. Your score is shown at the end."],
        params=[f'task = "{M}" #@param {jlist([B, M])}',
                'rounds = 8 #@param {type:"slider", min:4, max:20, step:2}'],
    ))
    questions.append((1, f"What was your score in the guessing game for *{M}*? Which classes were hard for **you** to tell apart, and what made them hard?"))
    cells.append(q(*questions[-1]))

    # ------------------------------------------------------------------ part 2
    cells.append(md(f"""
## Part 2 · Run a trained classifier ({B})

The **course model** was trained beforehand on {nb_train} labelled photos. Given a new photo it returns a **confidence** for every class (the confidences add up to 100 %) and answers with the class that has the highest confidence.
"""))
    cells.append(form(
        "▶ Step 2a · Classify one photo at a time",
        f'lab.pick_and_predict("{B}")',
        notes=["Click *🎲 Another photo* as often as you like. Watch the confidence bars: is the model always sure? Is it ever confidently wrong?"],
    ))
    cells.append(form(
        "▶ Step 2b · Test it on all the unseen test photos",
        f'lab.evaluate("{B}", how_many)',
        notes=["**Accuracy** = share of test photos classified correctly.",
               "The **confusion matrix** shows, for each true class (rows), what the model said (columns). Numbers on the diagonal are correct; everything else is a mistake."],
        params=['how_many = "all" #@param ["all", "100", "300"]'],
    ))
    questions.append((2, f"What accuracy did the course model reach for *{B}*, on how many test photos? Would you trust it as the *only* check in a building inspection? Explain in two or three sentences."))
    cells.append(q(*questions[-1]))
    cells.append(form(
        "▶ Step 2c · Look at the mistakes",
        f'lab.error_explorer("{B}")',
        notes=["Use the drop-downs to filter by true class and by what the model said. The most confident mistakes are shown first: those are the interesting ones."],
    ))
    cells.append(form(
        "▶ Step 2d · Where do you draw the line?",
        "lab.threshold_explorer()",
        notes=[f"By default the model says *{bin_classes[1]}* when its confidence is above 50 %. Move the slider to change that rule and watch the two kinds of mistakes: **missed {bin_classes[1]}s** and **false alarms**."],
    ))
    questions.append((3, f"Move the threshold to 0.1 and to 0.9. What happens to the number of missed *{bin_classes[1]}s* and the number of false alarms? Which of the two mistakes is worse for a building inspection, and which threshold would you choose? Why?"))
    cells.append(q(*questions[-1]))

    # ------------------------------------------------------------------ part 3
    cells.append(md(f"""
## Part 3 · Where does it break?

A model only knows the kind of photos it was trained on. Let's look for its limits. In the galleries below the caption says what each photo really is; the model's verdict is printed under it (**green** = agrees with the expected answer, **red** = disagrees, black = there is no single right answer).
"""))
    cells.append(form(
        "▶ Step 3a · Tricky photos",
        f'lab.tricky("{B}", group)',
        notes=["*hard_real*: real test photos the model gets wrong · *borderline*: real photos it is barely sure about · *synthetic*: real photos we edited · *other_domain*: photos from a different dataset · *out_of_scope*: not walls at all."],
        params=['group = "all" #@param ["all", "hard_real", "borderline", "synthetic", "other_domain", "out_of_scope"]'],
    ))
    if gradio:
        cells.append(form(
            "▶ Step 3b · Break it yourself",
            f'lab.playground_app("{B}")',
            notes=["An app appears below (give it 10–20 seconds; a *public URL* is printed above it, which you can open in a new tab or on your phone).",
                   "**Sliders tab:** rotate, zoom, blur, darken, cast a shadow, add noise, draw a dark line; the model re-runs on every change. Try to flip its answer with the *smallest* possible change.",
                   "**Draw on it tab:** paint a crack, a stain or a shadow on the photo with the brush, then click *Classify my drawing*. You can also upload or photograph your own wall and draw on that."],
        ))
    else:
        cells.append(form(
            "▶ Step 3b · Break it yourself",
            f'lab.playground("{B}")',
            notes=["Move the sliders: rotate, zoom, blur, darken, cast a shadow, add noise, draw a dark line. The model re-runs after every change. Try to flip its answer with the *smallest* possible change."],
        ))
    cells.append(form(
        "▶ Step 3c · A model from a different world",
        "lab.domain_shift()",
        notes=[f"The same test photos are given to a model trained on **{DATASETS[sb.other_domain].title if sb.other_domain else 'another dataset'}**. Both models answer 'defect or not', but only one has seen photos like these before."],
    ))
    cells.append(form(
        "▶ Step 3d · Your own photo",
        "lab.upload_app()",
        notes=["A small app appears below (also as a public link you can open on your phone to use the camera). Upload or photograph a wall, floor, pavement, a face, a drawing... anything.",
               f"Test at least {v['own_photos']} photo(s) of your own and take screenshots for your report."],
    ))
    questions.append((4, "List three photos (from the tricky gallery, the sliders, or your own) that fooled the model. For each one, say what the model answered and what you think confused it."))
    cells.append(q(*questions[-1]))
    questions.append((5, "What does the model say about a photo that is not a wall at all (the cat, the floor plan, the noise)? Why can it not answer *I don't know*? How would you deal with this if the model were used on a real project?"))
    cells.append(q(*questions[-1]))

    # ------------------------------------------------------------------ part 4
    cells.append(md(f"""
## Part 4 · More than two classes ({M})

{v['multiclass_intro']} The course model for this task was trained on {nm_train} labelled images and chooses between **{len(multi_classes)} classes**: {', '.join(multi_classes)}.
"""))
    cells.append(form("▶ Step 4a · Classify one image at a time", f'lab.pick_and_predict("{M}")',
                      notes=["With many classes the confidence is spread out. Look at the runner-up: is it a *reasonable* second guess?"]))
    cells.append(form("▶ Step 4b · Test it on all the unseen test images", f'lab.evaluate("{M}", "all")',
                      notes=["Read the confusion matrix row by row: which pairs of classes get mixed up?"]))
    cells.append(form("▶ Step 4c · Look at the mistakes", f'lab.error_explorer("{M}")',
                      notes=["Pick the pair of classes that is confused most often and look at the actual images."]))
    cells.append(form("▶ Step 4d · Tricky images", f'lab.tricky("{M}", group)',
                      params=['group = "all" #@param ["all", "hard_real", "borderline", "synthetic", "other_domain", "out_of_scope"]']))
    questions.append((6, f"For *{M}*: what is the accuracy, and which two classes are confused most often? Look at a few examples of that confusion. Would a person make the same mistake? Is accuracy alone a fair summary of this model?"))
    cells.append(q(*questions[-1]))

    # ------------------------------------------------------------------ part 5
    cells.append(md("""
## Part 5 · Classes you invent (no training at all)

**CLIP** is a model trained on hundreds of millions of internet photos *together with their captions*. It can compare a photo with any sentence you type. That means you can invent classes on the spot, with no labelled photos: type the class names, and CLIP picks the closest one for each image. This is called **zero-shot** classification.
"""))
    if gradio:
        cells.append(form(
            "▶ Step 5a · Type your own classes",
            'lab.zero_shot_app(class_names, how_many)',
            notes=["An app appears below (give it 10–20 seconds; a *public URL* is printed above it). Type class names separated by commas and click **Classify**. The photos stay the same until you click *New photos*, so change the wording and click again to see exactly what your words changed. Try short and descriptive names (*a brick wall*, *a cracked wall*, ...).",
                   "Second tab: classify your own photo with your own class names. The first click loads CLIP (about a minute)."],
            params=[f'class_names = "{v["zero_shot_default"]}" #@param {{type:"string"}}',
                    'how_many = 8 #@param {type:"slider", min:4, max:16, step:4}'],
        ))
    else:
        cells.append(form(
            "▶ Step 5a · Type your own classes",
            "lab.zero_shot(class_names, how_many, source)",
            notes=["Separate the class names with commas. Try short and descriptive names (*a brick wall*, *a cracked wall*, ...). Run it several times with different names.",
                   "The first run downloads CLIP (about a minute)."],
            params=[f'class_names = "{v["zero_shot_default"]}" #@param {{type:"string"}}',
                    'how_many = 8 #@param {type:"slider", min:4, max:16, step:4}',
                    'source = "test photos" #@param ["test photos", "tricky photos"]'],
        ))
    questions.append((7, "Which class names did you try, and did CLIP's answers make sense? Give one construction task where inventing classes like this would be good enough, and one where you would rather train a model on labelled photos. Explain the difference."))
    cells.append(q(*questions[-1]))

    # ------------------------------------------------------------------ part 6
    cells.append(md("""
## Part 6 · Train your own model

**Training** means showing the model labelled photos, letting it guess, and nudging it a little each time it is wrong. One pass over all the training photos is called an **epoch**.

The course models did not start from zero: they started from a network **pretrained** on 1.2 million everyday photos (ImageNet) and were then **fine-tuned** on our photos. You can start from that pretrained network, or from a **random** network that has never seen a photo in its life.

Suggested experiments (each run takes one to three minutes): 20 photos · 1 pass → 100 photos · 2 passes → 300 photos · 3 passes → then the best setting with a *random* start.
"""))
    cells.append(form(
        "▶ Step 6a · Train",
        f"""n = 10**9 if training_photos == "all" else int(training_photos)
lab.train_my_model(task, n, passes, start, run_name)""",
        notes=["Choose the settings, give the run a name, click ▶. Every run is added to the leaderboard in the next step."],
        params=[f'task = "{B}" #@param {jlist([B, M])}',
                'training_photos = "100" #@param ["20", "50", "100", "300", "all"]',
                'passes = 2 #@param {type:"slider", min:1, max:5, step:1}',
                'start = "pretrained" #@param ["pretrained", "random"]',
                'run_name = "run 1" #@param {type:"string"}'],
    ))
    cells.append(form("▶ Step 6b · Leaderboard", "lab.leaderboard()",
                      notes=["All your runs, best first. Copy this table into your report."]))
    cells.append(form("▶ Step 6c · Your model vs. the course model on the tricky photos", f'lab.compare_my_model("{B}")',
                      notes=["Each photo shows the verdict of the course model and of your latest model for this task."]))
    questions.append((8, "Copy your leaderboard. How did accuracy change with more training photos and more passes? What happened with a *random* start compared with a *pretrained* start, and why do you think that is?"))
    cells.append(q(*questions[-1]))
    if v["own_photos"] > 1:
        questions.append((9, f"Photograph {v['own_photos']} surfaces yourself (walls, floors, pavements, façades) and test them in Step 3d. Include the screenshots. Which verdicts were right? For the wrong ones, what made the photo hard?"))
        cells.append(q(*questions[-1]))
    questions.append((len(questions) + 1, "Name one place in a construction project where a classifier like this could be useful. What photos would you need to collect to train it, who would label them, and what could go wrong?"))
    cells.append(q(*questions[-1]))

    # ------------------------------------------------------------------ wrap-up
    cells.append(md("## Wrap-up"))
    cells.append(form("▶ Step 7 · Numbers for your report", "lab.report_summary()",
                      notes=["Prints the accuracies and your training runs in one place."]))
    src_lines = []
    for s in {sb.key: sb, sm.key: sm}.values():
        src_lines.append(f"- **{s.title}** — {s.description}")
    if sb.other_domain:
        o = DATASETS[sb.other_domain]
        src_lines.append(f"- **{o.title}** (used in Step 3c) — {o.description}")
    cells.append(md("### Data and model sources\n" + "\n".join(src_lines) + """
- Course models: ConvNeXt V2 (femto), pretrained on ImageNet-1k by Meta AI (Apache-2.0), fine-tuned for this course.
- Zero-shot model: CLIP ViT-B/32 by OpenAI (MIT).
- Out-of-scope sample images: scikit-image data (public domain / CC0).
- Lab code: https://github.com/Haolan-Zhang/CEM4644 (folder `mp2_image_classification`).
"""))

    nb = new_notebook(cells=cells)
    nb.metadata.update({
        "colab": {"provenance": [], "gpuType": "T4", "toc_visible": True},
        "accelerator": "GPU",
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    })
    out = REPO / (v["file"].replace(".ipynb", "_gradio.ipynb") if gradio else v["file"])
    nbformat.write(nb, str(out))
    print("wrote", out)
    if not gradio:  # the copies share the originals' report templates (same questions)
        write_report_template(variant, v, questions, B, M)


def write_report_template(variant, v, questions, B, M):
    lines = [f"# CEM4644 · MP2 report — {v['label']}", "",
             "Name: ______________________    Date: ____________", "",
             f"Notebook: `{v['file']}` — tasks: *{B}* and *{M}*.", "",
             "Answer every question in a few sentences. Paste screenshots where the question asks for photos or tables. "
             "Numbers must come from **your** run of the notebook (Step 7 prints them).", ""]
    for n, text in questions:
        lines += [f"## Question {n}", "", text, "", "*Your answer:*", "", "", ""]
    p = REPO / "docs" / f"MP2_{variant.capitalize()}_Report_Template.md"
    p.parent.mkdir(exist_ok=True)
    p.write_text("\n".join(lines))
    print("wrote", p)


if __name__ == "__main__":
    args = sys.argv[1:]
    gradio = "--gradio" in args
    for variant in ([a for a in args if not a.startswith("--")] or VARIANTS):
        build(variant, gradio=gradio)
