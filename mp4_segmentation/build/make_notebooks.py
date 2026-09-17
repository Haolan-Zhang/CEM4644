"""Generates the MP4 Workshop and Homework notebooks (+ report templates) from one template."""
import json
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from aec_seg.config import SETS, LEGEND, CUBICASA  # noqa: E402
from aec_seg.data import IntroPhotos  # noqa: E402
from aec_seg.ui import INTRO_PHRASES  # noqa: E402

GITHUB_URL = "https://github.com/Haolan-Zhang/CEM4644.git"

VARIANTS = {
    "workshop": dict(
        file="MP4_Workshop_Segmentation.ipynb", label="Workshop (in class)", minutes=90, dataset="homes_a", own_plans=1,
        # which things the steps start with
        seg_thing="room (any)", count_thing="window", lab_thing="door", inspect_thing="room (any)", fix_thing="room (any)",
        takeoff_plans="each of the three plans",
        q_takeoff=("From Step 3a on each of the three plans: your scale reading and its error, the table of rooms (your m², the drawing's m², the error), the total of your rooms "
                   "against the drawing's floor area, and the windows found / missed / extra. Which rooms came out worst, and why (your box, an open-plan space, the mask stopping at a door opening, the scale)?"),
    ),
    "homework": dict(
        file="MP4_Homework_Segmentation.ipynb", label="Homework (individual)", minutes=120, dataset="homes_b", own_plans=3,
        seg_thing="bedroom", count_thing="toilet", lab_thing="stairs", inspect_thing="living room", fix_thing="bedroom",
        takeoff_plans="three plans of your choice, one of them a two-storey sheet",
        q_takeoff=("From Step 3a on three plans of your choice (one of them a two-storey sheet): your scale reading and its error, the table of rooms (your m², the drawing's m², the error), "
                   "the total against the drawing's floor area, and the windows found / missed / extra. On plan 5018 the sheet prints KERROSALA and HUONEISTOALA (gross and net floor area per storey): "
                   "how do they relate to what you measured?"),
    ),
}


def plan_labels(spec):
    cred = json.loads((REPO / spec.folder / "credits.json").read_text())
    return [f"{c['id']}: {c['title']}" for c in cred], [c["id"] for c in cred]


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
    spec = SETS[v["dataset"]]
    labels, ids = plan_labels(spec)
    label_of = dict(zip(ids, labels))
    d0 = spec.default_plan
    things = spec.names
    conf = 'confidence = 0.3 #@param {type:"slider", min:0.1, max:0.9, step:0.05}'
    questions, cells = [], []

    cells.append(md(f"""
# 🏗️ CEM4644 · MP4 — Segmentation for quantity take-off on floor plans
## {v['label']}: *{spec.title}*

**No coding needed.** Each grey box below is one *step*: click the ▶ (play) button at its left, wait until it finishes, look at the result, then answer the report question that follows. Run the steps **from top to bottom**.

**What you will do (about {v['minutes']} minutes)**
1. Meet SAM 3 on an ordinary site photo: ask by name, draw a box, tap an object. Then look at real floor plans and what is drawn on them.
2. Ask a segmentation model, by name, for rooms, fixtures and openings: see the mask, the overlay, the count and the area, and compare with the drawing's own numbers.
3. Do a quantity take-off with boxes, plan by plan: read the scale, count the windows, measure every room in square metres.
4. Examine where the model goes wrong: wording, weak regions, and your own words.
5. Try a plan of your own.

**Before you start:** menu *Runtime → Change runtime type → T4 GPU → Save*. The model used here (SAM 3) is large: with a GPU each request takes well under a second; without one, the precomputed results still work but live requests take about a minute each.
"""))
    cells.append(form(
        "▶ Step 0 · Run me first (2–3 minutes)",
        f"""import importlib, os, shutil, subprocess, sys
REPO, FOLDER, PKG = "CEM4644", "mp4_segmentation", "aec_seg"

def _git(*args):
    return subprocess.run(["git", "-C", REPO, *args], capture_output=True, text=True).returncode == 0

if os.path.isdir(REPO):                      # a copy is already here: pull the newest course code over it
    if not (_git("fetch", "-q", "--depth", "1", "origin", "master")
            and _git("reset", "-q", "--hard", "FETCH_HEAD") and _git("clean", "-qfd")):
        shutil.rmtree(REPO, ignore_errors=True)          # broken copy: start again from scratch
if not os.path.isdir(REPO):
    subprocess.run(["git", "clone", "--depth", "1", "-q", "{GITHUB_URL}", REPO], check=True)
for _m in [m for m in list(sys.modules) if m == PKG or m.startswith(PKG + ".")]:
    del sys.modules[_m]                      # Python caches imported code: drop it, or this cell keeps the old version
importlib.invalidate_caches()
sys.path.insert(0, os.path.abspath(os.path.join(REPO, FOLDER)))
from aec_seg import lab
lab.setup(dataset="{spec.key}", load_model=load_model)""",
        notes=["Click ▶ and wait for the green ✅ line. This downloads the plans with their precomputed results and loads SAM 3 (about 3 GB).",
               "Untick *load_model* only if you have no GPU and want to skip the live steps."],
        params=['load_model = True #@param {type:"boolean"}'],
    ))

    # ------------------------------------------------------------------ Part 1
    intro = IntroPhotos(REPO / "data" / "intro")
    intro_labels = intro.labels()
    legend_items = [(k, val) for k, val in LEGEND.items() if not (spec.render and (k.startswith("SOVR") or k == "m²"))]
    legend_rows = "\n".join(f"| {k} | {val} |" for k, val in legend_items)
    labels_note = "Room labels are abbreviations in Finnish:" if spec.render else "Room labels are abbreviations in Finnish (one plan is Swedish):"
    cells.append(md("""
## Part 1 · Meet SAM 3

Detection (MP3) draws a **box** around an object. **Segmentation** goes one step further: it decides, *pixel by pixel*, what belongs to the object. Count the pixels and you have an area; know the scale and you have square metres. That is what makes it useful for **quantity take-off** later in this notebook.

The model is **SAM 3** (Segment Anything Model 3, Meta 2025). You do not train it, and it has no fixed list of classes. You tell it *what* or *where*, in one of three ways:

- a **phrase**, such as *helmet* or *wet concrete*: it returns every region that matches, each with a **confidence**;
- a **box** around one object: it cuts out that object's exact outline;
- a **click** on one object: same thing, from a single point.

Two steps on an ordinary site photo first, so you see what the model does before it meets a drawing.
"""))
    cells.append(form("▶ Step 1a · Ask by name", "lab.intro_phrase(photo, phrase, own_phrase, confidence)",
                      notes=["Pick a phrase, or type your own in *own_phrase* (it wins when it is not empty). Three panels: the photo, the mask (white = the model says *this is it*), the overlay. "
                             "Try a thing (*helmet*), a material (*wet concrete*), a part (*hand*), and something that is not there. Watch the confidences and move the slider."],
                      params=[f'photo = "{intro_labels[0]}" #@param {jlist(intro_labels)}', f'phrase = "person" #@param {jlist(INTRO_PHRASES)}',
                              'own_phrase = "" #@param {type:"string"}', conf]))
    cells.append(form("▶ Step 1b · Box it, or tap it", "lab.intro_draw(photo)",
                      notes=["Draw a box around an object and label it *box*; or draw a tiny box on an object and label it *point* (its centre is the click). Draw several, click *Submit*: "
                             "SAM 3 cuts out one object per box or click, no words needed. Needs the live model."],
                      params=[f'photo = "{intro_labels[0]}" #@param {jlist(intro_labels)}']))
    cells.append(md(f"""
### The plans

On a drawing the same three prompts work, and the answer key lets us check every result. {spec.description}

Every plan has a **5 m scale bar** at the bottom left. {labels_note}

| label | meaning |
|---|---|
{legend_rows}
"""))
    cells.append(form("▶ Step 1c · Browse the plans", "lab.show_plans(which)",
                      notes=["*all plans* shows every plan with what the drawing contains (rooms, floor area, doors, windows). These facts come from the plans' own annotations and are the answer key the notebook checks you against."],
                      params=[f'which = "all plans" #@param {jlist(["all plans"] + labels)}']))

    # ------------------------------------------------------------------ Part 2
    cells.append(md(f"""
## Part 2 · Ask for something by name

Pick a plan and a thing. You get three panels: the drawing, the **mask** (white = the model says *this is it*), and the **overlay**. Below them: how many regions, how many pixels, how many square metres (using the plan's scale), and what the **drawing's own answer key** says. The **confidence slider** hides the regions the model is unsure about: watch the count and the area change.
"""))
    cells.append(form("▶ Step 2a · Original → mask → overlay", "lab.segment(plan, thing, confidence)",
                      notes=["Try *room (any)*, then *bedroom* and *bathroom*; then the symbols *toilet*, *sink*, *stairs*; then *kitchen*, *door* and *window*. Some words work, some find nothing at all: that is part of the lesson."],
                      params=[f'plan = "{label_of[d0]}" #@param {jlist(labels)}', f'thing = "{v["seg_thing"]}" #@param {jlist(things)}', conf]))
    cells.append(form("▶ Step 2b · Hits, misses and extras", "lab.count(plan, thing, confidence)",
                      notes=["The answer key drawn on the plan: green = a real one the model found, red = a real one it missed, blue = a region that is not one."],
                      params=[f'plan = "{label_of[d0]}" #@param {jlist(labels)}', f'thing = "{v["count_thing"]}" #@param {jlist(things)}', conf]))
    questions.append((1, "From Step 2a and 2b: which words found what they should (rooms? toilets? windows? doors?), and which found nothing or something else? Give the found / missed / extra counts for two things on one plan at confidence 0.3, and say what the misses have in common."))
    cells.append(q(*questions[-1]))
    # ------------------------------------------------------------------ Part 3
    cells.append(md("""
## Part 3 · Quantity take-off with boxes

A phrase is quick, but a take-off needs control. So now you draw the boxes, in one cell per plan: **the scale** (one box along the 5 m scale bar: a 1 % error in the scale is a 2 % error in every area), **every window** (a small box each), and **every room** (a tight box each, edges on the inside faces of the walls). Draw the small things first, so that later boxes do not overlap them. Then *Submit*: the notebook reads the scale from your box, counts your windows against the drawing's, measures every room with SAM 3 and prints each next to the drawing's own area.

How a *room* box is measured: a box alone makes SAM 3 cut out the *furniture symbols* inside it rather than the empty floor (it was trained to find objects). So the notebook asks for *empty room* **and** gives your box as the example, keeps the region that fits your box, fills the holes left by symbols, removes the black walls, and converts the pixels with **your** scale.
"""))
    cells.append(form("▶ Step 3a · Scale, windows, rooms: the take-off of one plan", "lab.takeoff(plan)",
                      notes=["Zoom with the mouse wheel. Order: *scale bar* (0 to 5 m), then every *window*, then every *room*, then *Submit*. Rooms need the live model. "
                             f"Do this for {v['takeoff_plans']} and copy each table into your report."],
                      params=[f'plan = "{label_of[d0]}" #@param {jlist(labels)}']))
    questions.append((len(questions) + 1, v["q_takeoff"]))
    cells.append(q(*questions[-1]))

    # ------------------------------------------------------------------ Part 4
    cells.append(md("""
## Part 4 · Where it goes wrong

Three kinds of error to look for: the **words** you use (the model was trained on everyday photos, not on drawings), **weak regions** it proposes with low confidence, and words of your own that describe what is *drawn* rather than what it *means*.
"""))
    cells.append(form("▶ Step 4a · Does the wording matter?", "lab.phrase_lab(plan, thing, confidence)",
                      notes=["The same thing asked for with different words; all wordings are precomputed."],
                      params=[f'plan = "{label_of[d0]}" #@param {jlist(labels)}', f'thing = "{v["lab_thing"]}" #@param {jlist(things)}', conf]))
    cells.append(form("▶ Step 4b · Look at each region and its confidence", "lab.inspect(plan, thing)",
                      notes=["Every region the model proposed, numbered, with its confidence, its area and the room it sits on. Move the slider to see which ones survive."],
                      params=[f'plan = "{label_of[d0]}" #@param {jlist(labels)}', f'thing = "{v["inspect_thing"]}" #@param {jlist(things)}']))
    cells.append(form("▶ Step 4c · Your own words", "lab.your_phrase(plan, phrase, confidence)",
                      notes=["Type any phrase: a room, a symbol, a shape. Try *curved line* (the door swings), *thick black line* (the walls), *circle*, *small rectangle*. Needs the live model."],
                      params=[f'plan = "{label_of[d0]}" #@param {jlist(labels)}', 'phrase = "curved line" #@param {type:"string"}', conf]))
    questions.append((len(questions) + 1, "From Step 4a: which wording worked best for the thing you chose, and how different were the counts and areas? From Step 4b: describe one weak region (what it sits on, its confidence) and one plain mistake, and what you would tell a colleague who wants to use these square metres in a cost estimate."))
    cells.append(q(*questions[-1]))
    questions.append((len(questions) + 1, "From Step 4c: which of your own words found something the named things could not (for example *curved line* for the door swings, *thick black line* for the walls)? Why does a shape word work on a drawing where the name of the thing does not?"))
    cells.append(q(*questions[-1]))

    # ------------------------------------------------------------------ Part 5
    cells.append(md("## Part 5 · Your own plan"))
    cells.append(form("▶ Your plan, your words", "lab.upload_app()",
                      notes=["This cell prints a **link**: open it in a new tab (or on your phone). Upload a floor plan (a photo of a drawing works too), type what to find, move the threshold. "
                             "If the plan has a scale bar, measure how many pixels one metre is and enter the centimetres per pixel to get square metres.",
                             f"Test at least {v['own_plans']} plan(s) of your own and take screenshots for your report. Needs the live model."]))
    questions.append((len(questions) + 1, f"Test {v['own_plans']} plan(s) of your own (any floor plan from the internet or a course). For each: the phrase you used, the count and area measured, and whether the mask is right. What kind of drawing or wording failed?"))
    cells.append(q(*questions[-1]))
    questions.append((len(questions) + 1, "Where in a project would a take-off like this be useful, and where would it mislead? What would you need (clean drawings, a scale, a room schedule, a person checking) to turn it into numbers you would put in an estimate?"))
    cells.append(q(*questions[-1]))

    cells.append(md("## Wrap-up"))
    cells.append(form("▶ Numbers for your report", "lab.report_summary()"))
    cells.append(md("### Plan credits and model\n"
                    f"- Site photos in Part 1: " + "; ".join(f"{p.title} ({p.author}, {p.license}, {p.source})" for p in intro.photos) + ".\n"
                    f"- Floor plans: {CUBICASA['author']}, licence {CUBICASA['license']}, {CUBICASA['source']}. "
                    f"Sample ids in this notebook: {', '.join(ids)} (folder `data/plans/{spec.key}`, credits in `credits.json`). "
                    "The plans were resampled to a plan-specific scale and given a scale bar; the answer keys come from the dataset's vector annotations.\n"
                    "- Model: SAM 3 by Meta AI (SAM License), loaded from a public mirror of the official checkpoint; a copy of the licence is in `docs/SAM_LICENSE.txt`.\n"
                    "- Lab code: https://github.com/Haolan-Zhang/CEM4644 (folder `mp4_segmentation`).\n"))

    nb = new_notebook(cells=cells)
    nb.metadata.update({"colab": {"provenance": [], "gpuType": "T4", "toc_visible": True}, "accelerator": "GPU",
                        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                        "language_info": {"name": "python"}})
    out = REPO / v["file"]
    nbformat.write(nb, str(out))
    print("wrote", out)
    lines = [f"# CEM4644 · MP4 report — {v['label']}", "", "Name: ______________________    Date: ____________", "",
             f"Notebook: `{v['file']}` — plans: *{spec.title}*.", "",
             "Answer every question in a few sentences. Paste screenshots where the question asks for overlays or tables. "
             "Numbers must come from **your** run of the notebook.", ""]
    for n, text in questions:
        lines += [f"## Question {n}", "", text, "", "*Your answer:*", "", "", ""]
    p = REPO / "docs" / f"MP4_{variant.capitalize()}_Report_Template.md"
    p.parent.mkdir(exist_ok=True)
    p.write_text("\n".join(lines))
    print("wrote", p)


if __name__ == "__main__":
    for variant in (sys.argv[1:] or VARIANTS):
        build(variant)
