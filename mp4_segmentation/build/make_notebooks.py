"""Generates the MP4 Workshop and Homework notebooks (+ report templates) from one template."""
import json
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from aec_seg.config import SETS, LEGEND, CUBICASA  # noqa: E402

GITHUB_URL = "https://github.com/Haolan-Zhang/CEM4644.git"

VARIANTS = {
    "workshop": dict(
        file="MP4_Workshop_Segmentation.ipynb", label="Workshop (in class)", minutes=90, dataset="homes_a", own_plans=1,
        # which things the steps start with
        seg_thing="room (any)", count_thing="window", lab_thing="door", inspect_thing="room (any)", fix_thing="room (any)",
        takeoff_targets="two rooms of your choice, one window and one door",
        questions_extra=[
            "From Step 3b: which rooms did you measure, what did you get, and what does the drawing say? Give the error in percent for each and explain where it comes from (your box, the mask stopping at furniture or a door opening, the scale).",
            "From Step 3b with *find_all* on a window: how many windows does the drawing have, how many did SAM 3 find, how many did it miss and how many were extra? Which confidence worked best, and what did the extras have in common?",
        ],
    ),
    "homework": dict(
        file="MP4_Homework_Segmentation.ipynb", label="Homework (individual)", minutes=120, dataset="homes_b", own_plans=3,
        seg_thing="bedroom", count_thing="toilet", lab_thing="stairs", inspect_thing="living room", fix_thing="bedroom",
        takeoff_targets="every bedroom of one plan, the stairs, one toilet and one bathtub",
        questions_extra=[
            "From Step 3b on plan 5018 (two floors on one sheet): measure the living room and the kitchen on the ground floor and two bedrooms upstairs. Give SAM 3's area and the drawing's area for each, and the total for the four rooms. The sheet itself prints the floor areas (KERROSALA, HUONEISTOALA): how do they relate to what you measured?",
            "From Step 3b with *find_all*: count the toilets and the bathtubs on one plan, and the windows on plan 1217. Report found / missed / extra for each, with the confidence you used. Which symbol was easiest for the model and why?",
        ],
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
1. Look at real floor plans and what is drawn on them.
2. Ask a segmentation model, by name, for rooms, fixtures and openings: see the mask, the overlay, the count and the area, and compare with the drawing's own numbers.
3. Do a quantity take-off with boxes: check the scale, measure rooms in square metres, count windows and doors.
4. Compare two plans, and examine where the model goes wrong: wording, weak regions, and how to correct it.
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
    legend_items = [(k, val) for k, val in LEGEND.items() if not (spec.render and (k.startswith("SOVR") or k == "m²"))]
    legend_rows = "\n".join(f"| {k} | {val} |" for k, val in legend_items)
    labels_note = "Room labels are abbreviations in Finnish:" if spec.render else "Room labels are abbreviations in Finnish (one plan is Swedish):"
    cells.append(md(f"""
## Part 1 · Meet the plans

Detection (MP3) draws a **box** around an object. **Segmentation** goes one step further: it decides, *pixel by pixel*, what belongs to the object. On a drawing that is what makes it useful for **quantity take-off**: count the pixels of a room and you have its area in pixels; know the scale and you have square metres.

The model is **SAM 3** (Segment Anything Model 3, Meta 2025). You do not train it. You type a short phrase, such as *room* or *toilet*, and it returns every region of the drawing that matches, each with a **confidence**. You can also draw a **box** around something: SAM 3 cuts out its outline, and can look for everything else that looks like it.

{spec.description}

Every plan has a **5 m scale bar** at the bottom left. {labels_note}

| label | meaning |
|---|---|
{legend_rows}
"""))
    cells.append(form("▶ Step 1a · Browse the plans", "lab.show_plans(which)",
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
    cells.append(form("▶ Step 2c · Every room type at once", "lab.mix(plan, confidence)",
                      notes=["SAM 3's square metres per room type next to the drawing's, and the counts of fixtures and openings next to the truth."],
                      params=[f'plan = "{label_of[d0]}" #@param {jlist(labels)}', conf]))
    cells.append(form("▶ Step 2d · Can *you* estimate an area?", "lab.guess_game(rounds)",
                      notes=["A room is outlined on a plan: guess its area from the scale bar, then see the drawing's number and SAM 3's."],
                      params=['rounds = 4 #@param {type:"slider", min:2, max:8, step:1}']))
    questions.append((1, "From Step 2a and 2b: which words found what they should (rooms? toilets? windows? doors?), and which found nothing or something else? Give the found / missed / extra counts for two things on one plan at confidence 0.3, and say what the misses have in common."))
    cells.append(q(*questions[-1]))
    questions.append((2, "From Step 2c: for one plan, copy the table of square metres per room type (SAM 3 vs the drawing). Which room type is measured best and which worst, and why (merged rooms, furniture, open-plan kitchen and living room)? Then move the confidence to 0.2 and 0.7: what changes? Also give your score in the estimation game."))
    cells.append(q(*questions[-1]))

    # ------------------------------------------------------------------ Part 3
    cells.append(md("""
## Part 3 · Quantity take-off with boxes

A phrase is quick, but a take-off needs control. So now you draw the boxes. Three steps: **check the scale** on the scale bar (a 1 % error in the scale is a 2 % error in every area), **measure** rooms and elements by drawing a tight box around each, and **count** with *find_all*: draw one box around a window, and SAM 3 looks for every other thing that looks like it. The drawing's answer key tells you what it found, missed and added.

How a *room* box is measured: a box alone makes SAM 3 cut out the *furniture symbols* inside it rather than the empty floor (it was trained to find objects). So the notebook asks for *empty room* **and** gives your box as the example, keeps the region that fits your box, fills the holes left by symbols, removes the black walls, and converts the pixels with the scale. Every result is printed next to the drawing's own area.
"""))
    cells.append(form("▶ Step 3a · Check the scale", "lab.scale_check(plan)",
                      notes=["Draw a box from the 0 tick to the 5 m tick of the scale bar (zoom in with the mouse wheel for precision), label it *scale bar*, click *Submit*."],
                      params=[f'plan = "{label_of[d0]}" #@param {jlist(labels)}']))
    cells.append(form("▶ Step 3b · Measure and count with boxes", "lab.takeoff(plan, find_all, confidence)",
                      notes=[f"Draw tight boxes (zoom with the mouse wheel; put the edges on the inside faces of the walls): label each one *room*, *window*, *door*, *fixture* or *other*, then *Submit*. Measure at least {v['takeoff_targets']}. "
                             "With *find_all* ticked, SAM 3 also looks for everything like your first window / door / fixture / room box (blue boxes; thin green = the drawing's answer key). Needs the live model."],
                      params=[f'plan = "{label_of[d0]}" #@param {jlist(labels)}', 'find_all = True #@param {type:"boolean"}',
                              'confidence = 0.3 #@param {type:"slider", min:0.1, max:0.9, step:0.05}']))
    for text in v["questions_extra"]:
        questions.append((len(questions) + 1, text))
        cells.append(q(*questions[-1]))

    # ------------------------------------------------------------------ Part 4
    a, b = spec.compare_default
    cells.append(md("""
## Part 4 · Compare plans, and where it goes wrong

Two plans side by side: square metres per room type from SAM 3, and the drawings' own totals. Then three kinds of error to look for: the **words** you use (the model was trained on everyday photos, not on drawings), **weak regions** it proposes with low confidence, and plain **mistakes** that need a correction. The last step lets you correct the model by drawing a box over what it got wrong.
"""))
    cells.append(form("▶ Step 4a · Compare two plans", "lab.compare(plan_a, plan_b, confidence)",
                      params=[f'plan_a = "{label_of[a]}" #@param {jlist(labels)}', f'plan_b = "{label_of[b]}" #@param {jlist(labels)}', conf]))
    cells.append(form("▶ Step 4b · Does the wording matter?", "lab.phrase_lab(plan, thing, confidence)",
                      notes=["The same thing asked for with different words; all wordings are precomputed."],
                      params=[f'plan = "{label_of[d0]}" #@param {jlist(labels)}', f'thing = "{v["lab_thing"]}" #@param {jlist(things)}', conf]))
    cells.append(form("▶ Step 4c · Look at each region and its confidence", "lab.inspect(plan, thing)",
                      notes=["Every region the model proposed, numbered, with its confidence, its area and the room it sits on. Move the slider to see which ones survive."],
                      params=[f'plan = "{label_of[d0]}" #@param {jlist(labels)}', f'thing = "{v["inspect_thing"]}" #@param {jlist(things)}']))
    cells.append(form("▶ Step 4d · Correct it with a box", "lab.fix(plan, thing, confidence)",
                      notes=["Draw a box over a region that is wrong, click *Submit*: SAM 3 runs again with your box as a *not this* hint. Needs the live model."],
                      params=[f'plan = "{label_of[d0]}" #@param {jlist(labels)}', f'thing = "{v["fix_thing"]}" #@param {jlist(things)}', conf]))
    cells.append(form("▶ Step 4e · Your own words", "lab.your_phrase(plan, phrase, confidence)",
                      notes=["Type any phrase: a room, a symbol, a shape. Try *curved line* (the door swings), *thick black line* (the walls), *circle*, *small rectangle*. Needs the live model."],
                      params=[f'plan = "{label_of[d0]}" #@param {jlist(labels)}', 'phrase = "curved line" #@param {type:"string"}', conf]))
    questions.append((len(questions) + 1, "From Step 4a: which plan has more bedroom area and more bathroom area according to SAM 3, and does the drawing agree? From Step 4b: which wording worked best for the thing you chose, and how different were the areas?"))
    cells.append(q(*questions[-1]))
    questions.append((len(questions) + 1, "Describe one mistake you found in Step 4c or 4d (what was included or missed, at which confidence). Did the negative box fix it? What would you tell a colleague who wants to use these square metres in a cost estimate?"))
    cells.append(q(*questions[-1]))

    # ------------------------------------------------------------------ Part 5
    cells.append(md("## Part 5 · Your own plan"))
    cells.append(form("▶ Your plan, your words", "lab.upload_app()",
                      notes=["Upload a floor plan (a photo of a drawing works too, or open the public link on your phone), type what to find, move the threshold. "
                             "If the plan has a scale bar, measure how many pixels one metre is and enter the centimetres per pixel to get square metres.",
                             f"Test at least {v['own_plans']} plan(s) of your own and take screenshots for your report. Needs the live model."]))
    questions.append((len(questions) + 1, f"Test {v['own_plans']} plan(s) of your own (any floor plan from the internet or a course). For each: the phrase you used, the count and area measured, and whether the mask is right. What kind of drawing or wording failed?"))
    cells.append(q(*questions[-1]))
    questions.append((len(questions) + 1, "Where in a project would a take-off like this be useful, and where would it mislead? What would you need (clean drawings, a scale, a room schedule, a person checking) to turn it into numbers you would put in an estimate?"))
    cells.append(q(*questions[-1]))

    cells.append(md("## Wrap-up"))
    cells.append(form("▶ Numbers for your report", "lab.report_summary()"))
    cells.append(md("### Plan credits and model\n"
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
