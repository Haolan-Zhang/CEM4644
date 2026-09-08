"""Generates the MP4 Workshop and Homework notebooks (+ report templates) from one template."""
import json
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from aec_seg.config import SETS  # noqa: E402

GITHUB_URL = "https://github.com/Haolan-Zhang/CEM4644.git"

VARIANTS = {
    "workshop": dict(file="MP4_Workshop_Segmentation.ipynb", label="Workshop (in class)", minutes=80, dataset="site", plans=False,
                     own_photos=1, compare_default=("site_07", "site_09"), series_default="A", default_photo="site_01"),
    "homework": dict(file="MP4_Homework_Segmentation.ipynb", label="Homework (individual)", minutes=120, dataset="interior", plans=True,
                     own_photos=5, compare_default=("int_16", "int_18"), series_default="C", default_photo="int_19"),
}


def photo_labels(spec):
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
    labels, ids = photo_labels(spec)
    label_of = dict(zip(ids, labels))
    d0 = v["default_photo"]
    mats = spec.names
    series_names = list(spec.series)
    thr = 'threshold = 0.5 #@param {type:"slider", min:0.2, max:0.9, step:0.05}'
    questions, cells = [], []

    cells.append(md(f"""
# 🏗️ CEM4644 · MP4 — Segmentation for progress and quantity measurement
## {v['label']}: *{spec.title}*

**No coding needed.** Each grey box below is one *step*: click the ▶ (play) button at its left, wait until it finishes, look at the result, then answer the report question that follows. Run the steps **from top to bottom**.

**What you will do (about {v['minutes']} minutes)**
1. Look at site photos and the materials in them.
2. Ask a segmentation model, by name, for a material: see the mask, the overlay, and the share of the photo it covers.
3. Compare photos and follow a site through time.
4. Examine where the model goes wrong: wording, weak regions, and how to correct it.
{"5. Measure footing areas on a structural plan (quantity take-off)." if v["plans"] else "5. Try your own photo and your own words."}

**Before you start:** menu *Runtime → Change runtime type → T4 GPU → Save*. The model used here (SAM 3) is large: with a GPU each request takes well under a second; without one, precomputed results still work but live requests take about a minute each.
"""))
    cells.append(form(
        "▶ Step 0 · Run me first (2–3 minutes)",
        f"""import os, sys, subprocess
if not os.path.isdir("CEM4644/mp4_segmentation"):
    subprocess.run(["git", "clone", "--depth", "1", "-q", "{GITHUB_URL}"], check=True)
sys.path.insert(0, os.path.abspath("CEM4644/mp4_segmentation"))
from aec_seg import lab
lab.setup(dataset="{spec.key}", load_model=load_model, plans={v['plans']})""",
        notes=["Click ▶ and wait for the green ✅ line. This downloads the photos with their precomputed results and loads SAM 3 (about 3 GB).",
               "Untick *load_model* only if you have no GPU and want to skip the live steps."],
        params=['load_model = True #@param {type:"boolean"}'],
    ))

    # ------------------------------------------------------------------ Part 1
    cells.append(md(f"""
## Part 1 · Meet the photos

Detection (MP3) draws a **box** around an object. **Segmentation** goes one step further: it decides, *pixel by pixel*, what belongs to the object. That is what makes it useful for measurement: count the pixels of a material and you have its share of the photo; know the scale and you have an area.

The model in this notebook is **SAM 3** (Segment Anything Model 3, Meta 2025). You do not train it. You type a short phrase, such as *concrete* or *steel reinforcement bars*, and it returns every region in the photo that matches, each with a **confidence**.

{spec.description}
"""))
    cells.append(form("▶ Step 1a · Browse the photos", "lab.show_photos(which)",
                      notes=["*all* shows every photo with its credit. A series letter shows one site through time."],
                      params=[f'which = "all" #@param {jlist(["all"] + series_names)}']))

    # ------------------------------------------------------------------ Part 2
    cells.append(md(f"""
## Part 2 · Segment a material by name

Pick a photo and a material. You get three panels: the photo, the **mask** (white = the model says *this is it*), and the **overlay**. The share of the photo covered by the mask is printed below, together with the confidence of each region. The **confidence slider** hides the regions the model is unsure about: watch how the share changes.
"""))
    cells.append(form("▶ Step 2a · Original → mask → overlay", "lab.segment(photo, material, threshold)",
                      notes=["Try several materials on the same photo, then the same material on other photos."],
                      params=[f'photo = "{label_of[d0]}" #@param {jlist(labels)}', f'material = "{mats[0]}" #@param {jlist(mats)}', thr]))
    cells.append(form("▶ Step 2b · The material mix of one photo", "lab.material_mix(photo, threshold)",
                      notes=["Every material at once, with the share of the photo each one covers. Regions may overlap (a worker standing in front of concrete), so the shares need not add up to 100."],
                      params=[f'photo = "{label_of[d0]}" #@param {jlist(labels)}', thr]))
    cells.append(form("▶ Step 2c · Can *you* estimate the share?", "lab.guess_game(rounds)",
                      notes=["A photo and a material: guess how much of the photo it covers, then see what SAM 3 measures."],
                      params=['rounds = 4 #@param {type:"slider", min:2, max:8, step:1}']))
    questions.append((1, "What was your score in the estimation game? Pick one photo and give its material mix at confidence 0.5 (the numbers from Step 2b). Then move the threshold to 0.3 and 0.8 for one material: how much does the share change, and why?"))
    cells.append(q(*questions[-1]))

    # ------------------------------------------------------------------ Part 3
    cells.append(md("""
## Part 3 · Compare photos and follow progress

Two photos of the same site at different times tell a story: formwork and rebar disappear, concrete and brick appear. Comparing the material shares turns that story into numbers. Keep in mind what the number is: the share of the **photo**, not the share of the **work**: camera position, zoom and the sky change it without any progress on site.
"""))
    a, b = v["compare_default"]
    cells.append(form("▶ Step 3a · Compare two photos", "lab.compare(photo_a, photo_b, threshold)",
                      params=[f'photo_a = "{label_of[a]}" #@param {jlist(labels)}', f'photo_b = "{label_of[b]}" #@param {jlist(labels)}', thr]))
    cells.append(form("▶ Step 3b · One site through time", "lab.series(series, threshold)",
                      notes=["The photos of a series in order, and a chart of each material's share over time."],
                      params=[f'series = "{v["series_default"]}" #@param {jlist(series_names)}', thr]))
    questions.append((2, "From Step 3b: which materials rise and which fall over the series, and does that match what a site manager would expect? Give one example where the number changes for a reason that has nothing to do with progress (camera position, sky, an old black-and-white photo...)."))
    cells.append(q(*questions[-1]))

    # ------------------------------------------------------------------ Part 4
    lab_photo = spec.phrase_lab_photos[0]
    cells.append(md("""
## Part 4 · Where does it go wrong?

Three kinds of error to look for: the **words** you use (the model was trained on everyday language, not construction jargon), **weak regions** the model proposes with low confidence, and plain **mistakes** that need a correction. The last step lets you correct the model by drawing a box over what it got wrong.
"""))
    cells.append(form("▶ Step 4a · Does the wording matter?", "lab.phrase_lab(photo, material, threshold)",
                      notes=[f"The same material asked for with different words. Alternative wordings are precomputed for {', '.join(spec.phrase_lab_photos)}; other photos need the live model."],
                      params=[f'photo = "{label_of[lab_photo]}" #@param {jlist(labels)}', f'material = "{mats[1]}" #@param {jlist(mats)}', thr]))
    cells.append(form("▶ Step 4b · Look at each region and its confidence", "lab.inspect(photo, material)",
                      notes=["Every region the model proposed, numbered, with its confidence. Move the slider to see which ones survive."],
                      params=[f'photo = "{label_of[lab_photo]}" #@param {jlist(labels)}', f'material = "{mats[0]}" #@param {jlist(mats)}']))
    cells.append(form("▶ Step 4c · Correct it with a box", "lab.fix(photo, material, threshold)",
                      notes=["Draw a box over a region that is wrong, click *Submit*: SAM 3 runs again with your box as a *not this* hint. Needs the live model."],
                      params=[f'photo = "{label_of[lab_photo]}" #@param {jlist(labels)}', f'material = "{mats[0]}" #@param {jlist(mats)}', thr]))
    cells.append(form("▶ Step 4d · Your own words", "lab.your_phrase(photo, phrase, threshold)",
                      notes=["Type any phrase: a material, a tool, a machine, a colour. Needs the live model."],
                      params=[f'photo = "{label_of[d0]}" #@param {jlist(labels)}', 'phrase = "safety helmet" #@param {type:"string"}', thr]))
    questions.append((3, "From Step 4a: which wording gave the most sensible mask for the material you chose, and how far apart were the shares? Why would *rebar* and *steel reinforcement bars* give different answers?"))
    cells.append(q(*questions[-1]))
    questions.append((4, "Describe one mistake you found in Step 4b or 4c (what was included or missed, at which confidence). Did the negative box fix it? What would you tell a colleague who wants to use these percentages in a progress report?"))
    cells.append(q(*questions[-1]))

    # ------------------------------------------------------------------ Part 5
    if v["plans"]:
        plans = SETS["plans"].plans
        plan_labels = [f"{p['id']}" for p in plans]
        cells.append(md(f"""
## Part 5 · Quantity take-off on a structural plan

On a drawing, words do not help: SAM 3 finds nothing for *footing*. A **box** does. Draw a box around a footing and SAM 3 cuts out its exact outline; the number of pixels inside is its area in pixels. To turn pixels into square feet you need a **scale**: one footing whose real width you know. On plan 1 the reference is **{plans[0]['reference']}** ({plans[0]['reference_width_ft']} ft wide); on plan 2 it is **{plans[1]['reference']}** ({plans[1]['reference_width_ft']} ft).

Label one box *reference*, the others *footing* or *opening*, then *Submit*. Members to measure on plan 2: {', '.join(plans[1]['targets'])}.
"""))
        cells.append(form("▶ Step 5a · Draw boxes, get areas", "lab.takeoff(plan)",
                          notes=["Needs the live model. Tight boxes give clean masks; if a mask spills, redraw the box tighter and submit again."],
                          params=[f'plan = "{plan_labels[0]}" #@param {jlist(plan_labels)}']))
        questions.append((5, "Plan 1: what scale did you get (feet per pixel) and what area for the reference footing? A 12 ft square footing should measure 144 sq ft: how far off is SAM 3, and where does the error come from (the box, the mask edge, the drawing)?"))
        cells.append(q(*questions[-1]))
        questions.append((6, "Plan 2: give the scale and the areas of F4.0, E4-6, E4-10, E5-0 and the elevator shaft opening, with the overlay screenshot. Which one was hardest for the model and why? How would you check these numbers before using them in a cost estimate?"))
        cells.append(q(*questions[-1]))
        nxt = 7
    else:
        nxt = 5

    # ------------------------------------------------------------------ own photo
    cells.append(md("## Part 6 · Your own photo" if v["plans"] else "## Part 5 · Your own photo"))
    cells.append(form("▶ Your photo, your words", "lab.upload_app()",
                      notes=["Upload a photo (or open the public link on your phone), type what to find, move the threshold.",
                             f"Test at least {v['own_photos']} photo(s) of your own and take screenshots for your report. Needs the live model."]))
    questions.append((nxt, f"Test {v['own_photos']} photo(s) of your own (walls, floors, a site, a street). For each: the phrase you used, the share measured, and whether the mask is right. What kind of surface or wording failed?"))
    cells.append(q(*questions[-1]))
    questions.append((nxt + 1, "Where on a project would a measurement like *share of the photo covered by X* be useful, and where would it mislead? What would you need (camera position, reference lengths, drawings, several photos) to turn it into a real quantity?"))
    cells.append(q(*questions[-1]))

    cells.append(md("## Wrap-up"))
    cells.append(form("▶ Numbers for your report", "lab.report_summary()"))
    cells.append(md("### Photo credits and model\n"
                    "All photos are from Wikimedia Commons under the licence shown with each photo in Step 1a (public domain, CC0, CC BY or CC BY-SA; "
                    "credits are also in `data/photos/*/credits.json`). Plan drawings are course material.\n\n"
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
             f"Notebook: `{v['file']}` — photos: *{spec.title}*.", "",
             "Answer every question in a few sentences. Paste screenshots where the question asks for photos or overlays. "
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
