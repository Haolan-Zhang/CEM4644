"""Generates the two MP4 student notebooks (+ their report templates).

    python build/make_notebooks.py                 # both
    python build/make_notebooks.py workshop        # one
    python build/make_notebooks.py --root <dir>    # read the drawings from another tree (testing)

Every student-facing cell is a Colab form: the code is one line calling `aec_seg`, the controls are #@param
widgets, and the cell's metadata hides the source.
"""
import json
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from aec_seg.config import SETS, group_of  # noqa: E402
from aec_seg.data import IntroPhotos, Sheets  # noqa: E402
from aec_seg.ui import INTRO_PHRASES  # noqa: E402

GITHUB_URL = "https://github.com/Haolan-Zhang/CEM4644.git"
ROOT = REPO

STEP0 = """import importlib, os, shutil, subprocess, sys
REPO, FOLDER, PKG = "CEM4644", "mp4_segmentation", "aec_seg"
FOLDERS = ["mp4_segmentation"]               # only this lab folder is downloaded, not the whole course repository

def _git(*args):
    return subprocess.run(["git", "-C", REPO, *args], capture_output=True, text=True).returncode == 0

if os.path.isdir(REPO):                      # a copy is already here: pull the newest course code over it
    if not (_git("sparse-checkout", "set", *FOLDERS)     # also trims a full copy left by an earlier run
            and _git("fetch", "-q", "--depth", "1", "origin", "master")
            and _git("reset", "-q", "--hard", "FETCH_HEAD") and _git("clean", "-qfd")):
        shutil.rmtree(REPO, ignore_errors=True)          # broken copy: start again from scratch
if not os.path.isdir(REPO):
    subprocess.run(["git", "clone", "-q", "--depth", "1", "--filter=blob:none", "--sparse", "{url}", REPO], check=True)
    subprocess.run(["git", "-C", REPO, "sparse-checkout", "set", *FOLDERS], check=True)
for _m in [m for m in list(sys.modules) if m == PKG or m.startswith(PKG + ".")]:
    del sys.modules[_m]                      # Python caches imported code: drop it, or this cell keeps the old version
importlib.invalidate_caches()
sys.path.insert(0, os.path.abspath(os.path.join(REPO, FOLDER)))
from aec_seg import lab
lab.setup(dataset="{set}", load_model=load_model)"""


# --------------------------------------------------------------------------- cell helpers
def form(title, body, notes=(), params=()):
    title = "\u25b6 " + title.replace(" - ", " \u00b7 ", 1)      # the course's step style: a play mark and a middle dot
    src = f'#@title {title} {{ display-mode: "form" }}\n'
    src += "".join(f"#@markdown {n}\n" for n in notes)
    src += "".join(p + "\n" for p in params)
    src += body.rstrip() + "\n"
    c = new_code_cell(src)
    c.metadata["cellView"] = "form"
    return c


def md(text):
    import re
    text = re.sub(r"^(#{1,3} (?:Part \d+|Step \d+\w?)) - ", "\\1 \u00b7 ", text.strip("\n"), flags=re.M)   # "## Part 1 · ..." like the other labs
    return new_markdown_cell(text)


def jlist(items):
    return json.dumps(list(items), ensure_ascii=False)


def param_choice(name, value, options, quiet=False):
    return f'{name} = {json.dumps(value, ensure_ascii=False)} #@param {jlist(options)}'


CONF = 'confidence = 0.3 #@param {type:"slider", min:0.1, max:0.9, step:0.05}'


class Questions:
    def __init__(self):
        self.items = []

    def add(self, text):
        self.items.append((len(self.items) + 1, text))
        n, t = self.items[-1]
        return md(f"> ### \U0001F4DD Report question {n}\n> {t}")


def header(set_key, minutes, what):
    spec = SETS[set_key]
    return md(f"""
# CEM4644 - MP4: Segmentation for a quantity take-off

## {'Workshop (in class)' if set_key == 'workshop' else 'Homework (individual)'}: *{spec.title}*

**No coding needed.** Each grey box below is one *step*: click the (play) button at its left, wait until it finishes,
look at the result, then answer the report question that follows. Run the steps **from top to bottom**.

**What you will do (about {minutes} minutes)**
{what}

**Before you start:** menu *Runtime -> Change runtime type -> T4 GPU -> Save*. The model used here (SAM 3) is large:
with a GPU each request takes well under a second; without one the steps that need the model take about a minute each.

Everything is in **feet and square feet**. There is no scale printed on a drawing that you can trust blindly: you set
the scale yourself, from a dimension the drawing prints or from something whose real size you know.
""")


def step0(set_key):
    return form(
        "Step 0 - Run me first (2-3 minutes)",
        STEP0.format(url=GITHUB_URL, set=set_key),
        notes=["Click the play button and wait for the 'Ready' line. This downloads the drawings with their answer keys "
               "and loads SAM 3 (about 3 GB).",
               "Untick *load_model* only if you have no GPU and want to skip the steps that need the live model."],
        params=['load_model = True #@param {type:"boolean"}'],
    )


def credits_cell(sheets, intro, extra=""):
    lines = ["### Where the drawings come from, and the model", ""]
    for s in sheets:
        c = s.credit
        lines.append(f"- **{s.id}** - {c.get('title', s.title)}. {c.get('author', '')}. {c.get('license', '')}. {c.get('source', '')}")
    if intro:
        lines.append("- Site photos in Part 1: " + "; ".join(f"{p.title} ({p.author}, {p.license}, {p.source})" for p in intro.photos) + ".")
    lines.append("- Model: SAM 3 by Meta AI (SAM License), loaded from a public mirror of the official checkpoint; "
                 "a copy of the licence is in `docs/SAM_LICENSE.txt`.")
    lines.append("- Lab code: https://github.com/Haolan-Zhang/CEM4644 (folder `mp4_segmentation`).")
    if extra:
        lines.append(extra)
    return md("\n".join(lines))


def write(nb, path: Path):
    nb.metadata.update({"colab": {"provenance": [], "gpuType": "T4", "toc_visible": True}, "accelerator": "GPU",
                        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                        "language_info": {"name": "python"}})
    nbformat.write(nb, str(path))
    print("wrote", path)


def write_template(variant, label, file, questions):
    lines = [f"# CEM4644 - MP4 report: {label}", "", "Name: ______________________    Date: ____________", "",
             f"Notebook: `{file}`.", "",
             "Answer every question in a few sentences. Paste screenshots where the question asks for an overlay or a "
             "table. Every number must come from **your** run of the notebook.", ""]
    for n, text in questions.items:
        lines += [f"## Question {n}", "", text, "", "*Your answer:*", "", "", ""]
    p = REPO / "docs" / f"MP4_{variant.capitalize()}_Report_Template.md"
    p.parent.mkdir(exist_ok=True)
    p.write_text("\n".join(lines))
    print("wrote", p)


# --------------------------------------------------------------------------- the workshop notebook
def build_workshop():
    spec = SETS["workshop"]
    sheets = Sheets(ROOT / spec.folder, "workshop")
    intro = IntroPhotos(ROOT / "data" / "intro")
    photo_labels = intro.labels()
    labels = sheets.labels()
    d0 = labels[0]
    things = spec.names
    q = Questions()
    cells = [header("workshop", 90, """1. Meet SAM 3 on an ordinary site photo: ask by name, draw a box, tap an object. Then look at the three drawings.
2. Ask for rooms, doors and windows **by name** on a drawing, and check the rooms against the drawing's own answer key.
3. Do the **take-off**: set the scale from a printed dimension, then measure every room in square feet - one cell per drawing.
   Then box **one** door and let SAM 3 find all the others.
4. Look at where the model goes wrong: the words, the weak regions, and words of your own.
5. Try a drawing of your own."""), step0("workshop")]

    cells.append(md("""
## Part 1 - Meet SAM 3

Detection (MP3) draws a **box** around an object. **Segmentation** goes one step further: it decides, *pixel by pixel*,
what belongs to the object. Count the pixels and you have an area; know the scale and you have square feet. That is what
makes it useful for a **quantity take-off**.

The model is **SAM 3** (Segment Anything Model 3, Meta 2025). You do not train it, and it has no fixed list of classes.
You tell it *what* or *where*, in one of four ways:

- a **phrase**, such as *helmet* or *wet concrete*: it returns every region that matches, each with a **confidence**;
- a **box** around one object: it cuts out that object's exact outline;
- a **click** on one object: the same thing, from a single point;
- a box around one object **as an example**: it returns every object on the picture that looks like it. No words,
  and that is how a count is made (Step 3d).

Two steps on an ordinary site photo first, so that you see what the model does before it meets a drawing.
"""))
    cells.append(form("Step 1a - Ask by name", "lab.intro_phrase(photo, phrase, own_phrase, confidence)",
                      notes=["Pick a phrase, or type your own in *own_phrase* (it wins when it is not empty). Three panels: the photo, "
                             "the mask (white = the model says *this is it*), the overlay. Try a thing (*helmet*), a material "
                             "(*wet concrete*), a part (*hand*), and something that is not in the photo at all. Watch the confidences."],
                      params=[param_choice("photo", photo_labels[0], photo_labels),
                              param_choice("phrase", "person", INTRO_PHRASES),
                              'own_phrase = "" #@param {type:"string"}', CONF]))
    cells.append(form("Step 1b - Box it, or tap it", "lab.intro_draw(photo)",
                      notes=["Draw a box around an object and label it *box*; or draw a tiny box on an object and label it *point* "
                             "(its centre is the click). Draw several, click *Submit*: SAM 3 cuts out one object per box or click, "
                             "no words needed. Needs the live model."],
                      params=[param_choice("photo", photo_labels[0], photo_labels)]))
    cells.append(md(f"""
### The drawings

{spec.description}

Read a drawing before you measure it: the room names and the printed room sizes, the overall dimension lines along two
sides (that is where your scale comes from), the black walls, the windows drawn as a gap with thin lines in it, and the
quarter-circle arcs that are door swings.
"""))
    cells.append(form("Step 1c - Browse the drawings", "lab.show_sheets(drawing)",
                      notes=["*all drawings* shows all three with what is on them and what you will take off from each. "
                             "These facts come from the answer key the notebook checks your measurements against."],
                      params=[param_choice("drawing", "all drawings", ["all drawings"] + labels)]))

    cells.append(md("""
## Part 2 - Ask for something by name

Pick a drawing and a thing. You get three panels: the drawing, the **mask** (white = the model says *this is it*), and
the **overlay**. Below them: how many regions, how many pixels, how many square feet, and what the **drawing's own
answer key** says. The **confidence slider** hides the regions the model is unsure about: watch the count change.
"""))
    cells.append(form("Step 2a - Original, mask, overlay", "lab.segment(drawing, thing, confidence)",
                      notes=["Try *room (any)* first, then *bedroom*, then *kitchen*, *porch*, *door*, *window*. Some words work, "
                             "some find nothing at all: that is the lesson of this part."],
                      params=[param_choice("drawing", d0, labels), param_choice("thing", "room (any)", things), CONF]))
    cells.append(form("Step 2b - Hits, misses and extras", "lab.count(drawing, thing, confidence)",
                      notes=["The answer key drawn on the drawing: green = a real one the model found, red = a real one it missed, "
                             "blue = a region that is not one at all."],
                      params=[param_choice("drawing", d0, labels), param_choice("thing", "room (any)", things), CONF]))
    cells.append(q.add("From Steps 2a and 2b: which words found what they should (rooms? bedrooms? kitchens? doors? windows?), and "
                       "which found nothing or something else? These drawings' answer keys measure rooms, so the hit / miss / extra "
                       "overlay of Step 2b works on the room words: give the found / missed / extra numbers for two room words on one "
                       "drawing at confidence 0.3. Then say what the words that found nothing have in common."))

    cells.append(md(f"""
## Part 3 - The take-off

A phrase is quick, but a take-off needs control, so now **you** draw the boxes. One cell per drawing. In each cell, pick
the label above the picture before you draw, and draw in this order:

1. **the scale**: one box exactly along the printed overall dimension, from arrowhead to arrowhead. Only the length along
   that dimension is used. *A 1 % error in the scale is a 2 % error in every area, because area is scale squared.*
2. **the second dimension** down the side of the plan, as a check. The two readings never agree exactly, and the report
   tells you by how much they differ and which one the answer key trusts.
3. **every room**: a tight box each, the edges on the **inside faces** of the walls - rooms, porches and halls.

Then press *Submit*.

The take-off cells measure rooms and nothing else. Counting is a separate job, and it is a job for the model, not for
your pencil: boxing every door yourself and ticking the boxes off against an answer key would tell you nothing about
SAM 3. So Step 3d does it the model's way - you draw **one** box around one door and SAM 3 finds all the others -
and the answer key tells you how many it got. On the homework's structural and MEP sheets the same one box counts
footings and light fixtures.

How a *room* box becomes square feet: a box on its own makes SAM 3 cut out the *furniture symbols* inside it rather than
the floor (it was trained to find objects). So the notebook asks for *empty room* **and** hands it your box, keeps the
region that fits your box, fills the holes the symbols leave, removes the black walls, gives back the bites that door
swings take out of a rectangular room, and converts the pixels with **your** scale.
"""))
    for i, s in enumerate(sheets):
        letter = "abc"[i] if i < 3 else str(i)
        cells.append(form(f"Step 3{letter} - Take-off: {s.id}, {s.title}", f'lab.takeoff({json.dumps(s.label)})'))
    cells.append(q.add("From Step 3 on all three drawings: your two scale readings on each drawing, how far each one is from the "
                       "answer key and how far they are from each other; and the room table (your square feet, the drawing's, the "
                       "error) for the drawing you did best on. What is the total of your rooms against the drawing's indoor total?"))
    cells.append(q.add("Which rooms came out worst, and why? Look at the pictures and name the reason for at least three of them "
                       "(a loose box, a kitchen counter or a bathtub eaten out of the mask, a hall that is really a set of "
                       "doorways, an open space with no wall on one side, the scale). Did the notebook warn you about any of "
                       "them, and was the warning right?"))
    cells.append(md("""
### Box one, and SAM 3 finds the rest

In Step 2 the word *door* found nothing. Now give the model an **example** instead of a word: one box around one door,
the opening and its swing arc together. SAM 3 looks at what is inside your box and returns everything on the sheet that
looks like it - the fourth way of asking from Part 1. The answer key marks what it found and what it missed, so the
count is checked, not taken on trust. Which door you pick matters: a clean single door in a quiet spot is a good
example; a double door or a closet door in a cluttered corner is a poor one, and the count drops.
"""))
    cells.append(form("Step 3d - Box one door, and SAM 3 finds the rest", "lab.find_like(drawing)",
                      params=[param_choice("drawing", labels[1], labels)]))
    cells.append(q.add("From Step 3d: on each of the three drawings, the best count you got from one example box (found / "
                       "missed / extra, and the confidence), and how much the count changed when you picked a different door "
                       "as the example. Which drawing was hardest and why? In Step 2 the word *door* found nothing: why does one "
                       "example work where the word does not?"))

    cells.append(md("""
## Part 4 - Where it goes wrong

Three kinds of error to look for: the **words** you use (the model was trained on everyday photographs, not on drawings),
the **weak regions** it proposes with a low confidence, and words of your own that describe what is *drawn* rather than
what it *means*.
"""))
    cells.append(form("Step 4a - Does the wording matter?", "lab.phrase_lab(drawing, thing, confidence)",
                      notes=["The same thing asked for with three or four different words. All the wordings are precomputed, so this "
                             "is instant. Try *door* (against *curved line*) and *window* (against *short parallel lines*)."],
                      params=[param_choice("drawing", d0, labels), param_choice("thing", "door", things), CONF]))
    cells.append(form("Step 4b - Look at each region and its confidence", "lab.inspect(drawing, thing)",
                      notes=["Every region the model proposed, numbered, with its confidence, its area and the room it sits on. "
                             "Move the slider to see which ones survive."],
                      params=[param_choice("drawing", d0, labels), param_choice("thing", "room (any)", things)]))
    cells.append(form("Step 4c - Your own words", "lab.your_phrase(drawing, phrase, confidence)",
                      notes=["Type any phrase: a room, a symbol, a shape. Try *curved line* (the door swings), *thick black line* "
                             "(the walls), *circle*, *small rectangle*, *hatched square*. Needs the live model."],
                      params=[param_choice("drawing", d0, labels), 'phrase = "curved line" #@param {type:"string"}', CONF]))
    cells.append(q.add("From Step 4a: which wording worked best for the thing you chose, and how different were the counts? From "
                       "Step 4b: describe one weak region (what it sits on, its confidence) and one plain mistake. What would you "
                       "tell a colleague who wants to put these square feet in a cost estimate?"))
    cells.append(q.add("From Step 4c: which of your own words found something that the name of the thing could not (for example "
                       "*curved line* for the door swings, *thick black line* for the walls)? Why does a shape word work on a "
                       "drawing where the name of the thing does not?"))

    cells.append(md("## Part 5 - Your own drawing"))
    cells.append(form("Your drawing, your words", "lab.upload_app()",
                      notes=["This cell prints a **link**: open it in a new tab (it works on a phone too). Upload a drawing, then ask "
                             "the three ways of this lab: a box for the scale (a printed dimension, plus its length in feet), a "
                             "box for a room or an object, one example box that SAM 3 finds the rest of, or a phrase. A box is two "
                             "clicks on the drawing: top-left, then bottom-right. Test at least one drawing of your own and take "
                             "screenshots. Needs the live model."]))
    cells.append(q.add("Test one drawing of your own (any floor plan or construction drawing). Which phrase or box did you use, "
                       "what did it find, and was the result right? Then: where in a project would a take-off like this be useful, and where "
                       "would it mislead you? What would you need (clean drawings, a known dimension, a room schedule, a person "
                       "checking) before you would put these numbers in an estimate?"))

    cells.append(md("## Wrap-up"))
    cells.append(form("Numbers for your report", "lab.report_summary()",
                      notes=["Every take-off you submitted in Part 3, printed again in one place."]))
    cells.append(credits_cell(sheets, intro))
    nb = new_notebook(cells=cells)
    write(nb, REPO / "MP4_Workshop_Segmentation.ipynb")
    write_template("workshop", "Workshop (in class)", "MP4_Workshop_Segmentation.ipynb", q)


# --------------------------------------------------------------------------- the homework notebook
GROUP_TEXT = {
    "floor": ("Floor plans",
              "On a floor plan you take off **areas of rooms**. Set the scale from a printed dimension, never from a scale "
              "bar: one of these two sheets carries a graphic scale bar that is wrong by a factor of two, and boxing it as "
              "well as the printed dimension is how you find that out. Then box every room the task asks for. The number "
              "you get is the *net* floor area: what the drawing puts on the floor (a counter, a bathtub) is cut out of "
              "the mask unless the room is a plain rectangle. The clinic sheet also asks for two **counts**, the water "
              "closets and the lavatories, and each one comes from a single box labelled *example: water closet* or "
              "*example: lavatory*."),
    "structural": ("Structural (foundation) plans",
                   "On a foundation plan you take off **areas of footings** and a **count of them**. The scale comes from a "
                   "printed bay dimension or from a footing whose width its mark gives you (F4.0 = 4'-0\" wide). The count "
                   "is made by the model from your first *footing* box, which it uses as the example - you never count them yourself. Two "
                   "things to watch: read the *same* edge of the ink at both ends (outside-to-outside or centre-to-centre, "
                   "not one of each), and remember that a symbol smaller than about 60 pixels on the sheet is too small for "
                   "the model - the mask becomes a rounded copy of your box, and the 'area' you get is the area you drew."),
    "mep": ("MEP plans",
            "On an MEP sheet almost nothing is measured and almost everything is **counted**. The trade words - *light "
            "fixture*, *diffuser*, *sprinkler* - return nothing at all from the model, so counting is done the other way "
            "round: box **one** example of the symbol, labelled *example: 2x4 light fixture*, and the model finds every "
            "other symbol like it. Pick an example that is clean and lying the same way as most of the others; a rotated "
            "example loses about 40 % of the count."),
}


def build_homework():
    spec = SETS["homework"]
    sheets = Sheets(ROOT / spec.folder, "homework")
    q = Questions()
    groups = {}
    for s in sheets:
        groups.setdefault(group_of(s.discipline), []).append(s)
    cells = [header("homework", 150, """1. Look at the seven drawings: what each one is, what it shows, and what you have to take off from it.
2. Do a take-off on each discipline in turn - floor plans, structural plans, MEP plans - with the same one cell, then
   ask SAM 3 for the same things by phrase and see which way gets you a usable number.
3. Try a drawing of your own."""), step0("homework")]

    cells.append(md(f"""
## Part 1 - The drawings

{spec.description}

You already met SAM 3 in the workshop, so this notebook goes straight to the work. What stays the same on every sheet:

- **you** set the scale, from a dimension the sheet prints or from something whose size the sheet tells you. Never from a
  scale bar you have not checked.
- **you** draw the boxes. The model turns a box into an outline; it does not know what a footing or a diffuser is.
- **counts come from the model, never from a tally of your own boxes**: where a sheet asks for a count you box ONE
  example of the symbol, labelled *example: ...*, and SAM 3 finds all the others like it.
- everything you measure and everything you count is checked against an answer key, so you always see how far off you are.
"""))
    cells.append(form("Step 1a - Browse the seven drawings", "lab.show_sheets(drawing)",
                      notes=["*all drawings* shows all seven; pick one to see it large."],
                      params=[param_choice("drawing", "all drawings", ["all drawings"] + sheets.labels())]))
    cells.append(form("Step 1b - The symbol legend", "lab.show_legend(drawing)",
                      notes=["The MEP sheets carry a legend of their symbols: a bold rectangle with a diagonal and a small circle "
                             "is a light fixture (2 ft x 4 ft or 2 ft x 2 ft), a small circle with a cross is a recessed light. "
                             "The structural sheets name their footings in a schedule printed on the sheet instead."],
                      params=[param_choice("drawing", "all", ["all"] + sheets.labels())]))

    cells.append(md("""
## Two ways of asking, on every discipline

Each of the next three parts has two cells. The first is the **take-off**: pick the drawing, then pick the label above
the picture before each box you draw (the labels come from that sheet's answer key), then *Submit*. Three kinds of
label: **scale: ...** for a length whose size the sheet gives you, the plain word (**room**, **footing**, **pit**) for
something you want the area of, and **example: ...** for something you want counted. One box labelled *example: pile
footing* is all a count needs: SAM 3 goes and finds every other symbol on the sheet that looks like it, and that is how
a count of 41 light fixtures or 29 pile footings is made. Where the thing counted is also measured (the footings), there is
no separate example label: your **first** *footing* box is the example. The slider under the picture sets how sure the model has to be
before it keeps one of them.

The second cell is the other way of asking: **type a phrase** and SAM 3 looks for it on the whole sheet, with no box
from you at all. *compare* picks the part of the answer key the regions are scored against: green for a real one found,
red for one missed, blue for a region that is not one; for rooms and footings the notebook also measures every region
the phrase found. Try the trade word first (*footing*, *light fixture*), then a word for the **shape on the paper**
(*square*, *circle*, *rectangle with a diagonal line*), and move the confidence. Both cells feed the same report
question: which way gets you a number you would put in an estimate, and which way is quicker?
"""))
    part_no = {"floor": 2, "structural": 3, "mep": 4}
    for gid in ("floor", "structural", "mep"):
        gsheets = groups.get(gid, [])
        if not gsheets:
            continue
        gsheets.sort(key=lambda s: (len(s.discipline), s.id))   # the plainest sheet of the group first
        n = part_no[gid]
        head, text = GROUP_TEXT[gid]
        default_phrase, phrase_note = PHRASE_HINTS[gid]
        cells.append(md(f"## Part {n} - {head}\n\n{text}\n\n"
                        f"Do every drawing in the list, one at a time, in Step {n}a. Phrases to try in Step {n}b: {phrase_note}"))
        glabels = [s.label for s in gsheets]
        what = {"floor": "a floor plan", "structural": "a structural plan", "mep": "an MEP sheet"}[gid]
        cells.append(form(f"Step {n}a - Take-off on {what}: your boxes", "lab.takeoff(drawing)",
                          params=[param_choice("drawing", glabels[0], glabels)]))
        cats = []
        for s in gsheets:
            for c in s.area_categories + s.count_categories:
                if c not in cats:
                    cats.append(c)
        cells.append(form(f"Step {n}b - Ask by name on {what}: a phrase",
                          "lab.ask(drawing, phrase, confidence, compare)",
                          params=[param_choice("drawing", glabels[0], glabels),
                                  f'phrase = {json.dumps(default_phrase)} #@param {{type:"string"}}', CONF,
                                  param_choice("compare", cats[0], ["(nothing)"] + cats)]))
        cells.append(q.add(QUESTION_BY_GROUP[gid]))

    cells.append(md("## Part 5 - Your own drawing"))
    cells.append(form("Your drawing, your words", "lab.upload_app()",
                      notes=["This cell prints a **link**: open it in a new tab (it works on a phone too). Upload a drawing, then ask "
                             "the three ways of this lab: a box for the scale (a printed dimension, plus its length in feet), a "
                             "box for a room or an object, one example box that SAM 3 finds the rest of, or a phrase. A box is two "
                             "clicks on the drawing: top-left, then bottom-right. Test at least three drawings of your own, from at "
                             "least two disciplines, and take screenshots. Needs the live model."]))
    cells.append(q.add("Test three drawings of your own, from at least two disciplines (a floor plan, a structural plan, an MEP "
                       "sheet, a section, a site plan...). For each: the phrase or box you used, what came back, and whether it is "
                       "right. Which kind of drawing failed, and can you say why?"))
    cells.append(q.add("Across the three disciplines: which take-off task was the most reliable and which the least, and what "
                       "decides that (the size of the thing on the sheet, how often it repeats, whether it is drawn as a simple "
                       "outline)? Boxes or phrases: for each discipline, say which way you would use and why. Where would you use "
                       "this in practice, where would it mislead you, and what would you insist on having (a known dimension, a "
                       "schedule, a second pair of eyes) before putting these numbers in an estimate?"))

    cells.append(md("## Wrap-up"))
    cells.append(form("Numbers for your report", "lab.report_summary()",
                      notes=["Every take-off you submitted, printed again in one place, plus the drawings you have not done yet."]))
    cells.append(credits_cell(sheets, None))
    nb = new_notebook(cells=cells)
    write(nb, REPO / "MP4_Homework_Segmentation.ipynb")
    write_template("homework", "Homework (individual)", "MP4_Homework_Segmentation.ipynb", q)


QUESTION_BY_GROUP = {
    "floor": ("From Step 2a: your scale reading on each sheet and how far it is from the answer key. On the VA clinic sheet you "
              "were asked to box the graphic scale bar as well as the printed dimension - what did the two give, and which one "
              "is right? (Work out what the areas would have been if you had trusted the bar.) Then the room table of one sheet: "
              "your square feet, the drawing's, the error. Which rooms are worst and why? The two counts SAM 3 made from your "
              "example boxes on the clinic sheet (found / missed / extra): which would you hand on, which would you check first? "
              "Then from Step 2b: what did *room* find on each sheet (found / missed / extra, and the median error of the areas "
              "it measured) against the rooms from your boxes? What did *door* and *window* return, and what did *curved line*?"),
    "structural": ("From Step 3a: for each sheet, the scale and how you set it, the areas of the footings you measured against "
                   "the sizes their marks give, and the count SAM 3 made from your first *footing* box, its example (found / missed / "
                   "extra). One sheet asks you only to count, because its pile caps are too small to measure - what happens to the "
                   "measured area of something that small, and why? Then from Step 3b: does *footing* find anything? Which shape "
                   "word finds the footings, on which sheet, with how many extras, and how do the areas it measures compare with "
                   "the ones from your boxes? Did any phrase find the grid bubbles or the pile caps?"),
    "mep": ("From Step 4a: the count of each symbol that SAM 3 made from your one example box (found / missed / extra), the "
            "confidence you used, and what the extras were. Try a second example box of the same symbol, one that is rotated or "
            "sits in a cluttered spot, and report how the count changes. Then from Step 4b: which words find any light fixture at "
            "all - the trade words or the shape words - and does the best phrase get anywhere near the count from your one example "
            "box (found / missed / extra for both)? Why do *light fixture* and *diffuser* return nothing on a drawing?"),
}

PHRASE_HINTS = {
    "floor": ("room", "Try *room*, *bedroom*, *bathroom*, *door*, *window*; then *curved line* (a door swing is an arc on the "
                      "paper) and, on the clinic sheet, *toilet* and *sink*. Compare with *room* for the areas, *water closet* "
                      "or *lavatory* for the counts."),
    "structural": ("footing", "Try *footing*, *foundation*, *column*; then *square* and *hatched square* for the footings, *rectangle* "
                              "for the pile caps, *circle* for the grid bubbles. Compare with *footing* (hits and areas), *pit*, "
                              "*pile footing* or *grid bubble*, and move the confidence: the extras go, then the real ones."),
    "mep": ("light fixture", "Try *light fixture*, *light*, *diffuser*, *sprinkler*; then *rectangle with a diagonal line*, "
                             "*small circle*, *circle with a cross*. Compare with the three fixture types of the answer key."),
}


if __name__ == "__main__":
    args = [a for a in sys.argv[1:]]
    if "--root" in args:
        i = args.index("--root")
        ROOT = Path(args[i + 1])
        del args[i:i + 2]
    for variant in (args or ["workshop", "homework"]):
        {"workshop": build_workshop, "homework": build_homework}[variant]()
