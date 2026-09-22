"""Generates the MP6 Workshop and Homework notebooks (+ report templates).

    python build/make_notebooks.py [workshop] [homework] [--fresh-text]

The notebooks are where the wording lives: a rebuild keeps every markdown cell and every cell's #@markdown notes it
finds in the existing notebook (matched by the cell's first line / title) and only regenerates the code. Pass
--fresh-text to start again from the defaults in this file.
"""
import json
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from aec_tab import config as C  # noqa: E402
from aec_tab.data import load_meters  # noqa: E402
from aec_tab.models import KINDS  # noqa: E402
from aec_tab.series import METHODS  # noqa: E402

GITHUB_URL = "https://github.com/Haolan-Zhang/CEM4644.git"
KEEP_TEXT = True
EXISTING = {}

VARIANTS = {
    "workshop": dict(file="MP6_Workshop_Tabular_TimeSeries.ipynb", label="Workshop (in class)", minutes=90, guess=True),
    "homework": dict(file="MP6_Homework_Tabular_TimeSeries.ipynb", label="Homework (individual)", minutes=90, guess=False),
}


def load_existing(path: Path):
    EXISTING.clear()
    if not (KEEP_TEXT and path.exists()):
        return
    for c in nbformat.read(str(path), as_version=4).cells:
        first = c.source.split("\n", 1)[0].strip()
        if c.cell_type == "markdown":
            EXISTING[first] = c.source
        elif first.startswith("#@title"):
            EXISTING[first] = [l[len("#@markdown "):] for l in c.source.split("\n") if l.startswith("#@markdown ")]


def form(title, body, notes=(), params=()):
    head = f'#@title {title} {{ display-mode: "form" }}'
    if head in EXISTING:
        notes = EXISTING[head]
    src = head + "\n" + "".join(f"#@markdown {n}\n" for n in notes) + "".join(p + "\n" for p in params) + body.rstrip() + "\n"
    c = new_code_cell(src); c.metadata["cellView"] = "form"
    return c


def md(text):
    text = text.strip("\n")
    return new_markdown_cell(EXISTING.get(text.split("\n", 1)[0].strip(), text))


def q(n, text):
    return md(f"> ### 📝 Report question {n}\n> {text}")


def jlist(items):
    return json.dumps(list(items), ensure_ascii=False)


def choice(name, value, options):
    return f'{name} = {json.dumps(value, ensure_ascii=False)} #@param {jlist(options)}'


def build(variant):
    v = VARIANTS[variant]
    spec = C.SPECS[variant]; t = spec.table
    meters = load_meters(REPO, spec.series.meters)
    labels = [m.label for m in meters.values()]
    uses = sorted({m.use for m in meters.values()})
    load_existing(REPO / v["file"])
    cells = []

    cells.append(md(f"""
# 📊 CEM4644 · MP6 — Tables and time series
## {v['label']}: *{spec.title}*

**No coding needed.** Each grey box is one step: click ▶, wait, read the result, answer the report question. Run from top to bottom.

Photos and drawings were the last four labs. Most construction data is neither: it is a **table** (one row per mix, per building, per bid) or a **time series** (one value per hour, per day). This lab does the same things with them: predict a number, predict a class, forecast what comes next, and check every answer against what really happened. About {v['minutes']} minutes. No GPU needed.
"""))
    cells.append(form(
        "▶ Step 0 · Run me first (1–2 minutes)",
        f"""import importlib, os, shutil, subprocess, sys
REPO, FOLDER, PKG = "CEM4644", "mp6_tabular_timeseries", "aec_tab"
FOLDERS = ["mp6_tabular_timeseries"]          # only this lab folder is downloaded, not the whole course repository

def _git(*args):
    return subprocess.run(["git", "-C", REPO, *args], capture_output=True, text=True).returncode == 0

if os.path.isdir(REPO):                      # a copy is already here: pull the newest course code over it
    if not (_git("sparse-checkout", "set", *FOLDERS)
            and _git("fetch", "-q", "--depth", "1", "origin", "master")
            and _git("reset", "-q", "--hard", "FETCH_HEAD") and _git("clean", "-qfd")):
        shutil.rmtree(REPO, ignore_errors=True)          # broken copy: start again from scratch
if not os.path.isdir(REPO):
    subprocess.run(["git", "clone", "-q", "--depth", "1", "--filter=blob:none", "--sparse", "{GITHUB_URL}", REPO], check=True)
    subprocess.run(["git", "-C", REPO, "sparse-checkout", "set", *FOLDERS], check=True)
for _m in [m for m in list(sys.modules) if m == PKG or m.startswith(PKG + ".")]:
    del sys.modules[_m]                      # Python caches imported code: drop it, or this cell keeps the old version
importlib.invalidate_caches()
sys.path.insert(0, os.path.abspath(os.path.join(REPO, FOLDER)))
from aec_tab import lab
lab.setup(dataset="{variant}", load_forecaster=load_forecaster)""",
        notes=["Click ▶ and wait for the ✅ line. Untick *load_forecaster* to skip the pretrained forecasting model (Step 5 then has two methods instead of three)."],
        params=['load_forecaster = True #@param {type:"boolean"}'],
    ))

    # ------------------------------------------------------------------ Part 1
    cells.append(md(f"""
## Part 1 · A table

{t.description} Every row is one {t.row_word}; the last column, **{t.label(t.target)}**, is the answer the model will learn to predict from the others.
"""))
    cells.append(form("▶ Step 1a · Look at the table", "lab.show_table(rows)", params=['rows = 10 #@param [5, 10, 20] {type:"raw"}']))
    if v["guess"]:
        cells.append(form("▶ Step 1b · Guess it yourself", "lab.guess()",
                          notes=[f"Five {t.rows} without their answer. Type your guess for each and click the button; Step 2a shows what the model makes of the same {t.rows}."]))

    # ------------------------------------------------------------------ Part 2
    cells.append(md(f"""
## Part 2 · Two questions, one table

The same table can answer **how much?** (a number: *regression*) or **which class?** (a category: *classification*). Both models train on 80 % of the rows and are scored on the 20 % they never saw.
"""))
    cells.append(form("▶ Step 2a · Regression: predict the number", "lab.regression(model)",
                      params=[choice("model", list(KINDS)[1], list(KINDS))]))
    cells.append(form("▶ Step 2b · Classification: predict the class", "lab.classification(model, task, threshold)",
                      notes=[f"*grades* puts each {t.row_word} in one of {len(t.grades)} bands of {t.target_label}; *pass / fail* asks whether it reaches the *threshold* you set."],
                      params=[choice("model", list(KINDS)[1], list(KINDS)), choice("task", "grades", ["grades", "pass / fail against a specification"]),
                              f'threshold = {t.spec_default:g} #@param {{type:"slider", min:{t.spec_range[0]:g}, max:{t.spec_range[1]:g}, step:{t.spec_range[2]:g}}}']))
    n = 1
    cells.append(q(n, (f"From Step 2a: the average miss of the straight line and of the trees, in {t.unit}, and what the worst misses have in common. "
                       f"From Step 2b: how many {t.rows} land in the right grade, and at your pass / fail threshold how many false passes and false fails there are. "
                       f"What is the difference between predicting 33 {t.unit} and predicting 'pass', and which of the two mistakes costs more on a real project?")
                   if variant == "workshop" else
                   (f"From Step 2a and 2b: the average miss and the share of {t.rows} in the right band. These numbers are far better than the concrete table's in the workshop. "
                    "What is different about this table (read the description in Part 1), and why does that make it easier for a model? Would you trust a model trained on it for a real building?")))

    # ------------------------------------------------------------------ Part 3
    cells.append(md(f"""
## Part 3 · What the model learned

A model that scores well may still have learned the wrong thing. Two checks: which columns it leans on, and how its prediction moves when you change one input at a time.
"""))
    cells.append(form("▶ Step 3a · Which columns matter", "lab.importance()"))
    cells.append(form("▶ Step 3b · What if…", "lab.whatif(start_from)",
                      notes=["Move a slider; the prediction updates. Everything not on a slider stays as it is in the chosen row."],
                      params=[choice("start_from", "a typical row", ["a typical row", "row 12", "row 100", "row 500"])]))
    n += 1
    cells.append(q(n, (f"From Step 3: the three columns that matter most. Does the model agree with what you know about concrete (more water, longer curing, more cement)? "
                       "Push one slider to the edge of its range: where does the prediction stop making sense, and why can a model not know that?")
                   if variant == "workshop" else
                   ("From Step 3: which two columns decide the heating load, and in which direction? Set the sliders to a building you would design yourself and report its predicted load. "
                    "Does the model tell you anything a building-energy simulator would not?")))

    # ------------------------------------------------------------------ Part 4
    cells.append(md(f"""
## Part 4 · A time series

{spec.series.title}: electricity, hour by hour, from {len(meters)} real buildings on North American campuses, with the site's air temperature. A time series has a **rhythm** (days, weeks, seasons) that a table does not, and a model that knows the rhythm can say what comes next.
"""))
    cells.append(form("▶ Step 4a · Which building is which?", "lab.buildings()",
                      notes=[f"Four buildings, no names: a week and a year each. They are, in some order, {', '.join(uses)}."]))
    cells.append(form("▶ Step 4b · Your answer", "lab.buildings_answer(a, b, c, d)",
                      params=[choice(k, uses[0], uses) for k in "abcd"]))
    cells.append(form("▶ Step 4c · The anatomy of one building's year", "lab.anatomy(building)",
                      params=[choice("building", labels[0], labels)]))
    n += 1
    cells.append(q(n, "From Step 4: which buildings did you get right, and from what (the shape of the day, the weekend, the summer)? "
                      "Pick one building in Step 4c and describe its week in three sentences a facilities manager would recognise."))

    # ------------------------------------------------------------------ Part 5
    cells.append(md("""
## Part 5 · Next week

Three ways to forecast a week: copy last week; decision trees that learned from the past weeks, the calendar and the temperature; and a **pretrained forecasting model** that has seen millions of other time series and none of ours. Each is scored against what really happened.
"""))
    cells.append(form("▶ Step 5a · Forecast one week", "lab.forecast(building, method)",
                      params=[choice("building", labels[0], labels), choice("method", "all three", ["all three"] + list(METHODS))]))
    n += 1
    cells.append(q(n, ("From Step 5a on all four buildings: the average miss of each method (copy the tables). Which method wins where, and is 'same hour last week' ever hard to beat? "
                       "What does the shaded band of the pretrained model mean, and how would you use it when planning a site's power supply?")
                   if variant == "workshop" else
                   ("From Step 5a on all four buildings: the average miss of each method. One of these buildings forecasts far worse than the others, whichever method you use: "
                    "which one, why (look at Step 4c), and what extra information would a forecaster need?")))

    # ------------------------------------------------------------------ Part 6
    cells.append(md("""
## Part 6 · The odd days

Every building has a usual day for each weekday. A day that leaves the pattern is either explained (a holiday, a closure) or worth a phone call (a fault, a meter, something left running).
"""))
    cells.append(form("▶ Step 6a · Days that do not fit", "lab.odd_days(building, threshold)",
                      notes=["Lower the threshold and more days are flagged; raise it and only the strangest remain."],
                      params=[choice("building", labels[0], labels), 'threshold = 3.5 #@param {type:"slider", min:2, max:6, step:0.5}']))
    n += 1
    cells.append(q(n, "From Step 6a on two buildings: the flagged days at threshold 3.5. Which have an obvious cause (the calendar column), which do not? "
                      "For one unexplained day, say what you would check first. What threshold would you set for an automatic alert, and why?"))

    # ------------------------------------------------------------------ Part 7
    cells.append(md("""
## Part 7 · Your own table or time series

A small app, opened from a link: upload any CSV. A table gets decision trees and a score on held-out rows; a time series gets a forecast of its last period.
"""))
    cells.append(form("▶ Step 7 · Your own data", "lab.upload_app()", notes=["Open the printed link in a new tab."]))
    n += 1
    cells.append(q(n, ("Upload one table or one time series of your own (a cost table, a utility bill history, anything with numbers) and report what the app found: "
                       "the score, the columns that mattered or the forecast, and whether you believe it.")
                   if variant == "workshop" else
                   ("The main deliverable: find or make two datasets of your own, one table and one time series (a bid tabulation, a materials price list, a utility bill history, "
                    "weather at a site, anything with numbers). Run both through Step 7 and report: what the data is, what the app found, and what you would need to trust the numbers.")))

    cells.append(md("## Wrap-up"))
    cells.append(form("▶ Numbers for your report", "lab.report_summary()"))
    cr = json.loads((REPO / "data" / "credits.json").read_text())
    tk = t.key
    cells.append(md("### Credits\n"
                    f"- Table: {cr[tk]['title']}, {cr[tk]['author']}, {cr[tk]['license']}, {cr[tk]['source']}.\n"
                    f"- Meters: {cr['meters']['title']}, {cr['meters']['author']}, {cr['meters']['license']}, {cr['meters']['source']}.\n"
                    f"- Pretrained forecaster: {cr['forecaster']['title']} ({cr['forecaster']['author']}, {cr['forecaster']['license']}).\n"
                    "- Lab code: https://github.com/Haolan-Zhang/CEM4644 (folder `mp6_tabular_timeseries`).\n"))

    questions = []
    for c in cells:
        if c.cell_type == "markdown" and c.source.startswith("> ### 📝 Report question "):
            head, _, body = c.source.partition("\n")
            questions.append((int(head.rsplit(" ", 1)[1]), body[2:].strip() if body.startswith("> ") else body.strip()))
    nb = new_notebook(cells=cells)
    nb.metadata.update({"colab": {"provenance": [], "toc_visible": True},
                        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}})
    out = REPO / v["file"]
    nbformat.write(nb, str(out)); print("wrote", out)
    lines = [f"# CEM4644 · MP6 report — {v['label']}", "", "Name: ______________________    Date: ____________", "",
             f"Notebook: `{v['file']}` — data: *{spec.title}*.", "",
             "Answer every question in a few sentences. Paste screenshots where the question asks for pictures or tables. "
             "Numbers must come from **your** run of the notebook.", ""]
    for num, text in questions:
        lines += [f"## Question {num}", "", text, "", "*Your answer:*", "", "", ""]
    p = REPO / "docs" / f"MP6_{variant.capitalize()}_Report_Template.md"
    p.parent.mkdir(exist_ok=True); p.write_text("\n".join(lines)); print("wrote", p)


if __name__ == "__main__":
    KEEP_TEXT = "--fresh-text" not in sys.argv
    for variant in ([a for a in sys.argv[1:] if not a.startswith("--")] or VARIANTS):
        build(variant)
