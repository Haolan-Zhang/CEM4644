"""Generates the MP6 notebooks (+ report templates): MP6A (tables) and MP6B (time series), each a workshop and a homework.

    python build/make_notebooks.py [tabular|series] [workshop|homework] [--fresh-text]

The notebooks are where the wording lives: a rebuild keeps every markdown cell and every cell's #@markdown notes it
finds in the existing notebook (matched by the cell's first line / title) and only regenerates the code. Pass
--fresh-text to start again from the defaults in this file.
"""
import json
import re
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
from aec_tab.chat import CHAT_URL, GIVE, TEST_N, TOOL_MODELS  # noqa: E402
from aec_tab.ui import REGRESSION_TEXT  # noqa: E402

GITHUB_URL = "https://github.com/Haolan-Zhang/CEM4644.git"
KEEP_TEXT = True
EXISTING = {}
EXISTING_TEXTS = {}          # cell title -> {name: text} for the result wordings written in a cell (name_text = """...""")

VARIANTS = {
    ("tabular", "workshop"): dict(file="MP6A_Workshop_Tabular.ipynb", label="Workshop (in class)", minutes=75, guess=True),
    ("tabular", "homework"): dict(file="MP6A_Homework_Tabular.ipynb", label="Homework (individual)", minutes=75, guess=False),
    ("series", "workshop"): dict(file="MP6B_Workshop_TimeSeries.ipynb", label="Workshop (in class)", minutes=75),
    ("series", "homework"): dict(file="MP6B_Homework_TimeSeries.ipynb", label="Homework (individual)", minutes=75),
}
TAP_TEST = """
### 🔨 The tap test: data you collect yourself

Knock on a wall and you can hear whether a stud is behind it; inspectors do the same on concrete and tile to find hollow, delaminated spots (a *sounding test*). Here you record taps with your phone, turn each tap into one row of a table, and train a model to tell the surfaces apart.

1. **Plan.** Pick 4–5 surfaces (for example drywall between studs, drywall over a stud, concrete or masonry, a wooden door, glass or tile) and 3–4 separate spots of each. To be sure a spot is over a stud, use the magnetometer in the free *phyphox* app: it jumps at the drywall screws.
2. **Record.** At each spot, tap 10 times, about a second apart, always with the same object (a coin or a pen cap). Hold the phone about 20 cm away in a quiet room and use its voice recorder: one recording per spot, named by what it is (`hollow_bedroom_1.m4a`). About 200 taps in all.
3. **Make the table.** Upload the recordings to the notebook. It finds each tap and measures its pitch, brightness, ring time, loudness and low / mid / high energy: one row per tap, with the spot and the surface. Download the table: it is your dataset.
4. **Look first.** Write down what you expect (does hollow sound lower? ring longer?), then check it against each surface's average spectrum and a scatter plot of two measurements.
5. **Train and test twice:** once with the taps split at random, once with whole spots held out. Why do the two scores differ, and which would you believe?
6. **Blind test.** Tap a spot you did not record, let the model name the surface, and check.
7. **Ask the chat.** Give hokie.ai your table, ask it to do the same with its analysis tool, and compare.

Report which surfaces get confused, which measurements matter, and whether the model would still work in another room, with another phone or with another person tapping.
"""
ACTIVITY = """
### 🚶 Activity recognition: data you collect yourself

Researchers put motion sensors on construction workers to tell, minute by minute, whether they are walking, climbing, carrying or idle, for productivity and safety studies. Your phone has the same sensor. Here you record your own movements, cut each recording into short windows, and train a model to name the activity in each window.

1. **Plan.** Pick 4–5 activities you can repeat (for example standing, walking, going up stairs, going down stairs, carrying a heavy bag or box).
2. **Record.** In the free *phyphox* app, choose *Acceleration (without g)* and put the phone in the same pocket every time. Do each activity for about 2 minutes, in 3 separate sessions (different days, places or shoes): one recording per activity and session, exported as CSV and named by what it is (`stairs_up_session2.csv`).
3. **Make the table.** Upload the recordings to the notebook. It cuts each into 2-second windows and measures each window's average, spread, peak and rhythm (steps per second): one row per window, with the session and the activity. Download the table: it is your dataset.
4. **Look first.** Write down what you expect (is climbing slower? is carrying smoother?), then check it against a few seconds of each activity and a scatter plot of two measurements.
5. **Train and test twice:** once with the windows split at random, once with whole sessions held out. Why do the two scores differ, and which would you believe?
6. **Blind test.** Record one mixed sequence (walk, stairs, stand) and write down when you switched; see whether the model finds when each activity starts.
7. **Ask the chat.** Give hokie.ai your table, ask it to do the same with its analysis tool, and compare.

Report which activities get confused, which measurements matter, and whether the model would still work for another person, with the phone in another pocket, or on a real site.
"""
PART_NAME = {"tabular": ("MP6A", "Tables", "📊"), "series": ("MP6B", "Time series", "📈")}


def load_existing(path: Path):
    EXISTING.clear(); EXISTING_TEXTS.clear()
    if not (KEEP_TEXT and path.exists()):
        return
    for c in nbformat.read(str(path), as_version=4).cells:
        first = c.source.split("\n", 1)[0].strip()
        if c.cell_type == "markdown":
            EXISTING[first] = c.source
        elif first.startswith("#@title"):
            EXISTING[first] = [l[len("#@markdown "):] for l in c.source.split("\n") if l.startswith("#@markdown ")]
            EXISTING_TEXTS[first] = dict(re.findall(r'^(\w+_text) = """(.*?)"""', c.source, re.S | re.M))


def form(title, body, notes=(), params=(), texts=None):
    """texts: {name: wording} written into the cell as name = \"\"\"...\"\"\" (visible under Show code, kept on a rebuild)."""
    head = f'#@title {title} {{ display-mode: "form" }}'
    if head in EXISTING:
        notes = EXISTING[head]
    words = ""
    if texts:
        kept = EXISTING_TEXTS.get(head, {})
        words = ("# The wording of the result: edit the text between the triple quotes. Words in {braces} are filled in by the notebook;\n"
                 "# **bold** works, and each line is shown as its own line.\n"
                 + "".join(f'{k} = """{kept.get(k, v)}"""\n' for k, v in texts.items()))
    src = head + "\n" + "".join(f"#@markdown {n}\n" for n in notes) + "".join(p + "\n" for p in params) + words + body.rstrip() + "\n"
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


def step0(variant, part):
    body = f"""import importlib, os, shutil, subprocess, sys
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
"""
    if part == "tabular":
        return form("▶ Step 0 · Run me first (1 minute)", body + f'lab.setup(dataset="{variant}", part="tabular")',
                    notes=["Click ▶ and wait for the ✅ line."])
    return form("▶ Step 0 · Run me first (1–2 minutes)", body + f'lab.setup(dataset="{variant}", part="series", load_forecaster=load_forecaster)',
                notes=["Click ▶ and wait for the ✅ line. Untick *load_forecaster* to skip the pretrained forecasting model (Step 2a then has two methods instead of three)."],
                params=['load_forecaster = True #@param {type:"boolean"}'])


def chat_md(what):
    return (f"**hokie.ai** ({CHAT_URL}, Virginia Tech's free access to GPT models, sign in with your VT account) {what} "
            "You paste its reply back into the notebook, which scores it against what really happened, next to the notebook's own models. "
            "First the chat on its own, then the chat told to use its **data-analysis tool** (it writes and runs code on the files).")


def build_tabular(variant, v, spec, cells):
    t = spec.table
    cells.append(md(f"""
# 📊 CEM4644 · MP6A — Tables
## {v['label']}: *{t.title}*

**No coding needed.** Each grey box is one step: click ▶, wait, read the result, answer the report question. Run from top to bottom.

Photos and drawings were the last four labs. Most construction data is neither: it is a **table** (one row per mix, per building, per bid) or a **time series** (one value per hour, per day; that is MP6B). This part does the table: predict a number, predict a class, see what the model learned, then give the same job to a chat model. Every answer is checked against what really happened. About {v['minutes']} minutes. No GPU needed.
"""))
    cells.append(step0(variant, "tabular"))

    cells.append(md(f"""
## Part 1 · A table

{t.description} Every row is one {t.row_word}; the last column, **{t.label(t.target)}**, is the answer the model will learn to predict from the others.
"""))
    cells.append(form("▶ Step 1a · Look at the table", "lab.show_table(rows)", params=['rows = 10 #@param [5, 10, 20] {type:"raw"}']))
    if v["guess"]:
        hints = ([f"Five {t.rows} without their answer: which grade of {t.target_label} does each one reach? Two rules of thumb help:",
                  "1. Look at the **water / cement** column: the water divided by the cement, already worked out for you.",
                  "2. Below 0.5 is probably **high**, 0.5 to 1.0 **normal**, above 1.0 **low**.",
                  "3. Check the **age**: a sample 7 days old or younger has not reached its strength yet, so drop it one grade.",
                  f"Pick a grade for each {t.row_word}, click the button, and see. Step 2a shows what the model makes of the same {t.rows}."]
                 if t.guess_ratio else
                 [f"Five {t.rows} without their answer: which grade of {t.target_label} does each one reach? Pick, click the button, and see. "
                  f"Step 2a shows what the model makes of the same {t.rows}."])
        cells.append(form("▶ Step 1b · Guess it yourself", "lab.guess()", notes=hints))

    cells.append(md(f"""
## Part 2 · Two questions, one table

The same table can answer **which class?** (a category: *classification*, the question you just answered yourself) or **how much?** (a number: *regression*). Both models train on 80 % of the rows and are scored on the 20 % they never saw.
"""))
    cells.append(form("▶ Step 2a · Classification: predict the class", "lab.classification(model, task, threshold)",
                      notes=[f"*grades* puts each {t.row_word} in one of {len(t.grades)} bands of {t.target_label}, the game of Step 1b; *pass / fail* asks whether it reaches the *threshold* you set."],
                      params=[choice("model", list(KINDS)[1], list(KINDS)), choice("task", "grades", ["grades", "pass / fail against a specification"]),
                              f'threshold = {t.spec_default:g} #@param {{type:"slider", min:{t.spec_range[0]:g}, max:{t.spec_range[1]:g}, step:{t.spec_range[2]:g}}}']))
    cells.append(form("▶ Step 2b · Regression: predict the number", "lab.regression(model, result_text)",
                      params=[choice("model", list(KINDS)[1], list(KINDS))],
                      texts={"result_text": REGRESSION_TEXT.replace("{rows}", t.rows)}))
    cells.append(q(1, (f"From Step 2a: how many {t.rows} the model puts in the right grade against your own score in Step 1b, and at your pass / fail threshold how many false passes and false fails there are. "
                       f"From Step 2b: the average miss of the straight line and of the trees, in {t.unit}, and what the worst misses have in common. "
                       f"What is the difference between predicting 'pass' and predicting 33 {t.unit}, and which of the two mistakes costs more on a real project?")
                   if variant == "workshop" else
                   (f"From Step 2a and 2b: the average miss and the share of {t.rows} in the right band. These numbers are far better than the concrete table's in the workshop. "
                    "What is different about this table (read the description in Part 1), and why does that make it easier for a model? Would you trust a model trained on it for a real building?")))

    cells.append(md("""
## Part 3 · What the model learned

A model that scores well may still have learned the wrong thing. Two checks: which columns it leans on, and how its prediction moves when you change one input at a time.
"""))
    cells.append(form("▶ Step 3a · Which columns matter", "lab.importance()"))
    cells.append(form("▶ Step 3b · What if…", "lab.whatif(start_from)",
                      notes=["Move a slider; the prediction updates. Everything not on a slider stays as it is in the chosen row."],
                      params=[choice("start_from", "a typical row", ["a typical row", "row 12", "row 100", "row 500"])]))
    cells.append(q(2, ("From Step 3: the three columns that matter most. Does the model agree with what you know about concrete (more water, longer curing, more cement)? "
                       "Push one slider to the edge of its range: where does the prediction stop making sense, and why can a model not know that?")
                   if variant == "workshop" else
                   ("From Step 3: which two columns decide the heating load, and in which direction? Set the sliders to a building you would design yourself and report its predicted load. "
                    "Does the model tell you anything a building-energy simulator would not?")))

    cells.append(md(f"""
## Part 4 · The same job, by a chat model

{chat_md(f"gets the same training {t.rows} the models above learned from, and {TEST_N} of the held-out {t.rows} without their {t.target_label}.")} The {TEST_N} {t.rows} are the same for everyone, so you can compare with your neighbours.
"""))
    cells.append(form("▶ Step 4a · Ask the chat", "lab.chat_table(give)",
                      notes=["Run the same prompt in **two** new chats and score both replies: the table then compares them. "
                             "If attaching files does not work, choose *paste the data into the prompt* and run the cell again."],
                      params=[choice("give", list(GIVE)[0], list(GIVE))]))
    cells.append(form("▶ Step 4b · Ask the chat to use its analysis tool", "lab.chat_table_tool(model)",
                      notes=["Pick the model the chat should train, run the cell, and follow the steps. If the reply shows no code or analysis panel, "
                             "ask it again to *use your data-analysis tool*. Then try a second model: every reply you score stays in the table."],
                      params=[choice("model", list(TOOL_MODELS)[0], list(TOOL_MODELS))]))
    cells.append(q(3, (f"From Step 4a: the chat's average miss and right grades next to the trees', and how many numbers changed between your two new chats. "
                       "Ask the chat how it made those predictions: what does it say it did? From Step 4b: the model you asked for, its average miss, and why it lands where it does "
                       "against the notebook's trees and straight line (Step 2b). Compare the three columns the chat said mattered most with Step 3a. "
                       "When would you trust a chat's numbers on a real project, and what would you check first?")
                   if variant == "workshop" else
                   (f"From Step 4a and 4b: the chat's average miss on its own and with its analysis tool (two models), against the notebook's trees. "
                    "This table comes from a simulator and the trees nearly get it perfect (question 1): did the chat on its own come close? "
                    "What does that tell you about the difference between reasoning about a table and fitting a model to it?")))

    if variant == "homework":                  # the workshop ends with the chat; the students' own table is homework
        cells.append(md("""
## Part 5 · Your own table

A small app, opened from a link: upload any CSV, pick the column to predict, and it fits decision trees and scores them on held-out rows.
"""))
        cells.append(md(TAP_TEST))
        cells.append(form("▶ Step 5 · Your own table", "lab.upload_app()", notes=["Open the printed link in a new tab."]))
        cells.append(q(4, "The main deliverable: find or make a table of your own (a bid tabulation, a materials price list, anything with a numeric column to predict and 30+ rows). "
                          "Run it through Step 5 and report what the data is, what the app found, and what you would need to trust the numbers. "
                          "Then give the same file to the chat and ask it to use its analysis tool to do the same: does it agree with the app?"))
    return ["table"]


def build_series(variant, v, spec, cells):
    meters = load_meters(REPO, spec.series.meters)
    labels = [m.label for m in meters.values()]
    uses = sorted({m.use for m in meters.values()})
    cells.append(md(f"""
# 📈 CEM4644 · MP6B — Time series
## {v['label']}: *{spec.series.title}*

**No coding needed.** Each grey box is one step: click ▶, wait, read the result, answer the report question. Run from top to bottom.

MP6A was a table: one row per thing. This part is a **time series**: one value per hour, from the electricity meters of {len(meters)} real buildings on North American campuses, with the site's air temperature. A time series has a **rhythm** (days, weeks, seasons) that a table does not, and a model that knows the rhythm can say what comes next. You will tell the buildings apart, forecast a week three ways, find the days that do not fit, then give the same jobs to a chat model. About {v['minutes']} minutes. No GPU needed.
"""))
    cells.append(step0(variant, "series"))

    cells.append(md(f"""
## Part 1 · A time series

{spec.series.title}. Each building uses its electricity to its own rhythm: who is in it, when, and for what.
"""))
    cells.append(form("▶ Step 1a · Which building is which?", "lab.buildings()",
                      notes=[f"Four buildings, no names: a week and a year each. They are, in some order, {', '.join(uses)}."]))
    cells.append(form("▶ Step 1b · Your answer", "lab.buildings_answer(a, b, c, d)",
                      params=[choice(k, uses[0], uses) for k in "abcd"]))
    cells.append(form("▶ Step 1c · The anatomy of one building's year", "lab.anatomy(building)",
                      params=[choice("building", labels[0], labels)]))
    cells.append(q(1, "From Step 1: which buildings did you get right, and from what (the shape of the day, the weekend, the summer)? "
                      "Pick one building in Step 1c and describe its week in three sentences a facilities manager would recognise."))

    cells.append(md("""
## Part 2 · Next week

Three ways to forecast a week: copy last week; decision trees that learned from the past weeks, the calendar and the temperature; and a **pretrained forecasting model** that has seen millions of other time series and none of ours. Each is scored against what really happened.
"""))
    cells.append(form("▶ Step 2a · Forecast one week", "lab.forecast(building, method)",
                      params=[choice("building", labels[0], labels), choice("method", "all three", ["all three"] + list(METHODS))]))
    cells.append(q(2, ("From Step 2a on all four buildings: the average miss of each method (copy the tables). Which method wins where, and is 'same hour last week' ever hard to beat? "
                       "What does the shaded band of the pretrained model mean, and how would you use it when planning a site's power supply?")
                   if variant == "workshop" else
                   ("From Step 2a on all four buildings: the average miss of each method. One of these buildings forecasts far worse than the others, whichever method you use: "
                    "which one, why (look at Step 1c), and what extra information would a forecaster need?")))

    cells.append(md("""
## Part 3 · The odd days

Every building has a usual day for each weekday. A day that leaves the pattern is either explained (a holiday, a closure) or worth a phone call (a fault, a meter, something left running).
"""))
    cells.append(form("▶ Step 3a · Days that do not fit", "lab.odd_days(building, threshold)",
                      notes=["Lower the threshold and more days are flagged; raise it and only the strangest remain."],
                      params=[choice("building", labels[0], labels), 'threshold = 3.5 #@param {type:"slider", min:2, max:6, step:0.5}']))
    cells.append(q(3, "From Step 3a on two buildings: the flagged days at threshold 3.5. Which have an obvious cause (the calendar column), which do not? "
                      "For one unexplained day, say what you would check first. What threshold would you set for an automatic alert, and why?"))

    cells.append(md(f"""
## Part 4 · The same jobs, by a chat model

{chat_md("gets four weeks of one building's hourly electricity and next week's temperature, and forecasts the week the notebook forecast in Step 2a; then the building's daily totals for the year, to find the odd days of Step 3a.")}
"""))
    cells.append(form("▶ Step 4a · Ask the chat for next week", "lab.chat_forecast(building, give)",
                      notes=["The reply must list all 168 hours; if the chat stops early, ask it to continue and paste every part. "
                             "Run the same prompt in two new chats and score both. If attaching files does not work, choose *paste the data into the prompt*."],
                      params=[choice("building", labels[0], labels), choice("give", list(GIVE)[0], list(GIVE))]))
    cells.append(form("▶ Step 4b · Ask the chat to use its analysis tool", "lab.chat_forecast_tool(building, model)",
                      notes=["Pick the model the chat should train. If the reply shows no code or analysis panel, ask it again to *use your data-analysis tool*."],
                      params=[choice("building", labels[0], labels), choice("model", list(TOOL_MODELS)[0], list(TOOL_MODELS))]))
    cells.append(form("▶ Step 4c · Ask the chat for the odd days", "lab.chat_odd_days(building, give)",
                      notes=["The notebook compares the chat's days with the days its own rule flags in Step 3a (threshold 3.5) and with the public holidays."],
                      params=[choice("building", labels[0], labels), choice("give", list(GIVE)[0], list(GIVE))]))
    cells.append(q(4, ("From Step 4a and 4b on one building: the chat's average miss on its own and with its analysis tool, next to the three methods of Step 2a (copy the table). "
                       "Did it give all 168 hours, and did two new chats agree? From Step 4c: how many of the notebook's flagged days the chat found, which days it added, "
                       "and whether its reasons are believable (check one against the calendar). Which job suits the chat better, forecasting numbers or explaining odd days, and why?")
                   if variant == "workshop" else
                   ("Steps 4a to 4c on the building that forecast worst in Step 2a and on one other: the chat's average miss on its own and with its analysis tool against Step 2a's methods, "
                    "and the odd days it found. Does the chat do better than the notebook on the hard building? Does its explanation of that building's pattern help you, "
                    "and how would you check whether it is true?")))

    cells.append(md("""
## Part 5 · Your own time series

A small app, opened from a link: upload any CSV with a time column and a value column, and it forecasts the last period from the data before it.
"""))
    if variant == "homework":
        cells.append(md(ACTIVITY))
    cells.append(form("▶ Step 5 · Your own time series", "lab.upload_app()", notes=["Open the printed link in a new tab."]))
    cells.append(q(5, ("Upload one time series of your own (a utility bill history, a site's weather, daily deliveries, anything with a date and a number) and report what the app found: "
                       "the forecast, its average miss, and whether you believe it.")
                   if variant == "workshop" else
                   ("The main deliverable: find or make a time series of your own (a utility bill history, a site's weather, daily progress or deliveries). "
                    "Run it through Step 5 and report what the data is, what the app found, and what you would need to trust the forecast. "
                    "Then give the same file to the chat and ask it to use its analysis tool to forecast the same period: does it agree with the app?")))
    return ["meters", "forecaster"]


def build(part, variant):
    v = VARIANTS[(part, variant)]
    spec = C.SPECS[variant]
    load_existing(REPO / v["file"])
    cells = []
    uses = (build_tabular if part == "tabular" else build_series)(variant, v, spec, cells)

    cells.append(md("## Wrap-up"))
    cells.append(form("▶ Numbers for your report", "lab.report_summary()"))
    cr = json.loads((REPO / "data" / "credits.json").read_text())
    lines = ["### Credits"]
    if "table" in uses:
        tk = spec.table.key
        lines.append(f"- Table: {cr[tk]['title']}, {cr[tk]['author']}, {cr[tk]['license']}, {cr[tk]['source']}.")
    if "meters" in uses:
        lines.append(f"- Meters: {cr['meters']['title']}, {cr['meters']['author']}, {cr['meters']['license']}, {cr['meters']['source']}.")
        lines.append(f"- Pretrained forecaster: {cr['forecaster']['title']} ({cr['forecaster']['author']}, {cr['forecaster']['license']}).")
    lines.append("- Chat model: the GPT models behind hokie.ai (Virginia Tech).")
    lines.append("- Lab code: https://github.com/Haolan-Zhang/CEM4644 (folder `mp6_tabular_timeseries`).")
    cells.append(md("\n".join(lines) + "\n"))

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
    code, name, _ = PART_NAME[part]
    data_title = spec.table.title if part == "tabular" else spec.series.title
    lines = [f"# CEM4644 · {code} ({name}) report — {v['label']}", "", "Name: ______________________    Date: ____________", "",
             f"Notebook: `{v['file']}` — data: *{data_title}*.", "",
             "Answer every question in a few sentences. Paste screenshots where the question asks for pictures or tables. "
             "Numbers must come from **your** run of the notebook.", ""]
    for num, text in questions:
        lines += [f"## Question {num}", "", text, "", "*Your answer:*", "", "", ""]
    p = REPO / "docs" / f"{code}_{variant.capitalize()}_Report_Template.md"
    p.parent.mkdir(exist_ok=True); p.write_text("\n".join(lines)); print("wrote", p)

if __name__ == "__main__":
    KEEP_TEXT = "--fresh-text" not in sys.argv
    want = [a for a in sys.argv[1:] if not a.startswith("--")]
    for part, variant in VARIANTS:
        if not want or part in want or variant in want:
            build(part, variant)
