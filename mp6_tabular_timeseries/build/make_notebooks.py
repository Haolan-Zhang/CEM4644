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
from aec_tab.chat import (CHAT_URL, COMPARE_TEXT, FORECAST_PROMPT, FORECAST_TOOL_PROMPT, GIVE, ODD_DAYS_PROMPT, PLOT_TEXT, TEST_N,  # noqa: E402
                          TOOL_MODELS, paste_prompt_default, steps_default, tool_prompt_default)
from aec_tab.app import LABELS as APP_LABELS  # noqa: E402
from aec_tab.data import load_table  # noqa: E402
from aec_tab.tap_app import LABELS as TAP_LABELS  # noqa: E402
from aec_tab.texts import TEXTS, block as texts_block  # noqa: E402
from aec_tab.ui import ANATOMY_TEXT, REGRESSION_TEXT  # noqa: E402

GITHUB_URL = "https://github.com/Haolan-Zhang/CEM4644.git"
KEEP_TEXT = True
EXISTING = {}
EXISTING_TEXTS = {}          # cell title -> {name: text} for the result wordings written in a cell (name_text = """...""")
EXISTING_PARAM_NOTES = {}    # cell title -> {param name: [#@markdown lines written just above that form field]}

VARIANTS = {
    ("tabular", "workshop"): dict(file="MP6A_Workshop_Tabular.ipynb", label="Workshop (in class)", minutes=75, guess=True),
    ("tabular", "homework"): dict(file="MP6A_Homework_Tabular.ipynb", label="Homework (individual)", minutes=75, guess=False),
    ("series", "workshop"): dict(file="MP6B_Workshop_TimeSeries.ipynb", label="Workshop (in class)", minutes=75),
    ("series", "homework"): dict(file="MP6B_Homework_TimeSeries.ipynb", label="Homework (individual)", minutes=75),
}
TAP_TEST = """
## Part 4 · Collect Your Own Data: Tap Test
Knocking on a surface and listening is a common inspection technique (a *sounding test*): hollow or delaminated areas in concrete, tile, or drywall sound different from solid ones. In this part, you will record taps on different materials with your phone, convert each tap into a row of features, and train an AI model to classify the material.

**Data collection**
1. **Materials.** Choose 4–5 materials or surfaces (for example drywall between studs, drywall over a stud, concrete or masonry, a wooden door, a metal frame, glass, or tile) and 3–4 separate spots on each.
2. **Recording.** At each spot, tap 10 times, about half a second apart, always with the same object (a coin or a pen cap). Hold the phone about 8 inches away in a quiet room and record with the phone's voice recorder (*Voice Memos* on iPhone; *Recorder* or *Voice Recorder* on Android). Make one recording per spot.
3. **File names.** Name each recording by its material and spot number, for example `wooden table 1`, `wooden table 2`, `metal stand 1`. The app takes the material label from the file name, so all spots of one material must use the same words.

**Features extracted from each tap**
- *pitch (Hz)*: the frequency with the highest power
- *brightness (Hz)*: the spectral centroid (power-weighted mean frequency)
- *ring time (ms)*: the time for the tap to decay to 10 % of its peak amplitude
- *loudness (dB)*: the peak level above the room's background noise
- *low / middle / high share*: the fraction of power below 300 Hz, between 300 and 800 Hz, and above 800 Hz
"""
ACTIVITY = """
## Part 5 · Collect Your Own Data: Activity Recognition
Wearable motion sensors are used in construction research to recognize worker activities (walking, climbing, carrying, idling) for productivity and safety studies. Your phone has the same sensor, an accelerometer. In this part, you will record your own activities as time series and use AI to analyze them.

**Data collection**
1. **Activities.** Choose 4–5 activities you can repeat (for example standing, walking, going up stairs, going down stairs, and carrying a heavy bag or box).
2. **Recording.** In the free *phyphox* app, choose *Acceleration (without g)* and keep the phone in the same pocket every time. Record each activity for about 2 minutes in 3 separate sessions (different days, places, or shoes), and export each recording as a CSV file named by activity and session (for example `stairs_up_session2.csv`).

**Analysis**
1. **Forecasting (Step 5).** Upload one recording to the app in Step 5 to forecast its last part from the data before it.
2. **Classification (HokieAI).** Give HokieAI your recordings and ask it to use its data-analysis tool to cut each recording into 2-second windows, compute features for each window (for example mean, standard deviation, peak, and dominant frequency), and train a classifier for the activity. Ask it to evaluate the classifier twice: with a random split of the windows, and with whole sessions held out.
"""
PART_NAME = {"tabular": ("MP6A", "Tables", "📊"), "series": ("MP6B", "Time series", "📈")}


def load_existing(path: Path):
    EXISTING.clear(); EXISTING_TEXTS.clear(); EXISTING_PARAM_NOTES.clear()
    if not (KEEP_TEXT and path.exists()):
        return
    for c in nbformat.read(str(path), as_version=4).cells:
        first = c.source.split("\n", 1)[0].strip()
        if c.cell_type == "markdown":
            EXISTING[first] = c.source
        elif first.startswith("#@title"):
            notes, above, pending, seen_param = [], {}, [], False
            for l in c.source.split("\n"):
                if l.startswith("#@markdown "):
                    (pending if seen_param else notes).append(l[len("#@markdown "):])     # above a later field / the cell's notes
                elif "#@param" in l:
                    if pending:
                        above[l.split("=")[0].strip()] = pending
                    pending, seen_param = [], True
            EXISTING[first] = notes
            EXISTING_PARAM_NOTES[first] = above
            EXISTING_TEXTS[first] = dict(re.findall(r'^(\w+_text) = """(.*?)"""', c.source, re.S | re.M))


def form(title, body, notes=(), params=(), texts=None, param_notes=None):
    """texts: {name: wording} written into the cell as name = \"\"\"...\"\"\" (visible under Show code, kept on a rebuild).
    param_notes: {param name: note} shown just above that form field (kept on a rebuild like the other notes)."""
    head = f'#@title {title} {{ display-mode: "form" }}'
    above = {k: [v] for k, v in (param_notes or {}).items()}
    if head in EXISTING:
        notes = EXISTING[head]
        above = EXISTING_PARAM_NOTES.get(head, {})
    params = ["".join(f"#@markdown {n}\n" for n in above.get(p.split("=")[0].strip(), [])) + p for p in params]
    words = ""
    if texts:
        kept = EXISTING_TEXTS.get(head, {})
        comment = (("# The app's titles, labels, and result lines: edit the wording after each colon (keep the name before it).\n"
                    "# Words in {braces} are filled in by the app; \\n starts a new line.\n") if list(texts) == ["labels_text"] else
                   ("# The wording the notebook shows: edit the text between the triple quotes. Words in {braces} are filled in by the notebook;\n"
                    "# **bold** and *italic* work, and each line is shown as its own line.\n"))
        words = (comment
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

**No coding needed.** Each gray box is one step: click ▶, wait, read the result, answer the report question. Run from top to bottom.

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
                              f'threshold = {t.spec_default:g} #@param {{type:"slider", min:{t.spec_range[0]:g}, max:{t.spec_range[1]:g}, step:{t.spec_range[2]:g}}}'],
                      param_notes={"threshold": "*threshold* (only for *pass / fail against a specification*):"}))
    cells.append(form("▶ Step 2b · Regression: predict the number", "lab.regression(model, result_text)",
                      params=[choice("model", list(KINDS)[1], list(KINDS))],
                      texts={"result_text": REGRESSION_TEXT.replace("{rows}", t.rows)}))
    cells.append(q(1, (f"From Step 2a: how many {t.rows} the model puts in the right grade against your own score in Step 1b, and at your pass / fail threshold how many false passes and false fails there are. "
                       f"From Step 2b: the average error of the straight line and of the decision tree, in {t.unit}, and what the largest errors have in common. "
                       f"What is the difference between predicting 'pass' and predicting {t.fmt(t.spec_default * 1.2)} {t.unit}, and which of the two mistakes costs more on a real project?")
                   if variant == "workshop" else
                   (f"From Step 2a and 2b: the average error and the share of {t.rows} in the right band. These numbers are far better than the concrete table's in the workshop. "
                    "What is different about this table (read the description in Part 1), and why does that make it easier for a model? Would you trust a model trained on it for a real building?")))

    cells.append(md("""
## Part 3 · What the model learned

A model that scores well may still have learned the wrong thing. Two checks: which columns it leans on, and how its prediction moves when you change one input at a time.
"""))
    cells.append(form("▶ Step 3a · Which columns matter", "lab.importance()"))
    cells.append(form("▶ Step 3b · What if…", "lab.whatif(sample)",
                      notes=[f"- The decision tree of Step 2b predicts the {t.target_label} of one {t.row_word}.",
                             f"- *sample*: the {t.row_word} the sliders start from: *a typical {t.row_word}* (the middle value of every column) or one real {t.row_word} from the table.",
                             "- Each slider changes one input; everything else stays fixed, and the prediction updates.",
                             "- Check that the model behaves as you expect, and where it stops making sense."],
                      params=[choice("sample", f"a typical {t.row_word}", [f"a typical {t.row_word}"] + [f"{t.row_word} {i}" for i in (12, 100, 500)])]))
    cells.append(q(2, ("From Step 3: the three columns that matter most. Does the model agree with what you know about concrete (more water, longer curing, more cement)? "
                       "Push one slider to the edge of its range: where does the prediction stop making sense, and why can a model not know that?")
                   if variant == "workshop" else
                   ("From Step 3: which two columns decide the heating load, and in which direction? Set the sliders to a building you would design yourself and report its predicted load. "
                    "Does the model tell you anything a building-energy simulator would not?")))

    cells.append(md(f"""
## Part 4 · The same job, by a chat model

{chat_md(f"gets the same training {t.rows} the models above learned from, and {TEST_N} of the held-out {t.rows} without their {t.target_label}.")} The {TEST_N} {t.rows} are the same for everyone, so you can compare with your neighbors.
"""))
    cells.append(form("▶ Step 4a · Ask the chat", "lab.chat_table(give, steps_text, paste_steps_text, paste_prompt_text, compare_text, plot_text)",
                      texts={"steps_text": steps_default(2, True), "paste_steps_text": steps_default(2, False),
                             "paste_prompt_text": paste_prompt_default(t), "compare_text": COMPARE_TEXT, "plot_text": PLOT_TEXT},
                      notes=["Run the same prompt in **two** new chats and score both replies: the table then compares them. "
                             "If attaching files does not work, choose *paste the data into the prompt* and run the cell again."],
                      params=[choice("give", list(GIVE)[0], list(GIVE))]))
    cells.append(form("▶ Step 4b · Ask the chat to use its analysis tool", "lab.chat_table_tool(model, steps_text, compare_text, plot_text)",
                      texts={"steps_text": steps_default(2, True), "compare_text": COMPARE_TEXT, "plot_text": PLOT_TEXT},
                      notes=["Pick the model the chat should train, run the cell, and follow the steps. If the reply shows no code or analysis panel, "
                             "ask it again to *use your data-analysis tool*. Then try a second model: every reply you score stays in the table."],
                      params=[choice("model", list(TOOL_MODELS)[0], list(TOOL_MODELS))]))
    cells.append(q(3, (f"From Step 4a: HokieAI's average error and how many {t.rows} it predicts within {t.close_pct:g} %, next to the decision tree's, and how many numbers changed between your two new chats. "
                       "Ask the chat how it made those predictions: what does it say it did? From Step 4b: the model you asked for, its average error, and why it lands where it does "
                       "against the notebook's decision tree and straight line (Step 2b). Compare the three columns the chat said mattered most with Step 3a. "
                       "When would you trust a chat's numbers on a real project, and what would you check first?")
                   if variant == "workshop" else
                   (f"From Step 4a and 4b: the chat's average error on its own and with its analysis tool (two models), against the notebook's decision tree. "
                    "This table comes from a simulator and the decision tree nearly gets it perfect (question 1): did the chat on its own come close? "
                    "What does that tell you about the difference between reasoning about a table and fitting a model to it?")))

    if variant == "homework":                  # the workshop ends with the chat; the students' own table is homework
        cells.append(md("""
## Part 5 · Your own table

A small app, opened from a link, turns your own tap recordings into a table and trains a model on it.
"""))
        cells.append(md(TAP_TEST))
        cells.append(form("▶ Step 5 · The tap test", "lab.tap_app()",
                          notes=["**Using the app, step by step**",
                                 "1. **Open the app.** Run this cell and open the printed link in a new tab. It works on a phone too, so you can upload straight from the phone you recorded on.",
                                 "2. **Make the table.** Drop all your recordings into *your recordings* (or click it and pick them), then click **Make the table**. The line that appears says how many taps, recordings and materials it found.",
                                 "3. **Check the taps.** In *each recording, with the taps found*, every red line should sit on one of your taps. If a recording has fewer red lines than taps you made, tap harder or closer to the phone and record that spot again.",
                                 "4. **Download the table** under *download the table (CSV)*: one row per tap. Keep it for your report and for the chat.",
                                 "5. **Look before training.** Write down what you expect (does hollow sound lower? ring longer?). Compare the curves in *the average tap of each material*, then pick two measurements for *across* and *up*: do the materials form separate clouds of dots?",
                                 "6. **Train and test on random taps.** Pick a *model*, choose *random taps* under *test on*, click **Train and test**, and note the score.",
                                 "7. **Train and test with whole recordings held out.** Keep the same model, choose *whole recordings held out*, and click again: now each spot is tested by a model that never heard it. Compare the two scores.",
                                 "8. **Read the rules.** Choose the *small decision tree* model and read the rules it learned: which measurement does it ask about first?",
                                 "9. **Blind test.** Record one spot you did not use, upload it under *one new recording*, click **What is it?**, and check its answer.",
                                 "10. **Ask the chat.** Give hokie.ai the table you downloaded, ask it to do the same with its analysis tool, and compare its answer with the app's."]))
        cells.append(q(4, "From the tap test (Step 5): the materials you recorded (how many spots and taps of each), the score on random taps and with whole recordings held out, "
                          "and which materials get confused. Which measurements separate your materials best (the scatter plot and the small decision tree's rules)? "
                          "Did the blind test name the material right? Would the model still work in another room, with another phone or with another person tapping?"))
    return ["table"]


def build_series(variant, v, spec, cells):
    meters = load_meters(REPO, spec.series.meters)
    labels = [m.label for m in meters.values()]
    uses = sorted({m.use for m in meters.values()})
    cells.append(md(f"""
# 📈 CEM4644 · MP6B — Time series
## {v['label']}: *{spec.series.title}*

**No coding needed.** Each gray box is one step: click ▶, wait, read the result, answer the report question. Run from top to bottom.

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
    cells.append(form("▶ Step 1c · The anatomy of one building's year", "lab.anatomy(building, anatomy_text)",
                      params=[choice("building", labels[0], labels)], texts={"anatomy_text": ANATOMY_TEXT}))
    cells.append(q(1, "From Step 1: which buildings did you get right, and from what (the shape of the day, the weekend, the summer)? "
                      "Pick one building in Step 1c and describe its week in three sentences a facilities manager would recognize."))

    cells.append(md("""
## Part 2 · Next week

Three ways to forecast a week: copy last week; decision trees that learned from the past weeks, the calendar and the temperature; and a **pretrained forecasting model** that has seen millions of other time series and none of ours. Each is scored against what really happened.
"""))
    cells.append(form("▶ Step 2a · Forecast one week", "lab.forecast(building, method)",
                      params=[choice("building", labels[0], labels), choice("method", "all three", ["all three"] + list(METHODS))]))
    cells.append(q(2, ("From Step 2a on all four buildings: the average error of each method (copy the tables). Which method wins where, and is 'same hour last week' ever hard to beat? "
                       "What does the shaded band of the pretrained model mean, and how would you use it when planning a site's power supply?")
                   if variant == "workshop" else
                   ("From Step 2a on all four buildings: the average error of each method. One of these buildings forecasts far worse than the others, whichever method you use: "
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
    cells.append(form("▶ Step 4a · Ask the chat for next week", "lab.chat_forecast(building, give, steps_text, paste_steps_text)",
                      texts={"steps_text": steps_default(2, True), "paste_steps_text": steps_default(2, False)},
                      notes=["The reply must list all 168 hours; if the chat stops early, ask it to continue and paste every part. "
                             "Run the same prompt in two new chats and score both. If attaching files does not work, choose *paste the data into the prompt*."],
                      params=[choice("building", labels[0], labels), choice("give", list(GIVE)[0], list(GIVE))]))
    cells.append(form("▶ Step 4b · Ask the chat to use its analysis tool", "lab.chat_forecast_tool(building, model, steps_text)",
                      texts={"steps_text": steps_default(2, True)},
                      notes=["Pick the model the chat should train. If the reply shows no code or analysis panel, ask it again to *use your data-analysis tool*."],
                      params=[choice("building", labels[0], labels), choice("model", list(TOOL_MODELS)[0], list(TOOL_MODELS))]))
    cells.append(form("▶ Step 4c · Ask the chat for the odd days", "lab.chat_odd_days(building, give, steps_text, paste_steps_text)",
                      texts={"steps_text": steps_default(1, True), "paste_steps_text": steps_default(1, False)},
                      notes=["The notebook compares the chat's days with the days its own rule flags in Step 3a (threshold 3.5) and with the public holidays."],
                      params=[choice("building", labels[0], labels), choice("give", list(GIVE)[0], list(GIVE))]))
    cells.append(q(4, ("From Step 4a and 4b on one building: the chat's average error on its own and with its analysis tool, next to the three methods of Step 2a (copy the table). "
                       "Did it give all 168 hours, and did two new chats agree? From Step 4c: how many of the notebook's flagged days the chat found, which days it added, "
                       "and whether its reasons are believable (check one against the calendar). Which job suits the chat better, forecasting numbers or explaining odd days, and why?")
                   if variant == "workshop" else
                   ("Steps 4a to 4c on the building that forecast worst in Step 2a and on one other: the chat's average error on its own and with its analysis tool against Step 2a's methods, "
                    "and the odd days it found. Does the chat do better than the notebook on the hard building? Does its explanation of that building's pattern help you, "
                    "and how would you check whether it is true?")))

    if variant == "homework":                  # the workshop ends with the chat; the students' own series is homework
        cells.append(md("""
## Part 5 · Your own time series

A small app, opened from a link: upload any CSV with a time column and a value column, and it forecasts the last period from the data before it.
"""))
        cells.append(md(ACTIVITY))
        cells.append(form("▶ Step 5 · Your own time series", "lab.upload_app()", notes=["Open the printed link in a new tab."]))
        cells.append(q(5, "The main deliverable: find or make a time series of your own (a utility bill history, a site's weather, daily progress or deliveries). "
                          "Run it through Step 5 and report what the data is, what the app found, and what you would need to trust the forecast. "
                          "Then give the same file to the chat and ask it to use its analysis tool to forecast the same period: does it agree with the app?"))
    return ["meters", "forecaster"]


def build(part, variant):
    v = VARIANTS[(part, variant)]
    spec = C.SPECS[variant]
    load_existing(REPO / v["file"])
    cells = []
    revised = part == "series" or variant == "homework"          # the notebooks written in the In-Class activity's style
    builder = (build_tabular_v2 if part == "tabular" else build_series_v2) if revised else build_tabular
    uses = builder(variant, v, spec, cells)
    cr = json.loads((REPO / "data" / "credits.json").read_text())
    if revised:
        lines = ["### Data and model sources"]
        if "table" in uses:
            tk = spec.table.key
            lines.append(f"- Table: {cr[tk]['title']}, {cr[tk]['author']}, {cr[tk]['license']}, {cr[tk]['source']}.")
        if "meters" in uses:
            lines.append(f"- Electricity meters and weather: {cr['meters']['title']}, {cr['meters']['author']}, {cr['meters']['license']}, {cr['meters']['source']}.")
            lines.append(f"- Pretrained forecasting model: {cr['forecaster']['title']}, {cr['forecaster']['author']}, {cr['forecaster']['license']}, {cr['forecaster']['source']}.")
        lines.append("- Generative AI: GPT models via HokieAI (Virginia Tech).")
        cells.append(md("\n".join(lines) + "\n"))
        return _write(part, variant, v, spec, cells)

    cells.append(md("## Wrap-up"))
    cells.append(form("▶ Numbers for your report", "lab.report_summary()"))
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
    return _write(part, variant, v, spec, cells)


def _write(part, variant, v, spec, cells):
    questions = []
    for c in cells:
        src = c.source
        if c.cell_type == "markdown" and ("### 📝 Report question " in src.split("\n", 1)[0]):
            head, _, body = src.partition("\n")
            paras = [p.strip()[2:].strip() if p.strip().startswith("> ") else p.strip() for p in body.split("\n\n") if p.strip()]
            questions.append((int(head.rsplit(" ", 1)[1]), "\n\n".join(paras)))
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

# ============================================================================ the revised notebooks (MP6A homework, MP6B)
def qv2(n, *parts):
    """A report question in the In-Class activity's form: a heading, then one quoted paragraph per step."""
    return md(f"### 📝 Report question {n}\n" + "\n\n".join(f"> {p}" for p in parts))


def T(group, spec=None, **extra):
    """A step's default wordings for its cell, with the dataset's words written in ({rows}, {unit})."""
    out = {}
    for k, v in TEXTS[group].items():
        if spec is not None:
            v = v.replace("{rows}", spec.rows).replace("{unit}", spec.unit)
        out[k] = v
    out.update(extra)
    return out


def call(head, texts):
    """lab.step(args, name_text=name_text, ...) for the wordings a cell carries."""
    head = head.rstrip(")")
    kw = ", ".join(f"{k}={k}" for k in texts)
    return f"{head}{', ' if head[-1] != '(' and kw else ''}{kw})"


def wform(title, head, texts, notes=(), params=()):
    return form(title, call(head, texts), notes=notes, params=params, texts=texts)


HOKIEAI = "**multimodal/generative AI (GPT/HokieAI)**"
WORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
TOOL_NOTE = ("Choose the model HokieAI should train. With its data-analysis tool, HokieAI writes and runs Python code on the attached files "
             "instead of predicting from the text alone. If its output shows no code or analysis, ask it to use its data-analysis tool.")


def build_tabular_v2(variant, v, spec, cells):
    t = spec.table
    out_name = t.label(t.target).split(" (")[0]
    levels = ", ".join(f"{g[0].split(' (')[0]} ({g[0].split('(')[1].rstrip(')')})" for g in t.grades)
    cells.append(md(f"""
# Homework: Predicting the Heating Load of Buildings
In this homework, you will use AI models to predict the heating load of buildings from tabular data, compare them with generative AI (HokieAI), and then collect your own tabular dataset to train a model.
"""))
    cells.append(step0(variant, "tabular"))
    cells.append(wform("▶ Browse the Data", "lab.show_table(rows)", T("show_table", t),
                       notes=[f"The table contains {len(load_table(REPO, t)):,} simulated residential buildings of the same volume but different shapes. Each row represents one building; "
                              "the columns describe its geometry (relative compactness; surface, wall, and roof areas; overall height), orientation, and glazing, and the last column "
                              f"is the heating load computed by a building-energy simulation. The heating load is the output the AI model will learn to predict from the other columns. "
                              "Here, you can browse examples from the table."],
                       params=['rows = 10 #@param [5, 10, 20] {type:"raw"}']))

    cells.append(md("## Part 1 · Predicting Heating Load"))
    cells.append(wform("▶ Step 1a · Classification: Predict Heating Load Level", "lab.classification(model, task, threshold)",
                       {k: v for k, v in T("classification_grades", t).items() if k != "guess_text"},
                       notes=[f"The AI model uses the input variables to classify each building into one of {WORDS[len(t.grades)]} heating-load levels ({t.unit}): {levels}.",
                              "You can explore two AI models in this and the following steps:",
                              "Straight line model: learns a simple linear relationship between the inputs and the output.",
                              "Decision tree model: learns a series of splits in the data (for example, based on relative compactness or glazing area) to make predictions. "
                              "It can capture more complex relationships than a straight line model."],
                       params=[choice("model", "decision tree", list(KINDS)), 'task = "grades"', f"threshold = {t.spec_default:g}"]))
    cells.append(wform("▶ Step 1b · Classification: Predict Pass or Fail", "lab.classification(model, task, threshold)", T("classification_passfail", t),
                       notes=[f"Instead of predicting {WORDS[len(t.grades)]} levels, the AI model predicts whether each building meets a heating-load limit: a building passes if its "
                              "heating load is at or below the threshold. This is a binary classification task. You can adjust the threshold."],
                       params=[choice("model", "decision tree", list(KINDS)), 'task = "pass / fail against a specification"',
                               f'threshold = {t.spec_default:g} #@param {{type:"slider", min:{t.spec_range[0]:g}, max:{t.spec_range[1]:g}, step:{t.spec_range[2]:g}}}']))
    cells.append(wform("▶ Step 1c · Regression: Predict Heating Load", "lab.regression(model)",
                       {k: v for k, v in T("regression", t).items() if k != "guess_text"},
                       notes=["Finally, instead of predicting a category, let the AI model predict the heating load itself (regression)."],
                       params=[choice("model", "decision tree", list(KINDS))]))
    cells.append(qv2(1,
                     "From Step 1a: Explore both AI models. How accurately does each model classify the test buildings into the correct heating-load level? "
                     "Include the confusion matrix for each model.",
                     "From Step 1b: For the better-performing model, at your selected threshold, what is the accuracy? How many false passes (false positives) "
                     "and false fails (false negatives) did it make? Which is more concerning for an energy requirement, and why?",
                     "From Step 1c: Report the MAE and R² of both models and compare them with the concrete-strength results from the in-class activity. "
                     "The heating loads come from a building-energy simulation rather than measurements: why does that make the prediction task easier?"))

    cells.append(md("""
## Part 2 · What the Model Learned
The model learned relationships between the input columns and the heating load from the training data. In this part, we will explore which inputs matter most overall and how changing input values affects a prediction.
"""))
    cells.append(wform("▶ Step 2a · Which Inputs Matter", "lab.importance()", T("importance", t),
                       notes=["Run the cell to see the importance of each input for the decision tree model. Importance is measured by permutation: how much the model's R² "
                              "drops when the values of one input are shuffled. A higher importance means that the model relied more on that input."]))
    whatif_words = ", ".join(t.label(f).split(" (")[0] for f in t.whatif[:-1]) + ", and " + t.label(t.whatif[-1]).split(" (")[0]
    cells.append(wform("▶ Step 2b · What If Inputs Change?", "lab.whatif(building_sample)", T("whatif", t),
                       notes=[f"Move the sliders to change {whatif_words}. The decision tree model updates its predicted heating load each time; all other inputs stay at "
                              "the median values of the table. If the sliders are not showing, run the cell again."],
                       params=['building_sample = "a typical building"']))
    cells.append(qv2(2,
                     "From Step 2a: Which two inputs are most important to the model? Were these the inputs you expected to matter most?",
                     "From Step 2b: Change one input at a time. Which input causes the largest change in predicted heating load, and in which direction? "
                     "Then set the sliders to a building you would design and report its predicted heating load."))

    cells.append(md(f"""
## Part 3 · The Same Prediction Task with Generative/Multimodal AI

So far, you used AI models trained specifically to predict heating load (in **Step 1c**). Now you will give the same training data and test buildings to {HOKIEAI} and compare its predictions with those models.
"""))
    ct = {k: v for k, v in T("chat_table", t).items()}
    cells.append(wform("▶ Step 3a · Predict with HokieAI", 'lab.chat_table(give)',
                       dict(paste_steps_text=steps_default(2, False), paste_prompt_text=paste_prompt_default(t), **ct),
                       notes=["Run the cell to reveal the instructions and prompt. To check whether HokieAI is consistent, paste the same prompt into a second new chat "
                              "and score that output too: the notebook then compares the two."],
                       params=['give = "paste the data into the prompt"']))
    cells.append(wform("▶ Step 3b · Predict with HokieAI's Data-Analysis Tool", "lab.chat_table_tool(model)",
                       dict(steps_text=steps_default(2, True), tool_prompt_text=tool_prompt_default(t),
                            **{k: v for k, v in ct.items() if k != "repeat_text"}),
                       notes=[TOOL_NOTE], params=[choice("model", list(TOOL_MODELS)[0], list(TOOL_MODELS))]))
    cells.append(qv2(3,
                     "From Step 3a: How does HokieAI perform compared with the two models in Part 1? Use the error table and the predicted vs. actual plot to summarize. "
                     "If you ran the prompt twice, how consistent were the two outputs?",
                     "From Step 3b: Report the MAE of HokieAI with its data-analysis tool and the model you chose. Why is it closer to (or farther from) the decision tree "
                     "model than in Step 3a?",
                     "Finally, ask HokieAI how it made its predictions in Step 3a and briefly summarize its explanation."))

    cells.append(md(TAP_TEST))
    cells.append(form("▶ Step 4 · Tap Test App", "lab.tap_app(labels_text=labels_text)",
                      notes=["1. Run the cell and open the printed link in a new tab. The app also works on a phone, so you can upload directly from the phone you recorded with.",
                             "2. **Build the dataset.** Upload all recordings and click **Build dataset**. Check the detected taps (red lines): each tap you made should have one line. "
                             "If taps are missing, tap harder or closer to the phone and record that spot again.",
                             "3. **Download the dataset** (CSV, one row per tap) for your report and for HokieAI.",
                             "4. **Explore the features.** Compare the mean spectra of the materials, then choose two features for the scatter plot. Do the materials form separate clusters?",
                             "5. **Train and evaluate** with both evaluation methods: a *random split* (80 % of the taps for training, 20 % for testing) and *leave-one-recording-out* "
                             "(each recording is tested by a model trained on all other recordings, so the test spot is never seen in training). Compare the accuracies.",
                             "6. **Read the rules.** With the *decision tree* model, read the rules it learned: which feature does it split on first?",
                             "7. **Predict a new recording.** Record a spot that is not in the dataset, upload it, and click **Predict**. Check whether the predicted material is correct.",
                             "8. **Ask HokieAI.** Give HokieAI your dataset (CSV), ask it to train a classifier with its data-analysis tool, and compare its accuracy with the app's."],
                      texts={"labels_text": texts_block(TAP_LABELS)}))
    cells.append(qv2(4,
                     "Report the materials you recorded (number of spots and taps for each) and include the confusion matrix of your best model.",
                     "Compare the accuracy from the random split with the accuracy from leave-one-recording-out. Why are they different, and which one better estimates "
                     "the accuracy on a new spot?",
                     "Which features separate your materials best (use the scatter plot and the decision tree's rules)? Was the material of your new recording predicted correctly?",
                     "Would the model still work in another room, with another phone, or with another person tapping? Briefly explain."))
    return ["table", "chat"]


def build_series_v2(variant, v, spec, cells):
    meters = load_meters(REPO, spec.series.meters)
    labels = [m.label for m in meters.values()]
    uses = sorted({m.use for m in meters.values()})
    ws = variant == "workshop"
    cells.append(md("# In-Class Activity: Forecasting Building Electricity Use\nIn this activity, we will use AI to explore, forecast, and detect anomalies in the hourly electricity use of four campus buildings."
                    if ws else
                    "# Homework: Forecasting Building Electricity Use\nIn this homework, you will repeat the in-class analysis on four other buildings and then collect your own time-series data."))
    cells.append(step0(variant, "series"))

    cells.append(md(f"""
## Part 1 · Exploring the Time Series
The data are the hourly electricity use (kWh) of {WORDS[len(meters)]} buildings on North American university campuses in 2017, with the outdoor air temperature at each site. Unlike a table, a time series is ordered in time and has patterns that repeat (daily, weekly, and seasonal cycles).
"""))
    cells.append(wform("▶ Step 1a · Identify the Building Type", "lab.buildings()", T("buildings"),
                       notes=[f"The plots show one week in March and the whole year for {WORDS[len(meters)]} unlabeled buildings: {', '.join(_article_list(uses))}. "
                              "Use the daily and weekly patterns to decide which building is which."]))
    cells.append(wform("▶ Step 1b · Check Your Answers", "lab.buildings_answer(a, b, c, d)", T("buildings_answer"),
                       params=[choice(k, uses[0], uses) for k in "abcd"]))
    cells.append(wform("▶ Step 1c · Daily and Weekly Patterns", "lab.anatomy(building)", T("anatomy"),
                       notes=["Choose a building to see its hourly use over the year, one week in March, and its typical day for each weekday "
                              "(the median of all such days in 2017)."],
                       params=[choice("building", labels[0], labels)]))
    cells.append(qv2(1,
                     "From Steps 1a and 1b: How many buildings did you identify correctly? Which patterns (daily cycle, weekends, seasons) did you use?",
                     "From Step 1c: Choose one building and describe its typical weekday, weekend, and seasonal pattern in three sentences."))

    cells.append(md("""
## Part 2 · Forecasting Next Week
A forecast predicts future values of a time series from its past. Here, three methods forecast the electricity use for every hour of the week of October 16, 2017 (168 hours), and each forecast is compared with the actual use.
"""))
    m1, m2, m3 = list(METHODS)
    cells.append(wform("▶ Step 2a · Forecast One Week", "lab.forecast(building, method)", T("forecast"),
                       notes=[f"*{m1}*: repeats the use from the same hour one week earlier. It is the baseline any model should beat.",
                              f"*{m2}*: gradient-boosted decision trees trained on all data before the test week, with the use one and two weeks earlier, the hour of day, "
                              "the day of week, and the outdoor temperature as inputs.",
                              f"*{m3}*: a pretrained time-series foundation model, used zero-shot: it was trained on many other time series and receives only this "
                              "building's recent history. The shaded area is its 80 % prediction interval."],
                       params=[choice("building", labels[0], labels), choice("method", "all three", ["all three"] + list(METHODS))]))
    cells.append(qv2(2,
                     "From Step 2a: Report the MAE of each method for all four buildings (copy the tables). Which method performs best for each building? "
                     "Is the seasonal naive baseline ever hard to beat?",
                     "What does the 80 % prediction interval of Chronos-Bolt mean, and how could a facility manager use it?")
                 if ws else
                 qv2(2,
                     "From Step 2a: Report the MAE of each method for all four buildings (copy the tables). One building is forecast much less accurately than the "
                     "others by every method: which one, and why (use Step 1c)? What additional information would improve its forecast?"))

    cells.append(md("""
## Part 3 · Detecting Anomalous Days
An anomaly is a day whose electricity use departs from the building's typical pattern. For each day, the notebook computes the average deviation from the typical use for that weekday and hour, and converts it into a robust z-score: how many robust standard deviations the day is from normal. Days beyond the threshold are flagged.
"""))
    cells.append(wform("▶ Step 3a · Flag Anomalous Days", "lab.odd_days(building, threshold)", T("odd_days"),
                       notes=["Lower the threshold to flag more days; raise it to keep only the most unusual ones. The table lists the flagged days with the US public holidays."],
                       params=[choice("building", labels[0], labels), 'threshold = 3.5 #@param {type:"slider", min:2, max:6, step:0.5}']))
    cells.append(qv2(3,
                     "From Step 3a (two buildings, threshold 3.5): Which flagged days have an obvious cause (see the holiday column), and which do not? "
                     "For one unexplained day, what would you check first?",
                     "What threshold would you choose for an automatic alert, and why?"))

    cells.append(md(f"""
## Part 4 · The Same Tasks with Generative/Multimodal AI
Now you will give the same data to {HOKIEAI}: first to forecast the test week, then to identify anomalous days. The notebook compares its outputs with the methods of Steps 2a and 3a.
"""))
    cf = T("chat_forecast")
    cells.append(wform("▶ Step 4a · Forecast with HokieAI", "lab.chat_forecast(building)",
                       dict(paste_steps_text=steps_default(2, False), paste_prompt_text=FORECAST_PROMPT, **cf),
                       notes=["Choose a building and run the cell to reveal the instructions and prompt. HokieAI must return all 168 hourly values; "
                              "if its output stops early, ask it to continue and paste all parts."],
                       params=[choice("building", labels[0], labels)]))
    cells.append(wform("▶ Step 4b · Forecast with HokieAI's Data-Analysis Tool", "lab.chat_forecast_tool(building, model)",
                       dict(steps_text=steps_default(2, True), tool_prompt_text=FORECAST_TOOL_PROMPT, **cf),
                       notes=[TOOL_NOTE], params=[choice("building", labels[0], labels), choice("model", list(TOOL_MODELS)[0], list(TOOL_MODELS))]))
    cells.append(wform("▶ Step 4c · Identify Anomalous Days with HokieAI", "lab.chat_odd_days(building)",
                       dict(paste_steps_text=steps_default(1, False), paste_prompt_text=ODD_DAYS_PROMPT, **T("chat_odd_days")),
                       notes=["The notebook compares the days HokieAI lists with the days flagged in Step 3a (threshold 3.5) and with the US public holidays."],
                       params=[choice("building", labels[0], labels)]))
    cells.append(qv2(4,
                     "From Steps 4a and 4b (one building): Report HokieAI's MAE without and with its data-analysis tool, next to the three methods of Step 2a.",
                     "From Step 4c: How many of the flagged days did HokieAI find, and which days did it add? Check one of its reasons against the data or the "
                     "calendar: is it plausible?",
                     "Which task suits HokieAI better, forecasting numbers or explaining anomalies, and why?")
                 if ws else
                 qv2(4,
                     "Repeat Steps 4a–4c for the building with the largest forecast error in Step 2a and for one other building. Does HokieAI forecast the difficult "
                     "building better than the methods of Step 2a?",
                     "Is HokieAI's explanation of that building's pattern helpful, and how would you verify it?"))

    if not ws:
        cells.append(md(ACTIVITY))
        cells.append(form("▶ Step 5 · Your Own Time Series", "lab.upload_app(labels_text=labels_text)",
                          notes=["Run the cell and open the printed link in a new tab. Upload a CSV with a time column and a value column, choose *time series*, "
                                 "and click **Run**: the app forecasts the last period from the data before it, with a seasonal naive baseline and gradient-boosted trees."],
                          texts={"labels_text": texts_block(APP_LABELS)}))
        cells.append(qv2(5,
                         "Describe the data you collected: activities, number of sessions, sampling rate, and recording length.",
                         "From Step 5: Report the forecast MAE for one recording. Does the seasonal naive baseline or the gradient-boosted trees model forecast it better?",
                         "From HokieAI: Report the features it computed and the accuracy of its classifier with a random split and with whole sessions held out. "
                         "Why are the two accuracies different, and which one better estimates the accuracy for a new session?"))
    return ["meters", "forecaster", "chat"]


def _article_list(uses):
    return [("an " if u[0] in "aeiou" else "a ") + u for u in uses[:-1]] + ["and " + ("an " if uses[-1][0] in "aeiou" else "a ") + uses[-1]]


if __name__ == "__main__":
    KEEP_TEXT = "--fresh-text" not in sys.argv
    want = [a for a in sys.argv[1:] if not a.startswith("--")]
    parts = [a for a in want if a in ("tabular", "series")] or ["tabular", "series"]
    variants = [a for a in want if a in ("workshop", "homework")] or ["workshop", "homework"]
    for part, variant in VARIANTS:
        if part in parts and variant in variants:
            build(part, variant)
