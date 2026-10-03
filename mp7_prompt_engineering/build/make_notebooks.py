"""Generates the MP7 workshop notebook (+ report template).

    python build/make_notebooks.py [--fresh-text]

The notebook is where the wording lives: a rebuild keeps every markdown cell, every cell's #@markdown notes and every
wording written in a cell (name_text = \"\"\"...\"\"\") it finds in the existing notebook (matched by the cell's first
line / title) and only regenerates the code. Pass --fresh-text to start again from the defaults.
"""
import json
import re
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from aec_prompt.lab import INCIDENT_VERSIONS, SCHEDULE_VERSIONS, TAKEOFF_VERSIONS, drawing_label  # noqa: E402
from aec_prompt.texts import CHAT_URL, FOUNDATION_CONVENTIONS, ROOF_CONVENTIONS, TEXTS  # noqa: E402

GITHUB_URL = "https://github.com/Haolan-Zhang/CEM4644.git"
FILE = "MP7_Workshop_Prompt_Engineering.ipynb"
KEEP_TEXT = True
EXISTING, EXISTING_TEXTS = {}, {}


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


def form(title, call, notes=(), params=(), texts=None):
    """A form cell; texts: {name: wording} written as name = \"\"\"...\"\"\" and passed to the call as name=name."""
    head = f'#@title {title} {{ display-mode: "form" }}'
    if head in EXISTING:
        notes = EXISTING[head]
    words = ""
    if texts:
        kept = EXISTING_TEXTS.get(head, {})
        words = ("# The wording the notebook shows (prompts, instructions, results): edit the text between the triple quotes.\n"
                 "# Words in {braces} are filled in by the notebook; **bold** and *italic* work; each line is shown as its own line.\n"
                 + "".join(f'{k} = """{kept.get(k, v)}"""\n' for k, v in texts.items()))
        head_call = call.rstrip(")")
        call = head_call + (", " if not head_call.endswith("(") else "") + ", ".join(f"{k}={k}" for k in texts) + ")"
    src = head + "\n" + "".join(f"#@markdown {n}\n" for n in notes) + "".join(p + "\n" for p in params) + words + call + "\n"
    c = new_code_cell(src); c.metadata["cellView"] = "form"
    return c


def md(text):
    text = text.strip("\n")
    return new_markdown_cell(EXISTING.get(text.split("\n", 1)[0].strip(), text))


def question(n, *parts):
    return md(f"### 📝 Report question {n}\n" + "\n\n".join(f"> {p}" for p in parts))


def choice(name, value, options):
    return f'{name} = {json.dumps(value, ensure_ascii=False)} #@param {json.dumps(list(options), ensure_ascii=False)}'


def step0():
    body = f"""import importlib, os, shutil, subprocess, sys
REPO, FOLDER, PKG = "CEM4644", "mp7_prompt_engineering", "aec_prompt"
FOLDERS = ["mp7_prompt_engineering"]          # only this lab folder is downloaded, not the whole course repository

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
from aec_prompt import lab
lab.setup()"""
    return form("▶ Step 0 · Run me first (1 minute)", body, notes=["Click ▶ and wait for the ✅ line."])


def T(group, drop=()):
    return {k: v for k, v in TEXTS[group].items() if k not in drop}


def build():
    load_existing(REPO / FILE)
    probs = json.loads((REPO / "data" / "practice" / "problems.json").read_text())
    labels = {f: [drawing_label(p) for p in probs if p["family"] == f] for f in ("foundation", "roof", "schedule")}
    cells = []
    cells.append(md("""
# In-Class Activity: Prompt Engineering for Construction Documents and Drawings
In this activity, we will use generative AI (GPT/HokieAI) to classify construction incident reports and to read construction drawings, and benchmark how the wording of a prompt changes the accuracy of its outputs.
"""))
    cells.append(step0())

    # ------------------------------------------------------------------ Part 1
    cells.append(md(f"""
## Part 1 · Classifying Incident Reports
OSHA records every severe work-related injury reported by employers, with a short narrative and a code for the event that caused it. Here, you will use HokieAI to classify 30 construction incident reports into OSHA's construction Focus Four hazards (fall, struck-by, caught-in/between, electrocution) or other, and compare its output with OSHA's own classification.

In every HokieAI step, the notebook gives you a prompt; you send it to HokieAI ({CHAT_URL}), paste HokieAI's output back, and the notebook scores it against the answer key. Each prompt version changes one thing, and every scored output is added to the comparison step that follows.
"""))
    cells.append(form("▶ Step 1a · Browse the Reports", "lab.show_reports(rows)",
                      notes=["Run the cell to browse the incident reports HokieAI will classify. OSHA's classification of each report is the answer key; it stays hidden until you score an output."],
                      params=['rows = 10 #@param [5, 10, 30] {type:"raw"}'], texts=T("reports")))
    cells.append(form("▶ Step 1b · Classify with HokieAI", "lab.classify_incidents(prompt_version)",
                      notes=["Choose a prompt version, run the cell, and follow the instructions. Score every version, and run one version twice in new chats to check its consistency.",
                             "v1 · task only: the categories and the output format.",
                             "v2 · role and definitions: adds a role and OSHA's definition of each category.",
                             "v3 · definitions and worked examples: adds five reports with OSHA's classification (few-shot prompting).",
                             "my own prompt: starts from v3; edit the prompt in the prompt box before copying it.",
                             "If HokieAI declines a prompt or its output is blocked, record which version: that is a result too."],
                      params=[choice("prompt_version", list(INCIDENT_VERSIONS)[0], INCIDENT_VERSIONS)], texts=T("incidents")))
    cells.append(form("▶ Step 1c · Compare Prompt Versions", 'lab.compare("incidents")',
                      notes=["Run the cell after scoring the prompt versions in Step 1b. Each bar is the share of reports classified as OSHA classified them, over all outputs of that version."],
                      texts=T("compare")))
    cells.append(question(1,
                          "From Step 1b: Report the accuracy of each prompt version and include the confusion matrix of the best one.",
                          "From Step 1c: Which change to the prompt (role, definitions, worked examples) improved the accuracy most? Did repeating a version in a new chat give the same output?",
                          "Choose two reports where HokieAI's output differs from OSHA's classification. From the report text, explain why each is hard to classify, and how you would change the prompt to fix it."))

    # ------------------------------------------------------------------ Part 2
    cells.append(md("""
## Part 2 · Quantity Takeoff from Drawings
Five foundation plans and five gable roofs were drawn for this course. Each comes with estimating questions (CMUs and grout; roof area, underlayment, and shingles), and the answer key is computed exactly from the drawing's dimensions with standard estimating conventions. HokieAI reads the attached drawing and answers each question; quantities within 1–2 % of the answer key count as correct.
"""))
    vnotes = ["v1 · task only: the questions and the output format.",
              "v2 · step-by-step reasoning: asks HokieAI to list the dimensions it reads and to solve each question step by step.",
              "v3 · estimating conventions: adds the estimating conventions used for the answer key.",
              "my own prompt: starts from v3; edit the prompt in the prompt box before copying it."]
    cells.append(form("▶ Step 2a · Foundation Plans", "lab.takeoff(drawing, prompt_version)",
                      notes=["Choose a drawing and a prompt version, run the cell, and follow the instructions. Run all three versions on at least two drawings."] + vnotes,
                      params=[choice("drawing", labels["foundation"][0], labels["foundation"]), choice("prompt_version", list(TAKEOFF_VERSIONS)[0], TAKEOFF_VERSIONS)],
                      texts=dict(T("takeoff"), conventions_text=FOUNDATION_CONVENTIONS)))
    cells.append(form("▶ Step 2b · Gable Roofs", "lab.takeoff(drawing, prompt_version)",
                      notes=["Choose a drawing and a prompt version, run the cell, and follow the instructions. Run all three versions on at least two drawings."] + vnotes,
                      params=[choice("drawing", labels["roof"][0], labels["roof"]), choice("prompt_version", list(TAKEOFF_VERSIONS)[0], TAKEOFF_VERSIONS)],
                      texts=dict(T("takeoff"), conventions_text=ROOF_CONVENTIONS)))
    cells.append(form("▶ Step 2c · Compare Prompt Versions", 'lab.compare("takeoff")',
                      notes=["Run the cell after Steps 2a and 2b. Each bar is the share of quantities within tolerance of the answer key, over all drawings scored with that version."],
                      texts=T("compare")))
    cells.append(question(2,
                          "From Step 2c: Report the share of correct quantities for each prompt version, across at least two foundation plans and two roofs.",
                          "For one wrong quantity, find in HokieAI's output where the error came from: a misread dimension, a different estimating convention, or an arithmetic error.",
                          "Why does stating the estimating conventions help more than asking for step-by-step reasoning?"))

    # ------------------------------------------------------------------ Part 3
    cells.append(md("""
## Part 3 · Reading Schedules on a Drawing Sheet
Five 36 × 24 in drawing sheets, made for this course, each carry the door schedules of several unit types, a window schedule, hardware sets, and door-type elevations. The task: list every door of one unit type. The answer key is the schedule's own data; a door counts as correct only if all six of its fields are transcribed exactly.
"""))
    cells.append(form("▶ Step 3a · Door Schedules", "lab.schedule(sheet, prompt_version)",
                      notes=["Choose a sheet and a prompt version, run the cell, and follow the instructions. Run all three versions on at least two sheets.",
                             "v1 · whole sheet: HokieAI receives the whole sheet.",
                             "v2 · cropped schedule: HokieAI receives only the part of the schedule for the unit type.",
                             "v3 · cropped schedule and column names: adds the column names and asks for an exact transcription.",
                             "my own prompt: starts from v3 with both images; edit the prompt in the prompt box before copying it."],
                      params=[choice("sheet", labels["schedule"][0], labels["schedule"]), choice("prompt_version", list(SCHEDULE_VERSIONS)[0], SCHEDULE_VERSIONS)],
                      texts=T("schedule")))
    cells.append(form("▶ Step 3b · Compare Prompt Versions", 'lab.compare("schedule")',
                      notes=["Run the cell after Step 3a. Each bar is the share of doors transcribed exactly, over all sheets scored with that version."], texts=T("compare")))
    cells.append(question(3,
                          "From Step 3b: Report the number of doors transcribed exactly with the whole sheet and with the cropped schedule. Why does cropping change the result?",
                          "Before using a door count from HokieAI in an estimate, what would you check, and how?"))

    cells.append(md("""
### Data and model sources
- Incident reports: Severe Injury Reports (2015–2025), Occupational Safety and Health Administration, U.S. Department of Labor, public domain, https://www.osha.gov/severe-injury-reports.
- Drawings: original practice drawings made for CEM4644 (not for construction).
- Generative AI: GPT models via HokieAI (Virginia Tech).
"""))

    nb = new_notebook(cells=cells)
    nb.metadata.update({"colab": {"provenance": [], "toc_visible": True},
                        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}})
    nbformat.write(nb, str(REPO / FILE)); print("wrote", REPO / FILE)
    lines = ["# CEM4644 · MP7 report — In-Class Activity: Prompt Engineering", "", "Name: ______________________    Date: ____________", "",
             "Answer every question in a few sentences. Paste screenshots where the question asks for tables or charts. Numbers must come from **your** run of the notebook.", ""]
    for c in cells:
        if c.cell_type == "markdown" and c.source.startswith("### 📝 Report question "):
            head, _, body = c.source.partition("\n")
            paras = [p.strip()[2:] for p in body.split("\n\n") if p.strip()]
            lines += [f"## Question {head.rsplit(' ', 1)[1]}", ""] + sum(([p, ""] for p in paras), []) + ["*Your answer:*", "", "", ""]
    (REPO / "docs").mkdir(exist_ok=True)
    (REPO / "docs" / "MP7_Workshop_Report_Template.md").write_text("\n".join(lines)); print("wrote report template")


if __name__ == "__main__":
    KEEP_TEXT = "--fresh-text" not in sys.argv
    build()
