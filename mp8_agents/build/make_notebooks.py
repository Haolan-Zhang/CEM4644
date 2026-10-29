"""Generates the MP8 notebooks (+ report templates): the in-class activity (AI workflows, AI agents and their tools,
skills, a HokieAI agent) and the homework (tools, your own HokieAI agent, a schedule skill of your own).

    python build/make_notebooks.py [workshop | homework] [--fresh-text]      (both notebooks without a name)

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
from aec_agent.lab import DEFAULT_QUESTIONS as STEP2_QUESTIONS, DOCUMENTS  # noqa: E402
from aec_agent.texts import CHAT_URL, MODEL, TEXTS  # noqa: E402

GITHUB_URL = "https://github.com/Haolan-Zhang/CEM4644.git"
FIG = "https://raw.githubusercontent.com/Haolan-Zhang/CEM4644/master/mp8_agents/figures"   # the diagrams (build/make_figures.py)
FILE = "MP8_Workshop_AI_Agents.ipynb"
FILE_HW = "MP8_Homework_AI_Agents.ipynb"
HW_QUESTIONS = ["Q4", "Q5", "Q6"]          # the homework's tool-calling questions (Q1-Q3 are the in-class ones)
ALL_QUESTIONS = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]
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
        words = ("# The wording the notebook shows (prompts, tool descriptions, skills, results): edit the text between the triple quotes.\n"
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


def T(group, keep=None, drop=()):
    return {k: v for k, v in TEXTS[group].items() if (keep is None or k in keep) and k not in drop}


def step0(edits="editing a prompt, a tool description, or the skill needs the key."):
    body = f"""import importlib, os, shutil, subprocess, sys
REPO, FOLDER, PKG = "CEM4644", "mp8_agents", "aec_agent"
FOLDERS = ["mp8_agents"]                      # only this lab folder is downloaded, not the whole course repository

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
try:
    import google.genai
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "google-genai"], capture_output=True)
for _m in [m for m in list(sys.modules) if m == PKG or m.startswith(PKG + ".")]:
    del sys.modules[_m]                      # Python caches imported code: drop it, or this cell keeps the old version
importlib.invalidate_caches()
sys.path.insert(0, os.path.abspath(os.path.join(REPO, FOLDER)))
from aec_agent import lab
model = "{MODEL}"
lab.setup(api_key=api_key, model=model)"""
    return form("▶ Step 0 · Run me first (1 minute)", body,
                notes=["Click ▶ and wait for the ✅ line. Paste your Gemini API key from MP5 into *api_key* (or leave it empty to use the Colab secret GEMINI_API_KEY).",
                       "Without a key, every step runs with its default settings (stored outputs); " + edits],
                params=['api_key = "" #@param {type:"string"}'])


def build():
    load_existing(REPO / FILE)
    drawings = json.loads((REPO / "data" / "drawings" / "problems.json").read_text())
    questions = json.loads((REPO / "data" / "project" / "questions.json").read_text())
    step2 = [q for q in questions if q["id"] in STEP2_QUESTIONS]
    cells = [md("""
# In-Class Activity: AI Workflows and AI Agents
In this activity, we will use AI to take off quantities from foundation plans and answer questions about a construction project: first as a fixed workflow, then as an AI agent with tools and a skill.
"""), step0()]

    # ------------------------------------------------------------------ Part 1: workflows
    cells.append(md("""
## Part 1 · AI Workflows: Fixed Steps
A workflow splits a task into fixed steps, each done by an AI model or by code. We compare one prompt with a two-step workflow on a takeoff from five foundation plans.

![One prompt vs. a workflow]({FIG}/workflow.png)
""".replace("{FIG}", FIG)))
    cells.append(form("▶ Step 1a · Look at the Drawings", "lab.show_drawing(drawing)",
                      notes=["Choose a drawing to see it with its three questions. Some plans leave one side undimensioned."],
                      params=[choice("drawing", drawings[0]["label"], [p["label"] for p in drawings])]))
    cells.append(form("▶ Step 1b · One Prompt: The AI Model Does Everything", "lab.one_prompt()",
                      notes=["Run the cell: the AI model (Gemini) reads each drawing and computes every quantity in one reply."],
                      texts=T("one_prompt")))
    cells.append(form("▶ Step 1c · Workflow: The AI Model Reads, Code Computes", "lab.workflow()",
                      notes=["Run the cell: step 1, the AI model reads the length of every side; step 2, code adds them and applies the estimating conventions."],
                      texts=T("workflow")))
    cells.append(form("▶ Step 1d · Compare One Prompt and the Workflow", "lab.compare_workflow()",
                      notes=["Run the cell after Steps 1b and 1c."], texts=T("compare_workflow")))
    cells.append(question(1,
                          "From Step 1d: Report the number of quantities within tolerance for one prompt and for the workflow, and include the comparison chart.",
                          "From Steps 1b and 1c: Which quantities does one prompt get wrong, and why? Did the AI model read every side correctly in Step 1 of the workflow, including the undimensioned ones?",
                          "Which steps of a takeoff would you give to an AI model and which to code (or a spreadsheet) on a real project, and why?"))

    # ------------------------------------------------------------------ Part 2: agents and tools
    cells.append(md(f"""
## Part 2 · AI Agents: Tool Calling
An AI agent decides by itself which tool to call and when it has enough to answer. You choose its tools; it answers questions about a practice project whose documents it can reach only through the tools.

![How an AI agent calls a tool]({FIG}/agent.png)

- **search_spec**: searches the specification.
- **calculator**: exact arithmetic.
- **lookup_log**: reads the submittal log or the schedule.
""".replace("{FIG}", FIG)))
    cells.append(form("▶ Step 2a · Look at the Project Documents", "lab.show_document(document)",
                      notes=["Choose a document to see what the tools read."],
                      params=[choice("document", list(DOCUMENTS)[0], DOCUMENTS)]))
    cells.append(form("▶ Step 2b · Run the Agent with the Tools You Choose", "lab.agent_tools(search_spec, calculator, lookup_log, questions)",
                      notes=["Tick the tools and run the cell; each run is added to the table. Try no tools, each tool alone, and all three.",
                             "UNKNOWN: the agent said it could not find the value. Wrong: it guessed."],
                      params=['search_spec = True #@param {type:"boolean"}', 'calculator = True #@param {type:"boolean"}', 'lookup_log = True #@param {type:"boolean"}',
                              "questions = " + json.dumps(STEP2_QUESTIONS) + "          # the project questions the agent answers (Q1-Q6); Step 2c and Part 5 use the same ones"],
                      texts=T("tools")))
    cells.append(form("▶ Step 2c · Inspect the Tool Calls", "lab.show_trace(question)",
                      notes=["Choose a question to see, for each run of Step 2b, the tools the agent called, with their inputs and outputs."],
                      params=[choice("question", f"{step2[-1]['id']}. {step2[-1]['question']}", [f"{q['id']}. {q['question']}" for q in step2])],
                      texts=T("trace")))
    cells.append(question(2,
                          "From Step 2b: Include the table of runs. For each tool, report which questions it was needed for.",
                          "From Step 2c: For question Q3, describe the agent's steps with all three tools. Without the tools it needed, did the agent answer UNKNOWN or guess?",
                          "Which is more concerning on a real project, an agent that answers UNKNOWN or one that guesses, and how would you check an agent's answer?"))

    # ------------------------------------------------------------------ Part 3: code execution
    cells.append(md("""
## Part 3 · Code Execution: The Agent Builds Its Own Tool
In Part 2 the agent chose from tools written for it. Here HokieAI writes and runs its own code, and the result is something you can see and use: a chart or a calculator. Each result is checked against the ground truth, because code that runs can still give wrong numbers.

![Tool calling vs. code execution]({FIG}/codeexec.png)
""".replace("{FIG}", FIG)))
    cells.append(form("▶ Step 3a · Look-Ahead Schedule: A Gantt Chart", "lab.lookahead()",
                      notes=["Run the cell and follow the steps: HokieAI computes each activity's duration and start day, draws a Gantt chart, and marks in red the concrete placements whose mix design is not approved.",
                             "Look-ahead schedule: the near-term part of the project schedule that the field team plans from."],
                      texts=T("lookahead")))
    cells.append(form("▶ Step 3b · Build a Takeoff Calculator", "lab.calculator(version)",
                      notes=["Run the cell and follow the steps: HokieAI builds a calculator web page (an artifact); you test it with the three cases and enter its outputs."],
                      params=['version = "without the skill"'],
                      texts=T("calculator")))
    cells.append(question(3,
                          "From Step 3a: Include HokieAI's Gantt chart and the score. Which activities did it mark red, and did any start or finish day differ from the ground truth? Open the code it wrote: which step went wrong, if any?",
                          "From Step 3b: Include a screenshot of your calculator and the score on the three test cases. Which output was wrong, and which formula did the calculator use for it?",
                          "When would you let an agent write and run code for a project task, and what would you check before using its output?"))

    # ------------------------------------------------------------------ Part 4: skills
    cells.append(md("""
## Part 4 · Skills
A skill is a saved set of instructions with a name and a description. The agent sees only the descriptions and loads a skill's instructions when a task matches; a prompt is read every time.

![A prompt vs. a skill]({FIG}/skills.png)

| | Prompt | Skill |
|---|---|---|
| Written | in each message | once, saved with a name and a description |
| Read by the model | every time | only when the task matches the description |
| Example | the conventions pasted above every takeoff question | *cmu-foundation-takeoff*, loaded for CMU questions only |
""".replace("{FIG}", FIG)))
    cells.append(form("▶ Step 4a · Run the Agent with and without the Skill", "lab.agent_skill(use_skill)",
                      notes=["Run the cell with *use_skill* ticked and unticked: the agent answers two takeoff questions (T1, T2) and one project question (Q5).",
                             "Then make the skill's description vague (for example, *Useful notes.*) and run again: is the skill still loaded?"],
                      params=['use_skill = True #@param {type:"boolean"}'],
                      texts=T("skills")))
    cells.append(form("▶ Step 4b · Create the Skill in HokieAI", "lab.hokieai_skill()",
                      notes=["Create the same skill in HokieAI, then send the same prompt once without and once with the skill, and score both outputs."],
                      texts=T("hokieai_skill")))
    cells.append(form("▶ Step 4c · Rebuild the Calculator with the Skill", "lab.calculator(version)",
                      notes=["Run the cell and follow the steps: the same calculator prompt as in Step 3b, this time with the skill turned on. The table compares both calculators."],
                      params=['version = "with the skill"'],
                      texts=T("calculator_skill")))
    cells.append(question(4,
                          "From Step 4a: Report the quantities correct with and without the skill, and the questions the skill was loaded for. Did the agent load the skill for Q5? What happened with a vague description?",
                          "From Steps 4b and 4c: Report HokieAI's quantities correct without and with the skill, and your two calculators' scores. Which outputs did the skill fix?",
                          "What estimating conventions or company rules from your own experience would you write as a skill, and what would its description say?"))

    # ------------------------------------------------------------------ Part 5: your own HokieAI agent
    cells.append(md("""
## Part 5 · Your Own HokieAI Agent
A HokieAI agent is instructions plus knowledge sources (files it can read). Build one with the project files and ask it the questions of Part 2.
"""))
    cells.append(form("▶ Step 5a · Create Your HokieAI Agent", "lab.create_agent()",
                      notes=["Run the cell and follow the steps; edit the instructions in the box before pasting them."],
                      texts=T("hokieai_agent", keep=("create_steps_text", "instructions_text"))))
    cells.append(form("▶ Step 5b · Ask Your HokieAI Agent", "lab.hokieai_agent()",
                      notes=["Run the cell after Step 5a and follow the steps."],
                      texts=T("hokieai_agent", drop=("create_steps_text", "instructions_text"))))
    cells.append(question(5,
                          "From Step 5b: Report your agent's correct, UNKNOWN, and wrong answers, next to the Gemini agent's run with all three tools, and include the table.",
                          "Which instructions did you change, and what changed in the output? For one wrong answer, explain from the output which value or calculation went wrong.",
                          "What would you need to see before you let an agent answer project questions for your team?"))

    cells.append(md("""
### Data and model sources
- Practice project (specification excerpt, submittal log, schedule) and foundation plans: made for CEM4644 (fictional, not for construction); the specification excerpt follows the format of UFGS 03 30 00, U.S. Department of Defense, public domain, https://www.wbdg.org/dod/ufgs.
- AI model: Gemini (gemini-3.5-flash-lite), Google, via the Gemini API with your own key.
- Generative AI: GPT models via HokieAI (Virginia Tech).
"""))
    write(cells, FILE, "In-Class Activity: AI Workflows and AI Agents", "MP8_Workshop_Report_Template.md")


SOURCES = """
### Data and model sources
- Practice project (specification excerpt, submittal log, schedule): made for CEM4644 (fictional, not for construction); the specification excerpt follows the format of UFGS 03 30 00, U.S. Department of Defense, public domain, https://www.wbdg.org/dod/ufgs.
- AI model: Gemini (gemini-3.5-flash-lite), Google, via the Gemini API with your own key.
- Generative AI: GPT models via HokieAI (Virginia Tech).
"""


def build_homework():
    load_existing(REPO / FILE_HW)
    questions = json.loads((REPO / "data" / "project" / "questions.json").read_text())
    hw = [q for q in questions if q["id"] in HW_QUESTIONS]
    key = json.loads((REPO / "data" / "project" / "lookahead_key.json").read_text())
    cells = [md("""
# Homework: AI Tools, AI Agents, and Skills
In this homework, we will use AI to make a look-ahead schedule and answer questions about a construction project with tools, an AI agent, and a skill.
"""), step0("editing the agent's system prompt or a tool description needs the key.")]

    # ------------------------------------------------------------------ Part 1: tools
    cells.append(md("""
## Part 1 · Tools: Code Execution and Tool Calling
In this part, we will explore two kinds of tools: code that HokieAI writes and runs by itself, and tools written for the Gemini agent, which it chooses from to answer the project questions Q4 to Q6.

![Code execution and tool calling]({FIG}/codeexec_hw.png)

- **search_spec**: searches the specification.
- **calculator**: exact arithmetic.
- **lookup_log**: reads the submittal log or the schedule.
""".replace("{FIG}", FIG)))
    cells.append(form("▶ Step 1a · Look-Ahead Schedule: A Gantt Chart", "lab.lookahead()",
                      notes=["Run the cell and follow the steps: HokieAI computes each activity's duration and start day, draws a Gantt chart, and marks in red the concrete placements whose mix design is not approved.",
                             "Look-ahead schedule: the near-term part of the project schedule that the field team plans from."],
                      texts=T("lookahead")))
    cells.append(form("▶ Step 1b · Look at the Project Documents", "lab.show_document(document)",
                      notes=["Choose a document to see what the tools read."],
                      params=[choice("document", list(DOCUMENTS)[0], DOCUMENTS)]))
    cells.append(form("▶ Step 1c · Run the Agent with the Tools You Choose", "lab.agent_tools(search_spec, calculator, lookup_log, questions)",
                      notes=["Tick the tools and run the cell; each run is added to the table. Try no tools, each tool alone, and all three.",
                             "UNKNOWN: the agent said it could not find the value. Wrong: it guessed."],
                      params=['search_spec = True #@param {type:"boolean"}', 'calculator = True #@param {type:"boolean"}', 'lookup_log = True #@param {type:"boolean"}',
                              "questions = " + json.dumps(HW_QUESTIONS) + "          # the project questions the agent answers (Q1-Q3 were in class)"],
                      texts=T("tools")))
    cells.append(form("▶ Step 1d · Inspect the Tool Calls", "lab.show_trace(question)",
                      notes=["Choose a question to see, for each run of Step 1c, the tools the agent called, with their inputs and outputs."],
                      params=[choice("question", f"{hw[-1]['id']}. {hw[-1]['question']}", [f"{q['id']}. {q['question']}" for q in hw])],
                      texts={**T("trace"), "empty_text": "No runs yet: run Step 1c first."}))
    cells.append(question(1,
                          "From Step 1a: Include HokieAI's Gantt chart and the score. Which activities did it mark red, and did any start or finish day differ from the ground truth? Open the code it wrote: which step went wrong, if any?",
                          "Would you trust a look-ahead schedule that HokieAI computed with its own code, and what would you check before sending it to the field team?"))
    cells.append(question(2,
                          "From Step 1c: Include the table of runs. For each of Q4, Q5, and Q6, report which tools the agent needed to answer it correctly.",
                          "From Step 1d: For question Q6, describe the agent's steps with all three tools. Without the tools it needed, did the agent answer UNKNOWN or guess?",
                          "For which project tasks would you write a tool for an agent (as in Step 1c), and for which would you let it write its own code (as in Step 1a)?"))

    # ------------------------------------------------------------------ Part 2: your own HokieAI agent
    cells.append(md("""
## Part 2 · AI Agents: Your Own HokieAI Agent
A HokieAI agent is instructions plus knowledge sources (files it can read). In this part, we will build one with the project files and ask it all six project questions: Q1 to Q3 from the in-class activities and Q4 to Q6 from Part 1.
"""))
    cells.append(form("▶ Step 2a · Create Your HokieAI Agent", "lab.create_agent()",
                      notes=["Run the cell and follow the steps; edit the instructions in the box before pasting them."],
                      texts=T("hokieai_agent", keep=("create_steps_text", "instructions_text"))))
    cells.append(form("▶ Step 2b · Ask Your HokieAI Agent", "lab.hokieai_agent(questions)",
                      notes=["Run the cell after Step 2a and follow the steps. The table compares your agent with the Gemini agent's runs of Step 1c on Q4 to Q6."],
                      params=["questions = " + json.dumps(ALL_QUESTIONS) + "          # the project questions your agent answers"],
                      texts={**T("hokieai_agent", drop=("create_steps_text", "instructions_text")),
                             "steps_text": TEXTS["hokieai_agent"]["steps_text"].replace("Step 2b", "Step 1c"),
                             "compare_text": "Your HokieAI agent vs. the Gemini agent's runs in Step 1c (on the questions both answered):"}))
    cells.append(question(3,
                          "From Step 2b: Report your agent's correct, UNKNOWN, and wrong answers on all six questions, include the tables, and compare it on Q4 to Q6 with the Gemini agent's run with all three tools.",
                          "Which instructions did you change, and what changed in the output? For one wrong or UNKNOWN answer, explain from the output which value or calculation went wrong.",
                          "What would you need to see before you let an agent answer project questions for your team?"))

    # ------------------------------------------------------------------ Part 3: a skill
    cells.append(md("""
## Part 3 · Skills: A Schedule Skill of Your Own
In this part, we will write a skill for a scheduling job of your choice that ends in a Gantt chart, then use it when the look-ahead schedule of Step 1a changes. You can ask HokieAI to help you develop the skill.

![Your skill, used on the schedule with a change you choose]({FIG}/schedule_skill.png)
""".replace("{FIG}", FIG)))
    cells.append(form("▶ Step 3a · Write Your Schedule Skill", "lab.schedule_skill()",
                      notes=["Run the cell and create your skill in HokieAI; there is no single correct skill. Choose the job it does for your team, for example:",
                             "Subcontractor look-ahead: the next two weeks for one trade, with what it needs on site each day.",
                             "Crew histogram: the crews needed each day, under the chart, with the days that need more crews than are available marked.",
                             "Pour plan: the concrete trucks to order and the pump to book for each placement day.",
                             "Meeting agenda: the weekly coordination meeting items read from the chart.",
                             "Copy the required rule into the skill's body. You can ask HokieAI in a chat to draft or improve the skill, then edit it."],
                      texts=T("schedule_skill")))
    cells.append(form("▶ Step 3b · Use Your Skill When the Schedule Changes", "lab.schedule_change(change, activity, day, last_day)",
                      notes=["Choose a change, run the cell, and follow the steps. The notebook recomputes the schedule for the change and scores the start and finish days.",
                             "An activity starts late: *activity* cannot start before *day* (for example, its mix design is approved late).",
                             "No work on some days: no work from *day* to *last_day* (for example, rain).",
                             "One concrete placement at a time: concrete placements cannot overlap (for example, one pump on site).",
                             "If the output does not follow your skill, revise the skill in **My Skills** and run the step again."],
                      params=[choice("change", "an activity starts late", ["an activity starts late", "no work on some days", "one concrete placement at a time", "no change"]),
                              choice("activity", "A-210", [r["activity"] for r in key["rows"]]),
                              'day = 10 #@param {type:"integer"}', 'last_day = 11 #@param {type:"integer"}'],
                      texts=T("schedule_change")))
    cells.append(question(4,
                          "From Step 3a: Include your skill (name, description, and body) and say what job it does. If HokieAI helped you write it, what did you change in its draft?",
                          "From Step 3b: Include HokieAI's output (the Gantt chart and what your skill adds), the change you chose, and the score. Did the output follow every rule of your skill? If you revised the skill, what did you change, and did it help?",
                          "Would your field team use this output? What would they still check by hand before using it?"))

    cells.append(md(SOURCES))
    write(cells, FILE_HW, "Homework: AI Tools, AI Agents, and Skills", "MP8_Homework_Report_Template.md")


def write(cells, file, title, template):
    nb = new_notebook(cells=cells)
    nb.metadata.update({"colab": {"provenance": [], "toc_visible": True},
                        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}})
    nbformat.write(nb, str(REPO / file)); print("wrote", REPO / file)
    lines = [f"# CEM4644 · MP8 report — {title}", "", "Name: ______________________    Date: ____________", "",
             "Answer every question in a few sentences. Paste screenshots where the question asks for tables or charts. Numbers must come from **your** run of the notebook.", ""]
    for c in cells:
        if c.cell_type == "markdown" and c.source.startswith("### 📝 Report question "):
            head, _, body = c.source.partition("\n")
            paras = [p.strip()[2:] for p in body.split("\n\n") if p.strip()]
            lines += [f"## Question {head.rsplit(' ', 1)[1]}", ""] + sum(([p, ""] for p in paras), []) + ["*Your answer:*", "", "", ""]
    (REPO / "docs").mkdir(exist_ok=True)
    (REPO / "docs" / template).write_text("\n".join(lines)); print("wrote", REPO / "docs" / template)


if __name__ == "__main__":
    KEEP_TEXT = "--fresh-text" not in sys.argv
    which = [a for a in sys.argv[1:] if not a.startswith("--")] or ["workshop", "homework"]
    if "workshop" in which:
        build()
    if "homework" in which:
        build_homework()
