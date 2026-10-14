"""Generates the MP8 notebook (+ report template): AI workflows, AI agents and their tools, skills, and a HokieAI agent.

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
from aec_agent.lab import DOCUMENTS  # noqa: E402
from aec_agent.texts import CHAT_URL, MODEL, TEXTS  # noqa: E402

GITHUB_URL = "https://github.com/Haolan-Zhang/CEM4644.git"
FILE = "MP8_Workshop_AI_Agents.ipynb"
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


def step0():
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
                notes=["Click ▶ and wait for the ✅ line. Paste your Gemini API key from MP5 into *api_key*, or leave it empty to use the Colab secret GEMINI_API_KEY.",
                       "The default settings of every step run without a key (their outputs are stored); a key is needed only when you edit a prompt, a tool description, or the skill."],
                params=['api_key = "" #@param {type:"string"}'])


def build():
    load_existing(REPO / FILE)
    drawings = json.loads((REPO / "data" / "drawings" / "problems.json").read_text())
    questions = json.loads((REPO / "data" / "project" / "questions.json").read_text())
    cells = [md("""
# In-Class Activity: AI Workflows and AI Agents for a Construction Project
In this activity, we will use AI to take off quantities from foundation plans and to answer questions about a construction project from its specification, submittal log, and schedule, first as a fixed workflow, then as an AI agent that chooses its own tools, and finally with a skill and a HokieAI agent of your own.
"""), step0()]

    # ------------------------------------------------------------------ Part 1: workflows
    cells.append(md("""
## Part 1 · AI Workflows: Fixed Steps
In this part, we will explore an AI workflow: a task split into fixed steps, some done by an AI model and some by code. The task is a takeoff from five foundation plans (CMUs and grout), done once with one prompt and once as a workflow; both are scored against the ground truth.
"""))
    cells.append(form("▶ Step 1a · Look at the Drawings", "lab.show_drawing(drawing)",
                      notes=["Choose a drawing and run the cell to see it with its three takeoff questions. Some plans leave one side undimensioned: its length follows from the other dimensions."],
                      params=[choice("drawing", drawings[0]["label"], [p["label"] for p in drawings])]))
    cells.append(form("▶ Step 1b · One Prompt: The AI Model Does Everything", "lab.one_prompt()",
                      notes=["Run the cell to send each drawing with its questions to the AI model (Gemini) in one prompt: the model reads the dimensions and computes every quantity."],
                      texts=T("one_prompt")))
    cells.append(form("▶ Step 1c · Workflow: The AI Model Reads, Code Computes", "lab.workflow()",
                      notes=["Run the cell to do the same takeoff as a workflow of two fixed steps:",
                             "Step 1 (AI model): read the length of every side of the wall from the drawing (the prompt below).",
                             "Step 2 (code): add the sides and apply the estimating conventions (for example, grouting at 32 in on center fills 8/32 of the cells)."],
                      texts=T("workflow")))
    cells.append(form("▶ Step 1d · Compare One Prompt and the Workflow", "lab.compare_workflow()",
                      notes=["Run the cell after Steps 1b and 1c."], texts=T("compare_workflow")))
    cells.append(question(1,
                          "From Step 1d: Report the number of quantities within tolerance for one prompt and for the workflow, and include the comparison chart.",
                          "From Steps 1b and 1c: Which quantities does one prompt get wrong, and why? Did the AI model read every side correctly in Step 1 of the workflow, including the undimensioned ones?",
                          "Which steps of a takeoff would you give to an AI model and which to code (or a spreadsheet) on a real project, and why?"))

    # ------------------------------------------------------------------ Part 2: agents and tools
    cells.append(md(f"""
## Part 2 · AI Agents: Choose the Tools
In this part, we will explore an AI agent: an AI model that decides by itself which tool to call, with which inputs, and when it has enough information to answer. You choose which tools the agent may use, and every run answers the same six questions about the CEM4644 practice project (fictional): its specification, submittal log, and schedule are available only through the tools.

- **search_spec**: searches the specification (Section 03 30 00).
- **calculator**: evaluates an arithmetic expression exactly.
- **lookup_log**: reads rows of the submittal log or the schedule.
"""))
    cells.append(form("▶ Step 2a · Look at the Project Documents", "lab.show_document(document)",
                      notes=["Choose a document and run the cell to see what the tools read."],
                      params=[choice("document", list(DOCUMENTS)[0], DOCUMENTS)]))
    cells.append(form("▶ Step 2b · Run the Agent with the Tools You Choose", "lab.agent_tools(search_spec, calculator, lookup_log)",
                      notes=["Tick the tools the agent may use and run the cell; every run is added to the table of runs. Run at least: no tools, each tool alone, and all three.",
                             "Tool calling: the AI model does not run a tool itself; it asks for one (the tool's name and inputs), the notebook runs it, and the tool's output goes back to the model.",
                             "UNKNOWN: the agent answered that it could not find the value; a wrong answer means it guessed.",
                             "The system prompt and the tool descriptions below are what the agent reads. If you edit one, the run needs your Gemini key (Step 0)."],
                      params=['search_spec = True #@param {type:"boolean"}', 'calculator = True #@param {type:"boolean"}', 'lookup_log = True #@param {type:"boolean"}'],
                      texts=T("tools")))
    cells.append(form("▶ Step 2c · Inspect the Tool Calls", "lab.show_trace(question)",
                      notes=["Choose a question and run the cell to see, for every run of Step 2b, each tool the agent called, its inputs, and the tool's output."],
                      params=[choice("question", f"{questions[4]['id']}. {questions[4]['question']}", [f"{q['id']}. {q['question']}" for q in questions])],
                      texts=T("trace")))
    cells.append(question(2,
                          "From Step 2b: Include the table of runs. For each tool, report which questions it was needed for.",
                          "From Step 2c: For question P5, describe the agent's steps with all three tools. Without the tools it needed, did the agent answer UNKNOWN or guess?",
                          "Which is more concerning on a real project, an agent that answers UNKNOWN or one that guesses, and how would you check an agent's answer?"))

    # ------------------------------------------------------------------ Part 3: skills
    cells.append(md("""
## Part 3 · Skills: Instructions the Agent Loads When Needed
In this part, we will explore skills: written instructions, with a name and a description, that an agent loads only when a task needs them. The agent sees every skill's description; when the description matches the task, it loads the skill's full instructions (the body). The skill here holds the estimating conventions of Part 1, and the agent answers two takeoff questions (T1, T2) and one project question (P3).
"""))
    cells.append(form("▶ Step 3a · Run the Agent with and without the Skill", "lab.agent_skill(use_skill)",
                      notes=["Run the cell once with *use_skill* ticked and once unticked, and compare the quantities correct and the questions the skill was loaded for.",
                             "The agent has the calculator and the project logs; with *use_skill*, it also has the tool load_skill.",
                             "Then make the description vague (for example, *Useful notes.*) and run again: does the agent still load the skill? If you edit the skill, the run needs your Gemini key (Step 0)."],
                      params=['use_skill = True #@param {type:"boolean"}'],
                      texts=T("skills")))
    cells.append(form("▶ Step 3b · Create the Skill in HokieAI", "lab.hokieai_skill(skill)",
                      notes=["Create the skill of Step 3a in HokieAI, then send the takeoff questions T1 and T2 once without and once with the skill, and score both outputs."],
                      params=[choice("skill", "with the skill", ["without the skill", "with the skill"])],
                      texts=T("hokieai_skill")))
    cells.append(question(3,
                          "From Step 3a: Report the quantities correct with and without the skill, and the questions the skill was loaded for. Did the agent load the skill for P3? What happened with a vague description?",
                          "From Step 3b: Report HokieAI's quantities correct without and with the skill.",
                          "What estimating conventions or company rules from your own experience would you write as a skill, and what would its description say?"))

    # ------------------------------------------------------------------ Part 4: your own HokieAI agent
    cells.append(md("""
## Part 4 · Your Own HokieAI Agent
In this part, we will build an agent in HokieAI: instructions plus knowledge sources (the project files), asked the six questions of Part 2. Its output is scored against the ground truth and compared with the Gemini agent's runs.
"""))
    cells.append(form("▶ Step 4a · Create Your HokieAI Agent", "lab.create_agent()",
                      notes=["Run the cell and follow the steps. The instructions below are a starting point: edit them in the box before pasting."],
                      texts=T("hokieai_agent", keep=("create_steps_text", "instructions_text"))))
    cells.append(form("▶ Step 4b · Ask Your HokieAI Agent", "lab.hokieai_agent()",
                      notes=["Run the cell after Step 4a and follow the instructions. If you change your agent's instructions, start a new chat and score its output again."],
                      texts=T("hokieai_agent", drop=("create_steps_text", "instructions_text"))))
    cells.append(question(4,
                          "From Step 4b: Report your agent's correct, UNKNOWN, and wrong answers, next to the Gemini agent's run with all three tools, and include the table.",
                          "Which instructions did you change, and what changed in the output? For one wrong answer, explain from the output which value or calculation went wrong.",
                          "What would you need to see before you let an agent answer project questions for your team?"))

    cells.append(md("""
### Data and model sources
- Practice project (specification excerpt, submittal log, schedule) and foundation plans: made for CEM4644 (fictional, not for construction); the specification excerpt follows the format of UFGS 03 30 00, U.S. Department of Defense, public domain, https://www.wbdg.org/dod/ufgs.
- AI model: Gemini (gemini-3.5-flash-lite), Google, via the Gemini API with your own key.
- Generative AI: GPT models via HokieAI (Virginia Tech).
"""))
    write(cells)


def write(cells):
    nb = new_notebook(cells=cells)
    nb.metadata.update({"colab": {"provenance": [], "toc_visible": True},
                        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}})
    nbformat.write(nb, str(REPO / FILE)); print("wrote", REPO / FILE)
    lines = ["# CEM4644 · MP8 report — In-Class Activity: AI Workflows and AI Agents", "", "Name: ______________________    Date: ____________", "",
             "Answer every question in a few sentences. Paste screenshots where the question asks for tables or charts. Numbers must come from **your** run of the notebook.", ""]
    for c in cells:
        if c.cell_type == "markdown" and c.source.startswith("### 📝 Report question "):
            head, _, body = c.source.partition("\n")
            paras = [p.strip()[2:] for p in body.split("\n\n") if p.strip()]
            lines += [f"## Question {head.rsplit(' ', 1)[1]}", ""] + sum(([p, ""] for p in paras), []) + ["*Your answer:*", "", "", ""]
    (REPO / "docs").mkdir(exist_ok=True)
    (REPO / "docs" / "MP8_Workshop_Report_Template.md").write_text("\n".join(lines)); print("wrote", REPO / "docs" / "MP8_Workshop_Report_Template.md")


if __name__ == "__main__":
    KEEP_TEXT = "--fresh-text" not in sys.argv
    build()
