"""Default wordings of every prompt, tool description, skill, instruction and result the MP8 notebook shows.

Each notebook cell carries its own copy (name_text = \"\"\"...\"\"\", editable in Colab); these are only the fallbacks.
Words in {braces} are filled in when the cell runs; **bold** and *italic* work; each line is shown as its own line.
"""
from typing import Dict, Optional

CHAT_URL = "https://hokie.ai.vt.edu/"
MODEL = "gemini-3.5-flash-lite"

FINALS = "End your reply with one final line per question, in this format:\n{lines}"

AGENT_SYSTEM = ("You are an AI agent that answers questions about the CEM4644 practice project, a fictional building project. "
                "Project information (specification requirements, submittals, schedule) is available only through your tools. "
                "Do not guess a project value: if you cannot find it, answer UNKNOWN.\n"
                "End your reply with the final answer on its own line, in this format: FINAL: <answer>\n"
                "If the question asks for several quantities, give one FINAL line per quantity, for example: FINAL: Q1 = 832 CMU")

TOOLS = {
    "search_spec": "Searches the project specification (Section 03 30 00, cast-in-place concrete) and returns the lines that best match the search words, for example the requirements for one element.",
    "calculator": "Evaluates an arithmetic expression exactly and returns the result. Supports + - * / ** ( ), ceil, floor, round, sqrt, min and max, for example: ceil(212 / 48).",
    "lookup_log": "Returns the rows of a project log that contain the search text: log = submittals (the submittal log: status and the mix of each submittal) or schedule (the schedule: quantity and daily output of each activity).",
    "load_skill": "Loads the full instructions of one of the available skills by its name. Load a skill when its description matches the task.",
}

SKILL = {
    "name": "cmu-foundation-takeoff",
    "description": "Estimating conventions for CMU foundation walls: total wall length, number of CMUs, and grout volume. Use it for any CMU or grout quantity question.",
    "body": ("# CMU foundation takeoff: estimating conventions\n"
             "- Total wall length = the sum of the outside dimensions of all sides (no deduction at the corners).\n"
             "- Wall area = total wall length x wall height.\n"
             "- One 8 in x 16 in (nominal) CMU covers 0.889 ft² of wall.\n"
             "- CMUs to purchase = wall area / 0.889 ft², plus waste, rounded up to a whole CMU.\n"
             "- Grout, all cells = CMUs before waste x grout per CMU, converted to cubic yards (27 ft³ = 1 CY), plus grout waste.\n"
             "- Cells are 8 in apart, so grouting the cells at S in on center fills 8/S of the cells: grout = grout for all cells x 8 / S."),
}

PASTE_STEPS = ("1. Copy the prompt below.\n2. Open {url} (sign in with your VT account), start a new chat, and paste the prompt.\n"
               "3. Copy HokieAI's output with the **copy button** below its reply, and paste it into the box below.\n"
               "4. Click *Score HokieAI output* to compare it with the ground truth.")
BOX = {"button_text": "Score HokieAI output", "reply_text": "Paste HokieAI's full output here.", "prompt_label_text": "prompt", "reply_label_text": "output"}

HOKIE_RESULT_STEPS = ("3. Copy HokieAI's output with the **copy button** below its reply, and paste it into the box below.\n"
                      "4. Click *Score HokieAI output* to compare it with the ground truth.")
CALCULATOR_PROMPT = ("Build an interactive calculator as an artifact (a web page I can use): a takeoff calculator for a CMU foundation wall.\n"
                     "Inputs: the lengths of all sides of the wall, measured on the outside face (ft, separated by commas); the wall height (ft); "
                     "CMU waste (%); grout per fully grouted CMU (ft³); grout waste (%); the spacing of the grouted cells (in on center).\n"
                     "Outputs: the total wall length (ft), the number of 8 in x 16 in CMUs to purchase, grout for all cells (CY), and grout for "
                     "only the cells at the given spacing (CY).\nShow the formula used for each output.")
CALCULATOR = {
    "prompt_text": CALCULATOR_PROMPT,
    "steps_text": ("1. Copy the prompt below, open {url} (sign in with your VT account), start a new chat, and paste the prompt.\n"
                   "2. Open the calculator HokieAI builds (the artifact) and type in each test case below.\n"
                   "3. Enter your calculator's outputs in the box below, one line per case, and click *Score my calculator*."),
    "cases_text": "Test cases (type each one into your calculator):",
    "cases_columns_text": "Case | Sides (ft) | Height (ft) | CMU waste (%) | Grout per CMU (ft³) | Grout waste (%) | Grouted cells (in O.C.)",
    "button_text": "Score my calculator",
    "reply_text": "C1 | <CMUs> | <grout, all cells (CY)> | <grout at the spacing (CY)>\nC2 | ...\nC3 | ...",
    "prompt_label_text": "prompt", "reply_label_text": "results",
    "result_text": "**Your calculator ({version}):** {right} of {n} outputs within tolerance of the ground truth.",
    "table_text": "Ground truth vs. your calculator:",
    "columns_text": "Case | Quantity | Ground truth | Your calculator | Unit | Correct",
    "runs_text": "Your calculators so far:",
    "runs_columns_text": "Calculator | Outputs correct",
}

TEXTS: Dict[str, Dict[str, str]] = {
    "drawing": {},
    "one_prompt": {
        "prompt_text": "{problem}\n\n" + FINALS,
        "result_text": "**One prompt:** {right} of {n} quantities within tolerance of the ground truth (5 drawings).",
        "table_text": "Ground truth vs. model output:",
        "columns_text": "Drawing | Question | Quantity | Ground truth | Model output | Unit | Correct",
    },
    "workflow": {
        "read_prompt_text": ("The attached foundation plan shows a CMU foundation wall; the dimensions are to the outside face of the wall. "
                             "List the length of every side of the wall, going around the outside face. If a side is not dimensioned, "
                             "work out its length from the other dimensions.\nEnd your reply with one line, in this format:\n"
                             "SIDES: <length 1>, <length 2>, ... (in feet)"),
        "result_text": "**Workflow (AI model reads, code computes):** {right} of {n} quantities within tolerance of the ground truth (5 drawings).",
        "steps_text": "Step 1 (AI model) and step 2 (code) for each drawing:",
        "steps_columns_text": "Drawing | Sides read by the AI model (ft) | Total wall length (ft) | Ground truth (ft)",
        "table_text": "Ground truth vs. workflow output:",
        "columns_text": "Drawing | Question | Quantity | Ground truth | Workflow output | Unit | Correct",
    },
    "compare_workflow": {
        "title_text": "Quantities within tolerance of the ground truth (15 quantities, 5 drawings)",
        "columns_text": "Method | Correct | Model calls",
        "empty_text": "Run Steps 1b and 1c first.",
    },
    "documents": {},
    "tools": {
        "system_text": AGENT_SYSTEM,
        "search_spec_text": TOOLS["search_spec"],
        "calculator_text": TOOLS["calculator"],
        "lookup_log_text": TOOLS["lookup_log"],
        "result_text": "**Run {run} · tools: {tools}**\n**Correct:** {right} of {n}   **UNKNOWN:** {unknown}   **Wrong:** {wrong}",
        "none_text": "no tools",
        "table_text": "Ground truth vs. agent output (this run):",
        "columns_text": "Question | Ground truth | Agent output | Result | Tools called",
        "runs_text": "All runs so far:",
        "runs_columns_text": "Run | Tools | Correct | UNKNOWN | Wrong",
    },
    "trace": {
        "run_text": "**Run {run}**\ntools: {tools}",
        "question_label_text": "question",
        "call_label_text": "tool call",
        "output_label_text": "tool output",
        "answer_label_text": "answer",
        "empty_text": "No runs yet: run Step 2b first.",
    },
    "lookahead": {
        "prompt_text": ("Use your Python tool to make a look-ahead schedule from the two attached files, schedule.csv and submittal_log.csv.\n"
                        "1. For each activity, the duration in working days = quantity / daily output, rounded up to a whole day.\n"
                        "2. {sequence}\n"
                        "3. Draw a Gantt chart: one horizontal bar per activity, from its start day to its finish day, with working days on the x-axis. "
                        "Color a bar red if the activity places concrete (its description starts with \"Place\") and the mix-design submittal for its element "
                        "in submittal_log.csv is not Approved; color the other bars gray.\n"
                        "4. After the chart, write the schedule as plain text in your reply (not inside the chart image), one line per activity, "
                        "in this format (the last column: YES if the bar is red, otherwise NO):\nA-110 | <start day> | <finish day> | <YES or NO>"),
        "steps_text": ("1. Download {files} with the buttons below and copy the prompt.\n2. Open {url} (sign in with your VT account), start a new chat, "
                       "attach both files, and paste the prompt. Look at the chart and open the code HokieAI wrote.\n" + HOKIE_RESULT_STEPS),
        **BOX,
        "result_text": "**Look-ahead schedule:** {right} of {n} activities with the correct start day, finish day, and flag.",
        "chart_title_text": "Ground truth: the look-ahead schedule (red = the mix design for the element is not approved)",
        "axis_text": "working day",
        "table_text": "Ground truth vs. HokieAI's output:",
        "columns_text": "Activity | Ground truth (start – finish, red) | HokieAI's output | Correct",
    },
    "calculator": CALCULATOR,
    "calculator_skill": {**CALCULATOR, "steps_text": (
        "1. In HokieAI ({url}), start a new chat and turn on the skill of Step 4b (*cmu-foundation-takeoff*) from the chat menu.\n"
        "2. Copy the prompt below (the same as in Step 3b), paste it, and open the new calculator.\n"
        "3. Type in the same test cases, enter the outputs in the box below, and click *Score my calculator*.")},
    "skills": {
        "system_text": AGENT_SYSTEM,
        "calculator_text": TOOLS["calculator"],
        "lookup_log_text": TOOLS["lookup_log"],
        "load_skill_text": TOOLS["load_skill"],
        "skills_list_text": "Available skills (load one with load_skill when its description matches the task):\n- {name}: {description}",
        "card_text": "**The skill** · **name:** {name}\n**description:** {description}\n**body:**",
        "skill_name_text": SKILL["name"],
        "skill_description_text": SKILL["description"],
        "skill_body_text": SKILL["body"],
        "result_text": ("**Run {run} · skill: {skill}**\n**Quantities correct:** {right} of {n}   **Skill loaded for:** {loaded}\n"
                        "**Tool calls:** {calls}   **Model calls:** {model_calls}"),
        "no_skill_text": "not available",
        "none_loaded_text": "no question",
        "table_text": "Ground truth vs. agent output (this run):",
        "columns_text": "Question | Quantity | Ground truth | Agent output | Unit | Correct | Skill loaded",
        "runs_text": "All runs so far:",
        "runs_columns_text": "Run | Skill | Quantities correct | Skill loaded for | Tool calls | Model calls",
    },
    "hokieai_skill": {
        "prompt_text": "{problems}\n\n" + FINALS,
        "steps_text": ("1. In HokieAI ({url}), open **Account → My Skills → Create Skill**. Copy the skill's name, description, and body from Step 4a into the form, "
                       "switch *Always On* off, and click *Create Skill*.\n"
                       "2. Choose *with the skill* or *without the skill* above and run this cell. Start a new chat; for *with the skill*, turn the skill on "
                       "from the chat menu.\n3. Copy the prompt below, paste it into the chat, and send.\n"
                       "4. Copy HokieAI's output with the **copy button** below its reply, paste it into the box below, and click *Score HokieAI output*."),
        **BOX,
        "result_text": "**HokieAI, {skill}:** {right} of {n} quantities within tolerance of the ground truth.",
        "table_text": "Ground truth vs. HokieAI's output:",
        "columns_text": "Question | Quantity | Ground truth | HokieAI's output | Unit | Correct",
    },
    "hokieai_agent": {
        "instructions_text": ("You answer questions about the CEM4644 practice project, a fictional building project, using only the attached files: "
                              "the specification excerpt (spec_03_30_00.txt), the submittal log (submittal_log.csv), and the schedule (schedule.csv). "
                              "Show each value you look up and each calculation. Do not guess a project value: if the files do not contain it, answer UNKNOWN."),
        "create_steps_text": ("1. Download the three project files with the buttons below.\n"
                              "2. In HokieAI ({url}), open **Explore Agents** and click *Create* next to *My Agents*; enter a name, for example *CEM4644 project agent*.\n"
                              "3. Under **Agent Instructions → Specific**, paste the instructions below.\n"
                              "4. Under **Knowledge Sources → Add → Upload Files**, upload the three files.\n5. Click *Create*."),
        "prompt_text": ("Answer the questions below. Reply with one line per question and nothing else, in this format:\nP1 | <answer>\n\n"
                        "Questions:\n{questions}"),
        "steps_text": ("1. Open your agent in HokieAI ({url}) and start a new chat with it.\n2. Copy the prompt below, paste it into the chat, and send.\n"
                       "3. Copy the agent's output with the **copy button** below its reply, and paste it into the box below.\n"
                       "4. Click *Score HokieAI output* to compare it with the ground truth and with the Gemini agent's runs in Step 2b."),
        **BOX,
        "result_text": "**Your HokieAI agent:** {right} of {n} correct, {unknown} UNKNOWN, {wrong} wrong.",
        "table_text": "Ground truth vs. your HokieAI agent's output:",
        "columns_text": "Question | Ground truth | HokieAI agent output | Result",
        "compare_text": "Your HokieAI agent vs. the Gemini agent's runs in Step 2b:",
        "compare_columns_text": "Agent | Correct | UNKNOWN | Wrong",
    },
}


def text(texts: Optional[dict], group: str, key: str) -> str:
    """The notebook's wording if the cell passed one, else the default."""
    v = (texts or {}).get(key)
    return v if v else TEXTS[group][key]


def fill(template: str, **values) -> str:
    try:
        return template.format(**values)
    except (KeyError, IndexError, ValueError) as e:
        return template + f"  [unknown placeholder: {e}]"


def fill_prompt(template: str, **values) -> str:
    """Fill only the known placeholders, so a prompt may contain other braces."""
    for k, v in values.items():
        template = template.replace("{" + k + "}", str(v))
    return template


def say(texts: Optional[dict], group: str, key: str, **values):
    """Show one wording as markdown, one line per line, with the values filled in."""
    from IPython.display import Markdown, display
    s = fill(text(texts, group, key), **values)
    display(Markdown("  \n".join(s.strip().split("\n"))))
