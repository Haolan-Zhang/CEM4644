"""Default wordings of every prompt, instruction and result the MP7 notebook shows.

Each notebook cell carries its own copy (name_text = \"\"\"...\"\"\", editable in Colab); these are only the fallbacks.
Words in {braces} are filled in when the cell runs; **bold** and *italic* work; each line is shown as its own line.
"""
from typing import Dict, Optional

CHAT_URL = "https://hokie.ai.vt.edu/"

PASTE_STEPS = ("1. Copy the prompt below.\n2. Open {url} (sign in with your VT account), start a new chat, and paste the prompt.\n"
               "3. Copy HokieAI's output and paste it into the box below.\n4. Click *Score HokieAI output* to compare it with the ground truth.")
ATTACH_STEPS = ("1. Download {files} with the button below and copy the prompt.\n2. Open {url} (sign in with your VT account), start a new chat, "
                "attach the image, and paste the prompt.\n3. Copy HokieAI's output and paste it into the box below.\n"
                "4. Click *Score HokieAI output* to compare it with the ground truth.")
BOX = {"button_text": "Score HokieAI output", "reply_text": "Paste HokieAI's full output here.", "prompt_label_text": "prompt", "reply_label_text": "output"}

DEFINITIONS = """Definitions (OSHA construction Focus Four, as coded in OSHA's injury records):
- Fall: the worker fell to a lower level (from a ladder, roof, scaffold, through an opening), fell on the same level, or jumped down.
- Struck-by: the worker was hit by a moving object or equipment: a falling, flying, swinging, or rolling object, a vehicle, or a tool or object that slipped or swung while the worker was holding it (for example, a saw kickback or a slipping knife).
- Caught-in/between: the worker or a body part was caught in, squeezed, compressed, pinched, or crushed between objects or in running machinery, including trench, structure, or material collapses.
- Electrocution: the worker made contact with electrical energy (power lines, energized wiring or equipment).
- Other: anything else (heat, chemicals, fire or explosion, overexertion, struck against a stationary object, slips or trips without a fall)."""
INCIDENT_TASK = ("Classify each construction incident report below into one category: Fall, Struck-by, Caught-in/between, Electrocution, or Other.\n"
                 "Reply with one line per report and nothing else, in this format:\nN01, Fall\nN02, Other\n\nIncident reports:\n{reports}")
STEPS = ("First list every dimension and given value you read from the drawing and the text. Then solve each question step by step: "
         "state the formula, substitute the numbers, and compute. Check that each result is reasonable before giving it.")
DOOR_FORMAT = ("Reply with one line per door and nothing else, in this format:\n"
               "mark | door type | width | height | hardware set | location\n1 | DF | 3'-0\" | 7'-0\" | 32 | ENTRY")

TEXTS: Dict[str, Dict[str, str]] = {
    "incidents": {
        "v1_prompt_text": INCIDENT_TASK,
        "v2_prompt_text": "You are a construction safety analyst coding OSHA severe-injury reports.\n" + DEFINITIONS + "\n\n" + INCIDENT_TASK,
        "v3_prompt_text": ("You are a construction safety analyst coding OSHA severe-injury reports.\n" + DEFINITIONS +
                           "\n\nWorked examples:\n{examples}\n\n" + INCIDENT_TASK),
        "steps_text": PASTE_STEPS,
        **BOX,
        "accuracy_text": "**{version}:** {right} of {n} reports classified as OSHA classified them (**{accuracy} %**).",
        "matrix_text": "Confusion matrix (rows: OSHA's classification; columns: HokieAI's output):",
        "disagree_text": "Reports where HokieAI's output differs from OSHA's classification:",
        "missing_text": "HokieAI's output has {got} of the {n} reports; the missing ones count as wrong.",
    },
    "takeoff": {
        "v1_prompt_text": "{problem}\n\n{finals}",
        "v2_prompt_text": "{problem}\n\n" + STEPS + "\n{finals}",
        "v3_prompt_text": "{problem}\n\n{conventions}\n" + STEPS + "\n{finals}",
        "finals_text": "End your reply with one final line per question, in this format:\n{lines}",
        "steps_text": ATTACH_STEPS,
        **BOX,
        "result_text": "**{version}, {drawing}:** {right} of {n} quantities within tolerance of the ground truth.",
        "table_text": "Ground truth vs. HokieAI's output:",
    },
    "schedule": {
        "v1_prompt_text": "The attached drawing sheet contains a door schedule. List every door of UNIT TYPE {unit}.\n{format}",
        "v2_prompt_text": "The attached image is the part of a door schedule for UNIT TYPE {unit}. List every door.\n{format}",
        "v3_prompt_text": ("The attached image is the part of a door schedule for UNIT TYPE {unit}. The schedule's columns are, left to right: "
                           "MARK, DOOR TYPE, DESCRIPTION, WIDTH, HEIGHT, FRAME, FIRE RATING, HDWR SET, LOCATION. Transcribe every row exactly as "
                           "written; if a value is not legible, write ?.\n{format}"),
        "format_text": DOOR_FORMAT,
        "steps_text": ATTACH_STEPS,
        **BOX,
        "result_text": "**{version}, {sheet}:** {right} of {n} doors transcribed exactly ({fields_right} of {fields} fields right).",
        "table_text": "Ground truth vs. HokieAI's output:",
    },
    "submittals": {
        "v1_prompt_text": "Review the submittal below against the specification excerpt. List every noncompliance.\n{final}\n\nSpecification:\n{spec}\nSubmittal:\n{submittal}",
        "v2_prompt_text": ("You are a project engineer reviewing a submittal. First list every requirement in the specification excerpt that applies to the "
                           "submittal's intended use. Then compare the submittal with each requirement, one by one, and decide whether it complies.\n{final}"
                           "\n\nSpecification:\n{spec}\nSubmittal:\n{submittal}"),
        "v3_prompt_text": ("You are a project engineer reviewing a submittal. First list every requirement in the specification excerpt that applies to the "
                           "submittal's intended use. Then compare the submittal with each requirement, one by one, in a table with the columns: "
                           "requirement | required value | submitted value | complies (YES or NO). Compute any value the submittal does not state "
                           "directly (for example, w/cm = water / total cementitious materials).\n{final}\n\nSpecification:\n{spec}\nSubmittal:\n{submittal}"),
        "final_text": ("End your reply with one line per noncompliance, in this format:\nNONCOMPLIANT | <requirement> | <submitted value> | <required value>\n"
                       "If the submittal complies with every requirement, end with: NONCOMPLIANT | none"),
        "steps_text": PASTE_STEPS,
        **BOX,
        "result_text": "**{version}, {submittal}:** {right} of {n} requirements judged correctly; noncompliances found {found} of {planted}, false flags {false}.",
        "table_text": "Ground truth vs. HokieAI's output (each requirement the submittal is checked against):",
    },
    "contract": {
        "v1_prompt_text": "Answer the questions about the construction contract clauses below.\nReply with one line per question: Q01 | answer\n\nQuestions:\n{questions}\n\nContract clauses:\n{clauses}",
        "v2_prompt_text": ("Answer the questions about the construction contract clauses below. For each question, quote the sentence of the contract that "
                           "supports your answer.\nReply with one line per question: Q01 | answer | exact quote\n\nQuestions:\n{questions}\n\nContract clauses:\n{clauses}"),
        "v3_prompt_text": ("You are a contract administrator. Answer each question using only the contract clauses below; do not use outside knowledge.\n"
                           "If a question describes a situation, state the requirement and apply it to the facts (dates, amounts, quantities).\n"
                           "Reply with one line per question: Q01 | short answer | FAR clause number | exact quote of the supporting sentence, copied word for word\n"
                           "If the clauses do not answer the question, reply: Q01 | NOT IN TEXT | - | -\n\nQuestions:\n{questions}\n\nContract clauses:\n{clauses}"),
        "steps_text": PASTE_STEPS,
        **BOX,
        "result_text": ("**{version}:** {right} of {n} questions answered correctly; {made_up} of the {unanswerable} questions the clauses do not answer were "
                        "given an answer anyway; {verified} of {quotes} quotes found word for word in the clauses."),
        "table_text": "Ground truth vs. HokieAI's output:",
    },
    "compare": {
        "title_text": "Share of correct outputs by prompt version",
        "table_text": "All scored outputs so far:",
        "empty_text": "No scored outputs yet: run the step above first.",
        "axis_text": "correct (%)",
    },
}

# The one-prompt notebook (MP7_Workshop_Construction_Documents): which prompt version each part gives, and the result
# wordings without a version.
ONE_PROMPT = {"incidents": "v2", "submittals": "v1", "contract": "v2"}
ONE_PROMPT_TEXTS = {
    "incidents": {"accuracy_text": "**Accuracy:** {right} of {n} reports classified as OSHA classified them (**{accuracy} %**)."},
    "submittals": {"result_text": "**{submittal}:** {right} of {n} requirements judged correctly; noncompliances found {found} of {planted}, false flags {false}."},
    "contract": {"result_text": ("**Contract questions:** {right} of {n} questions answered correctly; {made_up} of the {unanswerable} questions the clauses "
                                 "do not answer were given an answer anyway; {verified} of {quotes} quotes found word for word in the clauses.")},
}

FOUNDATION_CONVENTIONS = ("Estimating conventions: total wall length = sum of the outside dimensions (no deduction at corners); wall area = length x height; "
                          "one CMU covers 8 in x 16 in = 0.889 ft²; round CMUs up after adding waste. Grout = CMUs before waste x grout per CMU, "
                          "converted to CY (27 ft³ = 1 CY), then add grout waste. Cells are 8 in apart, so grouting at S in on center fills 8/S of the cells.")
ROOF_CONVENTIONS = ("Estimating conventions: run = half the eave-to-eave distance; rise = run x slope; rake length = sqrt(run² + rise²); "
                    "net roof area = rake length x eave length x 2 roof planes. Underlayment area = net area + ridge lap (eave length x 1 ft) + "
                    "rake laps (4/12 ft x rake length x 4 rake edges); rolls = area / (roll length x width), rounded up. Starter course = eave length x "
                    "exposure x 2 eaves; ridge coverage = eave length x 1 ft; rake waste = 4 rake edges x rake length x 0.25 ft²/ft; gross area = net + "
                    "starter + ridge + rake waste; shingles = gross area / (3 ft x exposure), rounded up.")


def text(texts: Optional[dict], group: str, key: str) -> str:
    """The notebook's wording if the cell passed one, else the default."""
    v = (texts or {}).get(key)
    return v if v else TEXTS[group][key]


def fill(template: str, **values) -> str:
    try:
        return template.format(**values)
    except (KeyError, IndexError, ValueError) as e:
        return template + f"  [unknown placeholder: {e}]"


def say(texts: Optional[dict], group: str, key: str, **values):
    """Show one wording as markdown, one line per line, with the values filled in."""
    from IPython.display import Markdown, display
    s = fill(text(texts, group, key), **values)
    display(Markdown("  \n".join(s.strip().split("\n"))))


def fill_prompt(template: str, **values) -> str:
    """Fill only the known placeholders, so a prompt may contain other braces (a JSON example, say)."""
    for k, v in values.items():
        template = template.replace("{" + k + "}", str(v))
    return template
