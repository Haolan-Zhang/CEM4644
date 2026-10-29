# MP8 · AI workflows and AI agents for a construction project

Teaching material for **CEM4644**. Students take off quantities from foundation plans with one prompt and as a fixed
workflow (the AI model reads the drawing, code computes), then let an AI agent answer questions about a practice
project by calling tools they choose (a specification search, a calculator, the project logs), give the agent a
skill (estimating conventions it loads when a task needs them), and build an agent of their own in HokieAI. Every
output is scored against the ground truth. No programming and no GPU needed.

| Notebook | Open | Data |
|---|---|---|
| `MP8_Workshop_AI_Agents.ipynb` (in class, two sessions) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Haolan-Zhang/CEM4644/blob/master/mp8_agents/MP8_Workshop_AI_Agents.ipynb) | 5 foundation plans; a practice project (specification excerpt, submittal log, schedule) with 6 questions; 2 takeoff questions in words; all made for the course |
| `MP8_Homework_AI_Agents.ipynb` (homework) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Haolan-Zhang/CEM4644/blob/master/mp8_agents/MP8_Homework_AI_Agents.ipynb) | the same practice project (questions Q4-Q6, then all six); the student's own schedule in Part 3 if they like |

## What the students do

1. **AI workflows** (Gemini, vision): five foundation plans, three quantities each (CMUs, grout in all cells, grout at
   a bar spacing). One prompt, in which the model reads and computes everything, vs. a workflow of two fixed steps: the
   model reads the length of every side, code adds them and applies the estimating conventions.
2. **AI agents and their tools** (Gemini, function calling): six questions about the practice project, answered by an
   agent that may call the tools the student ticks: `search_spec` (the specification), `calculator` (exact
   arithmetic), `lookup_log` (the submittal log and the schedule). Each tool is needed by at least two questions; the
   runs table counts correct, UNKNOWN (the agent said it could not find the value) and wrong (it guessed) answers,
   tool calls and time; a trace shows every tool call with its inputs and output.
3. **Code execution** (HokieAI): the agent writes and runs its own code, and the result is something to see: a
   look-ahead Gantt chart that marks the concrete placements whose mix design is not approved, and a takeoff
   calculator built as an artifact and tested on three cases. Each is scored against the ground truth.
4. **Skills** (Gemini, then HokieAI): the estimating conventions written as a skill (name, description, body) that the
   agent loads with `load_skill` only when the description matches the task; two takeoff questions and one project
   question, with and without the skill, and with a vague description. Then the same skill created in HokieAI
   (*Account → My Skills*) and the takeoff questions sent without and with it, and the Part 3 calculator rebuilt with
   the skill turned on.
5. **Your own HokieAI agent**: instructions plus the three project files as knowledge sources (*Explore Agents →
   Create*), asked the six questions of Part 2; scored and compared with the Gemini agent's runs.

**The homework** (no Gemini key needed; Step 1c runs from the stored outputs):
1. **Tools:** the look-ahead Gantt chart (code execution), then the Gemini agent on questions Q4-Q6 with the tools the
   student chooses.
2. **Agents:** the student's own HokieAI agent asked all six questions.
3. **Skills:** an open-ended schedule skill, not scored. Students save the rules of a scheduling task like Step 1a's,
   of their own choice and with a Gantt chart in its output, as a HokieAI skill. They test it on a schedule, change the
   schedule once, check numbers by hand and revise the skill, and report with screenshots.

## How the AI model runs

Parts 1, 2 and 4a call Gemini (`gemini-3.5-flash-lite`, the free tier) with the student's own key from MP5 (typed in Step 0
or the Colab secret `GEMINI_API_KEY`). Every default setting (Part 1, all eight tool combinations of Part 2, Step 4a
with and without the skill) is stored in `data/cache/`, so it runs at once without a key and without the free tier's
rate limits; a key is needed only for an edited prompt, tool description, system prompt or skill. Part 3, Steps 4b-4c and Part 5 use
HokieAI in the browser (paste-back, as in MP6 and MP7).

## What is in this folder

```
MP8_Workshop_AI_Agents.ipynb   in-class notebook (generated)
MP8_Homework_AI_Agents.ipynb   homework notebook (generated)
aec_agent/       the hidden code: texts (every default prompt, tool description, skill and result wording), client
                 (Gemini requests, the agent loop, the cache), tools (search_spec, calculator, lookup_log, load_skill),
                 score (reading and scoring outputs), ui (tables, traces, the HokieAI box), lab (the steps)
data/            project/ (spec_03_30_00.txt, submittal_log.csv, schedule.csv, questions.json, takeoff_questions.json);
                 lookahead_key.json, calculator_cases.json;
                 drawings/ (5 foundation plans + problems.json); cache/ (stored Gemini outputs)
figures/         the concept diagrams of the markdown cells (workflow, agent loop, code execution, prompt vs. skill; the
                 homework's code execution and schedule skill)
build/           generate_project.py (the project and the drawings), make_figures.py (the diagrams), precompute.py (fills
                 the cache), make_notebooks.py
docs/            report template, instructor guide
```

**Editing the text.** Every prompt, tool description, skill and result line is written in its cell
(`name_text = """..."""`, under *Show code*), as are the notes and markdown cells. Edit them in Colab, then *File →
Save a copy in GitHub*; a later `make_notebooks.py` run keeps them (`--fresh-text` starts again from the defaults in
`aec_agent/texts.py`). An edited prompt, tool description or skill is a new setting: it runs live (with a key), or
rerun `precompute.py` after changing a default.

## Rebuilding (instructors only)

```
python build/generate_project.py                       # the project files, the questions and the drawings
python build/make_figures.py                           # the diagrams (shown in the notebook from GitHub)
GEMINI_API_KEY=... python build/precompute.py          # the stored outputs of every default setting
python build/make_notebooks.py [workshop|homework]     # the notebooks and their report templates
```

## Sources and licenses

| Item | Source | License |
|---|---|---|
| Practice project and foundation plans | made for CEM4644 (fictional, not for construction); the specification excerpt follows the format of UFGS 03 30 00 | course material; UFGS is public domain |
| AI model | Gemini (`gemini-3.5-flash-lite`), Google, via the Gemini API with the student's key | Gemini API terms |
| Generative AI | GPT models via HokieAI (Virginia Tech), used by the students in their browser | — |
