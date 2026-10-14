# MP8 · Instructor guide

One in-class notebook, `MP8_Workshop_AI_Agents.ipynb`, about 75–90 minutes. Parts 1–3 call Gemini
(`gemini-3.5-flash-lite`) through the students' own API key from MP5; every default setting is stored in
`data/cache/`, so it runs at once without a key. Parts 3b and 4 use HokieAI in the browser.

## 1. What the students get

| Part | Steps | What it shows |
|---|---|---|
| 1 · AI Workflows | 1a drawings, 1b one prompt, 1c workflow, 1d compare | A task split into fixed steps: the AI model reads the drawing, code applies the estimating conventions |
| 2 · AI Agents | 2a documents, 2b run with chosen tools, 2c trace | An AI model that decides which tool to call; the student ticks the tools; correct / UNKNOWN / wrong (guessed) per run |
| 3 · Skills | 3a agent with / without the skill, 3b the same skill in HokieAI | Instructions with a name and a description that the agent loads only when the task matches |
| 4 · Your Own HokieAI Agent | 4a create, 4b ask and score | Instructions + knowledge sources (the three project files), compared with the Gemini agent |

**The practice project** (`build/generate_project.py`, fictional): a concrete specification table with
project-specific limits (for example maximum w/cm 0.48 for foundation walls, 0.42 for exterior slabs), a submittal log
of five mix designs (batch weights, air, slump, status) and a schedule of six activities (quantity, daily output). The
six questions: P1 a specification value; P2 footing concrete volume (arithmetic only); P3 a submittal status; P4
working days (log + arithmetic); P5 w/cm of a submittal against its element (log + arithmetic + specification); P6
slump against its element (log + specification). Ground truth in `data/project/questions.json`.

**The tools** (`aec_agent/tools.py`): `search_spec` returns the three specification lines that share the most words
with the query (plus the table header); `lookup_log` returns the log rows containing the search text; `calculator`
evaluates arithmetic exactly (ceil, floor, round, sqrt, min, max); `load_skill` returns a skill's body. The model only
asks for a tool; the notebook runs it and returns its output. An agent run stops after 15 rounds of tool calls, then
one last request with the tools switched off forces an answer.

## 2. Measured numbers (the stored outputs)

**Part 1 · one prompt vs. workflow** (5 drawings, 15 quantities):

| Method | Correct | Notes |
|---|---|---|
| One prompt | 11 of 15 (8 of 15 in an earlier run) | Misses the partial grout on plans 1 and 5 (6.13 CY instead of 3.04; 8.35 CY, the all-cell value, instead of 4.17) and both grout quantities on plan 3; the earlier run missed every partial grout and all of plan 4 (its undimensioned side) |
| Workflow | 15 of 15 (both runs) | Gemini read every side right, including the two undimensioned ones; code did the rest |

**Part 2 · tools** (6 questions per run):

| Tools | Correct | UNKNOWN | Wrong (guessed) | Tool calls |
|---|---|---|---|---|
| none | 1 | 5 | 0 | 5 |
| lookup_log | 3 | 2 | 1 | 20 |
| calculator | 1 | 4 | 1 | 13 |
| calculator, lookup_log | 3 | 2 | 1 | 16 |
| search_spec | 2 | 4 | 0 | 22 |
| search_spec, lookup_log | **6** | 0 | 0 | 9 |
| search_spec, calculator | 2 | 4 | 0 | 23 |
| all three | **6** | 0 | 0 | 13 |

The documents decide the score: only the runs with both `search_spec` and `lookup_log` answer all six. The calculator
barely matters: the model does this arithmetic (P2, P4, the w/cm of P5) correctly by itself, and with all three tools
it uses the calculator anyway (13 tool calls instead of 9). The guesses: with only the logs, P1 is answered "0.50"
(a typical w/cm, not this project's 0.48); with only the calculator, P4 is answered "12 days". Without tools the
model once asked for a tool that does not exist; the trace shows the notebook's reply ("There is no tool called ...").

**Part 3 · skill** (T1 rectangle, T2 L-shape, 3 quantities each; P3 as the question the skill does not apply to):

| Skill | Quantities correct | Skill loaded for |
|---|---|---|
| not available | 3 of 6 (both partial-grout quantities, and T2's all-cell grout) | — |
| cmu-foundation-takeoff | 6 of 6 | T1, T2 (not P3) |

A vague description (for example *Useful notes.*) is not stored: it runs live with a key, and is the students'
experiment.

Gemini outputs vary from run to run; the stored outputs are one run each, and a live run (an edited setting) may
differ. That is worth saying in class: Part 1's one prompt gave 8 and 11 of 15 in two runs.

## 3. Marking notes

1. One prompt fails on the conventions (8/S of the cells) and on deriving a side; the workflow gives the model only the
   reading and leaves the rules to code, which never misapplies them. Good answers put reading and judgment with the
   model, fixed rules and arithmetic with code or a spreadsheet, and keep a person checking the model's reading.
2. Each document tool is needed by the questions that use its document; the calculator turned out not to be needed
   for these questions (a fair finding). With missing tools the agent mostly said UNKNOWN, but it also guessed
   (P1 = 0.50 with only the logs). A guess is the more dangerous failure; checking the trace (which tool, which
   output) is how to verify an answer.
3. The skill fixes the partial grout and is loaded only for the takeoff questions. With a vague description the agent
   may not load it: the description is what the agent reads to decide. Skills students suggest: company estimating
   rules, a report format, a submittal-review checklist.
4. The HokieAI agent (gpt-5) with the three files usually answers all six; differences come from the instructions
   (UNKNOWN vs. a guess) and from reading the CSV files. Not measured from here: students' HokieAI outputs (Steps 3b, 4b).

## 4. Running it

- Students paste their Gemini key in Step 0 (or keep it as the Colab secret `GEMINI_API_KEY`). Without a key, every
  default runs from the stored outputs; an edited prompt, tool description, system prompt or skill prints that it
  needs the key.
- The free tier allows a few requests per minute; the client waits and retries. A live agent run takes 10–60 s.
- After changing a default wording that the model reads (a prompt, a tool description, the system prompt, the skill),
  rerun `build/precompute.py` with a key so the class does not need one.
