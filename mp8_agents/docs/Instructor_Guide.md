# MP8 · Instructor guide

One in-class notebook, `MP8_Workshop_AI_Agents.ipynb` (and a homework, section 5), about 90–120 minutes with every step; Part 3 (code execution) and
Part 5 are the natural pieces to move to a homework. Parts 1, 2 and 4a call Gemini (`gemini-3.5-flash-lite`) through
the students' own API key from MP5; every default setting is stored in `data/cache/`, so it runs at once without a
key. Part 3, Steps 4b-4c and Part 5 use HokieAI in the browser (paste-back).

## 1. What the students get

| Part | Steps | What it shows |
|---|---|---|
| 1 · AI Workflows | 1a drawings, 1b one prompt, 1c workflow, 1d compare | A task split into fixed steps: the AI model reads the drawing, code applies the estimating conventions |
| 2 · AI Agents | 2a documents, 2b run with chosen tools, 2c trace | An AI model that decides which tool to call; the student ticks the tools; correct / UNKNOWN / wrong (guessed) per run |
| 3 · Code Execution | 3a look-ahead Gantt chart (HokieAI's Python tool), 3b takeoff calculator (artifact) | The agent writes and runs its own code; each result is something to see and is scored against the ground truth |
| 4 · Skills | 4a agent with / without the skill, 4b the same skill in HokieAI, 4c the calculator rebuilt with the skill | Instructions with a name and a description that the agent loads only when the task matches |
| 5 · Your Own HokieAI Agent | 5a create, 5b ask and score | Instructions + knowledge sources (the three project files), compared with the Gemini agent |

**Part 3 ground truth** (`build/generate_project.py`: `lookahead_key.json`, `calculator_cases.json`):
- 3a: duration = quantity / daily output rounded up (3, 3, 6, 5, 3, 3 working days); the sequence is given in the prompt
  (A-110 → A-120 → A-210 → A-310 → A-410, A-510 after A-210): A-110 1–3, A-120 4–6, A-210 7–12, A-310 13–17,
  A-410 18–20, A-510 13–15. Red (mix design not approved): A-210 (revise and resubmit), A-410 and A-510 (under
  review). Scored per activity on start, finish and flag; the notebook draws the correct chart next to the student's.
  If you edit the sequence in the prompt, the ground truth no longer matches it.
- 3b / 4c: C1 = T1 (840 CMU, 5.24 CY, 1.31 CY), C2 = T2 (1,224, 8.23, 2.06), C3 = foundation plan 4 (783, 7.60, 3.80);
  tolerances 1 % (CMUs) and 2 % (grout). The prompt does not give the conventions, so the expected failure without the
  skill is the partial grout (as in Part 1); with the skill it should pass. Students type the cases into the artifact
  and enter its outputs; the table keeps the two calculators side by side.
- 3a tried in HokieAI (instructor, October 2026): the chart and all six lines exactly right. The prompt asks for the
  lines as plain text in the reply, because HokieAI otherwise prints them inside the chart image. 3b not yet tried.

**The practice project** (`build/generate_project.py`, fictional): a concrete specification table with
project-specific limits (for example maximum w/cm 0.48 for foundation walls, 0.42 for exterior slabs), a submittal log
of five mix designs (batch weights, air, slump, status) and a schedule of six activities (quantity, daily output). The
six questions: Q1 a specification value; Q2 working days (log + arithmetic); Q3 w/cm of a submittal against its
element (log + arithmetic + specification); Q4 footing concrete volume (arithmetic only); Q5 a submittal status; Q6
slump against its element (log + specification). Q1-Q3 are the workshop's Step 2b, Q4-Q6 the homework's (they were
P1, P4, P5, P2, P3, P6 before October 2026). Ground truth in `data/project/questions.json`.

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

**Part 2 · the questions of Step 2b.** The cell's `questions = ["Q1", "Q2", "Q3"]` (under *Show code*; the default is
`DEFAULT_QUESTIONS` in `aec_agent/lab.py`) picks the questions the agent answers; Step 2c and the HokieAI agent of Part 5
use the same ones, and any subset runs from the stored outputs. Q1 needs the specification (with only the logs the agent
guesses 0.50), Q2 the schedule (with only the calculator it guesses 12 days), Q3 all three tools. Q4, Q5 and Q6 are
for the homework. On Q1, Q2 and Q3:

| Tools | Correct | UNKNOWN | Wrong (guessed) |
|---|---|---|---|
| none | 0 | 3 | 0 |
| lookup_log | 1 | 1 | 1 |
| calculator | 0 | 2 | 1 |
| calculator, lookup_log | 1 | 1 | 1 |
| search_spec | 1 | 2 | 0 |
| search_spec, lookup_log | **3** | 0 | 0 |
| search_spec, calculator | 1 | 2 | 0 |
| all three | **3** | 0 | 0 |

**Part 2 · tools, all six questions** (6 questions per run):

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
barely matters: the model does this arithmetic (Q4, Q2, the w/cm of Q3) correctly by itself, and with all three tools
it uses the calculator anyway (13 tool calls instead of 9). The guesses: with only the logs, Q1 is answered "0.50"
(a typical w/cm, not this project's 0.48); with only the calculator, Q2 is answered "12 days". Without tools the
model once asked for a tool that does not exist; the trace shows the notebook's reply ("There is no tool called ...").

**Part 4 · skill** (T1 rectangle, T2 L-shape, 3 quantities each; Q5 as the question the skill does not apply to):

| Skill | Quantities correct | Skill loaded for |
|---|---|---|
| not available | 3 of 6 (both partial-grout quantities, and T2's all-cell grout) | — |
| cmu-foundation-takeoff | 6 of 6 | T1, T2 (not Q5) |

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
   (Q1 = 0.50 with only the logs). A guess is the more dangerous failure; checking the trace (which tool, which
   output) is how to verify an answer.
3. Code execution: a good answer shows the chart and the calculator, says which number was wrong and finds
   the line of code or the formula behind it (often the partial grout, a rounding of the duration, or a finish-to-start
   off by one day), and concludes that code that runs is not code that is right: test it on cases with a known answer.
4. The skill fixes the partial grout and is loaded only for the takeoff questions. With a vague description the agent
   may not load it: the description is what the agent reads to decide. Skills students suggest: company estimating
   rules, a report format, a submittal-review checklist.
5. The HokieAI agent (gpt-5) with the three files usually answers all six; differences come from the instructions
   (UNKNOWN vs. a guess) and from reading the CSV files. Not measured from here: students' HokieAI outputs (Steps 3b, 4b).

## 4. Running it

- Students paste their Gemini key in Step 0 (or keep it as the Colab secret `GEMINI_API_KEY`). Without a key, every
  default runs from the stored outputs; an edited prompt, tool description, system prompt or skill prints that it
  needs the key.
- The free tier allows a few requests per minute; the client waits and retries. A live agent run takes 10–60 s.
- After changing a default wording that the model reads (a prompt, a tool description, the system prompt, the skill),
  rerun `build/precompute.py` with a key so the class does not need one.

## 5. The homework (`MP8_Homework_AI_Agents.ipynb`)

| Part | Steps | Taken from the workshop |
|---|---|---|
| 1 · Tools: Code Execution and Tool Calling | 1a look-ahead Gantt chart, 1b documents, 1c agent with chosen tools on Q4-Q6, 1d trace | Step 3a; Part 2 with the homework questions |
| 2 · AI Agents: Your Own HokieAI Agent | 2a create, 2b ask all six questions (Q1-Q6) | Part 5 |
| 3 · Skills: A Look-Ahead Schedule Skill | 3a write the skill, 3b use it on project A, 3c reuse it on project B | new |

**Part 1 stored outputs (Q4, Q5, Q6).** Q4 (footing volume) is answered correctly by every run, even without tools.
Q5 needs `lookup_log`, and Q6 needs `lookup_log` and `search_spec`. The calculator adds nothing. With a missing tool
the agent always answered UNKNOWN; no run guessed.

| Tools | Correct | UNKNOWN | Wrong |
|---|---|---|---|
| none, calculator, search_spec, search_spec + calculator | 1 | 2 | 0 |
| lookup_log, calculator + lookup_log | 2 | 1 | 0 |
| search_spec + lookup_log, all three | **3** | 0 | 0 |

**Part 2.** Step 2b compares the HokieAI agent with the Gemini runs of Step 1c on the questions both answered (Q4-Q6);
its score on all six is reported separately.

**Part 3 (open-ended).** Step 3a's starter skill (`LOOKAHEAD_SKILL` in `aec_agent/texts.py`) holds the required rules
(duration, sequence, flag rule, chart, the `activity | start | finish | YES/NO` lines). Students add at least three
conventions of their own. The requests of Steps 3b and 3c carry only the project's files and sequence. Scoring reads
only the text lines, so a student's extra tables that name the activities (an approvals list, for example) do not
confuse it, but it does not check the chart. Project B (`project_b_schedule.csv`, `project_b_submittal_log.csv`,
`lookahead_key_b.json`): B-100 1-3, B-110 4-8, B-120 9-13 red (under review), B-210 14-18, B-220 14-17 red (revise and
resubmit), B-310 19-20 red, because its element has **no submittal** in the log (the case the Step 1a prompt never
meets). Example answer (local, not in the repository): `MP8_Homework_Answer_Sheet_Part3.md`. With gpt-5 through the
HokieAI API, an example skill scored 6 of 6 on both projects. It followed all five conventions, but version 1 drew
every bar one day long and the legend hid a bar. Two added conventions fixed both.
