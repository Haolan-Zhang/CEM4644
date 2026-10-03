# MP7 · Prompt engineering for construction documents and drawings

Teaching material for **CEM4644**. Students use generative AI (HokieAI, Virginia Tech's access to GPT models) on
construction text and drawings, and benchmark how the wording of a prompt changes the accuracy of its outputs. The
notebook gives the prompts; students send them to HokieAI, paste the output back, and the notebook scores it against
an answer key and compares the prompt versions. No programming and no GPU needed.

| Notebook | Open | Data |
|---|---|---|
| `MP7_Workshop_Prompt_Engineering.ipynb` | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Haolan-Zhang/CEM4644/blob/master/mp7_prompt_engineering/MP7_Workshop_Prompt_Engineering.ipynb) | 30 OSHA construction incident reports; 12 practice submittals against UFGS excerpts; 29 FAR construction clauses; 15 practice drawings made for the course |

## What the students do

1. **Classifying incident reports** (text): 30 construction reports from OSHA's Severe Injury Reports, classified into
   OSHA's Focus Four (fall, struck-by, caught-in/between, electrocution) or other, with three prompt versions (task only;
   role and definitions; definitions and worked examples) and their own; scored against OSHA's classification.
2. **Reviewing submittals** (text): 12 practice submittals (6 concrete mix designs, 6 mortar and grout mixes) checked
   against specification excerpts adapted from UFGS 03 30 00 and 04 20 00. Each states its intended use rather than the
   specification's portion of the structure; some values must be computed or sit exactly at a limit. Prompt versions task
   only, requirements first then compare (task decomposition), and a comparison table; scored on six requirements per
   submittal against the planted noncompliances.
3. **Answering questions from a contract** (text): 27 questions on 29 FAR construction clauses, ten of them project
   situations to check against a clause (notice deadlines, retainage, warranty periods), seven not answerable from the
   text; prompt versions questions only, answers with quotes, and only-from-the-text with quotes and NOT IN TEXT;
   quotes checked word for word.
4. **Quantity takeoff from drawings** (vision): five foundation plans (CMUs, grout) and five gable roofs (area,
   underlayment, shingles); prompt versions task only, step-by-step reasoning, and estimating conventions.
5. **Reading schedules on a drawing sheet** (vision): five 36 × 24 in door-schedule sheets; whole sheet, cropped
   schedule, cropped schedule with column names.

Each part ends with a comparison of the prompt versions over every output the student scored.

## What is in this folder

```
MP7_Workshop_Prompt_Engineering.ipynb   student notebook (generated)
aec_prompt/      the hidden code: texts (every default wording and prompt), lab (the steps), score (reading and
                 scoring HokieAI's output), ui (the HokieAI box, tables, comparison chart)
data/            incidents.csv and incident_examples.csv (OSHA, public domain); submittals/ (spec excerpts + submittals);
                 contract/ (FAR clauses + questions); practice/ (drawings + problems.json)
build/           prepare_incidents.py (from OSHA's download), generate_submittals.py, prepare_contract.py (from acquisition.gov),
                 generate_practice.py (the drawings), make_notebooks.py
docs/            report template, instructor guide
```

**Editing the text.** Every prompt, instruction and result line is written in its cell (`name_text = """..."""`,
under *Show code*), as are the notes and markdown cells. Edit them in Colab, then *File → Save a copy in GitHub*; a
later `make_notebooks.py` run keeps them (`--fresh-text` starts again from the defaults in `aec_prompt/texts.py`).

## Rebuilding (instructors only)

```
python build/prepare_incidents.py <OSHA January2015toNovember2025.csv>   # data/incidents.csv
python build/generate_submittals.py                                       # data/submittals/ (seed 4644)
python build/prepare_contract.py                                          # data/contract/ (fetches the FAR clauses)
python build/generate_practice.py                                         # data/practice/ (seed 4644; --seed for a new set)
python build/make_notebooks.py                                            # the notebook + report template
```

## Sources and licences

| Item | Source | Licence |
|---|---|---|
| Incident reports | [OSHA Severe Injury Reports](https://www.osha.gov/severe-injury-reports), U.S. Department of Labor | Public domain |
| Specification excerpts | Adapted from [UFGS](https://www.wbdg.org/dod/ufgs) 03 30 00 and 04 20 00, U.S. Department of Defense | Public domain |
| Contract clauses | [Federal Acquisition Regulation](https://www.acquisition.gov/far), Part 52 | Public domain |
| Practice submittals, drawings and problems | Made for CEM4644 by `build/generate_submittals.py` and `build/generate_practice.py` | Course material |
| Generative AI | HokieAI (Virginia Tech), GPT models | — |
