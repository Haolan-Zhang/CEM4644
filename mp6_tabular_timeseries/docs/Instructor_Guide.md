# MP6 · Instructor guide

MP6 comes in two parts, each with a workshop and a homework notebook: **MP6A · Tables** and **MP6B · Time series**.
Both end with a section in which the students give the same job to a chat model (hokie.ai) and paste its reply back.

## 1. What the students get

| | Workshop | Homework |
|---|---|---|
| **MP6A** table | 1,030 real concrete mixes → compressive strength (MPa) | 768 simulated building shapes → heating load (kWh/m²) |
| MP6A steps | 1a look, 1b guess five grades (with a water / cement rule of thumb), 2a classification, 2b regression, 3a importance, 3b what-if, **4a chat, 4b chat + analysis tool** | the same without 1b, plus 5: the tap test and their own table |
| **MP6B** series | `Hog_office_Marlena` (office), `Bear_education_Lila` (school), `Bear_lodging_Evan` (residence hall), `Bear_assembly_Jose` (assembly hall) | `Hog_office_Gustavo`, `Moose_education_Leland`, `Robin_lodging_Janie`, `Rat_assembly_Rolland` (a swimming pool) |
| MP6B steps | 1a-1b which building is which, 1c anatomy, 2a forecast three ways, 3a odd days, **4a chat forecast, 4b chat + analysis tool, 4c chat odd days**, 5 own series | the same |
| Report questions | MP6A 3, MP6B 5 | MP6A 4, MP6B 5 (own data is the main deliverable) |
| Time | about 75 min each | about 75 min each |

Step 0 clones only this folder and prints one line. MP6A installs `gradio` and loads the table; MP6B also installs
`chronos-forecasting` and (if ticked) downloads Chronos-Bolt-small (about 190 MB). Nothing needs a GPU.

**The chat steps.** Each shows download buttons for the files, a link to hokie.ai, the prompt to copy, a box for the
reply and a *Score* button (the MP5 pattern). The files are made from the lab's own data when the cell runs:
`<table>_train.csv` (the 80 % training split, with the answer) and `<table>_test.csv` (30 of the held-out rows, balanced
across the grades, ids T01-T30, no answer), the same for every student; for a building, `<id>_last_4_weeks.csv`
(hourly kWh and temperature, 18 September to 15 October 2017), `<id>_next_week_temperature.csv` and
`<id>_daily_2017.csv`. A form option puts the data inside the prompt instead, for when attaching fails. The reply is
read loosely (one line or many, tables, fences, extra words); a short reply is scored on the rows or hours it gave.
Every reply scored stays in the step's table, so two new chats with the same prompt can be compared; the analysis-tool
steps let the student pick the model the chat should train (gradient-boosted trees, a random forest, a straight line,
a small neural network).

## 2. Measured numbers (the answer keys)

**Step 1b rule of thumb** (water / cement below 0.5 high, 0.5-1.0 normal, above 1.0 low; 7 days or younger one grade
lower): 4 of the 5 game mixes right, about 57 % of the 206 held-out mixes, against the trees' 83 %.

**Concrete, 206 held-out mixes (split seed 4644):** straight line MAE 7.8 MPa (R² 0.69); trees **MAE 2.8 MPa (R² 0.94)**.
Importance: age, cement, then slag and water, then superplasticizer; the aggregates and fly ash near zero. What-if on the
median mix at 28 days: water 140 → 230 kg/m³ takes the prediction from about 48 to 35 MPa; age 3 → 28 → 365 days gives
about 18 → 38 → 49 MPa. Grades (< 25 / 25-45 / > 45): trees **83 %** (logistic 75 %), mistakes only between neighbouring
grades. Pass / fail at 30 MPa: **94 %**, 7 false passes, 5 false fails, 6 uncertain mixes; at 40 MPa 90 %, 10 and 10.

**Energy efficiency, 154 held-out buildings:** trees MAE 0.35 (R² 0.998), 98 % in the right band, 100 % at the 20 kWh/m²
threshold. That is the homework's question 1: the loads come from a simulator, so there is no measurement noise and the
same eight inputs fully determine the answer. Compactness and glazing area decide it.

**Forecasts, week of 16-22 October 2017, MAE in kWh per hour** (mean load in brackets):

| building | last week | trees | Chronos-Bolt |
|---|---|---|---|
| Hog_office_Marlena (80) | 3.5 | **2.5** | 4.7 |
| Bear_education_Lila (202) | 10.1 | **9.1** | 9.6 |
| Bear_lodging_Evan (182) | 7.9 | 6.6 | **6.3** |
| Bear_assembly_Jose (274) | 20.6 | **17.5** | 21.9 |

Chronos-Bolt's 80 % band covers 66-92 % of the hours.

**Homework buildings, the same week, MAE in kWh per hour and as a share of the mean load:**

| building | last week | trees | Chronos-Bolt |
|---|---|---|---|
| Hog_office_Gustavo (732) | 14.9 (2 %) | 15.2 (2 %) | **12.8 (2 %)** |
| Moose_education_Leland (999) | 60.4 (6 %) | 47.2 (5 %) | **28.3 (3 %)** |
| Robin_lodging_Janie (85) | 9.4 (11 %) | 9.6 (11 %) | **8.8 (10 %)** |
| Rat_assembly_Rolland, a swimming pool (37) | 6.5 (17 %) | 6.9 (18 %) | **5.8 (16 %)** |

The pool is the answer to homework question 4: whichever method, its error is a sixth of its load, against a fiftieth
for the big office, because a small pool building has no weekly rhythm to speak of (pumps and heating run on demand).
Here the pretrained model wins on all four buildings, most clearly on the large school.

**Odd days** (office, threshold 3.5): 25 days flagged, 9 on public holidays (New Year, MLK, Memorial, 3-4 July, Labor Day,
Thanksgiving, Christmas), plus a Saturday spike on 22 April, a dip on 8 August and a three-day dip in mid September.

**The chat on the concrete table** (hokie.ai, September 2026, the 30 test mixes; trees on the same 30: MAE 2.7 MPa,
27 of 30 grades; straight line 8.5 MPa):

| | average miss | worst miss | right grade |
|---|---|---|---|
| chat on its own, first new chat | 2.9 MPa | 13.0 | 28 of 30 |
| chat on its own, second new chat | 3.3 MPa | 15.3 | 26 of 30 |
| chat asked for grades directly | | | 27 of 30 |
| chat + analysis tool (it said HistGradientBoostingRegressor) | 2.7 MPa | 11.6 | 27 of 30 |

On its own the chat copies the strength of the most similar training mix (its two-decimal answers are exact training
values; the UCI table records some mixes twice with rounded numbers, which also flatters the trees). It is close to
the trees but gives a different number for about half the mixes in a second chat, and misses badly when no training
mix is close. With the analysis tool it fits the same kind of model as the notebook and misses the same mixes.

## 3. Marking notes

MP6A:

1. Regression vs classification: trees beat the line by a factor of nearly three; the worst misses are high-strength
   mixes at unusual ages. Predicting 33 MPa says how far above 30 the mix is; "pass" does not. A false pass is the
   dangerous mistake; the model's probability shows which mixes to send for a test cube.
2. The model agrees with the textbook on water, age and cement; a slider beyond the data (water 230+, age 365) flattens
   or wanders, because the trees cannot extrapolate.
3. The chat on its own is about as good as the trees on concrete and inconsistent between chats; asked how, it should
   describe finding similar mixes. With the analysis tool it runs the notebook's method (same family, same misses),
   which is the answer to "why does it agree". Not yet measured: the other tool models (a straight line should land
   near the notebook's 8.5 MPa), the chat's column ranking against Step 3a, and the homework table, where the trees
   are near perfect and the test is whether the chat on its own gets anywhere close.
4. (Homework) Own data: any table with a numeric answer column and 30+ rows works; the tap test is described in Part 5.

MP6B:

1. The office has flat weekends and a summer that looks like the rest of the year; the school has the flattest profile;
   the residence hall has an evening peak and empties in December; the assembly hall is the noisiest.
2. Last week is hard to beat on regular buildings; trees win on three of four; the pretrained model is close with no
   training at all, and its band is the honest answer to "how sure are you".
3. Holidays explain about a third of the flags; the rest need a question to the building. A threshold of 3 to 4.
4. Not yet measured with hokie.ai. The chat sees only four weeks (the notebook's methods see the year), and 168 values
   is a long reply: look for a stopped or drifting list (the notebook says how many hours it read). For odd days,
   compare its list with the rule's flags and the holiday column; a plausible reason for a day the rule does not flag
   (a heat wave, an event) still needs checking, which is the point of the question.
5. Own data: a series needs a time column and a value column.

## 4. Rebuilding

`build/prepare_data.py` reads the downloads in `_candidates/` (gitignored; `LAB_OUTLINE.md` there records how the
buildings were chosen) and writes `data/`. `build/make_notebooks.py` builds the four notebooks and their report
templates (`python build/make_notebooks.py tabular` or `series` for one part) and keeps text edited in the notebooks
unless `--fresh-text` is passed.
