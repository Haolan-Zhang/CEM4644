# MP6 · Instructor guide

MP6 comes in two parts, each with a workshop and a homework notebook: **MP6A · Tables** and **MP6B · Time series**.
Both end with a section in which the students give the same job to a chat model (hokie.ai) and paste its reply back.

## 1. What the students get

| | Workshop | Homework |
|---|---|---|
| **MP6A** table | 1,030 real concrete mixes (lb/yd³) → compressive strength (psi) | 768 simulated building shapes (ft²) → heating load (kBtu/ft²) |
| MP6A steps | 1a look, 1b guess five grades (with a water / cement rule of thumb), 2a classification, 2b regression, 3a importance, 3b what-if, **4a chat, 4b chat + analysis tool** | browse; 1a level classification, 1b pass / fail (pass = heating load at or below the limit), 1c regression; 2a importance, 2b what-if; **3a HokieAI, 3b HokieAI's data-analysis tool**; 4 the tap-test app |
| **MP6B** series | `Hog_office_Marlena` (office), `Bear_education_Lila` (school), `Bear_lodging_Evan` (residence hall), `Bear_assembly_Jose` (assembly hall) | `Hog_office_Gustavo`, `Moose_education_Leland`, `Robin_lodging_Janie`, `Rat_assembly_Rolland` (a swimming pool) |
| MP6B steps | 1a-1b identify the building type, 1c daily and weekly patterns, 2a forecast one week (same hour last week, decision tree, a pretrained model (Chronos-Bolt)), 3a anomalous days (robust z-score), **4a HokieAI forecast, 4b HokieAI's data-analysis tool, 4c HokieAI anomalous days** | the same, plus 5: activity recognition (phyphox; classified with HokieAI's data-analysis tool) and their own series in the upload app |
| Report questions | MP6A 3, MP6B 4 | MP6A 4, MP6B 5 |
| Time | about 75 min each | about 75 min each |

Step 0 clones only this folder and prints one line. MP6A installs `gradio` and loads the table; MP6B also installs
`chronos-forecasting` and (if ticked) downloads Chronos-Bolt-small (about 190 MB). Nothing needs a GPU.

**Style and wording.** The MP6A homework and both MP6B notebooks follow the In-Class activity's style (Title Case step
titles, technical terms, one quoted report paragraph per step, "Data and model sources"). Every result line, caption,
plot title, button and prompt is a wording written in its cell (`name_text = """..."""`; the two apps take a
`labels_text` block of `key: wording` lines), so it can be edited in Colab; `aec_tab/texts.py` (and `LABELS` in
`app.py` / `tap_app.py`) only hold the fallbacks. Column and slider names come from the dataset definitions in
`config.py`, and error messages stay in the code.

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
lower): 4 of the 5 game mixes right, about 64 % of the 206 held-out mixes, against the trees' 84 %.

The concrete table is in US units: the UCI data (kg/m³, MPa) converted to lb/yd³ and psi by `build/prepare_data.py`.

**Concrete, 206 held-out mixes (split seed 4644):** straight line MAE 1,137 psi (R² 0.69); trees **MAE 419 psi (R² 0.94)**.
Importance: age, cement, then slag and water, then superplasticizer; the aggregates and fly ash near zero. What-if on the
median mix at 28 days: water 236 → 388 lb/yd³ takes the prediction from about 6,600 to 5,000 psi; age 3 → 28 → 365 days
gives about 2,600 → 5,300 → 6,800 psi. Grades (< 3,500 / 3,500-6,500 / > 6,500 psi): trees **84 %** (logistic 75 %),
mistakes only between neighboring grades. Pass / fail at 4,000 psi: **95 %**, 8 false passes, 2 false fails, 5 uncertain
mixes; at 6,000 psi 92 %, 6 and 11.

**Energy efficiency, 154 held-out buildings** (US units: areas in ft², height in ft, heating load in kBtu/ft², converted
from the UCI m² and kWh/m²; bands A < 4, B 4-6.5, C 6.5-9.5, D > 9.5): trees MAE 0.11 kBtu/ft² (R² 0.998), straight
line 0.72 (R² 0.91), 97 % in the right band, 100 % at the 6.5 kBtu/ft² threshold. That is the homework's question 1: the loads come from a simulator, so there is no measurement noise and the
same eight inputs fully determine the answer. Compactness and glazing area decide it.

**Forecasts, week of 16-22 October 2017, MAE in kWh per hour** (mean load in brackets):

| building | last week | trees | Chronos-Bolt |
|---|---|---|---|
| Hog_office_Marlena (80) | 3.5 | **2.5** | 4.7 |
| Bear_education_Lila (202) | 10.1 | **9.1** | 9.6 |
| Bear_lodging_Evan (182) | 7.9 | 6.4 | **6.3** |
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

**The chat on the concrete table** (hokie.ai, September 2026, before the switch to US units: the table was then in
kg/m³ and MPa, and the 30 test mixes were balanced on the old grades, so today's 30 differ; psi in brackets, converted.
Trees on those 30: MAE 2.7 MPa (≈ 390 psi), 27 of 30 grades; straight line 8.5 MPa (≈ 1,230 psi)):

| | average error | worst error | right grade |
|---|---|---|---|
| chat on its own, first new chat | 2.9 MPa (≈ 420 psi) | 13.0 MPa | 28 of 30 |
| chat on its own, second new chat | 3.3 MPa (≈ 480 psi) | 15.3 MPa | 26 of 30 |
| chat asked for grades directly | | | 27 of 30 |
| chat + analysis tool (it said HistGradientBoostingRegressor) | 2.7 MPa (≈ 390 psi) | 11.6 MPa | 27 of 30 |

With today's 30 mixes the notebook's references are trees 486 psi (16 of 30 within 10 %) and straight line 1,385 psi
(7 of 30). The chat table now shows average error, worst error and how many predictions fall within 10 % of the
measured strength; the chat's runs are named HokieAI, HokieAI (chat 2), HokieAI + analysis tool, ...

On its own the chat copies the strength of the most similar training mix (its two-decimal answers are exact training
values; the UCI table records some mixes twice with rounded numbers, which also flatters the trees). It is close to
the trees but gives a different number for about half the mixes in a second chat, and misses badly when no training
mix is close. With the analysis tool it fits the same kind of model as the notebook and misses the same mixes.

## 3. Marking notes

MP6A:

1. Regression vs classification: trees beat the line by a factor of nearly three; the largest errors are high-strength
   mixes at unusual ages. Predicting 4,800 psi says how far above 4,000 the mix is; "pass" does not. A false pass is the
   dangerous mistake; the model's probability shows which mixes to send for a test cylinder.
2. The model agrees with the textbook on water, age and cement; a slider beyond the data (water 385+ lb/yd³, age 365) flattens
   or wanders, because the trees cannot extrapolate.
3. The chat on its own is about as good as the trees on concrete and inconsistent between chats; asked how, it should
   describe finding similar mixes. With the analysis tool it runs the notebook's method (same family, same misses),
   which is the answer to "why does it agree". Not yet measured: the other tool models (a straight line should land
   near the notebook's straight line), the chat's column ranking against Step 3a, and the homework table, where the trees
   are near perfect and the test is whether the chat on its own gets anywhere close.
4. (Homework) The tap test (Step 5): students name each recording by material and spot number (`wooden table 2.m4a`);
   the app reads the material from the name with the number dropped. It finds the taps (onsets well above the room's
   loudness; the first 0.15 s, the recorder's start click, is skipped), measures seven numbers per tap, and trains one
   of four models (a small decision tree whose rules are shown, nearest neighbors, logistic regression, gradient-boosted
   trees) on random taps or with whole recordings held out (a material with one recording is left out of that test).
   Measured on three example recordings (an interior wall, a metal stand, a wooden table; one spot each, 10 taps, an
   iPhone's Voice Memos): typical pitch about 190, 800 and 380-500 Hz; every model 100 % on random taps; the decision
   tree splits on the share of sound below 300 Hz (under 17 % metal, over 58 % wall). With one spot per material the
   honest test cannot run, which is the point of asking for 3-4 spots. The iPhone recordings lose everything above
   about 2 kHz, so the measurements use the sound below that.

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
5. (Homework) Own data: a series needs a time column and a value column; activity recognition is described in Part 5.

## 4. Rebuilding

`build/prepare_data.py` reads the downloads in `_candidates/` (gitignored; `LAB_OUTLINE.md` there records how the
buildings were chosen) and writes `data/`. `build/make_notebooks.py` builds the four notebooks and their report
templates (`python build/make_notebooks.py tabular` or `series` for one part) and keeps text edited in the notebooks
unless `--fresh-text` is passed.
