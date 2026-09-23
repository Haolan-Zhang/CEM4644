# MP6 · Instructor guide

## 1. What the students get

| | Workshop | Homework |
|---|---|---|
| Table | 1,030 real concrete mixes → compressive strength (MPa) | 768 simulated building shapes → heating load (kWh/m²) |
| Series | `Hog_office_Marlena` (office), `Bear_education_Lila` (school), `Bear_lodging_Evan` (residence hall), `Bear_assembly_Jose` (assembly hall) | `Hog_office_Gustavo`, `Moose_education_Leland`, `Robin_lodging_Janie`, `Rat_assembly_Rolland` (a swimming pool) |
| Guess game | yes (Step 1b: pick the strength grade of five mixes; classification comes first in Part 2 because a class is easier to guess and to judge than a number) | no |
| Report questions | 6 | 6 (own data is the main deliverable) |
| Time | about 90 min | about 90 min |

Step 0 clones only this folder, installs `gradio` and `chronos-forecasting` quietly, loads the tables and meters and
(if ticked) downloads Chronos-Bolt-small (about 190 MB) and prints one line. Nothing needs a GPU.

## 2. Measured numbers (the answer keys)

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

## 3. Marking notes

1. Regression vs classification: trees beat the line by a factor of nearly three; the worst misses are high-strength
   mixes at unusual ages. Predicting 33 MPa says how far above 30 the mix is; "pass" does not. A false pass is the
   dangerous mistake; the model's probability shows which mixes to send for a test cube.
2. The model agrees with the textbook on water, age and cement; a slider beyond the data (water 230+, age 365) flattens
   or wanders, because the trees cannot extrapolate.
3. The office has flat weekends and a summer that looks like the rest of the year; the school has the flattest profile;
   the residence hall has an evening peak and empties in December; the assembly hall is the noisiest.
4. Last week is hard to beat on regular buildings; trees win on three of four; the pretrained model is close with no
   training at all, and its band is the honest answer to "how sure are you".
5. Holidays explain about a third of the flags; the rest need a question to the building. A threshold of 3 to 4.
6. Own data: any table with a numeric answer column and 30+ rows works; a series needs a time column.

## 4. Rebuilding

`build/prepare_data.py` reads the downloads in `_candidates/` (gitignored; `LAB_OUTLINE.md` there records how the
buildings were chosen) and writes `data/`. `build/make_notebooks.py` keeps text edited in the notebooks unless
`--fresh-text` is passed.
