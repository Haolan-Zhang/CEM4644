# MP6 · Tables and time series for construction

Teaching material for **CEM4644**: the same things the earlier labs did with photos and drawings, done with the two
kinds of data most construction work actually produces, in two parts. **MP6A · Tables** (one row per concrete mix or
per building) predicts a number (regression) and a class (classification); **MP6B · Time series** (a building's
electricity, hour by hour) forecasts a week ahead three ways and searches for days that do not fit. Both parts end by
giving the same job to a chat model (hokie.ai) and scoring its pasted reply next to the notebook's models. Every answer
is scored against what really happened. Students need **no programming**: every cell is a Colab form and the code is
hidden. No GPU needed.

| Notebook | Open | Data |
|---|---|---|
| `MP6A_Workshop_Tabular.ipynb` | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Haolan-Zhang/CEM4644/blob/master/mp6_tabular_timeseries/MP6A_Workshop_Tabular.ipynb) | 1,030 concrete mixes and their strength |
| `MP6A_Homework_Tabular.ipynb` | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Haolan-Zhang/CEM4644/blob/master/mp6_tabular_timeseries/MP6A_Homework_Tabular.ipynb) | 768 simulated building shapes and their heating load |
| `MP6B_Workshop_TimeSeries.ipynb` | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Haolan-Zhang/CEM4644/blob/master/mp6_tabular_timeseries/MP6B_Workshop_TimeSeries.ipynb) | four campus buildings' electricity in 2017 |
| `MP6B_Homework_TimeSeries.ipynb` | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Haolan-Zhang/CEM4644/blob/master/mp6_tabular_timeseries/MP6B_Homework_TimeSeries.ipynb) | four other buildings' electricity in 2017 |

## What the students do

**MP6A · Tables**

1. **A table**: look at it, then put five mixes in their strength grade yourself, with a water / cement rule of thumb
   (workshop).
2. **Two questions, one table**: classification first (strength grades, the game they just played, or pass / fail
   against a specification the student sets; false passes counted separately), then regression (a straight line vs
   gradient-boosted trees, scored on rows the model never saw).
3. **What the model learned**: which columns matter, and what-if sliders on a mix of their choosing.
4. **The same job, by a chat model**: the training rows and 30 held-out rows go to hokie.ai, first with a plain
   prompt (twice, in two new chats, to see whether the numbers stay the same), then with the chat told to train a
   model of the student's choice with its data-analysis tool; each reply is scored next to the trees and the line.
5. **Their own table** (homework): the tap test (phone recordings of taps on different surfaces, one row per tap) or
   any CSV, in a small app opened from a link: trees and a score on held-out rows.

**MP6B · Time series**

1. **A time series**: four buildings unlabelled, a week and a year each: which is the office, the school, the residence
   hall, the assembly hall? Then the anatomy of one building's year.
2. **Next week**: same hour last week, trees on the past weeks + calendar + temperature, and Chronos-Bolt (a pretrained
   forecasting model used zero-shot, with an uncertainty band), each scored on the week of 16 October 2017.
3. **The odd days**: each day against the building's usual pattern, a threshold slider, the US calendar next to the flags.
4. **The same jobs, by a chat model**: four weeks of a building's meter and next week's temperature go to hokie.ai for
   a 168-hour forecast (on its own, then with its analysis tool), and the year's daily totals for the odd days; each is
   scored next to the notebook's methods.
5. **Their own time series** (homework): activity recognition (the phone's accelerometer in a pocket, one row per
   2-second window) or any CSV with a time column, in the same app: a forecast of its last period.

## What is in this folder

```
MP6A_Workshop_Tabular.ipynb, MP6A_Homework_Tabular.ipynb         student notebooks, tables (generated)
MP6B_Workshop_TimeSeries.ipynb, MP6B_Homework_TimeSeries.ipynb   student notebooks, time series (generated)
aec_tab/          the hidden code: config (what each notebook works on), data, models (regression / classification),
                  series (forecasts, odd days), ui (what each step shows), chat (the hokie.ai steps: files, prompts,
                  reading and scoring the pasted replies), app (the upload app), lab (the one object)
data/             concrete.csv, energy_efficiency.csv, meters/<building>.csv (2017, hourly kWh + air temperature),
                  meters.json, credits.json  (2.3 MB in total)
build/            instructor-side scripts: prepare_data.py (from the raw downloads), make_notebooks.py
docs/             report templates, instructor guide
```

**Editing the text.** The notebooks are where the wording lives. Edit any markdown cell, any cell's `#@markdown`
notes, or a wording written in a cell (`result_text = """..."""` in MP6A Step 2b, `anatomy_text` in MP6B Step 1c, `steps_text` in the chat steps) in Colab (*Show code* on
a form cell), then *File → Save a copy in GitHub* (clear the outputs first). A later `make_notebooks.py` run keeps
every markdown cell, note and result wording it finds in the existing notebook and rebuilds only the code;
`--fresh-text` starts again from the generator's defaults. Step 0 prints only the ✅ line.

## Rebuilding (instructors only)

```
python build/prepare_data.py       # needs the raw downloads in _candidates/ (see _candidates/LAB_OUTLINE.md)
python build/make_notebooks.py     # the four notebooks + report templates (add "tabular" or "series" for one part)
```

## Sources and licences

| Item | Source | Licence |
|---|---|---|
| Concrete compressive strength | I-Cheng Yeh (1998), [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength) | CC BY 4.0 |
| Energy efficiency | A. Tsanas and A. Xifara (2012), [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/242/energy+efficiency) | CC BY 4.0 |
| Building electricity meters and site weather (2017) | [Building Data Genome Project 2](https://github.com/buds-lab/building-data-genome-project-2), Miller et al. (2020) | MIT |
| Chronos-Bolt (small) | [Amazon Science](https://huggingface.co/amazon/chronos-bolt-small), loaded in Step 0 | Apache-2.0 |
| Models | scikit-learn (linear / logistic regression, gradient-boosted trees) | BSD-3 |
| Chat model | hokie.ai (Virginia Tech's access to GPT models), used by the students in their browser | — |
