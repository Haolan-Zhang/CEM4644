# MP6 · Tables and time series for construction

Teaching material for **CEM4644**: the same things the earlier labs did with photos and drawings, done with the two
kinds of data most construction work actually produces. A **table** (one row per concrete mix or per building) is used
to predict a number (regression) and a class (classification); a **time series** (a building's electricity, hour by
hour) is forecast a week ahead three ways and searched for days that do not fit. Every answer is scored against what
really happened. Students need **no programming**: every cell is a Colab form and the code is hidden. No GPU needed.

| Notebook | Open | Data |
|---|---|---|
| `MP6_Workshop_Tabular_TimeSeries.ipynb` | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Haolan-Zhang/CEM4644/blob/master/mp6_tabular_timeseries/MP6_Workshop_Tabular_TimeSeries.ipynb) | 1,030 concrete mixes and their strength; four campus buildings' electricity in 2017 |
| `MP6_Homework_Tabular_TimeSeries.ipynb` | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Haolan-Zhang/CEM4644/blob/master/mp6_tabular_timeseries/MP6_Homework_Tabular_TimeSeries.ipynb) | 768 simulated building shapes and their heating load; four other buildings' electricity |

## What the students do

1. **A table**: look at it, guess five answers yourself (workshop).
2. **Two questions, one table**: regression (a straight line vs gradient-boosted trees, scored on rows the model never
   saw) and classification (strength grades, or pass / fail against a specification the student sets; false passes
   counted separately).
3. **What the model learned**: which columns matter, and what-if sliders on a mix of their choosing.
4. **A time series**: four buildings unlabelled, a week and a year each: which is the office, the school, the residence
   hall, the assembly hall? Then the anatomy of one building's year.
5. **Next week**: same hour last week, trees on the past weeks + calendar + temperature, and Chronos-Bolt (a pretrained
   forecasting model used zero-shot, with an uncertainty band), each scored on the week of 16 October 2017.
6. **The odd days**: each day against the building's usual pattern, a threshold slider, the US calendar next to the flags.
7. **Their own CSV** in a small app opened from a link: a table gets trees and a score, a series gets a forecast.

## What is in this folder

```
MP6_Workshop_Tabular_TimeSeries.ipynb   student notebook (generated)
MP6_Homework_Tabular_TimeSeries.ipynb   student notebook (generated)
aec_tab/          the hidden code: config (what each notebook works on), data, models (regression / classification),
                  series (forecasts, odd days), ui (what each step shows), app (the upload app), lab (the one object)
data/             concrete.csv, energy_efficiency.csv, meters/<building>.csv (2017, hourly kWh + air temperature),
                  meters.json, credits.json  (2.3 MB in total)
build/            instructor-side scripts: prepare_data.py (from the raw downloads), make_notebooks.py
docs/             report templates, instructor guide
```

**Editing the text.** The notebooks are where the wording lives. Edit any markdown cell or any cell's `#@markdown`
notes in Colab (*Show code* on a form cell), then *File → Save a copy in GitHub* (clear the outputs first). A later
`make_notebooks.py` run keeps every markdown cell and every note it finds in the existing notebook and rebuilds only
the code; `--fresh-text` starts again from the generator's defaults. Step 0 prints only the ✅ line.

## Rebuilding (instructors only)

```
python build/prepare_data.py       # needs the raw downloads in _candidates/ (see _candidates/LAB_OUTLINE.md)
python build/make_notebooks.py     # both notebooks + report templates
```

## Sources and licences

| Item | Source | Licence |
|---|---|---|
| Concrete compressive strength | I-Cheng Yeh (1998), [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength) | CC BY 4.0 |
| Energy efficiency | A. Tsanas and A. Xifara (2012), [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/242/energy+efficiency) | CC BY 4.0 |
| Building electricity meters and site weather (2017) | [Building Data Genome Project 2](https://github.com/buds-lab/building-data-genome-project-2), Miller et al. (2020) | MIT |
| Chronos-Bolt (small) | [Amazon Science](https://huggingface.co/amazon/chronos-bolt-small), loaded in Step 0 | Apache-2.0 |
| Models | scikit-learn (linear / logistic regression, gradient-boosted trees) | BSD-3 |
