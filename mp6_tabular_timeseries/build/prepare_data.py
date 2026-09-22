"""Instructor-side: the lab's data files from the downloads in _candidates/ (see LAB_OUTLINE.md there).

    python build/prepare_data.py

Writes data/concrete.csv, data/energy_efficiency.csv, data/meters/<building>.csv (2017, hourly kWh + air temperature),
data/meters.json (which building is which) and data/credits.json.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
C = REPO / "_candidates"
D = REPO / "data"

WORKSHOP = ["Hog_office_Marlena", "Bear_education_Lila", "Bear_lodging_Evan", "Bear_assembly_Jose"]
HOMEWORK = ["Hog_office_Gustavo", "Moose_education_Leland", "Robin_lodging_Janie", "Rat_assembly_Rolland"]
USE = {"Office": "office", "Education": "school", "Lodging/residential": "residence hall", "Entertainment/public assembly": "assembly hall",
       "Public services": "public building"}


def tables():
    c = pd.read_excel(C / "concrete/Concrete_Data.xls")
    c.columns = ["cement", "slag", "fly_ash", "water", "superplasticizer", "coarse_agg", "fine_agg", "age_days", "strength_MPa"]
    c = c.round(2); c.to_csv(D / "concrete.csv", index=False)
    e = pd.read_excel(C / "energy/ENB2012_data.xlsx")
    e.columns = ["compactness", "surface_area", "wall_area", "roof_area", "height", "orientation", "glazing_area",
                 "glazing_distribution", "heating_load", "cooling_load"]
    e.to_csv(D / "energy_efficiency.csv", index=False)
    print("tables:", c.shape, e.shape)


def meters():
    meta = pd.read_csv(C / "metadata.csv").set_index("building_id")
    el = pd.read_csv(C / "electricity_cleaned.csv", parse_dates=["timestamp"]).set_index("timestamp").loc["2017"]
    wx = pd.read_csv(C / "weather.csv", parse_dates=["timestamp"])
    wx = wx[wx.timestamp.dt.year == 2017]
    (D / "meters").mkdir(exist_ok=True)
    info = {}
    for group, ids in (("workshop", WORKSHOP), ("homework", HOMEWORK)):
        for b in ids:
            m = meta.loc[b]
            s = el[b].interpolate(limit=6).bfill().ffill().round(2)
            t = wx[wx.site_id == m.site_id].set_index("timestamp").airTemperature.reindex(s.index).interpolate(limit=12).bfill().ffill().round(1)
            df = pd.DataFrame({"timestamp": s.index, "kWh": s.values, "air_temp_C": t.values})
            df.to_csv(D / "meters" / f"{b}.csv", index=False)
            info[b] = {"set": group, "use": USE.get(m.primaryspaceusage, m.primaryspaceusage), "sub_use": str(m.sub_primaryspaceusage),
                       "sqm": round(float(m.sqm)), "site": m.site_id, "timezone": m.timezone, "year_built": None if pd.isna(m.yearbuilt) else int(m.yearbuilt),
                       "mean_kWh": round(float(s.mean()), 1), "missing_hours": int(el[b].isna().sum())}
            print(f"{group:8} {b:26} {info[b]['use']:15} {info[b]['sqm']:>7,} m2  mean {info[b]['mean_kWh']:>6} kWh  missing {info[b]['missing_hours']}")
    json.dump(info, open(D / "meters.json", "w"), indent=1)


def credits():
    json.dump({
        "concrete": {"title": "Concrete Compressive Strength", "author": "I-Cheng Yeh (1998), UCI Machine Learning Repository",
                     "license": "CC BY 4.0", "source": "https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength"},
        "energy_efficiency": {"title": "Energy Efficiency", "author": "A. Tsanas and A. Xifara (2012), UCI Machine Learning Repository",
                              "license": "CC BY 4.0", "source": "https://archive.ics.uci.edu/dataset/242/energy+efficiency"},
        "meters": {"title": "Building Data Genome Project 2 (hourly electricity meters and site weather, 2017)",
                   "author": "Miller et al. (2020), Scientific Data 7:368", "license": "MIT",
                   "source": "https://github.com/buds-lab/building-data-genome-project-2"},
        "forecaster": {"title": "Chronos-Bolt (small)", "author": "Amazon Science", "license": "Apache-2.0",
                       "source": "https://huggingface.co/amazon/chronos-bolt-small"},
    }, open(D / "credits.json", "w"), indent=1)


if __name__ == "__main__":
    tables(); meters(); credits(); print("done")
