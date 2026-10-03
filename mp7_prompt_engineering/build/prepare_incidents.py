"""The incident reports for MP7 Part 1, from OSHA's Severe Injury Reports (public domain, US Department of Labor).

    python build/prepare_incidents.py <path to January2015toNovember2025.csv>

Download: https://www.osha.gov/severe-injury-reports ("Download the full SIR data set"; a scripted download needs a
browser User-Agent). Construction records (NAICS 23), 2023 onward, 120-600 characters; six reports per Focus Four
class and six "other", plus one worked example per class for the few-shot prompt. The class comes from OSHA's own OIICS
event title of each record.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CATS = ["Fall", "Struck-by", "Caught-in/between", "Electrocution", "Other"]
OUT = Path(__file__).resolve().parents[1] / "data"


def focus_four(title: str) -> str:
    t = str(title).lower().strip()
    if "electric" in t:
        return "Electrocution"
    if t.startswith(("fall", "other fall", "jump", "other jump")) or "fall to lower level" in t or "fall on same level" in t:
        return "Fall"
    if t.startswith(("caught", "compressed", "pinched")) or "collaps" in t or "caught in" in t:
        return "Caught-in/between"
    if t.startswith("struck by") or "struck by" in t or "held by injured worker" in t or "held or wielded" in t:
        return "Struck-by"
    if ("fall" in t and "falling" not in t) or "jump to lower level" in t:
        return "Fall"
    return "Other"


def main(src):
    df = pd.read_csv(src, encoding="latin-1", low_memory=False)
    df = df[df["Primary NAICS"].astype(str).str.startswith("23")].copy()
    df["year"] = pd.to_datetime(df.EventDate, errors="coerce").dt.year
    df["focus_four"] = df.EventTitle.map(focus_four)
    df["report"] = df["Final Narrative"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df = df[(df.year >= 2023) & df.report.str.len().between(120, 600)]
    rng = np.random.default_rng(4644)
    test, ex = [], []
    for c in CATS:
        idx = rng.permutation(df.index[df.focus_four == c].to_numpy())
        test += list(idx[:6]); ex.append(idx[6])
    test = list(rng.permutation(test))
    t = df.loc[test].reset_index(drop=True)
    t.insert(0, "id", [f"N{i + 1:02d}" for i in range(len(t))])
    cols = {"EventTitle": "osha_event", "EventDate": "event_date"}
    t[["id", "report", "focus_four", "EventTitle", "EventDate"]].rename(columns=cols).to_csv(OUT / "incidents.csv", index=False)
    e = df.loc[ex].reset_index(drop=True)
    e[["report", "focus_four", "EventTitle"]].rename(columns=cols).to_csv(OUT / "incident_examples.csv", index=False)
    print(f"{len(t)} reports, {len(e)} worked examples -> {OUT}")
    print(t.focus_four.value_counts().to_dict())


if __name__ == "__main__":
    main(sys.argv[1])
