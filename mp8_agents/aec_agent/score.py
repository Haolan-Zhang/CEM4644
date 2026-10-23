"""Reading a model's output and scoring it against the ground truth."""
import math
import re
from typing import Dict, List, Optional

import pandas as pd

NUM = r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?"


def _num(s: str) -> Optional[float]:
    m = re.search(NUM, s or "")
    return float(m.group(0).replace(",", "")) if m else None


def final_lines(text: str) -> List[str]:
    """The FINAL lines of a reply (the whole last line if there are none)."""
    lines = [l.strip().strip("*").strip() for l in (text or "").split("\n") if l.strip()]
    fin = [re.sub(r"^\**\s*FINAL\s*\**\s*[:|]\s*", "", l, flags=re.I) for l in lines if re.match(r"^\**\s*FINAL\b", l, re.I)]
    return fin or lines[-1:]


NO_ANSWER = "(no answer)"


def final_answer(text: str) -> str:
    return " / ".join(final_lines(text)) or NO_ANSWER


def judge(answer: str, q: dict) -> str:
    """correct, wrong or UNKNOWN (the agent said it could not find the value, or gave no answer)."""
    a = answer or ""
    if not a.strip() or a.strip() == NO_ANSWER:
        return "UNKNOWN"                                     # no answer at all: the agent did not guess
    if re.search(r"\bunknown", a, re.I) and not re.search(r"\b(yes|no)\b", a, re.I) and _num(a) is None:
        return "UNKNOWN"
    if q["kind"] == "number":
        nums = [float(m.replace(",", "")) for m in re.findall(NUM, a)]          # any number of the answer ("212 / 48 = 4.42, so 5 days")
        ok = any(abs(v - q["value"]) <= q["tolerance"] * max(abs(q["value"]), 1e-9) + 1e-9 for v in nums)
    elif q["kind"] == "yesno":
        m = re.search(r"\b(yes|no)\b", a, re.I)
        ok = bool(m) and m.group(1).upper() == q["value"]
    else:
        ok = any(re.search(p, a, re.I) for p in q["accept"])
    return "correct" if ok else "wrong"


def truth_text(q: dict) -> str:
    if q["kind"] == "number":
        return f"{q['value']:g} {q.get('unit', '')}".strip()
    return str(q["value"])


def quantities(text: str, prefix: str = "") -> Dict[str, float]:
    """'Q1 = 832', 'FINAL | Q1 | 832 | CMU', 'T1 Q2: 5.2 CY' ... -> {"Q1": 832.0} (with prefix: only lines naming it)."""
    out = {}
    for line in (text or "").split("\n"):
        if prefix and not re.search(rf"\b{prefix}\b", line):
            continue
        for q, v in re.findall(rf"\b(Q\d)\b[^0-9\n-]*?({NUM})", line):
            out[q] = float(v.replace(",", ""))                   # the last mention wins (the final lines come last)
    return out


def takeoff(text: str, key: List[dict], prefix: str = "") -> pd.DataFrame:
    got = quantities(text, prefix)
    rows = []
    for k in key:
        v = got.get(k["q"])
        ok = v is not None and abs(v - k["value"]) <= k["tolerance"] * abs(k["value"]) + 1e-9
        rows.append({"question": k["q"], "quantity": k["quantity"], "ground truth": f"{k['value']:g}", "output": "—" if v is None else f"{v:g}",
                     "unit": k["unit"], "correct": "✓" if ok else "✗"})
    return pd.DataFrame(rows)


FT_IN = re.compile(r"(\d+(?:\.\d+)?)\s*(?:'|ft)\s*-?\s*(?:(\d+(?:\.\d+)?)\s*(?:\"|in))?")


def sides(text: str) -> List[float]:
    """The side lengths of the SIDES line, in feet (44, 44.5, 44'-6", 44 ft 6 in)."""
    line = next((l for l in reversed((text or "").split("\n")) if re.search(r"\bSIDES\b", l, re.I)), "")
    body = re.split(r"SIDES\s*\**\s*[:|]?", line, flags=re.I)[-1]
    out = []
    for part in re.split(r"[,;]", body):
        m = FT_IN.search(part)
        if m:
            out.append(float(m.group(1)) + (float(m.group(2)) / 12 if m.group(2) else 0)); continue
        v = _num(part)
        if v is not None:
            out.append(v)
    return out


CMU_FACE = 8 * 16 / 144


def compute(length: float, p: dict) -> Dict[str, float]:
    """The workflow's code step: the estimating conventions applied to the wall length."""
    cmu = length * p["height"] / CMU_FACE
    grout_all = cmu * p["grout_per_cmu"] / 27 * (1 + p["grout_waste"] / 100)
    return {"Q1": math.ceil(cmu * (1 + p["waste"] / 100) - 1e-9), "Q2": round(grout_all, 3), "Q3": round(grout_all * 8 / p["spacing"], 3)}


def answers_by_id(text: str, ids: List[str]) -> Dict[str, str]:
    """'P1 | 0.48' lines -> {"P1": "0.48"} (the HokieAI agent's output)."""
    out = {}
    for line in (text or "").split("\n"):
        m = re.match(r"^\W*(P\d)\W*[|:.\-–]\s*(.+)$", line.strip())
        if m and m.group(1) in ids:
            out[m.group(1)] = m.group(2).strip().strip("|").strip()
    return out


# ----------------------------------------------------------------------------- Part 3: code execution
def lookahead(text: str, key: List[dict]) -> pd.DataFrame:
    """'A-110 | 1 | 3 | NO' lines (tables, bold and extra words are fine) against the look-ahead ground truth."""
    got = {}
    for line in (text or "").split("\n"):
        m = re.search(r"\b(A-\d{3})\b(.*)", line)
        if not m:
            continue
        nums = re.findall(r"(?<![\d.])\d+(?![\d.])", m.group(2))
        yn = re.findall(r"\b(YES|NO)\b", m.group(2), re.I)              # "red: NO" is NO: the YES / NO decides
        colour = re.findall(r"\b(red|gray|grey)\b", m.group(2), re.I)
        if len(nums) >= 2:
            f = yn[-1].upper() if yn else ("YES" if colour and colour[-1].lower() == "red" else ("NO" if colour else None))
            got[m.group(1)] = (int(nums[0]), int(nums[1]), f)            # the last line per activity wins
    rows = []
    for k in key:
        g = got.get(k["activity"])
        ok = g is not None and g == (k["start"], k["finish"], k["flag"])
        rows.append({"activity": f"{k['activity']} {k['description']}", "truth": f"{k['start']} – {k['finish']}, {k['flag']}",
                     "output": "—" if g is None else f"{g[0]} – {g[1]}, {g[2] or '?'}", "correct": "✓" if ok else "✗"})
    return pd.DataFrame(rows)


def calculator(text: str, cases: List[dict]) -> pd.DataFrame:
    """'C1 | 840 | 5.24 | 1.31' lines (CMUs, grout all cells, grout at the spacing) against each case's ground truth."""
    got = {}
    for line in (text or "").split("\n"):
        m = re.search(r"\b(C\d)\b(.*)", line)
        if m:
            got[m.group(1)] = [float(v.replace(",", "")) for v in re.findall(NUM, m.group(2))]
    rows = []
    for c in cases:
        vals = got.get(c["id"], [])
        for i, k in enumerate(c["key"]):
            v = vals[i] if i < len(vals) else None
            ok = v is not None and abs(v - k["value"]) <= k["tolerance"] * abs(k["value"]) + 1e-9
            rows.append({"case": c["id"], "quantity": k["quantity"].replace(f"{c['spacing']} in O.C.", "at the spacing"), "truth": f"{k['value']:g}",
                         "output": "—" if v is None else f"{v:g}", "unit": k["unit"], "correct": "✓" if ok else "✗"})
    return pd.DataFrame(rows)
