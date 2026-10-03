"""Reading HokieAI's pasted output and comparing it with the answer keys (loosely: extra words, tables and fences are fine)."""
import re
from typing import Dict, List, Tuple

import pandas as pd

CATS = ["Fall", "Struck-by", "Caught-in/between", "Electrocution", "Other"]


def _cat(word: str) -> str:
    w = word.lower()
    for c, keys in (("Caught-in/between", ("caught",)), ("Struck-by", ("struck",)), ("Electrocution", ("electro", "electric")),
                    ("Fall", ("fall",)), ("Other", ("other",))):
        if any(k in w for k in keys):
            return c
    return word.strip()


def incidents(reply: str, key: pd.DataFrame) -> dict:
    """'N01, Fall' lines → accuracy, per-class counts, a confusion matrix and the disagreements."""
    got = {}
    for rid, label in re.findall(r"\b(N\d\d)\b\**\s*[,:|\-–]\s*\**\s*([A-Za-z][A-Za-z/\- ]*)", reply):
        got.setdefault(rid, _cat(label))
    truth = dict(zip(key.id, key.focus_four))
    right = sum(got.get(i) == c for i, c in truth.items())
    m = pd.crosstab(pd.Series([truth[i] for i in truth], name="OSHA"), pd.Series([got.get(i, "(none)") for i in truth], name="HokieAI"))
    cols = CATS + [c for c in m.columns if c not in CATS]
    m = m.reindex(index=CATS, columns=cols, fill_value=0)
    m.index = pd.MultiIndex.from_product([["OSHA"], m.index]); m.columns = pd.MultiIndex.from_product([["HokieAI"], m.columns])
    dis = key[[got.get(i) != c for i, c in zip(key.id, key.focus_four)]].copy()
    dis["HokieAI"] = [got.get(i, "(none)") for i in dis.id]
    return {"right": right, "n": len(truth), "got": len(got), "matrix": m,
            "disagree": dis[["id", "report", "focus_four", "HokieAI"]].rename(columns={"focus_four": "OSHA"})}


def finals(reply: str) -> Dict[str, float]:
    """'FINAL | Q3a | 33.3 | ft2' lines → {'Q3a': 33.3}."""
    out = {}
    for line in reply.splitlines():
        m = re.match(r"\s*[*`]*\s*FINAL\s*[*`]*\s*\|\s*(Q\d+[a-eA-E]?)\s*\|\s*([^|]+)", line, re.I)
        if m:
            nums = re.findall(r"-?\d[\d,]*\.?\d*", m.group(2))
            if nums:
                q = m.group(1); q = q[:2].upper() + q[2:].lower()
                out.setdefault(q, float(nums[0].replace(",", "")))
    return out


def takeoff(reply: str, key: List[dict]) -> dict:
    got = finals(reply)
    rows, right = [], 0
    for k in key:
        g, v = got.get(k["q"]), k["value"]
        tol = max(abs(v) * k["tolerance"], 0.5 if k["unit"] in ("rolls", "shingles", "CMU") else 0.01)
        ok = g is not None and abs(g - v) <= tol
        if not ok and k["q"] == "Q3a" and g is not None:        # the starter course given per eave: accepted
            ok = abs(g - v / 2) <= tol / 2
        right += ok
        rows.append({"question": k["q"], "quantity": k["quantity"], "answer key": round(v, 2), "HokieAI": g if g is not None else "—",
                     "unit": k["unit"].replace("ft2", "ft²"), "": "✓" if ok else "✗"})
    return {"right": right, "n": len(key), "table": pd.DataFrame(rows)}


def _n(s: str) -> str:
    return re.sub(r"\s+", "", str(s).replace("’", "'").replace("″", '"').replace("”", '"').replace("“", '"')).upper()


FIELDS = ["mark", "type", "width", "height", "hw", "location"]


def schedule(reply: str, key: List[dict]) -> dict:
    got = []
    for line in reply.splitlines():
        x = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(x) >= 6 and re.match(r"^\d+$", x[0]):
            got.append([c for c in x[:6]])
    rows, exact, fields = [], 0, 0
    by_mark = {g[0]: g for g in got}
    for k in key:
        g = by_mark.get(k["mark"])
        same = [g is not None and _n(g[i]) == _n(k[f]) for i, f in enumerate(FIELDS)]
        exact += all(same); fields += sum(same)
        rows.append({"mark": k["mark"], "answer key": " | ".join(k[f] for f in FIELDS[1:]),
                     "HokieAI": " | ".join(g[1:]) if g else "—", "": "✓" if all(same) else "✗"})
    return {"right": exact, "n": len(key), "fields_right": fields, "fields": len(key) * len(FIELDS), "listed": len(got), "table": pd.DataFrame(rows)}
