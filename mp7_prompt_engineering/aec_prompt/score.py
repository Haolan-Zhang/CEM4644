"""Reading HokieAI's pasted output and comparing it with the ground truth (loosely: extra words, tables and fences are fine)."""
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
        rows.append({"question": k["q"], "quantity": k["quantity"], "ground truth": round(v, 2), "HokieAI": g if g is not None else "—",
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
        rows.append({"mark": k["mark"], "ground truth": " | ".join(k[f] for f in FIELDS[1:]),
                     "HokieAI": " | ".join(g[1:]) if g else "—", "": "✓" if all(same) else "✗"})
    return {"right": exact, "n": len(key), "fields_right": fields, "fields": len(key) * len(FIELDS), "listed": len(got), "table": pd.DataFrame(rows)}


# ---------------------------------------------------------------------------- submittal review
RULES = {
    "concrete": [("content", r"cementitious (material(s)? )?content|minimum cementitious|\bcement content|total cementitious|content adequate|adequate[^|]*content"),
                 ("wcm", r"w\s*/\s*c|water[- /]*(to[- ])?cement|water-cementitious|water/cementitious"),
                 ("slump", r"slump"), ("air", r"\bair\b|entrain"), ("aggregate", r"aggregate|nominal max"), ("strength", r"strength|f.?c\b|psi")],
    "masonry": [("masonry_cement", r"masonry cement"), ("mortar_air", r"\bair\b|entrain"), ("grout_consistency", r"slump|flow|vsi|visual stability"),
                ("grout_strength", r"strength|psi"), ("grout_type", r"\bfine\b|coarse|grout type|c476"), ("mortar_type", r"mortar type|type [nsmo]\b|c270")],
}
FALLBACK = {"concrete": [], "masonry": [("mortar_type", r"mortar")]}


def _category(body: str, family: str):
    """The requirement column names the requirement; the whole line decides only when that column does not."""
    for part, rules in ((body.split("|")[0], RULES[family]), (body, RULES[family]), (body, FALLBACK[family])):
        cat = next((c for c, rx in rules if re.search(rx, part)), None)
        if cat:
            return cat


REQUIREMENT_NAMES = {"strength": "compressive strength (f'c at 28 days)", "slump": "slump", "aggregate": "nominal maximum aggregate size",
                     "wcm": "water-cementitious ratio (w/cm)", "air": "air content / air entrainment", "content": "minimum cementitious content",
                     "mortar_type": "mortar type", "masonry_cement": "masonry cement", "mortar_air": "air entrainment in mortar",
                     "grout_type": "grout type (fine / coarse)", "grout_consistency": "grout slump / slump flow / VSI", "grout_strength": "grout compressive strength"}


def submittal(reply: str, problem: dict) -> dict:
    """NONCOMPLIANT lines → the requirements flagged; every checked requirement counts as one decision."""
    flagged, unknown = set(), []
    for line in reply.splitlines():
        m = re.match(r"[\s>*`#-]*NONCOMPLIANT[\s*`]*\|\s*(.+)", line, re.I)
        if not m:
            continue
        body = m.group(1).strip().lower()
        if body.startswith("none"):
            continue
        cat = _category(body, problem["family"])
        (flagged.add(cat) if cat else unknown.append(body))
    planted = {k["item"] for k in problem["key"]}
    names = {k["item"]: k["name"] for k in problem["key"]}
    rows, right = [], 0
    for item in problem["checked"]:
        dev, flag = item in planted, item in flagged
        right += dev == flag
        rows.append({"requirement": REQUIREMENT_NAMES.get(item, names.get(item, item)), "ground truth": "noncompliant" if dev else "complies",
                     "HokieAI": "noncompliant" if flag else "complies", "": "✓" if dev == flag else "✗"})
    false = len(flagged - planted) + len(unknown)
    return {"right": right, "n": len(problem["checked"]), "found": len(planted & flagged), "planted": len(planted), "false": false,
            "table": pd.DataFrame(rows)}


# ---------------------------------------------------------------------------- contract questions
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"').replace("–", "-").replace("—", "-")).strip().lower()


NONE_RX = (r"not in (the )?(provided |supplied )?(text|clauses?)|not (stated|specified|provided|addressed|inserted|included|given|listed|contained|found|mentioned)"
           r"|left blank|\bblank\b|(does|do) not (say|state|specify|provide|address|include|contain|give)|\bsilent\b|cannot be determined"
           r"|\bno(ne)?\b[^.;|]{0,60}\b(is |are )?(stated|given|specified|provided|inserted|listed)\b")
NOT_IN_TEXT = r"^\W*not in (the )?(provided |supplied )?(text|clauses?)"


def _quote_found(quote: str, ctext: str) -> bool:
    """Every quoted passage (several may be joined with ' / ' or '...') appears word for word in the clauses. Only the
    letters and digits are compared, so spacing, punctuation and quotation marks do not matter (the published clause
    text has a few spacing errors, such as "(1)by"); a changed, added or missing word does."""
    words = lambda t: re.sub(r"[^a-z0-9]", "", t)
    parts = [words(x) for x in re.split(r"\s+/\s+|\s*\.\.\.\s*|\s*…\s*", _norm(quote))]
    parts = [x for x in parts if len(x) > 15]
    return bool(parts) and all(x in words(ctext) for x in parts)


def contract(reply: str, questions: List[dict], clauses: str) -> dict:
    lines = {}
    for m in re.finditer(r"^\s*[*`]*\s*(Q\d\d)\s*[*`]*\s*[|.:)]\s*(.+)$", reply, re.M):
        lines.setdefault(m.group(1), m.group(2))
    ctext = _norm(clauses)
    rows, right, made_up, not_provided, quotes, verified = [], 0, 0, 0, 0, 0
    for q in questions:
        parts = [p.strip() for p in lines.get(q["id"], "").split("|")]
        ans = parts[0] if parts and parts[0] else ""
        a = _norm(ans)
        said_none = bool(re.search(NONE_RX, a))
        if q["answerable"]:            # the patterns decide; an answer that only says NOT IN TEXT is wrong
            ok = bool(a) and not re.search(NOT_IN_TEXT, a) and all(
                (sum(bool(re.search(rx, a)) for rx in g["any"]) >= g["min"]) if isinstance(g, dict) else any(re.search(rx, a) for rx in g)
                for g in q["patterns"])
        else:
            ok = said_none
            made_up += bool(a) and not said_none
            not_provided += said_none
        right += ok
        quote = parts[-1] if len(parts) >= 2 else ""
        qv = "—"
        no_quote = re.match(r"^(-|—|none|n/?a|not applicable|no (supporting|relevant|such) )", _norm(quote)) or re.match(r"^(far )?52\.\d", _norm(quote))
        if quote and not no_quote:
            quotes += 1
            hit = _quote_found(quote, ctext)
            verified += hit; qv = "found" if hit else "misquote"
        rows.append({"question": q["id"], "ground truth": q["key"], "HokieAI": ans[:120] or "—", "": "✓" if ok else "✗", "quote found": qv})
    return {"right": right, "n": len(questions), "made_up": made_up, "not_provided": not_provided, "unanswerable": sum(not q["answerable"] for q in questions),
            "quotes": quotes, "verified": verified, "table": pd.DataFrame(rows)}
