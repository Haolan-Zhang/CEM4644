"""The contract clauses and questions for MP7 Part 3 (contract questions with quotes).

    python build/prepare_contract.py

Fetches 21 construction clauses of the Federal Acquisition Regulation (public domain) from acquisition.gov and writes
data/contract/clauses.txt and questions.json. Each question has a short key, the clauses it comes from, and the
patterns an answer must contain (every group must match; a group with "min" needs that many of its patterns).
"""
import html
import json
import re
import time
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "data" / "contract"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0 Safari/537.36"}
CLAUSES = ["52.211-10", "52.211-12", "52.232-5", "52.236-2", "52.236-3", "52.236-4", "52.236-5", "52.236-6", "52.236-7", "52.236-9",
           "52.236-10", "52.236-11", "52.236-12", "52.236-13", "52.236-15", "52.236-21", "52.242-14", "52.243-4", "52.246-12", "52.246-21", "52.249-10"]

Q = [
    ("Within how many days after the work commences must the Contractor submit its construction schedule, and how many copies?",
     "Within five days; three copies", ["52.236-15"], [[r"\b(five|5)\b"], [r"\b(three|3)\b"]]),
    ("What may the Contracting Officer do if the Contractor fails to submit the schedule within the time prescribed?",
     "Withhold approval of progress payments until the schedule is submitted", ["52.236-15"], [[r"withh[oe]ld"], [r"progress payment"]]),
    ("What is the maximum percentage the Contracting Officer may retain from a progress payment when satisfactory progress has not been made?",
     "10 percent", ["52.232-5"], [[r"\b10\b", r"\bten\b"]]),
    ("Progress payments on undefinitized contract actions may not exceed what percentage of the work accomplished?",
     "80 percent", ["52.232-5"], [[r"\b80\b", r"eighty"]]),
    ("How many calendar days after receiving the notice to proceed must the Contractor commence work?",
     "NOT IN TEXT (the number is left blank for the Contracting Officer to insert)", ["52.211-10"], None),
    ("How long does the Contractor's warranty of construction last, and from what date?",
     "1 year from the date of final acceptance (or from the date the Government takes possession, if earlier)", ["52.246-21"],
     [[r"\b(1|one)[- ]year"], [r"accept"]]),
    ("If the Government takes possession of part of the work before final acceptance, when does the warranty on that part begin, and does the possession count as acceptance?",
     "The 1-year warranty runs from the date the Government takes possession; possession is not an acceptance of the work", ["52.246-21", "52.236-11"],
     [[r"possession"], [r"not (be )?(deemed |considered )?(an )?accept", r"no acceptance", r"does not (constitute|count as|amount to)", r"isn.t (an )?accept", r"\bnot\b.*\baccept"]]),
    ("Within how many days from the beginning of a delay must the Contractor notify the Contracting Officer in writing of its causes, for the delay to be excused under the Default clause?",
     "10 days", ["52.249-10"], [[r"\b10\b", r"\bten\b"]]),
    ("What is the dollar amount of liquidated damages per calendar day of delay?",
     "NOT IN TEXT (the amount is left blank)", ["52.211-12"], None),
    ("Name three causes of delay that the Default clause gives as examples of unforeseeable causes beyond the Contractor's control.",
     "Any three of: acts of God or of the public enemy, acts of the Government, acts of another contractor, fires, floods, epidemics, quarantine restrictions, strikes, freight embargoes, unusually severe weather, certain subcontractor or supplier delays",
     ["52.249-10"], [{"min": 3, "any": [r"god", r"public enemy", r"acts? of the government", r"another contractor", r"\bfires?\b", r"\bfloods?\b", r"epidemic",
                                         r"quarantine", r"strikes?\b", r"embargo", r"severe weather", r"subcontractor"]}]),
    ("Under the Changes clause, within how many days after receiving a written change order must the Contractor assert its right to an adjustment?",
     "30 days", ["52.243-4"], [[r"\b30\b", r"thirty"]]),
    ("Both the Changes clause (for changes under its paragraph (b)) and the Suspension of Work clause limit the costs that can be recovered from before the Contractor's written notice. How many days is the limit in each clause?",
     "20 days in both", ["52.243-4", "52.242-14"], [[r"\b20\b", r"twenty"]]),
    ("Is profit included in the adjustment for an unreasonable suspension, delay, or interruption of the work?",
     "No (excluding profit)", ["52.242-14"], [[r"\bno\b", r"exclud", r"not (be )?(included|recoverable|allowed)"]]),
    ("What insurance coverage limits must the Contractor carry?",
     "NOT IN TEXT", [], None),
    ("If the drawings and the specifications differ, which governs?",
     "The specifications", ["52.236-21"], [[r"specifications? (shall |will )?(govern|control|prevail|take precedence)", r"^the specifications?\b", r"^specifications?\b"]]),
    ("Unless otherwise indicated, how many copies of each shop drawing must the Contractor submit, and how many does the Contracting Officer keep?",
     "Four submitted; three retained", ["52.236-21"], [[r"\b(four|4)\b"], [r"\b(three|3)\b"]]),
    ("Is the Contractor entitled to an extension of the performance schedule for a stop-work order issued because of a safety noncompliance?",
     "No", ["52.236-13"], [[r"\bno\b", r"not (be )?entitled"]]),
    ("Which federal safety standards must the Contractor comply with under the Accident Prevention clause?",
     "29 CFR Part 1926 and 29 CFR Part 1910", ["52.236-13"], [[r"1926"], [r"1910"]]),
    ("How many days does the Contracting Officer have to approve or disapprove a shop drawing?",
     "NOT IN TEXT", ["52.236-21"], None),
    ("If, before acceptance, the Government removes completed work to examine it and the work meets the contract requirements, what is the Contractor entitled to?",
     "An equitable adjustment for the examination and reconstruction, and an extension of time if completion was delayed", ["52.246-12"],
     [[r"equitable adjustment"]]),
    ("While the Government has possession or use of part of the work before completion, is the Contractor responsible for loss of or damage to that work caused by the Government's possession or use?",
     "No; the Contractor is relieved of that responsibility", ["52.236-11"], [[r"\bno\b", r"reliev", r"not (be )?responsible"]]),
]


def clean(raw):
    t = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", raw, flags=re.S)
    t = re.sub(r"<br\s*/?>|</p>|</li>|</h\d>|</div>", "\n", t)
    t = html.unescape(re.sub(r"<[^>]+>", "", t))
    i = t.find("As prescribed"); i = t.find("\n", i)
    j = t.find("(End of clause)", i)
    body = "\n".join(re.sub(r"\s+", " ", l).strip() for l in t[i:j].splitlines() if l.strip())
    return re.sub(r"\((\w{1,4})\)\s*\n", r"(\1) ", body).strip()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    parts = []
    for c in CLAUSES:
        raw = urllib.request.urlopen(urllib.request.Request(f"https://www.acquisition.gov/far/{c}", headers=UA), timeout=60).read().decode("utf-8", "replace")
        parts.append(f"FAR {c} {clean(raw)}")
        time.sleep(0.5)
    (OUT / "clauses.txt").write_text("\n\n".join(parts) + "\n")
    qs = [{"id": f"Q{i + 1:02d}", "question": q, "key": k, "clauses": c, "answerable": pats is not None, "patterns": pats} for i, (q, k, c, pats) in enumerate(Q)]
    (OUT / "questions.json").write_text(json.dumps(qs, indent=1))
    print(f"{len(parts)} clauses ({sum(len(p) for p in parts):,} characters), {len(qs)} questions ({sum(not q['answerable'] for q in qs)} not answerable)")


if __name__ == "__main__":
    main()
