"""The contract clauses and questions for MP7 Part 3 (contract questions with quotes).

    python build/prepare_contract.py

Fetches 29 construction clauses of the Federal Acquisition Regulation (public domain) from acquisition.gov and writes
data/contract/clauses.txt and questions.json. Ten questions describe a project situation that must be checked against a
clause (a notice deadline, a retainage amount, a warranty period); seven cannot be answered from the clauses, and most of
those have a well-known answer outside them. Each question has a short key, the clauses it comes from, and the patterns
an answer must contain (every group must match; a group with "min" needs that many of its patterns).
"""
import html
import json
import re
import time
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "data" / "contract"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0 Safari/537.36"}
CLAUSES = ["52.211-10", "52.211-12", "52.211-13", "52.211-18", "52.232-5", "52.236-1", "52.236-2", "52.236-3", "52.236-4", "52.236-5",
           "52.236-6", "52.236-7", "52.236-8", "52.236-9", "52.236-10", "52.236-11", "52.236-12", "52.236-13", "52.236-14", "52.236-15",
           "52.236-16", "52.236-17", "52.236-21", "52.236-26", "52.242-14", "52.243-4", "52.246-12", "52.246-21", "52.249-10"]

NO = r"\bno\b"
# (question, key, clauses, patterns); patterns None = the clauses do not answer it. For a question that describes a situation,
# the answer needs the requirement and its application to the facts (a date, an amount, a quantity).
Q = [
    ("Within how many days after the work commences must the Contractor submit its construction schedule, and how many copies?",
     "Within five days; three copies", ["52.236-15"], [[r"\b(five|5)\b"], [r"\b(three|3)\b"]]),
    ("A progress payment request is for $240,000, and the Contracting Officer finds that satisfactory progress has not been made. What is the most the Contracting Officer may retain from this payment?",
     "$24,000 (10 percent)", ["52.232-5"], [[r"24,?000", r"\$24\s?k"]]),
    ("How many calendar days after receiving the notice to proceed must the Contractor commence work?",
     "NOT IN TEXT (the number is left blank for the Contracting Officer to insert)", ["52.211-10"], None),
    ("The Government takes possession of the completed gymnasium on June 1, 2027; final acceptance of the whole project is on October 1, 2027. On July 15, 2028, a roof leak caused by defective flashing is found in the gymnasium. Is the gymnasium still under the Contractor's warranty of construction?",
     "No: for a part the Government took possession of before final acceptance, the 1-year warranty runs from possession, so it ended June 1, 2028",
     ["52.246-21"], [[NO, r"expired", r"not (still )?(under|covered|within)", r"\bended\b", r"lapsed"], [r"june 1,? 2028", r"possession"]]),
    ("A strike at the ready-mix supplier stops concrete deliveries from March 3. The Contractor first notifies the Contracting Officer in writing of the cause of the delay on March 16. Did the Contractor meet the notice requirement of the Default clause for an excusable delay?",
     "No: the notice was due within 10 days from the beginning of the delay (by March 13), unless the Contracting Officer extended the period",
     ["52.249-10"], [[NO, r"not (meet|met|timely)", r"\blate\b", r"missed", r"did not"], [r"\b(10|ten)\b", r"march 13"]]),
    ("What is the dollar amount of liquidated damages per calendar day of delay?",
     "NOT IN TEXT (the amount is left blank)", ["52.211-12"], None),
    ("If the Government takes possession of part of the work before final acceptance, does the possession count as acceptance of that work?",
     "No; possession is not an acceptance of any work (and the 1-year warranty on that part runs from the date of possession)", ["52.236-11", "52.246-21"],
     [[r"not (be )?(deemed |considered )?(an )?accept", r"no acceptance", r"does not (constitute|count as|amount to)", r"isn.t (an )?accept", r"\bnot\b.*\baccept", NO]]),
    ("Name three causes of delay that the Default clause gives as examples of unforeseeable causes beyond the Contractor's control.",
     "Any three of: acts of God or of the public enemy, acts of the Government, acts of another contractor, fires, floods, epidemics, quarantine restrictions, strikes, freight embargoes, unusually severe weather, certain subcontractor or supplier delays",
     ["52.249-10"], [{"min": 3, "any": [r"god", r"public enemy", r"acts? of the government", r"another contractor", r"\bfires?\b", r"\bfloods?\b", r"epidemic",
                                         r"quarantine", r"strikes?\b", r"embargo", r"severe weather", r"subcontractor"]}]),
    ("On April 1, the Contracting Officer orally directs the Contractor to change the anchor bolt pattern, and the Contractor's extra costs begin on April 2. On April 26, the Contractor gives written notice that it regards the direction as a change order. From what date are the extra costs recoverable?",
     "From April 6: no adjustment is made for costs incurred more than 20 days before the written notice (the costs of April 2-5 are not recoverable)",
     ["52.243-4"], [[r"april 6\b", r"apr\.? 6\b", r"\b4/6\b"]]),
    ("Under the Changes clause, within how many days after receiving a written change order must the Contractor assert its right to an adjustment?",
     "30 days", ["52.243-4"], [[r"\b30\b", r"thirty"]]),
    ("The Contracting Officer unreasonably delays the approval of a shop drawing, which interrupts the work. The Contractor's added costs are $18,000, and it asks for $18,000 plus 10 percent profit ($1,800). What adjustment does the Suspension of Work clause allow?",
     "$18,000: the adjustment is for the increase in cost, excluding profit", ["52.242-14"],
     [[r"18,?000"], [r"exclud", r"no profit", r"without (the )?profit", r"profit (is |would be )?not", r"not (include|allow|recover)"]]),
    ("What insurance coverage limits must the Contractor carry?",
     "NOT IN TEXT", [], None),
    ("The drawings show a floor drain in the mechanical room, but the specifications do not mention it. Must the Contractor provide the floor drain?",
     "Yes: anything shown on the drawings and not mentioned in the specifications is of like effect as if shown or mentioned in both",
     ["52.236-21"], [[r"\byes\b", r"must (provide|install|furnish)", r"\brequired\b", r"like effect"]]),
    ("Unless otherwise indicated, how many copies of each shop drawing must the Contractor submit, and how many does the Contracting Officer keep?",
     "Four submitted; three retained", ["52.236-21"], [[r"\b(four|4)\b"], [r"\b(three|3)\b"]]),
    ("Is the Contractor entitled to an extension of the performance schedule for a stop-work order issued because of a safety noncompliance?",
     "No", ["52.236-13"], [[NO, r"not (be )?entitled"]]),
    ("Which federal safety standards must the Contractor comply with under the Accident Prevention clause?",
     "29 CFR Part 1926 and 29 CFR Part 1910", ["52.236-13"], [[r"1926"], [r"1910"]]),
    ("While excavating for the footings, the Contractor hits rock that the boring logs in the contract do not show. It removes the rock and sends written notice to the Contracting Officer the following week. Is the Contractor's request for an equitable adjustment protected under the Differing Site Conditions clause?",
     "No (at risk): written notice was required promptly and before the conditions were disturbed; without it, no request is allowed unless the Contracting Officer extends the time",
     ["52.236-2"], [[NO, r"not (be )?(allowed|protected|entitled)", r"at risk", r"(may|could|can|will) (likely )?(be )?(disallow|den|reject|bar)", r"jeopard", r"\bbarred\b"], [r"disturb"]]),
    ("The estimated quantity of a unit-priced item, rock excavation, is 1,000 cubic yards; the actual quantity is 1,250 cubic yards. How many cubic yards of the variation can the equitable adjustment in the contract price be based on?",
     "100 cubic yards: the quantity above 115 percent of the estimate (1,150 cubic yards)", ["52.211-18"], [[r"(?<![\d,.])100\b"]]),
    ("What minimum percentage of the total amount of work must the Contractor perform on the site with its own organization?",
     "NOT IN TEXT (the percentage is left blank for the Contracting Officer to insert)", ["52.236-1"], None),
    ("If, before acceptance, the Government removes completed work to examine it and the work meets the contract requirements, what is the Contractor entitled to?",
     "An equitable adjustment for the examination and reconstruction, and an extension of time if completion was delayed", ["52.246-12"],
     [[r"equitable adjustment"]]),
    ("How many days after the designated billing office receives a proper payment request must the Government pay a progress payment before an interest penalty applies?",
     "NOT IN TEXT (the Prompt Payment clause is not among the clauses)", [], None),
    ("While grading, the Contractor's crew destroys survey stakes that the Contracting Officer established, before their removal was authorized. Who pays to replace them, and how?",
     "The Contractor: the Contracting Officer may replace them and deduct the expense from amounts due to the Contractor",
     ["52.236-17"], [[r"contractor\b"], [r"deduct"]]),
    ("What markup percentage for overhead and profit may the Contractor add to the cost of changed work?",
     "NOT IN TEXT", [], None),
    ("A change order extends the completion date for the gymnasium only. Does the change order necessarily change the completion dates for the rest of the project?",
     "No: the change order may extend only the elements related to the changed work, and the completion dates for all other portions of the work will not be altered",
     ["52.211-13"], [[NO, r"not (necessarily|be )?(altered|changed|extended)", r"unchanged", r"\bremain"]]),
    ("Who pays for the temporary connections, distribution lines, and meters for the utilities the Government makes available, and when must they be removed?",
     "The Contractor, at its expense; before final acceptance of the work", ["52.236-14"], [[r"contractor\b"], [r"before (the )?final acceptance"]]),
    ("Within how many days after receiving the Contracting Officer's final decision on a claim must the Contractor appeal?",
     "NOT IN TEXT (the Disputes clause is not among the clauses)", [], None),
    ("While the Government has possession or use of part of the work before completion, is the Contractor responsible for loss of or damage to that work caused by the Government's possession or use?",
     "No; the Contractor is relieved of that responsibility", ["52.236-11"], [[NO, r"reliev", r"not (be )?responsible"]]),
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
