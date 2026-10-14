"""The MP8 practice project, made for the course: a specification excerpt, a submittal log and a schedule (the
documents the AI agent's tools read), the project questions, the takeoff questions of the skills part, and original
foundation plans for the workflow part, each with its exact ground truth.

    python build/generate_project.py

The foundation plans are drawn with the MP7 drawing code (../mp7_prompt_engineering/build/generate_practice.py),
with new seeds and an MP8 title block. Everything here is fictional and made for the course.
"""
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
sys.path.insert(0, str(REPO.parent / "mp7_prompt_engineering" / "build"))
import generate_practice as G  # noqa: E402

CMU_FACE = 8 * 16 / 144                       # ft² of wall per CMU (8 in x 16 in nominal face)

# ----------------------------------------------------------------------------- the documents
SPEC = """SECTION 03 30 00 · CAST-IN-PLACE CONCRETE (excerpt)
CEM4644 practice project (fictional). Not for construction.

2.5 CONCRETE MIX DESIGN
2.5.1 Water-cementitious materials ratio (w/cm): the total weight of water divided by the total weight of cementitious materials (portland cement plus fly ash plus slag), per cubic yard.
2.5.2 Air content: the total air content of the fresh concrete, in percent, within the range given for the element.
2.5.3 Slump: the target slump of the mix must not exceed the maximum slump given for the element.
2.5.4 Requirements for each element of the structure:

Element                    | Minimum f'c at 28 days | Maximum w/cm | Air content            | Maximum slump
Footings                   | 3,500 psi              | 0.50         | not required           | 5 in
Foundation walls           | 4,000 psi              | 0.48         | 4.5 to 7.5 percent     | 5 in
Interior slab-on-ground    | 4,000 psi              | 0.50         | 3.0 percent maximum    | 5 in
Exterior slabs and walks   | 4,500 psi              | 0.42         | 5.0 to 8.0 percent     | 4 in
Columns                    | 5,000 psi              | 0.45         | not required           | 6 in
"""

SUBMITTALS = [  # id, description, element, submitted, status, cement, fly ash, water (lb/yd³), air (%), slump (in), f'c (psi)
    ("03 30 00-01", "Footing mix F-3500", "Footings", "2026-08-03", "Approved", 470, 0, 230, 1.5, 4, 3500),
    ("03 30 00-02", "Exterior flatwork mix X-4500", "Exterior slabs and walks", "2026-08-05", "Under review", 560, 100, 290, 6.0, 5, 4500),
    ("03 30 00-03", "Foundation wall mix W-4000", "Foundation walls", "2026-08-05", "Revise and resubmit", 500, 120, 310, 4.0, 4, 4000),
    ("03 30 00-04", "Slab mix S-4000", "Interior slab-on-ground", "2026-08-10", "Approved", 520, 0, 255, 2.0, 5, 4000),
    ("03 30 00-05", "Column mix C-5000", "Columns", "2026-08-12", "Under review", 600, 150, 330, 1.5, 7, 5000),
]
SUBMITTAL_HEAD = ["submittal", "description", "element", "date submitted", "status", "cement (lb/yd³)", "fly ash (lb/yd³)",
                  "water (lb/yd³)", "air content (%)", "target slump (in)", "f'c (psi)"]

SCHEDULE = [  # activity, description, element, quantity, unit, daily output
    ("A-110", "Excavate footings", "Footings", 420, "CY", "140 CY per day"),
    ("A-120", "Place footings", "Footings", 96, "CY", "32 CY per day"),
    ("A-210", "Place foundation walls", "Foundation walls", 186, "CY", "31 CY per day"),
    ("A-310", "Place slab-on-ground", "Interior slab-on-ground", 212, "CY", "48 CY per day"),
    ("A-410", "Place exterior slabs and walks", "Exterior slabs and walks", 74, "CY", "30 CY per day"),
    ("A-510", "Place columns", "Columns", 18, "CY", "6 CY per day"),
]
SCHEDULE_HEAD = ["activity", "description", "element", "quantity", "unit", "daily output"]


def wcm(s):
    return round(s[7] / (s[5] + s[6]), 2)


def project_questions():
    """Six questions: each tool is needed by at least two, two need more than one tool."""
    s2, s5 = SUBMITTALS[1], SUBMITTALS[4]
    footing = 187 * (31 / 12) * (14 / 12) / 27 * 1.07
    return [
        {"id": "P1", "question": "What is the maximum water-cementitious ratio (w/cm) for the foundation walls on this project?",
         "kind": "number", "value": 0.48, "tolerance": 0.001, "unit": "", "tools": ["search_spec"]},
        {"id": "P2", "question": "A continuous footing is 187 ft long, 2 ft 7 in wide, and 1 ft 2 in deep. How many cubic yards of concrete are needed, including 7 % waste?",
         "kind": "number", "value": round(footing, 2), "tolerance": 0.01, "unit": "CY", "tools": ["calculator"]},
        {"id": "P3", "question": "What is the review status of submittal 03 30 00-03?",
         "kind": "text", "value": "Revise and resubmit", "accept": [r"revise\s*(and|&)\s*resubmit"], "tools": ["lookup_log"]},
        {"id": "P4", "question": "How many working days are scheduled to place the slab-on-ground: the activity's quantity divided by its daily output, rounded up to a whole day?",
         "kind": "number", "value": math.ceil(212 / 48), "tolerance": 0.001, "unit": "days", "tools": ["lookup_log", "calculator"]},
        {"id": "P5", "question": "Does the mix in submittal 03 30 00-02 meet this project's maximum w/cm for its element? Answer YES or NO.",
         "kind": "yesno", "value": "NO", "note": f"w/cm = {s2[7]} / ({s2[5]} + {s2[6]}) = {wcm(s2):.2f} > 0.42",
         "tools": ["lookup_log", "calculator", "search_spec"]},
        {"id": "P6", "question": "Does the target slump of the mix in submittal 03 30 00-05 meet this project's maximum slump for its element? Answer YES or NO.",
         "kind": "yesno", "value": "NO", "note": f"target slump {s5[9]} in > 6 in (columns)", "tools": ["lookup_log", "search_spec"]},
    ]


# ----------------------------------------------------------------------------- foundation takeoffs
def sheet_frame(ax, w, h, title, number):
    """The MP7 title block with the MP8 label."""
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((0.15, 0.15), w - 0.3, h - 0.3, fill=False, lw=1.6, ec=G.INK))
    tb_w = min(3.2, w * 0.28)
    ax.add_patch(Rectangle((w - 0.15 - tb_w, 0.15), tb_w, 0.9, fill=False, lw=1.0, ec=G.INK))
    ax.text(w - 0.15 - tb_w + 0.1, 0.85, "CEM4644 · MP8 PRACTICE DRAWING", fontsize=6.5, weight="bold", va="center", **G.FONT)
    ax.text(w - 0.15 - tb_w + 0.1, 0.6, title, fontsize=6, va="center", **G.FONT)
    ax.text(w - 0.15 - tb_w + 0.1, 0.35, "NOT FOR CONSTRUCTION", fontsize=5.5, va="center", color="#555", **G.FONT)
    ax.text(w - 0.25, 0.48, number, fontsize=11, weight="bold", ha="right", va="center", **G.FONT)


G.sheet_frame = sheet_frame


def segments(pts):
    return [round(math.hypot(pts[(i + 1) % len(pts)][0] - pts[i][0], pts[(i + 1) % len(pts)][1] - pts[i][1]), 3) for i in range(len(pts))]


def takeoff_key(length, p):
    """The ground truth by the estimating conventions (the skill's body)."""
    cmu = length * p["height"] / CMU_FACE
    grout_all = cmu * p["grout_per_cmu"] / 27 * (1 + p["grout_waste"] / 100)
    return [{"q": "Q1", "quantity": "CMUs to purchase", "value": math.ceil(cmu * (1 + p["waste"] / 100)), "unit": "CMU", "tolerance": 0.01},
            {"q": "Q2", "quantity": "grout, all cells", "value": round(grout_all, 3), "unit": "CY", "tolerance": 0.02},
            {"q": "Q3", "quantity": f"grout, {p['spacing']} in O.C.", "value": round(grout_all * 8 / p["spacing"], 3), "unit": "CY", "tolerance": 0.02}]


def takeoff_text(p, where):
    return (f"{where} a {p['thickness']} in CMU foundation wall, {p['height']} ft high, with no openings. Use 8 in x 16 in (nominal) CMUs.\n"
            f"Q1. How many CMUs must be purchased, including {p['waste']} % waste?\n"
            f"Q2. If every cell is grouted, how many cubic yards of grout are needed, including {p['grout_waste']} % waste? "
            f"A fully grouted {p['thickness']} in CMU takes {p['grout_per_cmu']} ft³ of grout.\n"
            f"Q3. If only the cells with vertical reinforcement at {p['spacing']} in on center are grouted, how many cubic yards of grout are "
            f"needed, including {p['grout_waste']} % waste?")


def params(rng):
    thick = int(rng.choice([6, 8])); return {
        "thickness": thick, "height": int(rng.choice([4, 6, 8, 10])), "waste": int(rng.choice([3, 4, 5, 6])),
        "grout_waste": int(rng.choice([5, 7, 10])), "spacing": int(rng.choice([16, 24, 32, 48])), "grout_per_cmu": G.GROUT_PER_CMU[thick]}


def drawings(n=5, seed=4653):
    rng = np.random.default_rng(seed)
    out = DATA / "drawings"; out.mkdir(parents=True, exist_ok=True)
    problems = []
    for i in range(n):
        kind, pts = G.foundation_shape(rng)
        hide = int(rng.integers(0, len(pts))) if i % 2 == 1 else -1
        p = params(rng)
        name = f"foundation_{i + 1}"
        G.draw_foundation(pts, hide, p["thickness"], out / f"{name}.png", f"S-20{i + 1}")
        segs = segments(pts)
        problems.append({"id": name, "label": f"Foundation plan {i + 1}", "image": f"{name}.png", "shape": kind, "hidden_segment": hide,
                         "segments": segs, "wall_length_ft": round(sum(segs), 3), **p,
                         "text": takeoff_text(p, "The attached foundation plan shows"), "key": takeoff_key(sum(segs), p)})
    return problems


def skill_questions(seed=4649):
    """Takeoff questions in words (no drawing): the skills part. Two of them are L-shaped, to test the length rule."""
    rng = np.random.default_rng(seed)
    out = []
    shapes = [("rectangle", [(0, 0), (48, 0), (48, 40), (0, 40)]),
              ("L", [(0, 0), (30, 0), (30, 20), (16, 20), (16, 36), (0, 36)])]
    for k, (kind, pts) in enumerate(shapes):
        p = params(rng); segs = segments(pts)
        where = ("The foundation is a rectangle with outside dimensions of " + f"{segs[0]:g} ft x {segs[1]:g} ft. It has"
                 if kind == "rectangle" else
                 "The foundation is L-shaped; going around the outside face of the wall, its sides are " +
                 ", ".join(f"{s:g} ft" for s in segs) + ". It has")
        out.append({"id": f"T{k + 1}", "shape": kind, "segments": segs, **p, "text": takeoff_text(p, where), "key": takeoff_key(sum(segs), p)})
    return out


# ----------------------------------------------------------------------------- write
def write_csv(path, head, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(head); w.writerows(rows)


def main():
    proj = DATA / "project"; proj.mkdir(parents=True, exist_ok=True)
    (proj / "spec_03_30_00.txt").write_text(SPEC, encoding="utf-8")
    write_csv(proj / "submittal_log.csv", SUBMITTAL_HEAD, SUBMITTALS)
    write_csv(proj / "schedule.csv", SCHEDULE_HEAD, SCHEDULE)
    (proj / "questions.json").write_text(json.dumps(project_questions(), indent=1, ensure_ascii=False), encoding="utf-8")
    (proj / "takeoff_questions.json").write_text(json.dumps(skill_questions(), indent=1, ensure_ascii=False), encoding="utf-8")
    (DATA / "drawings" / "problems.json").write_text(json.dumps(drawings(), indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote", proj, DATA / "drawings")


if __name__ == "__main__":
    main()
