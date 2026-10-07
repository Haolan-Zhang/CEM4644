"""Specification excerpts and practice submittals for MP7 Part 2 (submittal review), with the exact ground truth.

    python build/generate_submittals.py [--seed 4644]

The two specification excerpts are adapted from the Unified Facilities Guide Specifications (public domain, US DoD):
UFGS 03 30 00 Cast-in-Place Concrete (2.5 Concrete Mix Design) and UFGS 04 20 00 Masonry (2.5 Mortar and Grout Mixes),
with one project's choices filled in for the guide specification's options. The 12 submittals are designed by hand for a
fictional project: each describes its intended use the way a project would (so the applicable requirements must be
identified), carries 0-3 planted noncompliances (some only visible after a calculation) and some values exactly at a
limit, and the ground truth lists the noncompliances.
"""
import argparse
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "data" / "submittals"

CONCRETE_SPEC = """SECTION 03 30 00 · CAST-IN-PLACE CONCRETE (excerpt)
Adapted from UFGS 03 30 00 with this project's selections. Practice project: not for construction.

2.5   CONCRETE MIX DESIGN
2.5.1   Properties and Requirements
a.  Cementitious material content must be adequate for concrete to satisfy the specified requirements for strength, w/cm, durability, and finishability.
b.  The minimum cementitious material content for concrete used in floors (slabs-on-ground and elevated slabs) must meet the following requirements:
    Nominal maximum size of aggregate 1 in: 520 pounds per cubic yard
    Nominal maximum size of aggregate 3/4 in: 540 pounds per cubic yard
c.  Selected target slump must not exceed 9 in, unless a lower maximum slump is given in the table below.
d.  Concrete must be air entrained for members assigned to Exposure Class F1 or F2, with the total air content in accordance with paragraph 2.5.2.
e.  Concrete for slabs to receive a hard-troweled finish must not contain an air-entraining admixture or have a total air content greater than 3 percent.
f.  Concrete properties and requirements for each portion of the structure are specified in the table below.

    Portion of structure           | Minimum f'c at 28 days | Exposure class | Miscellaneous requirements
    Footings                       | 3000 psi               | F0             | Max. slump 6 in; nominal maximum aggregate size 1 in
    Foundation walls (exterior)    | 4000 psi               | F1             | Nominal maximum aggregate size 3/4 in
    Interior slabs-on-ground       | 4000 psi               | F0             | Hard-troweled finish; nominal maximum aggregate size 3/4 in
    Exterior slabs, walks, steps   | 4500 psi               | F2             | Nominal maximum aggregate size 3/4 in

2.5.2   Durability (freezing and thawing)
    Exposure class F0: no air-content or w/cm requirement beyond those above.
    Exposure class F1: maximum w/cm 0.55; total air content 5.0 percent, tolerance plus or minus 1.5 percent.
    Exposure class F2: maximum w/cm 0.45; total air content 6.0 percent, tolerance plus or minus 1.5 percent.
"""

MASONRY_SPEC = """SECTION 04 20 00 · MASONRY (excerpt)
Adapted from UFGS 04 20 00 with this project's selections. Practice project: not for construction.

2.5   MORTAR AND GROUT MIXES
2.5.1   Mortar Mix
a.  Provide mortar that conforms to ASTM C270. Provide Type S mortar for exterior walls and for walls below grade.
b.  Provide Type N or Type S mortar for non-load-bearing interior masonry.
c.  Do not use masonry cement in the mortar. Do not use air-entrainment in the mortar.
2.5.2   Grout and Ready Mix Grout Mix
a.  Use grout that conforms to ASTM C476, coarse.
b.  Use conventional grout with a slump between 8 and 11 inches.
c.  Use self-consolidating grout with a slump flow of 24 to 30 inches and a visual stability index (VSI) not greater than 1.
d.  Provide a minimum grout strength of 2000 psi in 28 days, as tested in accordance with ASTM C1019.
"""

# What each portion of the structure must meet (from the excerpt above), for reference.
ELEMENTS = {
    "Footings": dict(fc=3000, F="F0", slump=6, agg="1 in", floor=False, trowel=False),
    "Foundation walls (exterior)": dict(fc=4000, F="F1", slump=9, agg="3/4 in", floor=False, trowel=False),
    "Interior slabs-on-ground": dict(fc=4000, F="F0", slump=9, agg="3/4 in", floor=True, trowel=True),
    "Exterior slabs, walks, steps": dict(fc=4500, F="F2", slump=9, agg="3/4 in", floor=True, trowel=False),
}
REQUIREMENT_NAMES = {"strength": "compressive strength (f'c at 28 days)", "slump": "slump", "aggregate": "nominal maximum aggregate size",
                     "wcm": "water-cementitious ratio (w/cm)", "air": "air content / air entrainment", "content": "minimum cementitious content",
                     "mortar_type": "mortar type", "masonry_cement": "masonry cement", "mortar_air": "air entrainment in mortar",
                     "grout_type": "grout type (fine / coarse)", "grout_consistency": "grout slump / slump flow / VSI", "grout_strength": "grout compressive strength"}

# Each submittal: the portion of the structure (hidden; the submittal and its label describe the use the way a project would), its values, and
# the noncompliances planted in them. Traps that comply are noted in "notes" for the instructor.
CONCRETE = [
    dict(element="Foundation walls (exterior)", short="basement walls", use="Basement walls, backfilled on the outside face, exposed to freezing and to moisture",
         fc=4000, age=28, cement=480, fly=120, water=342, show_wcm=False, agg="3/4 in", slump="5 in", aea=True, air=7.0,
         devs=["wcm", "air"], notes="w/cm = 342 / 600 = 0.57 > 0.55 (computed); air 7.0 % > 6.5 % (F1 tolerance)"),
    dict(element="Interior slabs-on-ground", short="warehouse floor slab", use="Warehouse floor slab-on-ground, power-troweled (hard-troweled) finish",
         fc=4000, age=28, cement=470, fly=60, water=244, show_wcm=True, agg="3/4 in", slump="5 in", aea=True, air=2.5,
         devs=["air", "content"], notes="air-entraining admixture in a hard-troweled slab although air is 2.5 %; cementitious 470 + 60 = 530 < 540"),
    dict(element="Exterior slabs, walks, steps", short="entrance stoop and sidewalks", use="Entrance stoop, steps, and sidewalks, exposed to freezing and to deicing salts",
         fc=4500, age=56, cement=560, fly=0, water=252, show_wcm=True, agg="3/4 in", slump="4 in", aea=True, air=4.5,
         devs=["strength"], notes="4500 psi at 56 days instead of 28 days; w/cm 0.45 and air 4.5 % are exactly at the limits (comply)"),
    dict(element="Footings", short="spread footings", use="Spread footings and column pads, below the frost line",
         fc=3000, age=28, cement=500, fly=0, water=290, show_wcm=True, agg="1 in", slump="7 in", aea=False, air=1.5,
         devs=["slump"], notes="slump 7 in: below the general 9 in but above the 6 in footing limit; w/cm 0.58 has no limit for F0 (complies)"),
    dict(element="Foundation walls (exterior)", short="stair-tower foundation walls", use="Stair-tower foundation walls, backfilled on one side, exposed to freezing and to moisture",
         fc=4000, age=28, cement=480, fly=120, water=330, show_wcm=False, agg="3/4 in", slump="4 in", aea=True, air=6.5,
         devs=[], notes="fully compliant: w/cm = 330 / 600 = 0.55 and air 6.5 % are exactly at the limits"),
    dict(element="Exterior slabs, walks, steps", short="loading-dock apron", use="Exterior loading-dock apron, exposed to freezing and to deicing salts",
         fc=5000, age=28, cement=520, fly=40, water=263, show_wcm=False, agg="1 in", slump="4 in", aea=True, air=6.0,
         devs=["aggregate", "wcm"], notes="1 in aggregate where 3/4 in is required; w/cm = 263 / 560 = 0.47 > 0.45 (computed)"),
]
MASONRY = [
    dict(exterior=True, short="maintenance shop walls", use="Exterior CMU walls of the vehicle maintenance shop", mtype="M", materials="portland cement (ASTM C150 Type I) and hydrated lime (ASTM C207 Type S)",
         mair="none", gtype="coarse", grout="conventional grout, slump 11 in (ASTM C143)", gstr=2000, devs=["mortar_type"],
         notes="Type M where Type S is specified (stronger, still noncompliant); grout slump 11 in and strength 2000 psi exactly at the limits (comply)"),
    dict(exterior=False, short="office partitions", use="Non-load-bearing CMU partitions between offices", mtype="N", materials="mortar cement (ASTM C1329 Type N)",
         mair="none", gtype="coarse", grout="self-consolidating grout, slump flow 23 in, visual stability index (VSI) 1 (ASTM C1611)", gstr=2500,
         devs=["grout_consistency"], notes="mortar cement is allowed (only masonry cement is prohibited); slump flow 23 in < 24 in"),
    dict(exterior=True, short="elevator pit walls", use="Elevator pit walls below grade", mtype="S", materials="masonry cement (ASTM C91 Type S)",
         mair="air-entraining admixture, 0.5 oz per bag of cement", gtype="fine", grout="conventional grout, slump 9 in (ASTM C143)", gstr=3000,
         devs=["masonry_cement", "mortar_air", "grout_type"], notes="three noncompliances"),
    dict(exterior=True, short="brick veneer backup walls", use="Exterior CMU backup walls behind the brick veneer", mtype="S", materials="portland cement (ASTM C150 Type I) and hydrated lime (ASTM C207 Type S)",
         mair="none", gtype="coarse", grout="self-consolidating grout, slump flow 30 in, visual stability index (VSI) 1 (ASTM C1611)", gstr=2000,
         devs=[], notes="fully compliant: slump flow 30 in, VSI 1 and 2000 psi are exactly at the limits"),
    dict(exterior=False, short="storage room partitions", use="Non-load-bearing CMU partitions around the storage rooms", mtype="O", materials="portland cement (ASTM C150 Type I) and hydrated lime (ASTM C207 Type S)",
         mair="none", gtype="coarse", grout="conventional grout, slump 10 in (ASTM C143)", gstr=1800, devs=["mortar_type", "grout_strength"],
         notes="Type O where Type N or S is allowed; grout 1800 psi < 2000 psi"),
    dict(exterior=True, short="loading-dock retaining wall", use="Below-grade CMU retaining wall at the loading dock", mtype="N", materials="portland cement (ASTM C150 Type I) and hydrated lime (ASTM C207 Type S)",
         mair="none", gtype="coarse", grout="conventional grout, slump 7 in (ASTM C143)", gstr=2500, devs=["mortar_type", "grout_consistency"],
         notes="Type N where Type S is specified for walls below grade; slump 7 in < 8 in"),
]


def concrete_submittal(k, c, rng):
    # Every concrete submittal is checked on all six requirements: where the excerpt sets no limit for the portion of the
    # structure (w/cm for F0, say), "complies" is the right decision and a flag is a false flag.
    checked = ["strength", "slump", "aggregate", "wcm", "air", "content"]
    cm = c["cement"] + c["fly"]
    wcm_line = f"Water-cementitious materials ratio (w/cm): {c['water'] / cm:.2f}" if c["show_wcm"] else "Water-cementitious materials ratio (w/cm): see batch weights"
    seven = round(c["fc"] * float(rng.uniform(0.66, 0.74)) / 10) * 10
    text = f"""CONCRETE MIX DESIGN SUBMITTAL · Submittal 03 30 00-{k:02d}
Project: CEM4644 practice project (fictional) · Supplier: Practice Ready-Mix Co. (fictional), Plant 2
Mix ID: P2-{c['fc'] // 100}-{k:02d} · Intended use: {c['use']}

Design data
Specified compressive strength: {c['fc']} psi at {c['age']} days
Required average strength f'cr: {c['fc'] + 1200} psi, from field records (30 consecutive tests, standard deviation 520 psi)
Trial batch: 7-day strength {seven} psi (average of 3 cylinders)
Batch weights per cubic yard (saturated surface-dry aggregates):
  Portland cement, ASTM C150 Type I/II: {c['cement']} lb
  {'Fly ash, ASTM C618 Class F: ' + str(c['fly']) + ' lb' if c['fly'] else 'Fly ash: none'}
  Water: {c['water']} lb
  Coarse aggregate, ASTM C33, nominal maximum size {c['agg']}: {int(rng.integers(1700, 1900))} lb
  Fine aggregate, ASTM C33 natural sand: {int(rng.integers(1150, 1350))} lb
{wcm_line}
Admixtures: {'air-entraining admixture, ASTM C260, 0.6 oz per 100 lb cementitious; ' if c['aea'] else ''}water-reducing admixture, ASTM C494 Type A, 4 oz per 100 lb cementitious
Target slump: {c['slump']} (ASTM C143)
Target total air content: {c['air']:.1f} percent (ASTM C231)
Unit weight: {round(float(rng.uniform(141, 147)), 1)} lb per cubic foot
Water-soluble chloride ion content: 0.05 percent by weight of cement
"""
    return {"id": f"concrete_{k}", "family": "concrete", "spec": "concrete", "element": c["element"], "short": c["short"], "use": c["use"], "text": text,
            "key": [{"item": d, "name": REQUIREMENT_NAMES[d]} for d in c["devs"]], "checked": checked, "notes": c["notes"]}


def masonry_submittal(k, m):
    wall = ("Exterior walls and walls below grade" if m["exterior"] else "Non-load-bearing interior walls")
    text = f"""MORTAR AND GROUT MIX SUBMITTAL · Submittal 04 20 00-{k:02d}
Project: CEM4644 practice project (fictional) · Supplier: Practice Masonry Supply (fictional)
Intended use: {m['use']}

Mortar
Specification: ASTM C270, Type {m['mtype']}, proportion specification
Cementitious materials: {m['materials']}
Aggregate: masonry sand, ASTM C144
Admixtures: {m['mair']}
Batching: field-batched by volume with measuring boxes

Grout
Specification: ASTM C476, {m['gtype']} grout, ready-mixed
Grout consistency: {m['grout']}
Compressive strength: {m['gstr']} psi at 28 days (ASTM C1019, three specimens, average)
"""
    return {"id": f"masonry_{k}", "family": "masonry", "spec": "masonry", "element": wall, "short": m["short"], "use": m["use"], "text": text,
            "key": [{"item": d, "name": REQUIREMENT_NAMES[d]} for d in m["devs"]],
            "checked": ["mortar_type", "masonry_cement", "mortar_air", "grout_type", "grout_consistency", "grout_strength"], "notes": m["notes"]}


def main(seed):
    rng = np.random.default_rng(seed)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "spec_concrete.txt").write_text(CONCRETE_SPEC); (OUT / "spec_masonry.txt").write_text(MASONRY_SPEC)
    probs = [concrete_submittal(k, c, rng) for k, c in enumerate(CONCRETE, 1)] + [masonry_submittal(k, m) for k, m in enumerate(MASONRY, 1)]
    (OUT / "problems.json").write_text(json.dumps(probs, indent=1))
    for p in probs:
        print(f"{p['id']:11} | {p['use'][:60]:60} | {[k['item'] for k in p['key']]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=4644)
    main(ap.parse_args().seed)
