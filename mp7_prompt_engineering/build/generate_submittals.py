"""Specification excerpts and practice submittals for MP7 Part 2 (submittal review), with exact answer keys.

    python build/generate_submittals.py [--seed 4644]

The two specification excerpts are adapted from the Unified Facilities Guide Specifications (public domain, US DoD):
UFGS 03 30 00 Cast-in-Place Concrete (2.5 Concrete Mix Design) and UFGS 04 20 00 Masonry (2.5 Mortar and Grout Mixes),
with one project's choices filled in for the guide specification's options. The submittals are generated: each is a
mix-design submittal for a fictional project with 0-3 planted noncompliances, and the key lists them.
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

ELEMENTS = {   # portion -> requirements
    "Footings": dict(fc=3000, F="F0", slump=6, agg="1 in", floor=False, trowel=False),
    "Foundation walls (exterior)": dict(fc=4000, F="F1", slump=9, agg="3/4 in", floor=False, trowel=False),
    "Interior slabs-on-ground": dict(fc=4000, F="F0", slump=9, agg="3/4 in", floor=True, trowel=True),
    "Exterior slabs, walks, steps": dict(fc=4500, F="F2", slump=9, agg="3/4 in", floor=True, trowel=False),
}
F_REQ = {"F0": None, "F1": dict(wcm=0.55, air=5.0), "F2": dict(wcm=0.45, air=6.0)}
MIN_CEM = {"1 in": 520, "3/4 in": 540}


def concrete_submittal(rng, k, element, n_dev, show_wcm):
    req = ELEMENTS[element]
    fr = F_REQ[req["F"]]
    applicable = ["strength", "slump", "aggregate"] + (["wcm", "air"] if fr else []) + (["air"] if req["trowel"] else []) + (["content"] if req["floor"] else [])
    applicable = list(dict.fromkeys(applicable))
    devs = list(rng.choice(applicable, size=min(n_dev, len(applicable)), replace=False)) if n_dev else []
    # compliant baseline
    fc = req["fc"] + int(rng.choice([0, 0, 500]))
    agg = req["agg"]
    cem_min = MIN_CEM[agg] if req["floor"] else 470
    cement = int(rng.integers(cem_min + 20, cem_min + 110) // 5 * 5); fly = int(rng.choice([0, 0, 75, 100]))
    cm = cement + fly
    wcm_max = fr["wcm"] if fr else 0.55
    wcm = round(float(rng.uniform(wcm_max - 0.08, wcm_max - 0.02)), 2)
    slump = int(rng.integers(3, min(req["slump"], 6))) if req["slump"] > 4 else 3
    aea = bool(fr)
    air = (fr["air"] + float(rng.choice([-0.5, 0, 0.5]))) if fr else float(rng.choice([1.5, 2.0, 2.5]))
    # planted noncompliances
    if "strength" in devs:
        fc = req["fc"] - int(rng.choice([500, 1000]))
    if "slump" in devs:
        slump = req["slump"] + int(rng.choice([1, 2]))
    if "aggregate" in devs:
        agg = "1 in" if req["agg"] == "3/4 in" else "1-1/2 in"
    if "wcm" in devs:
        wcm = round(fr["wcm"] + float(rng.choice([0.03, 0.05])), 2)
    if "air" in devs:
        if req["trowel"]:
            aea, air = True, float(rng.choice([4.0, 5.0]))
        else:
            air = fr["air"] + float(rng.choice([-2.5, 2.5]))
    if "content" in devs:
        cement = MIN_CEM[req["agg"]] - int(rng.choice([30, 50])) - fly; cm = cement + fly
    water = round(wcm * cm)
    wcm_line = f"Water-cementitious materials ratio (w/cm): {wcm:.2f}" if show_wcm else "Water-cementitious materials ratio (w/cm): see batch weights"
    text = f"""CONCRETE MIX DESIGN SUBMITTAL · Submittal 03 30 00-{k:02d}
Project: CEM4644 practice project (fictional) · Supplier: Practice Ready-Mix Co. (fictional), Plant 2
Mix ID: {fc}-{int(slump)}{'-AE' if aea else ''}-{k:02d} · Intended use: {element}

Design data
Specified compressive strength: {fc} psi at 28 days (required average strength f'cr = {fc + 1200} psi, from field records)
Batch weights per cubic yard (saturated surface-dry aggregates):
  Portland cement, ASTM C150 Type I/II: {cement} lb
  {'Fly ash, ASTM C618 Class F: ' + str(fly) + ' lb' if fly else 'Fly ash: none'}
  Total cementitious materials: {cm} lb
  Water: {water} lb
  Coarse aggregate, ASTM C33, nominal maximum size {agg}: {int(rng.integers(1700, 1900))} lb
  Fine aggregate, ASTM C33 natural sand: {int(rng.integers(1150, 1350))} lb
{wcm_line}
Admixtures: {'air-entraining admixture, ASTM C260, 0.6 oz per 100 lb cementitious; ' if aea else ''}water-reducing admixture, ASTM C494 Type A, 4 oz per 100 lb cementitious
Target slump: {slump} in (ASTM C143)
Target total air content: {air:.1f} percent (ASTM C231)
Unit weight: {round(float(rng.uniform(141, 147)), 1)} lb per cubic foot
"""
    names = {"strength": "compressive strength (f'c)", "slump": "slump", "aggregate": "nominal maximum aggregate size", "wcm": "water-cementitious ratio (w/cm)",
             "air": "air content / air entrainment", "content": "minimum cementitious content"}
    return {"id": f"concrete_{k}", "family": "concrete", "spec": "concrete", "element": element, "text": text,
            "w_cm_stated": show_wcm, "key": [{"item": d, "name": names[d]} for d in devs], "checked": applicable}


def masonry_submittal(rng, k, wall, n_dev, grout_kind):
    exterior = wall != "Interior non-load-bearing walls"
    applicable = ["mortar_type", "masonry_cement", "mortar_air", "grout_type", "grout_consistency", "grout_strength"]
    devs = list(rng.choice(applicable, size=n_dev, replace=False)) if n_dev else []
    mtype = "S" if exterior else str(rng.choice(["N", "S"]))
    materials = "portland cement (ASTM C150 Type I) and hydrated lime (ASTM C207 Type S)"
    mair = "none"; gtype = "coarse"; gstr = int(rng.choice([2500, 3000, 3500]))
    slump = int(rng.choice([9, 10])); flow = int(rng.choice([26, 27, 28])); vsi = 1 if rng.random() < 0.5 else 0
    if "mortar_type" in devs:
        mtype = "N" if exterior else "O"
    if "masonry_cement" in devs:
        materials = "masonry cement (ASTM C91 Type S)"
    if "mortar_air" in devs:
        mair = "air-entraining admixture, 0.5 oz per bag"
    if "grout_type" in devs:
        gtype = "fine"
    if "grout_consistency" in devs:
        if grout_kind == "conventional":
            slump = int(rng.choice([6, 7, 12]))
        else:
            flow, vsi = (int(rng.choice([20, 32])), vsi) if rng.random() < 0.5 else (flow, 2)
    if "grout_strength" in devs:
        gstr = int(rng.choice([1500, 1800]))
    grout_line = (f"Grout consistency: conventional grout, slump {slump} in (ASTM C143)" if grout_kind == "conventional" else
                  f"Grout consistency: self-consolidating grout, slump flow {flow} in, visual stability index (VSI) {vsi} (ASTM C1611)")
    text = f"""MORTAR AND GROUT MIX SUBMITTAL · Submittal 04 20 00-{k:02d}
Project: CEM4644 practice project (fictional) · Supplier: Practice Masonry Supply (fictional)
Intended use: {wall}

Mortar
Specification: ASTM C270, Type {mtype}, proportion specification
Cementitious materials: {materials}
Aggregate: masonry sand, ASTM C144
Admixtures: {mair}
Batching: field-batched by volume with measuring boxes

Grout
Specification: ASTM C476, {gtype} grout, ready-mixed
{grout_line}
Compressive strength: {gstr} psi at 28 days (ASTM C1019, three specimens, average)
"""
    names = {"mortar_type": "mortar type", "masonry_cement": "masonry cement", "mortar_air": "air entrainment in mortar", "grout_type": "grout type (fine / coarse)",
             "grout_consistency": "grout slump / slump flow / VSI", "grout_strength": "grout compressive strength"}
    return {"id": f"masonry_{k}", "family": "masonry", "spec": "masonry", "element": wall, "text": text,
            "key": [{"item": d, "name": names[d]} for d in devs], "checked": applicable}


def main(seed):
    rng = np.random.default_rng(seed)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "spec_concrete.txt").write_text(CONCRETE_SPEC); (OUT / "spec_masonry.txt").write_text(MASONRY_SPEC)
    probs = []
    plan_c = [("Foundation walls (exterior)", 2, True), ("Interior slabs-on-ground", 2, True), ("Exterior slabs, walks, steps", 3, False),
              ("Footings", 1, True), ("Foundation walls (exterior)", 0, False)]
    for k, (el, n, show) in enumerate(plan_c, 1):
        probs.append(concrete_submittal(rng, k, el, n, show))
    plan_m = [("Exterior walls", 2, "conventional"), ("Interior non-load-bearing walls", 1, "self-consolidating"), ("Walls below grade", 3, "conventional"),
              ("Exterior walls", 0, "self-consolidating"), ("Interior non-load-bearing walls", 2, "conventional")]
    for k, (w, n, g) in enumerate(plan_m, 1):
        probs.append(masonry_submittal(rng, k, w, n, g))
    (OUT / "problems.json").write_text(json.dumps(probs, indent=1))
    for p in probs:
        print(p["id"], "|", p["element"], "|", [k["item"] for k in p["key"]])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=4644)
    main(ap.parse_args().seed)
