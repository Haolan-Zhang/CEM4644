"""Photo sets, material vocabulary (the phrases sent to SAM 3), time series and plan facts."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Material:
    key: str
    name: str                     # what students see
    prompt: str                   # phrase sent to SAM 3 (the wording that works best)
    alternatives: List[str]       # other wordings, for the "does the phrase matter?" exercise
    color: str


@dataclass
class SetSpec:
    key: str
    title: str
    folder: str                   # repo-relative photo folder
    masks: str                    # repo-relative precomputed-mask folder
    description: str
    materials: List[Material]
    series: Dict[str, dict] = field(default_factory=dict)   # name -> {"title", "photos": [ids], "labels": [..]}
    phrase_lab_photos: List[str] = field(default_factory=list)  # photos with all alternative phrasings precomputed
    guess_photos: List[str] = field(default_factory=list)
    plans: List[dict] = field(default_factory=list)

    def material(self, name_or_key: str) -> Material:
        for m in self.materials:
            if name_or_key in (m.key, m.name, m.prompt):
                return m
        raise KeyError(name_or_key)

    @property
    def names(self) -> List[str]:
        return [m.name for m in self.materials]


SETS: Dict[str, SetSpec] = {}


def register(spec: SetSpec):
    SETS[spec.key] = spec
    return spec


register(SetSpec(
    key="site",
    title="Structural work on site: concrete, rebar, formwork, steel, scaffolding",
    folder="data/photos/site",
    masks="data/masks/site",
    description=("Photos of foundations, frames and concrete pours, from a 1937 public-domain project record to sites of the 2020s. "
                 "All from Wikimedia Commons under open licences (credits at the bottom of the notebook)."),
    materials=[
        Material("concrete", "concrete", "concrete", ["wet concrete", "cement", "concrete wall", "concrete floor"], "#9aa5b1"),
        Material("rebar", "rebar (reinforcing steel)", "steel reinforcement bars", ["rebar", "reinforcing steel", "metal rods", "rebar mesh"], "#e63946"),
        Material("formwork", "formwork", "wooden formwork", ["formwork", "wooden boards", "timber shuttering", "plywood panels"], "#f4a261"),
        Material("scaffolding", "scaffolding", "scaffolding", ["scaffold", "metal scaffolding", "scaffold poles"], "#ffd166"),
        Material("steel", "structural steel", "steel beam", ["steel frame", "structural steel", "steel column", "girder"], "#4cc9f0"),
        Material("brick", "brick / blockwork", "brick wall", ["bricks", "masonry", "concrete blocks", "block wall"], "#b5651d"),
        Material("soil", "soil / ground", "soil", ["dirt", "ground", "gravel", "mud"], "#8d6e63"),
        Material("wood", "timber", "wood", ["timber", "wooden planks", "lumber"], "#c9a227"),
        Material("worker", "worker", "worker", ["person", "construction worker", "man"], "#06d6a0"),
        Material("machine", "machine / vehicle", "construction machine", ["excavator", "crane", "truck", "concrete pump"], "#7b2cbf"),
        Material("sky", "sky", "sky", ["clouds"], "#bde0fe"),
    ],
    series={
        "A": {"title": "House footings, Trimingham (UK), January 2021", "photos": ["site_07", "site_08", "site_09"],
              "labels": ["8 Jan: pouring footings", "13 Jan: blockwork on footings", "18 Jan: concrete oversite"]},
        "B": {"title": "Administration building, Belchertown (USA), 1937-1939", "photos": ["site_23", "site_24", "site_25", "site_26", "site_27", "site_28", "site_29", "site_30", "site_31", "site_32"],
              "labels": ["Aug 1937: foundation", "Sep 1937: forms + rebar", "Oct 1937: foundation walls", "Nov 2 1937", "Nov 30 1937: steel", "Dec 1937: brickwork", "Jan 1938", "May 1938", "Aug 1938", "Sep 1939: finished"]},
    },
    phrase_lab_photos=["site_01", "site_07", "site_11", "site_14", "site_18", "site_27"],
    guess_photos=["site_01", "site_04", "site_09", "site_12", "site_18", "site_19", "site_28"],
))

register(SetSpec(
    key="interior",
    title="Interior finishing: drywall, studs, insulation, pipes, tiles",
    folder="data/photos/interior",
    masks="data/masks/interior",
    description=("Photos of interior fit-out: drywall (plasterboard) being installed, stud walls, insulation, services and tiling, "
                 "plus three 1938 interiors from the Belchertown project record. All from Wikimedia Commons under open licences."),
    materials=[
        Material("drywall", "drywall (plasterboard)", "drywall panel", ["drywall", "sheetrock", "plasterboard sheet", "gypsum board"], "#e9ecef"),
        Material("stud", "studs / framing", "metal framing", ["metal stud", "wooden stud", "wall framing", "steel frame"], "#c9a227"),
        Material("insulation", "insulation", "insulation", ["fiberglass insulation", "pink insulation", "mineral wool"], "#ff70a6"),
        Material("pipe", "pipes / ducts", "pipe", ["plumbing pipes", "duct", "metal pipe", "conduit"], "#4cc9f0"),
        Material("tile", "tiles", "tile", ["wall tiles", "ceramic tiles", "floor tiles"], "#06d6a0"),
        Material("concrete", "concrete", "concrete", ["concrete floor", "concrete wall", "cement"], "#9aa5b1"),
        Material("ceiling", "ceiling", "ceiling", ["ceiling panels", "roof"], "#bde0fe"),
        Material("floor", "floor", "floor", ["ground", "flooring"], "#8d6e63"),
        Material("window", "windows / openings", "window", ["window frame", "door", "opening"], "#ffd166"),
        Material("worker", "worker", "worker", ["person", "man"], "#7b2cbf"),
    ],
    series={
        "C": {"title": "Belchertown interiors, 1938", "photos": ["int_11", "int_12", "int_13"],
              "labels": ["Jan 1938: boiler room", "Mar 1938: plumbing", "Jun 1938: tiling"]},
    },
    phrase_lab_photos=["int_07", "int_13", "int_17", "int_21"],
    guess_photos=["int_06", "int_07", "int_09", "int_16", "int_19", "int_23"],
))

register(SetSpec(
    key="plans",
    title="Real foundation plans: quantity take-off of footings",
    folder="data/photos/plans",
    masks="data/masks/plans",
    description=("Plan-view regions of real foundation plans: two 1956 US Army standard drawings (US Army Corps of Engineers) and two "
                 "Library of Congress HABS/HAER sheets, all public domain, via Wikimedia Commons. Construction words find nothing on a "
                 "drawing; shape words ('small square', 'circle') and boxes do."),
    materials=[],
    plans=[
        {"id": "plan_dormitory",
         "short": "Airmen's dormitory (US Army, 1956): a grid of square footings F1–F6 with a footing schedule",
         "reference": "the 19'-4\" bay between column lines 1 and 2 (top dimension line)", "reference_width_ft": 19.33,
         "reference_hint": "draw the box from column line 1 to column line 2; only its WIDTH is used for the scale",
         "targets": ["one F2 footing on the top row (schedule: 4'-6\" square = 20.25 sq ft; it is only about 30 px wide here, so the box must sit exactly on its outline)",
                     "one footing of the middle row (read its mark and look it up in the schedule)",
                     "then tick find_all to count every footing like your box"],
         "phrases": ["footing", "small square", "circle"], "phrase_default": "footing",
         "expected": {"F2 footing": 20.25, "F3 footing": 25.0}},
        {"id": "plan_mess_hall",
         "short": "Kitchen and mess hall (US Army, 1956): 14 column footings, each 6'-0\" square, on 24'-0\" bays",
         "reference": "the 24'-0\" bay between column lines E and F (bottom dimension line)", "reference_width_ft": 24.0,
         "reference_hint": "box from bubble E to bubble F along the bottom; only its WIDTH is used",
         "targets": ["one column footing (detail: 6'-0\" square = 36 sq ft)", "find_all: 14 footings, 504 sq ft in total"],
         "phrases": ["footing", "small square", "circle"], "phrase_default": "footing",
         "expected": {"column footing": 36.0, "all 14 column footings": 504.0}},
        {"id": "plan_whittier",
         "short": "Whittier State School hospital (HABS, 1930s): 36 numbered columns on octagonal footings of six sizes",
         "reference": "one of the two 12'-0\" bays in the middle of the bottom dimension line", "reference_width_ft": 12.0,
         "reference_hint": "box between the two ticks of a 12'-0\" bay; only its WIDTH is used",
         "targets": ["the footing of column 7, 11, 15 or 19 (detail: 5'-4\" across, octagonal, about 23.6 sq ft)",
                     "the footing of column 6, 10, 14 or 18 (the other interior size)", "find_all: which columns does it miss?"],
         "phrases": ["footing", "small square", "circle", "octagon"], "phrase_default": "footing",
         "expected": {"footing of columns 7-11-15-19": 23.6}},
        {"id": "plan_mill",
         "short": "Shenandoah-Dives Mill (HAER, CAD drawing): foundation walls, two circular tanks and a graphic scale bar",
         "reference": "the graphic scale bar, from 0 to 50 FEET", "reference_width_ft": 50.0,
         "reference_hint": "box exactly from the 0 tick to the 50 tick of the FEET bar; only its WIDTH is used",
         "targets": ["one of the two circular tanks at the bottom left", "the hatched circle next to them",
                     "try the word 'circle' in Step 5a first"],
         "phrases": ["tank", "circle", "wall", "small square"], "phrase_default": "tank",
         "expected": {}},
    ],
))
