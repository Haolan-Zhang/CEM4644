"""The drawing sets of the take-off lab and the vocabulary (the phrases students send to SAM 3).

Everything here is American practice: feet, inches and square feet. A "sheet" is one drawing (PNG) with an
answer key next to it (`<id>.key.json`), see README.md for the schema.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Room types an answer key may use for an area whose category is "room".
ROOM_TYPES = ["bedroom", "kitchen", "living room", "bathroom", "hall", "porch", "closet", "dining room", "other"]


@dataclass
class Thing:
    """Something students can ask SAM 3 for by name on a drawing.

    kind:  room (an enclosed space), opening (door / window).
    truth: how to check the answer against the key:
           ("areas", None)   -> every area whose category is "room"
           ("areas", <type>) -> the rooms of that type
           ("counts", <cat>) -> the boxes under that counting category
           None              -> the drawing has no answer key for this thing.
    """
    key: str
    name: str
    prompt: str
    alternatives: List[str]
    color: str
    kind: str = "room"
    truth: Optional[Tuple[str, Optional[str]]] = None


# The workshop vocabulary: the rooms and the two openings. The alternatives are the other wordings Step 4a
# compares, and they include the SHAPE words that are the only ones that work on a drawing: "curved line"
# for a door swing, "short parallel lines" for a window. (Shape words for things that are NOT in this list -
# "thick black line" for a wall, "hatched square" for a fireplace - are typed live in Step 4c.)
THINGS: List[Thing] = [
    Thing("room", "room (any)", "room", ["a room", "empty room", "floor area"], "#9aa5b1", "room", ("areas", None)),
    Thing("bedroom", "bedroom", "bedroom", ["bed", "sleeping room"], "#457b9d", "room", ("areas", "bedroom")),
    Thing("kitchen", "kitchen", "kitchen", ["kitchen counter", "cooker", "stove"], "#f4a261", "room", ("areas", "kitchen")),
    Thing("living", "living room", "living room", ["lounge", "sitting room"], "#e9c46a", "room", ("areas", "living room")),
    Thing("bathroom", "bathroom", "bathroom", ["bath", "washroom"], "#4cc9f0", "room", ("areas", "bathroom")),
    Thing("porch", "porch", "porch", ["veranda", "deck"], "#06d6a0", "room", ("areas", "porch")),
    Thing("closet", "closet", "closet", ["cupboard", "wardrobe"], "#b5838d", "room", ("areas", "closet")),
    Thing("door", "door", "door", ["curved line", "door arc", "arc"], "#ffd166", "opening", ("counts", "door")),
    Thing("window", "window", "window", ["window opening", "short parallel lines", "gap in the wall"], "#bde0fe", "opening", ("counts", "window")),
]


@dataclass
class SheetSet:
    """One set of drawings: a folder of <id>.png + <id>.key.json + credits.json."""
    key: str
    title: str
    folder: str
    description: str
    masks: str = ""                      # "" = no precomputed masks (the homework does not need any)
    default_sheet: str = ""              # "" = the first one in credits.json
    takeoff_counts: bool = True          # False: the take-off cells measure only; counting from one example has its own step
    guided: bool = True                  # False: the cells and widgets do not repeat each sheet's tasks (the notebook text says it once)
    scale_check: bool = True             # False: the student's scale box is used but the SCALE CHECK lines are not printed
    things: List[Thing] = field(default_factory=lambda: list(THINGS))

    def thing(self, name_or_key: str) -> Thing:
        for t in self.things:
            if name_or_key in (t.key, t.name, t.prompt):
                return t
        raise KeyError(name_or_key)

    @property
    def names(self) -> List[str]:
        return [t.name for t in self.things]


SETS: Dict[str, SheetSet] = {}


def register(spec: SheetSet) -> SheetSet:
    SETS[spec.key] = spec
    return spec


register(SheetSet(
    key="workshop",
    takeoff_counts=False,                # Step 3d counts doors from one example; the take-off cells (3a-3c) are rooms only
    guided=False,                        # the Part 3 text explains the boxes once; the cells and widgets stay clean
    title="Three 1940 USDA farmhouse plans",
    folder="data/sheets/workshop",
    masks="data/masks/workshop",
    description=("Three small farmhouse floor plans published by the U.S. Department of Agriculture in 1940 (public domain). "
                 "Black walls, drawn windows and door swings, a printed size inside most rooms and overall dimension lines "
                 "along two sides. There is no scale bar on any of them: you set the scale yourself from a printed dimension. "
                 "Every room\'s real area was measured off each drawing, so the notebook can check "
                 "your measurements."),
    default_sheet="usda_5544",
))

register(SheetSet(
    key="homework",
    scale_check=False,                   # the scale box is drawn and used; only its printout is left out
    guided=False,                        # the Part texts explain each sheet once; cells, widgets and Step 0 stay clean
    title="Seven sheets from three disciplines",
    folder="data/sheets/homework",
    description=("Five real drawings: a floor plan (a modern VA clinic), three structural foundation plans and one reflected "
                 "ceiling plan. Each one carries a different take-off task - areas of rooms, areas of footings, counts of "
                 "repeated symbols - and each one comes with what is really on it, so every number you get is checked."),
))

# How the homework groups its sheets into the three take-off steps (2a floor plans, 2b structural, 2c MEP).
# The group is decided by the `discipline` field of the answer key.
DISCIPLINE_GROUPS = [
    ("floor", "floor plans", ("floor plan",)),
    ("structural", "structural plans", ("structural",)),
    ("mep", "MEP plans", ("electrical", "mechanical", "plumbing", "mep")),
]


def group_of(discipline: str) -> str:
    """'floor plan + plumbing' -> 'floor', 'structural' -> 'structural', 'electrical' -> 'mep'."""
    d = (discipline or "").strip().lower()
    for gid, _title, prefixes in DISCIPLINE_GROUPS:
        for p in prefixes:
            if d.startswith(p):
                return gid
    for gid, _title, prefixes in DISCIPLINE_GROUPS:
        for p in prefixes:
            if p in d:
                return gid
    return "floor"
