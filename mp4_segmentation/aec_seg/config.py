"""Plan sets and the vocabulary (the phrases sent to SAM 3) of the floor-plan take-off lab."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

CUBICASA = {
    "author": "CubiCasa5K (Kalervo, Ylioinas, Häikiö, Karhu, Kannala 2019), CubiCasa Oy",
    "license": "CC BY-NC-SA 4.0",
    "source": "https://zenodo.org/records/2613548",
}


@dataclass
class Thing:
    """Something students can ask SAM 3 for on a plan.
    kind: room (an area), fixture (a symbol), opening (door / window), structure (walls).
    truth: how to look it up in the answer key: ("rooms", <type>) / ("rooms", None) / ("fixtures", <type>) /
           ("doors", None) / ("windows", None) / ("walls", None) / None (no ground truth)."""
    key: str
    name: str
    prompt: str
    alternatives: List[str]
    color: str
    kind: str = "room"
    truth: Optional[Tuple[str, Optional[str]]] = None


THINGS: List[Thing] = [
    Thing("room", "room (any)", "room", ["a room", "space", "floor area"], "#9aa5b1", "room", ("rooms", None)),
    Thing("bedroom", "bedroom", "bedroom", ["bed", "sleeping room"], "#457b9d", "room", ("rooms", "bedroom")),
    Thing("bathroom", "bathroom", "bathroom", ["shower room", "wc", "toilet room"], "#4cc9f0", "room", ("rooms", "bathroom")),
    Thing("kitchen", "kitchen", "kitchen", ["kitchen counter", "cooker", "kitchen cabinets"], "#f4a261", "room", ("rooms", "kitchen")),
    Thing("living", "living room", "living room", ["lounge", "sofa", "living area"], "#e9c46a", "room", ("rooms", "living room")),
    Thing("balcony", "balcony / terrace", "balcony", ["terrace", "outdoor area", "patio"], "#06d6a0", "room", ("rooms", "balcony / terrace")),
    Thing("toilet", "toilet", "toilet", ["wc", "lavatory", "toilet bowl"], "#7b2cbf", "fixture", ("fixtures", "toilet")),
    Thing("sink", "sink", "sink", ["wash basin", "basin", "washbasin"], "#2a9d8f", "fixture", ("fixtures", "sink")),
    Thing("bathtub", "bathtub", "bathtub", ["bath", "tub", "shower"], "#e63946", "fixture", ("fixtures", "bathtub")),
    Thing("stairs", "stairs", "stairs", ["staircase", "steps", "stairway"], "#8d6e63", "fixture", ("fixtures", "stairs")),
    Thing("door", "door", "door", ["door arc", "curved line", "arc"], "#ffd166", "opening", ("doors", None)),
    Thing("window", "window", "window", ["window opening", "glass", "window in a wall"], "#bde0fe", "opening", ("windows", None)),
    Thing("wall", "wall", "wall", ["thick black line", "black bar", "black wall"], "#333333", "structure", ("walls", None)),
]

# Plan-language legend: the drawings come from Finland (and one from Sweden). Room labels are abbreviations.
LEGEND = {
    "OH": "olohuone = living room", "MH": "makuuhuone = bedroom", "H": "huone = room", "K / KT / KEITTIÖ": "keittiö = kitchen",
    "KK": "keittokomero = kitchenette", "RT / RUOK": "ruokailutila = dining area", "KH / KPH": "kylpyhuone = bathroom",
    "PH / PESUH": "pesuhuone = washroom", "WC": "toilet", "S": "sauna", "ET": "eteinen = entrance hall", "TK": "tuulikaappi = vestibule",
    "KÄYTÄVÄ": "corridor", "VH": "vaatehuone = walk-in closet", "PUKUH": "pukuhuone = dressing room", "KHH": "kodinhoitohuone = utility room",
    "VAR / VARASTO": "varasto = storage", "TEKN": "tekninen tila = technical room", "PARVEKE / PARV": "balcony", "TERASSI": "terrace",
    "KUISTI": "porch", "ULKOTILA": "outdoor area", "AT / AUTOTALLI / AUTOKATOS": "garage / carport", "TUPA": "farmhouse living room",
    "SOVR / KÖK / BAD / HALL": "Swedish: bedroom / kitchen / bathroom / hall", "m²": "square metres (printed on some plans)",
}


@dataclass
class PlanSet:
    key: str
    title: str
    folder: str
    masks: str
    description: str
    plans: Dict[str, dict]                    # id -> {"factor", "title", "note"} (factor: resampling of the 1 px = 1 cm source)
    default_plan: str
    compare_default: Tuple[str, str]
    things: List[Thing] = field(default_factory=lambda: list(THINGS))
    render: bool = False                      # True: clean rendering of the vector drawing; False: the dataset's scanned image

    def thing(self, name_or_key: str) -> Thing:
        for t in self.things:
            if name_or_key in (t.key, t.name, t.prompt):
                return t
        raise KeyError(name_or_key)

    @property
    def names(self) -> List[str]:
        return [t.name for t in self.things]

    @property
    def ids(self) -> List[str]:
        return list(self.plans)


SETS: Dict[str, PlanSet] = {}


def register(spec: PlanSet):
    SETS[spec.key] = spec
    return spec


register(PlanSet(
    key="homes_a",
    title="Residential floor plans, set A",
    folder="data/plans/homes_a",
    masks="data/masks/homes_a",
    description=("Six real floor plans of Finnish homes from the CubiCasa5K dataset (CC BY-NC-SA 4.0), drawn cleanly from the dataset's "
                 "vector data (black walls, light-blue windows, door arcs, fixture symbols) at a known scale with a 5 m scale bar. Each plan "
                 "comes with an answer key (every room's real area, every door, window and fixture) that the notebook uses to check your "
                 "measurements."),
    plans={   # CubiCasa5K sample id -> resampling factor (1 px = 1/factor cm) and title; 'idx' = position in the dataset's test split
        "1293": {"factor": 1.15, "title": "flat with four large rooms", "note": "idx 58; living room, bedroom, kitchen, hall, bathroom"},
        "2536": {"factor": 0.85, "title": "flat with three bedrooms", "note": "idx 01; kitchen, bathroom with bathtub, WC, walk-in closet"},
        "2090": {"factor": 1.10, "title": "small flat with a balcony", "note": "idx 06; open kitchen, washroom"},
        "6457": {"factor": 0.80, "title": "house with a dining area and stairs", "note": "idx 14; eight rooms, stairs, utility room"},
        "207": {"factor": 0.60, "title": "large house, 15 rooms", "note": "idx 35; corridor, washroom, storage, five bedrooms, fireplace"},
        "7696": {"factor": 1.30, "title": "studio flat", "note": "idx 79; one room, kitchenette, bathroom, hall"},
    },
    default_plan="1293",
    compare_default=("1293", "2536"),
    render=True,
))

register(PlanSet(
    key="homes_b",
    title="Residential floor plans, set B",
    folder="data/plans/homes_b",
    masks="data/masks/homes_b",
    description=("Seven other real floor plans from CubiCasa5K (CC BY-NC-SA 4.0): larger houses, two-storey plans with both floors on one sheet, "
                 "a Swedish-labelled plan and a small flat with printed room areas. Same scale bar, same answer keys."),
    plans={
        "14341": {"factor": 0.60, "title": "long single-storey house, 20 rooms", "note": "CAD plan with furniture"},
        "5018": {"factor": 0.45, "title": "two-storey house: ground floor (left) and upper floor (right)", "note": "both floors on one sheet; printed floor areas"},
        "1217": {"factor": 0.50, "title": "two-storey villa, Swedish labels", "note": "both floors on one sheet"},
        "8138": {"factor": 0.80, "title": "apartment with bold walls and furniture", "note": "clean plan, clear symbols"},
        "9136": {"factor": 0.85, "title": "two-storey house, both floors stacked on the sheet", "note": "CAD plan, ground floor on top"},
        "11615": {"factor": 1.30, "title": "small flat A3 with printed room areas", "note": "printed m² next to the room names"},
        "10715": {"factor": 0.90, "title": "apartment (bold walls)", "note": "bold-wall plan"},
    },
    default_plan="14341",
    compare_default=("14341", "1217"),
))
