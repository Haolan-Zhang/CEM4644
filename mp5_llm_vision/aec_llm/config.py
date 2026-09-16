"""What the MP5 notebooks work with: the example sets (with answer keys and the earlier labs' specialist models' answers),
the Gemini models, and the prompts and JSON schemas that the steps start from."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

MODELS = ["gemini-3.5-flash-lite", "gemini-3.5-flash", "gemini-3.8-flash", "gemini-3.6-flash"]   # first = default: the only one the free tier
                                                                                                  # lets a class use (the others: 20 requests/day)
DEFAULT_MODEL = MODELS[0]
THINKING = "low"                     # the model's private reasoning before it answers; "low" keeps replies fast and cheap


@dataclass
class PhotoSet:
    """Classification examples: a folder of photos whose true class is known (from MP2's test split)."""
    key: str
    title: str
    classes: List[str]                 # display names, fixed order
    intro: str                         # the first sentence of the classification prompt (who the model is, what the photo is)
    hints: Dict[str, str]              # class -> one-line description used by the 'with descriptions' prompt
    source: str
    specialist: str                    # the earlier lab's trained model, for comparison
    rules: str = ""                    # extra decision rules for the 'with descriptions and rules' prompt
    mp2_key: str = ""                  # dataset key in mp2_image_classification (used by build/prepare_data.py)
    mp2_model: str = ""


@dataclass
class SiteSet:
    """Detection examples: photos with every object boxed (from MP3's test split)."""
    key: str
    title: str
    classes: List[str]                 # label names the model must use
    display: Dict[str, str]
    colors: Dict[str, str]
    intro: str                         # what to detect
    label_help: str                    # how the labels are meant
    count_question: str                # the counting step's question
    count_field: str                   # the JSON field for the counted class
    count_class: str                   # which label is counted, for the answer key
    source: str
    specialist: str
    mp3_key: str = ""
    mp3_model: str = ""
    mp3_map: Dict[str, str] = field(default_factory=dict)   # MP3's raw class name -> the label used here


@dataclass
class PlanRef:
    """Segmentation examples: floor plans with answer keys, copied from MP4."""
    title: str
    mp4_set: str
    ids: List[str]
    description: str


@dataclass
class LabSpec:
    key: str
    title: str
    photos: PhotoSet
    sites: SiteSet
    plans: PlanRef
    folder: str = ""                   # data/<key>

    def __post_init__(self):
        self.folder = self.folder or f"data/{self.key}"


SPECS: Dict[str, LabSpec] = {}


def register(spec: LabSpec):
    SPECS[spec.key] = spec
    return spec


register(LabSpec(
    key="workshop",
    title="Façade defects, site safety, floor plans",
    photos=PhotoSet(
        key="facade_defects", title="Building façade / wall surface defects",
        classes=["plain wall (no defect)", "minor crack", "major crack", "spalling", "peeling paint / plaster", "stain", "algae / biological growth"],
        intro="You are inspecting the wall of a building. This is a close-up photo of a concrete or rendered wall surface, taken about one metre away.",
        hints={
            "plain wall (no defect)": "an intact surface: texture, joints, dirt-free or lightly weathered, nothing broken",
            "minor crack": "a hairline crack, thin as a pencil line, the surface otherwise intact",
            "major crack": "a wide or branching crack you could put a coin into, often with displaced edges",
            "spalling": "a piece of the concrete or render has broken off, leaving a crater with a rough bottom",
            "peeling paint / plaster": "paint or a thin plaster layer lifting off in flakes or sheets",
            "stain": "a discoloured patch (rust, water, efflorescence) on an otherwise intact surface",
            "algae / biological growth": "green, black or dark grey biological growth, usually where the wall stays damp",
        },
        source="BD3 Building Defect Dataset (CC-BY-4.0), the test photos of MP2",
        specialist="MP2 course model (ConvNeXt V2 femto trained on 1,400 photos)",
        rules=("Rules: look for the most severe defect first; a crack you could put a coin into is 'major crack' even if it is short; "
               "if the surface is only discoloured and nothing is broken or lifting, it is 'stain' or 'algae / biological growth' "
               "(green or black growth = algae); if nothing at all is wrong, 'plain wall (no defect)'."),
        mp2_key="facade_defects", mp2_model="models/facade_multiclass",
    ),
    sites=SiteSet(
        key="construction_safety", title="Workers and PPE on construction sites",
        classes=["person", "helmet", "no-helmet", "vest", "no-vest"],
        display={"person": "person", "helmet": "helmet", "no-helmet": "NO helmet", "vest": "vest", "no-vest": "NO vest"},
        colors={"person": "#4cc9f0", "helmet": "#2dc653", "vest": "#80ed99", "no-helmet": "#e63946", "no-vest": "#f77f00"},
        intro="This is a photo of a construction site. Detect every worker and their safety equipment.",
        label_help=("'person' is the whole worker; 'helmet' is a head with a hard hat and 'no-helmet' a worker's head without one; "
                    "'vest' is a torso with a high-visibility vest and 'no-vest' a torso without one. One box per worker, one per head, one per torso."),
        count_question="How many workers in this photo are NOT wearing a helmet, and how many workers are there in total?",
        count_field="workers_without_helmet", count_class="no-helmet",
        source="Roboflow 100 'construction-safety' (CC-BY-4.0), the test photos of MP3",
        specialist="MP3 course model (YOLO11n trained on 300 photos)",
        mp3_key="construction_safety", mp3_model="models/construction_safety_yolo11n.pt",
        mp3_map={"person": "person", "helmet": "helmet", "no-helmet": "no-helmet", "vest": "vest", "no-vest": "no-vest"},
    ),
    plans=PlanRef(title="Floor plans (clean drawings)", mp4_set="homes_a", ids=["1293", "2536", "2090"],
                  description="three of the MP4 workshop plans: clean renderings of real Finnish homes with every room's real area in the answer key"),
))

register(LabSpec(
    key="homework",
    title="Architectural styles, machinery, scanned floor plans",
    photos=PhotoSet(
        key="facade_styles", title="Architectural styles of building façades (computer-generated images)",
        classes=["Bauhaus International", "Brutalist", "Art Deco", "Neoclassical", "Gothic Revival", "Georgian",
                 "Victorian Terrace", "Mid-century Modern", "Contemporary Curtain Wall", "Industrial Warehouse"],
        intro="You are an architectural historian. This is a computer-generated reference image of a building façade.",
        hints={
            "Bauhaus International": "flat white or plain walls, ribbon windows, no ornament, 1920s to 1950s",
            "Brutalist": "raw exposed concrete, massive blocky forms, 1950s to 1970s",
            "Art Deco": "stepped forms, vertical emphasis, geometric ornament, 1920s to 1930s",
            "Neoclassical": "columns, pediments, symmetry, stone, inspired by Greek and Roman temples",
            "Gothic Revival": "pointed arches, tracery, pinnacles, steep gables",
            "Georgian": "brick, symmetrical sash windows in rows, a plain classical door surround, 18th century",
            "Victorian Terrace": "rows of brick houses with bay windows and decorated brick or stone details, 19th century",
            "Mid-century Modern": "low horizontal lines, large glass, wood and stone, 1945 to 1970",
            "Contemporary Curtain Wall": "a glass skin over the whole façade, recent office or apartment tower",
            "Industrial Warehouse": "brick or steel shed with large repeated windows or loading doors, functional",
        },
        source="Jonathandav/facade-styles (MIT licence; computer-generated images, no real building), the test images of MP2",
        specialist="MP2 course model (ConvNeXt V2 femto trained on 380 photos)",
        rules=("Rules: judge by the façade itself (ornament, window rhythm, materials), not by the weather or the age of the photo; "
               "a glass skin over the whole façade is 'Contemporary curtain wall' even next to old buildings; raw concrete without ornament is "
               "'Brutalist'; columns and a pediment are 'Neoclassical' unless the windows are pointed (then 'Gothic Revival')."),
        mp2_key="facade_styles", mp2_model="models/styles_multiclass",
    ),
    sites=SiteSet(
        key="excavators", title="Construction machinery: excavators, dump trucks, wheel loaders",
        classes=["excavator", "dump truck", "wheel loader"],
        display={"excavator": "excavator", "dump truck": "dump truck", "wheel loader": "wheel loader"},
        colors={"excavator": "#f4a261", "dump truck": "#457b9d", "wheel loader": "#2a9d8f"},
        intro="This is a photo of construction machinery at work. Detect every machine.",
        label_help=("'excavator' has tracks or wheels, a rotating cab and a digging arm with a bucket; 'dump truck' is a road or off-road "
                    "truck with a tipping body; 'wheel loader' has four large wheels, a fixed cab and a front bucket on lifting arms. One box per machine."),
        count_question="How many excavators are in this photo, and how many machines in total?",
        count_field="excavators", count_class="excavator",
        source="Roboflow 100 'excavators' (CC-BY-4.0), the test photos of MP3",
        specialist="MP3 course model (YOLO11n trained on the machinery set)",
        mp3_key="excavators", mp3_model="models/excavators_yolo11n.pt",
        mp3_map={"EXCAVATORS": "excavator", "dump truck": "dump truck", "wheel loader": "wheel loader"},
    ),
    plans=PlanRef(title="Floor plans (scanned drawings)", mp4_set="homes_b", ids=["8138", "10715", "11615"],
                  description="three of the MP4 homework plans: scanned real drawings with furniture and dimension strings, with the same answer keys"),
))


# ----------------------------------------------------------------------------- prompts and schemas
# Every prompt is a template; {classes}, {intro}, {label_help}, {question} and {hints} are filled in from the spec.
PROMPTS: Dict[str, str] = {
    "describe": "{intro} {question}",
    # classification: three wordings the students compare, and one JSON-in-the-prompt version for the structured-output step
    "classify_basic": ("{intro} Classify it into exactly one of these categories: {classes}. "
                       "Reply with JSON only, no other text, in this form: "
                       "{{\"label\": <one category, spelled exactly as in the list>, \"confidence\": <a number from 0 to 1>, \"reason\": <one short sentence>}}"),
    "classify_described": ("{intro} Classify it into exactly one of these categories:\n{hints}\n"
                           "Reply with JSON only, no other text, in this form: "
                           "{{\"label\": <one category, spelled exactly as in the list>, \"confidence\": <a number from 0 to 1>, \"reason\": <one short sentence>}}"),
    "classify_careful": ("{intro} Classify it into exactly one of these categories:\n{hints}\n{rules} "
                         "Reply with JSON only, no other text, in this form: "
                         "{{\"label\": <one category, spelled exactly as in the list>, \"confidence\": <a number from 0 to 1>, \"reason\": <one short sentence>}}"),
    # detection
    "detect": ("{intro} Output a JSON list where each entry has \"label\" (one of: {classes}) and \"box_2d\" as [ymin, xmin, ymax, xmax] "
               "normalized to 0-1000. {label_help}"),
    "count": ("{question} Reply with JSON only, no other text, in this form: "
              "{{\"{count_field}\": <integer>, \"total\": <integer>, \"reason\": <one short sentence>}}"),
    # segmentation on a plan
    "rooms_masks": ("This is an architectural floor plan. Give the segmentation masks for every room: the floor area inside the walls of each room, "
                    "not the outdoor areas (balcony, terrace, ULKOTILA). Output a JSON list of segmentation masks where each entry contains the 2D bounding "
                    "box in the key \"box_2d\" ([ymin, xmin, ymax, xmax] normalized to 0-1000), the segmentation mask in the key \"mask\", and the room "
                    "label as printed on the plan in the key \"label\"."),
    "rooms_boxes": ("This is an architectural floor plan. Detect every room: the floor area inside the walls of each room, not the outdoor areas "
                    "(balcony, terrace, ULKOTILA). Output a JSON list where each entry has \"label\" (the room label printed on the plan, e.g. OH, MH, K) "
                    "and \"box_2d\" as [ymin, xmin, ymax, xmax] normalized to 0-1000, tight to the inside faces of the walls."),
}
CLASSIFY_PROMPTS = {"basic": "classify_basic", "with descriptions": "classify_described", "with descriptions and rules": "classify_careful"}
DESCRIBE_QUESTIONS = ["What do you see in this photo? Answer in three sentences.",
                      "Is there anything a building inspector should worry about here? Answer in two sentences.",
                      "Describe this photo as a JSON object with the keys \"what\", \"condition\" and \"action\"."]


def fill(template_key: str, spec: LabSpec, **extra) -> str:
    """The prompt text a step starts from."""
    p, s = spec.photos, spec.sites
    fields = dict(
        intro=p.intro, classes="; ".join(p.classes),
        hints="\n".join(f"- {c}: {p.hints.get(c, '')}" for c in p.classes),
        label_help=s.label_help, question=s.count_question, count_field=s.count_field, rules=p.rules,
    )
    if template_key == "detect":
        fields["intro"] = s.intro; fields["classes"] = ", ".join(s.classes)
    fields.update(extra)
    return PROMPTS[template_key].format(**fields)


def classify_schema(classes: List[str]) -> dict:
    return {"type": "object",
            "properties": {"label": {"type": "string", "enum": list(classes)}, "confidence": {"type": "number"}, "reason": {"type": "string"}},
            "required": ["label", "confidence", "reason"]}


def detect_schema(classes: List[str]) -> dict:
    return {"type": "array", "items": {"type": "object",
            "properties": {"label": {"type": "string", "enum": list(classes)},
                           "box_2d": {"type": "array", "items": {"type": "integer"}, "minItems": 4, "maxItems": 4}},
            "required": ["label", "box_2d"]}}


def count_schema(count_field: str) -> dict:
    return {"type": "object", "properties": {count_field: {"type": "integer"}, "total": {"type": "integer"}, "reason": {"type": "string"}},
            "required": [count_field, "total", "reason"]}


def rooms_schema(with_mask: bool) -> dict:
    props = {"label": {"type": "string"}, "box_2d": {"type": "array", "items": {"type": "integer"}, "minItems": 4, "maxItems": 4}}
    if with_mask:
        props["mask"] = {"type": "array", "items": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2}}
    return {"type": "array", "items": {"type": "object", "properties": props, "required": list(props)}}


def describe_schema() -> dict:
    return {"type": "object", "properties": {"what": {"type": "string"}, "condition": {"type": "string"}, "action": {"type": "string"}},
            "required": ["what", "condition", "action"]}
