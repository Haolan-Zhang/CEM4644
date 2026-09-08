"""Dataset registry for the MP3 object-detection lab (YOLO-format image sets)."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class DetSpec:
    key: str
    title: str
    zip_name: str
    classes: List[str]                    # YOLO class ids in order
    display: Dict[str, str]               # class -> friendly name
    colors: Dict[str, str]                # class -> box colour
    description: str
    course_model: str                     # repo-relative .pt of the course detector
    tricky_dir: str
    other_domain: Optional[str] = None    # key of the other dataset (domain-shift demo)
    count_question: str = "How many objects are in this photo?"
    count_classes: List[str] = field(default_factory=list)   # classes counted in the game / dashboard
    dashboard: Optional[dict] = None      # how to turn boxes into a site metric
    unit: str = "photo"

    def pretty(self, c: str) -> str:
        return self.display.get(c, c)


DETSETS: Dict[str, DetSpec] = {}


def register(spec: DetSpec):
    DETSETS[spec.key] = spec
    return spec


register(DetSpec(
    key="construction_safety",
    title="Workers and PPE on construction sites",
    zip_name="construction_safety.zip",
    classes=["helmet", "no-helmet", "no-vest", "person", "vest"],
    display={"helmet": "helmet", "no-helmet": "NO helmet", "no-vest": "NO vest", "person": "person", "vest": "vest"},
    colors={"person": "#4cc9f0", "helmet": "#2dc653", "vest": "#80ed99", "no-helmet": "#e63946", "no-vest": "#f77f00"},
    description=(
        "Photos of construction sites with every worker boxed as 'person', plus a box for the head (helmet or NO helmet) "
        "and the torso (vest or NO vest). Source: Roboflow 100 'construction-safety' benchmark set, CC-BY-4.0."
    ),
    course_model="models/construction_safety_yolo11n.pt",
    tricky_dir="data/tricky/construction_safety",
    other_domain="excavators",
    count_question="How many workers are NOT wearing a helmet?",
    count_classes=["no-helmet"],
    dashboard={"kind": "ppe", "good": "helmet", "bad": "no-helmet", "label": "heads with a helmet",
               "good2": "vest", "bad2": "no-vest", "label2": "torsos with a vest"},
))

register(DetSpec(
    key="excavators",
    title="Construction machinery: excavators, dump trucks, wheel loaders",
    zip_name="excavators.zip",
    classes=["EXCAVATORS", "dump truck", "wheel loader"],
    display={"EXCAVATORS": "excavator", "dump truck": "dump truck", "wheel loader": "wheel loader"},
    colors={"EXCAVATORS": "#ffb703", "dump truck": "#4cc9f0", "wheel loader": "#e63946"},
    description=(
        "Photos of earth-moving equipment on sites and roads, each machine boxed as excavator, dump truck or wheel loader. "
        "Source: Roboflow 100 'excavators' benchmark set, CC-BY-4.0."
    ),
    course_model="models/excavators_yolo11n.pt",
    tricky_dir="data/tricky/excavators",
    other_domain="construction_safety",
    count_question="How many machines (excavators, dump trucks, wheel loaders) are in this photo?",
    count_classes=["EXCAVATORS", "dump truck", "wheel loader"],
    dashboard={"kind": "count", "label": "machines counted per photo"},
    unit="photo",
))
