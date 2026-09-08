"""Dataset registry for the MP2 lab. Folder names are the class ids used on disk;
`display` maps them to the friendly names students see."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class BinarySpec:
    positive: str                 # display name of the "something is wrong" side
    negative: str                 # display name of the "all good" side
    mapping: Dict[str, str]       # folder class -> positive/negative display name

    @property
    def classes(self) -> List[str]:
        return [self.negative, self.positive]


@dataclass
class DatasetSpec:
    key: str
    title: str
    zip_name: str
    classes: List[str]                       # folder names, fixed order
    display: Dict[str, str]                  # folder name -> friendly name
    description: str
    binary: Optional[BinarySpec] = None      # how to collapse to defect / no defect
    binary_model: Optional[str] = None       # repo-relative dir of the course binary model
    multiclass_model: Optional[str] = None   # repo-relative dir of the course multi-class model
    tricky_dir: Optional[str] = None         # repo-relative dir with hand-picked hard images
    other_domain: Optional[str] = None       # key of a dataset from a different domain (for the domain-shift demo)
    zero_shot_examples: List[str] = field(default_factory=list)
    unit: str = "photo"                      # what one image is, for wording

    def pretty(self, folder: str) -> str:
        return self.display.get(folder, folder.replace("_", " "))


DATASETS: Dict[str, DatasetSpec] = {}


def register(spec: DatasetSpec):
    DATASETS[spec.key] = spec
    return spec


register(DatasetSpec(
    key="facade_defects",
    title="Building façade / wall surface defects",
    zip_name="facade_defects.zip",
    classes=["plain", "minor_crack", "major_crack", "spalling", "peeling", "stain", "algae"],
    display={
        "plain": "plain wall (no defect)",
        "minor_crack": "minor crack",
        "major_crack": "major crack",
        "spalling": "spalling",
        "peeling": "peeling paint / plaster",
        "stain": "stain",
        "algae": "algae / biological growth",
    },
    description=(
        "Close-up photos of concrete and stone walls of more than 50 buildings (10 to 60 years old), "
        "taken about 1 m from the wall with a smartphone. Source: BD3 Building Defect Dataset, CC-BY-4.0."
    ),
    binary=BinarySpec(
        positive="defect",
        negative="no defect",
        mapping={"plain": "no defect", "minor_crack": "defect", "major_crack": "defect", "spalling": "defect",
                 "peeling": "defect", "stain": "defect", "algae": "defect"},
    ),
    binary_model="models/facade_binary",
    multiclass_model="models/facade_multiclass",
    tricky_dir="data/tricky/facade_defects",
    other_domain="concrete_cracks",
    zero_shot_examples=["a brick wall", "a concrete wall", "a painted wall", "a stone wall"],
))

register(DatasetSpec(
    key="concrete_cracks",
    title="Concrete surface cracks",
    zip_name="concrete_cracks.zip",
    classes=["no_crack", "crack"],
    display={"no_crack": "no crack", "crack": "crack"},
    description=(
        "227x227 photos of concrete surfaces (floors, walls, columns) of campus buildings, half with a visible crack. "
        "Source: Concrete Crack Images for Classification (Özgenel 2019), CC-BY-4.0."
    ),
    binary=BinarySpec(positive="crack", negative="no crack", mapping={"no_crack": "no crack", "crack": "crack"}),
    binary_model="models/concrete_binary",
    multiclass_model=None,
    tricky_dir="data/tricky/concrete_cracks",
    other_domain="facade_defects",
    zero_shot_examples=["smooth concrete", "cracked concrete", "wet concrete", "a tile floor"],
))

register(DatasetSpec(
    key="facade_styles",
    title="Architectural styles of building façades (computer-generated images)",
    zip_name="facade_styles.zip",
    classes=["bauhaus_international", "brutalist", "art_deco", "neoclassical", "gothic_revival", "georgian",
             "victorian_terrace", "mid_century_modern", "contemporary_curtain_wall", "industrial_warehouse"],
    display={
        "bauhaus_international": "Bauhaus International", "brutalist": "Brutalist", "art_deco": "Art Deco",
        "neoclassical": "Neoclassical", "gothic_revival": "Gothic Revival", "georgian": "Georgian",
        "victorian_terrace": "Victorian Terrace", "mid_century_modern": "Mid-century Modern",
        "contemporary_curtain_wall": "Contemporary Curtain Wall", "industrial_warehouse": "Industrial Warehouse",
    },
    description=(
        "Computer-generated reference images of building façades in 10 architectural styles, with controlled viewing "
        "angle, crop and lighting. None of them shows a real building. Source: Jonathandav/facade-styles, MIT licence."
    ),
    binary=None,
    binary_model=None,
    multiclass_model="models/styles_multiclass",
    tricky_dir="data/tricky/facade_styles",
    other_domain="facade_defects",
    zero_shot_examples=["a modern glass office tower", "a historic stone church", "an old brick warehouse", "a concrete apartment block"],
    unit="image",
))
