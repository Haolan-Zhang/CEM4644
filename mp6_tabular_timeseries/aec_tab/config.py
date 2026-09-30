"""What each MP6 notebook works on: one table (regression and classification) and four building meters."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

SEED = 4644
TEST_SIZE = 0.2
TEST_START = "2017-10-16"        # the week the forecasts are scored on (Monday)
HORIZON = 168                    # hours
FORECASTER_ID = "amazon/chronos-bolt-small"


@dataclass
class TableSpec:
    key: str
    title: str
    file: str
    target: str
    unit: str
    target_label: str                                   # "strength", "heating load"
    features: List[str]
    labels: Dict[str, str]                              # column -> words
    grades: List[Tuple[str, float, float]]              # (name, low, high) on the target
    spec_default: float                                 # the pass / fail specification the slider starts at
    spec_range: Tuple[float, float, float]              # min, max, step of that slider
    whatif: List[str]                                   # the inputs that get a slider in the what-if step
    row_word: str = "mix"                               # what one row is
    guess_n: int = 5
    description: str = ""
    chat_about: str = ""                                # the table in words, for the chat prompt ("824 " + this)
    close_pct: float = 10                               # an error within this share of the measured value counts as close (chat steps)
    decimals: int = 1                                   # how the target's numbers are shown
    guess_ratio: Optional[Tuple[str, str, str]] = None  # (label, numerator, denominator) worked out for the guessing game

    @property
    def rows(self) -> str:
        """The plural of one row's word."""
        return {"mix": "mixes"}.get(self.row_word, self.row_word + "s")

    def fmt(self, v: float) -> str:
        """A value of the target (or an error on it), as the notebook shows it."""
        return f"{v:,.{self.decimals}f}"

    def label(self, col: str) -> str:
        return self.labels.get(col, col.replace("_", " "))

    def grade_of(self, v: float) -> str:
        for name, lo, hi in self.grades:
            if lo <= v < hi:
                return name
        return self.grades[-1][0]


@dataclass
class SeriesSpec:
    key: str
    title: str
    meters: List[str]
    description: str = ""


@dataclass
class LabSpec:
    key: str
    title: str
    table: TableSpec
    series: SeriesSpec


CONCRETE = TableSpec(
    key="concrete", title="Concrete mixes and their strength", file="concrete.csv", target="strength_psi", unit="psi",
    target_label="compressive strength",
    features=["cement", "water", "coarse_agg", "fine_agg", "slag", "fly_ash", "superplasticizer", "age_days"],
    labels={"cement": "cement (lb/yd³)", "water": "water (lb/yd³)", "coarse_agg": "coarse aggregate (lb/yd³)", "fine_agg": "fine aggregate (lb/yd³)",
            "slag": "slag (lb/yd³)", "fly_ash": "fly ash (lb/yd³)", "superplasticizer": "superplasticizer (lb/yd³)",
            "age_days": "age at test (days)", "strength_psi": "compressive strength (psi)"},
    grades=[("low (< 3,500 psi)", 0, 3500), ("normal (3,500-6,500 psi)", 3500, 6500), ("high (> 6,500 psi)", 6500, 1e9)],
    spec_default=4000, spec_range=(2000, 9000, 500), whatif=["cement", "water", "age_days"], row_word="mix",
    description="1,030 concrete mixes tested in a laboratory: what went into each cubic yard, how old the sample was, and the strength it reached.",
    chat_about=("concrete mixes that were tested in a laboratory: the amount of each ingredient in pounds per cubic yard of concrete "
                "(cement, water, coarse aggregate, fine aggregate, blast-furnace slag, fly ash, superplasticizer), the age of the "
                "sample when it was tested (age_days), and the compressive strength it reached in psi (strength_psi)"),
    close_pct=10, decimals=0, guess_ratio=("water / cement", "water", "cement"),
)

ENERGY = TableSpec(
    key="energy_efficiency", title="Building shapes and their heating load", file="energy_efficiency.csv", target="heating_load",
    unit="kBtu/ft²", target_label="heating load",
    features=["compactness", "surface_area", "wall_area", "roof_area", "height", "orientation", "glazing_area", "glazing_distribution"],
    labels={"compactness": "relative compactness", "surface_area": "surface area (ft²)", "wall_area": "wall area (ft²)", "roof_area": "roof area (ft²)",
            "height": "overall height (ft)", "orientation": "orientation (2 = N, 3 = E, 4 = S, 5 = W)", "glazing_area": "glazing area (share of floor)",
            "glazing_distribution": "glazing distribution (0-5)", "heating_load": "heating load (kBtu/ft²)"},
    grades=[("A (< 4)", 0, 4), ("B (4-6.5)", 4, 6.5), ("C (6.5-9.5)", 6.5, 9.5), ("D (> 9.5)", 9.5, 1e9)],
    spec_default=6.5, spec_range=(2.5, 12.5, 0.5), whatif=["compactness", "glazing_area", "height", "surface_area"], row_word="building",
    description="768 simulated residential buildings of the same volume but different shapes, glazing and orientation, with the heating load a building-energy simulator computed for each.",
    chat_about=("residential buildings of the same volume but different shapes, simulated with a building-energy program: relative "
                "compactness, surface area, wall area and roof area (ft²), overall height (ft), orientation (2 = north, 3 = east, "
                "4 = south, 5 = west), glazing area (as a share of the floor area), glazing distribution (0-5), and the heating "
                "load the simulator computed (heating_load, kBtu per ft² of floor)"),
    close_pct=10, decimals=2,
)

SPECS: Dict[str, LabSpec] = {
    "workshop": LabSpec("workshop", "Concrete mixes, and four buildings' electricity", CONCRETE,
                        SeriesSpec("workshop", "Four buildings, hourly electricity in 2017",
                                   ["Hog_office_Marlena", "Bear_education_Lila", "Bear_lodging_Evan", "Bear_assembly_Jose"])),
    "homework": LabSpec("homework", "Building shapes, and four other buildings' electricity", ENERGY,
                        SeriesSpec("homework", "Four other buildings, hourly electricity in 2017",
                                   ["Hog_office_Gustavo", "Moose_education_Leland", "Robin_lodging_Janie", "Rat_assembly_Rolland"])),
}

# a US calendar for the odd-days step (the meters are on North American campuses)
HOLIDAYS_2017 = {"2017-01-01": "New Year's Day", "2017-01-02": "New Year's Day (observed)", "2017-01-16": "Martin Luther King Day",
                 "2017-02-20": "Presidents' Day", "2017-05-29": "Memorial Day", "2017-07-04": "Independence Day",
                 "2017-09-04": "Labor Day", "2017-10-09": "Columbus Day", "2017-11-10": "Veterans Day (observed)",
                 "2017-11-23": "Thanksgiving", "2017-11-24": "day after Thanksgiving", "2017-12-25": "Christmas Day",
                 "2017-12-26": "day after Christmas"}
