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
    close_enough: float = 5.0                           # a miss this small counts as close, in the chat steps

    @property
    def rows(self) -> str:
        """The plural of one row's word."""
        return {"mix": "mixes"}.get(self.row_word, self.row_word + "s")

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
    key="concrete", title="Concrete mixes and their strength", file="concrete.csv", target="strength_MPa", unit="MPa",
    target_label="compressive strength",
    features=["cement", "slag", "fly_ash", "water", "superplasticizer", "coarse_agg", "fine_agg", "age_days"],
    labels={"cement": "cement (kg/m³)", "slag": "blast-furnace slag (kg/m³)", "fly_ash": "fly ash (kg/m³)", "water": "water (kg/m³)",
            "superplasticizer": "superplasticizer (kg/m³)", "coarse_agg": "coarse aggregate (kg/m³)", "fine_agg": "fine aggregate (kg/m³)",
            "age_days": "age at test (days)", "strength_MPa": "compressive strength (MPa)"},
    grades=[("low (< 25 MPa)", 0, 25), ("normal (25-45 MPa)", 25, 45), ("high (> 45 MPa)", 45, 1e9)],
    spec_default=30, spec_range=(15, 60, 5), whatif=["cement", "water", "age_days", "superplasticizer"], row_word="mix",
    description="1,030 concrete mixes tested in a laboratory: what went into each cubic metre, how old the sample was, and the strength it reached.",
    chat_about=("concrete mixes that were tested in a laboratory: the amount of each ingredient in kg per cubic metre of concrete "
                "(cement, blast-furnace slag, fly ash, water, superplasticizer, coarse aggregate, fine aggregate), the age of the "
                "sample when it was tested (age_days), and the compressive strength it reached (strength_MPa)"),
    close_enough=5.0,
)

ENERGY = TableSpec(
    key="energy_efficiency", title="Building shapes and their heating load", file="energy_efficiency.csv", target="heating_load",
    unit="kWh/m²", target_label="heating load",
    features=["compactness", "surface_area", "wall_area", "roof_area", "height", "orientation", "glazing_area", "glazing_distribution"],
    labels={"compactness": "relative compactness", "surface_area": "surface area (m²)", "wall_area": "wall area (m²)", "roof_area": "roof area (m²)",
            "height": "overall height (m)", "orientation": "orientation (2 = N, 3 = E, 4 = S, 5 = W)", "glazing_area": "glazing area (share of floor)",
            "glazing_distribution": "glazing distribution (0-5)", "heating_load": "heating load (kWh/m²)"},
    grades=[("A (< 12)", 0, 12), ("B (12-20)", 12, 20), ("C (20-30)", 20, 30), ("D (> 30)", 30, 1e9)],
    spec_default=20, spec_range=(8, 40, 2), whatif=["compactness", "glazing_area", "height", "surface_area"], row_word="building",
    description="768 simulated residential buildings of the same volume but different shapes, glazing and orientation, with the heating load a building-energy simulator computed for each.",
    chat_about=("residential buildings of the same volume but different shapes, simulated with a building-energy program: relative "
                "compactness, surface area, wall area and roof area (m²), overall height (m), orientation (2 = north, 3 = east, "
                "4 = south, 5 = west), glazing area (as a share of the floor area), glazing distribution (0-5), and the heating "
                "load the simulator computed (heating_load, kWh per m² of floor)"),
    close_enough=2.0,
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
