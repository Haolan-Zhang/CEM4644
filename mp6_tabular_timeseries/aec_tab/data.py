"""The lab's data files: one table, four meters."""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .config import TableSpec


def load_table(root: Path, spec: TableSpec) -> pd.DataFrame:
    df = pd.read_csv(root / "data" / spec.file)
    return df[spec.features + [spec.target]]


@dataclass
class Meter:
    id: str
    df: pd.DataFrame                # index: hourly timestamps; columns kWh, air_temp_C
    info: dict

    @property
    def kwh(self) -> pd.Series:
        return self.df["kWh"]

    @property
    def temp(self) -> pd.Series:
        return self.df["air_temp_C"]

    @property
    def use(self) -> str:
        return self.info["use"]

    @property
    def label(self) -> str:
        return f"{self.id}: {self.use}, {self.info['sqm']:,} m²"


def load_meters(root: Path, ids: List[str]) -> Dict[str, Meter]:
    info = json.loads((root / "data" / "meters.json").read_text())
    out = {}
    for b in ids:
        df = pd.read_csv(root / "data" / "meters" / f"{b}.csv", parse_dates=["timestamp"]).set_index("timestamp")
        out[b] = Meter(b, df, info[b])
    return out


def credits(root: Path) -> dict:
    return json.loads((root / "data" / "credits.json").read_text())
