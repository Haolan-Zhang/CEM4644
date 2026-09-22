"""The building meters: next week three ways, and the days that do not fit the building's usual pattern."""
import contextlib
import logging
import os
import time
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .config import FORECASTER_ID, HOLIDAYS_2017, HORIZON, TEST_START
from .data import Meter

METHODS = {"same hour last week": "naive", "decision trees (last weeks + calendar + temperature)": "trees",
           "Chronos-Bolt (a pretrained forecasting model, zero-shot)": "chronos"}


@contextlib.contextmanager
def quiet():
    """Nothing printed while a model downloads or loads."""
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    restore = []
    try:
        from huggingface_hub.utils import disable_progress_bars, enable_progress_bars
        disable_progress_bars(); restore.append(enable_progress_bars)
    except Exception:  # noqa: BLE001
        pass
    for name in ("huggingface_hub", "transformers", "chronos", "torch"):
        lg = logging.getLogger(name); lvl = lg.level; lg.setLevel(logging.ERROR); restore.append(lambda lg=lg, lvl=lvl: lg.setLevel(lvl))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                yield
    finally:
        for f in restore:
            try:
                f()
            except Exception:  # noqa: BLE001
                pass


class Forecaster:
    """Chronos-Bolt: a model pretrained on many time series, used as it comes (no training on our meters)."""

    def __init__(self, model_id: str = FORECASTER_ID):
        import torch
        from chronos import BaseChronosPipeline
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        with quiet():
            self.pipe = BaseChronosPipeline.from_pretrained(model_id, device_map=self.device)
        self.torch = torch

    def predict(self, history: pd.Series, horizon: int = HORIZON, context: int = 512):
        ctx = self.torch.tensor(history.values[-context:], dtype=self.torch.float32)
        with quiet():
            q, _ = self.pipe.predict_quantiles(ctx, prediction_length=horizon, quantile_levels=[0.1, 0.5, 0.9])
        q = q[0].numpy()
        return q[:, 1], q[:, 0], q[:, 2]


@dataclass
class Forecast:
    method: str
    median: np.ndarray
    low: Optional[np.ndarray] = None
    high: Optional[np.ndarray] = None
    seconds: float = 0.0
    mae: float = 0.0
    coverage: Optional[float] = None


def test_window(meter: Meter, start: str = TEST_START, horizon: int = HORIZON):
    t0 = pd.Timestamp(start)
    idx = pd.date_range(t0, periods=horizon, freq="h")
    return meter.kwh[meter.kwh.index < t0], meter.kwh.reindex(idx)


def _features(s: pd.Series, temp: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"y": s})
    for lag in (168, 336):
        df[f"lag{lag}"] = df.y.shift(lag)
    df["hour"] = df.index.hour; df["dow"] = df.index.dayofweek
    df["temp"] = temp.reindex(df.index).interpolate().bfill().ffill().values
    return df


def forecast(meter: Meter, method: str, forecaster: Optional[Forecaster] = None, start: str = TEST_START, horizon: int = HORIZON) -> Forecast:
    m = METHODS.get(method, method)
    hist, actual = test_window(meter, start, horizon)
    t = time.time()
    low = high = None
    if m == "naive":
        med = meter.kwh.shift(168).reindex(actual.index).values
    elif m == "trees":
        df = _features(meter.kwh, meter.temp)
        train = df[df.index < actual.index[0]].dropna()
        feats = ["lag168", "lag336", "hour", "dow", "temp"]
        gb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, random_state=0).fit(train[feats], train.y)
        med = gb.predict(df.reindex(actual.index)[feats])
    elif m == "chronos":
        if forecaster is None:
            raise RuntimeError("The pretrained forecaster is not loaded: run Step 0 with load_forecaster ticked.")
        med, low, high = forecaster.predict(hist, horizon)
    else:
        raise KeyError(method)
    f = Forecast(m, np.asarray(med, dtype=float), low, high, time.time() - t)
    ok = ~np.isnan(actual.values)
    f.mae = float(np.mean(np.abs(f.median[ok] - actual.values[ok])))
    if low is not None:
        f.coverage = float(np.mean((actual.values[ok] >= low[ok]) & (actual.values[ok] <= high[ok])))
    return f


def usual_pattern(s: pd.Series) -> pd.Series:
    """The building's typical value for each weekday and hour (the median over the year)."""
    prof = s.groupby([s.index.dayofweek, s.index.hour]).median()
    return pd.Series([prof[(d, h)] for d, h in zip(s.index.dayofweek, s.index.hour)], index=s.index)


def odd_days(meter: Meter, threshold: float = 3.5) -> pd.DataFrame:
    """Each day's mean deviation from the usual pattern, in robust standard units; flagged above the threshold."""
    s = meter.kwh
    dev = (s - usual_pattern(s)).resample("D").mean()
    med = dev.median(); mad = (dev - med).abs().median()
    z = (dev - med) / (1.4826 * mad + 1e-9)
    out = pd.DataFrame({"deviation_kWh_per_h": dev.round(1), "z": z.round(2)})
    out["flag"] = z.abs() > threshold
    out["weekday"] = out.index.day_name()
    out["holiday"] = [HOLIDAYS_2017.get(d.strftime("%Y-%m-%d"), "") for d in out.index]
    return out
