"""The tap test: find each tap in a phone recording, measure it, and train a model to name the surface.

One recording = one spot on one surface; the file name says the material ("wooden table 2.m4a" -> "wooden table").
"""
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SR = 48000
AUDIO = (".m4a", ".wav", ".mp3", ".aac", ".caf", ".ogg", ".webm", ".3gp", ".amr", ".flac", ".mp4", ".mov")

# measurement -> the words students see
MEASURES = {
    "pitch_Hz": "pitch (Hz)",
    "ring_ms": "ring time (ms)",
    "low_share": "low share (below 300 Hz)",
}
# the three models of the app, in the order of its model list; the straight line and decision tree models are the same
# kinds as in Parts 1-3 (models.fit_classifier), with leaves small enough for a few dozen taps
MODELS = {
    "line": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
    "tree": lambda: HistGradientBoostingClassifier(min_samples_leaf=3, random_state=0),
    "net": lambda: make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=3000, random_state=0)),
}


# ----------------------------------------------------------------------------- reading
def material_of(name: str) -> str:
    """'Wooden Table 2.m4a' -> 'wooden table'; 'metal_stand-3.wav' -> 'metal stand'."""
    stem = Path(name).stem
    stem = re.sub(r"[\s_\-.()]*\d+[\s_\-.()]*$", "", stem)            # a spot number at the end
    stem = re.sub(r"[_\-]+", " ", stem)
    return re.sub(r"\s+", " ", stem).strip().lower() or "unnamed"


def decode(path) -> np.ndarray:
    """Mono samples at 48 kHz, from any phone recording (ffmpeg in Colab; PyAV or a plain WAV reader otherwise)."""
    path = str(path)
    if shutil.which("ffmpeg"):
        out = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                             capture_output=True, check=True).stdout
        return np.frombuffer(out, dtype=np.float32).astype(np.float64)
    try:
        import av
        c = av.open(path); s = c.streams.audio[0]
        rs = av.AudioResampler(format="flt", layout="mono", rate=SR)
        parts = []
        for fr in c.decode(s):
            for g in rs.resample(fr):
                parts.append(g.to_ndarray().reshape(-1))
        return np.concatenate(parts).astype(np.float64)
    except ImportError:
        pass
    from scipy.io import wavfile
    sr, x = wavfile.read(path)
    x = x.astype(np.float64)
    if x.ndim > 1:
        x = x.mean(1)
    if np.abs(x).max() > 1.5:
        x = x / 32768.0
    if sr != SR:
        x = np.interp(np.arange(0, len(x) * SR / sr) * sr / SR, np.arange(len(x)), x)
    return x


# ----------------------------------------------------------------------------- finding and measuring taps
def envelope(x: np.ndarray) -> np.ndarray:
    hop = int(0.005 * SR)
    return np.sqrt(np.convolve(x ** 2, np.ones(hop) / hop, mode="same"))


def find_taps(x: np.ndarray, min_gap: float = 0.25, skip_start: float = 0.15) -> Tuple[List[int], np.ndarray, float]:
    """Onsets where the 5-ms loudness jumps well above the room; the first 0.15 s (the recording's start click) is skipped."""
    env = envelope(x)
    floor = float(np.median(env)) + 1e-9
    thr = max(floor * 8, float(env[int(skip_start * SR):].max(initial=0)) * 0.08)
    above = env > thr
    on = np.flatnonzero(above & ~np.r_[False, above[:-1]])
    on = on[on > int(skip_start * SR)]
    keep: List[int] = []
    for i in on:
        if not keep or (i - keep[-1]) / SR > min_gap:
            keep.append(int(i))
    return keep, env, floor


def spectrum(x: np.ndarray, i: int, n: int = 4096) -> Tuple[np.ndarray, np.ndarray]:
    seg = x[max(i - int(0.002 * SR), 0): max(i - int(0.002 * SR), 0) + n]
    seg = np.pad(seg, (0, n - len(seg)))
    return np.fft.rfftfreq(n, 1 / SR), np.abs(np.fft.rfft(seg * np.hanning(n)))


def measure(x: np.ndarray, i: int, env: np.ndarray, win: float = 0.25) -> Dict[str, float]:
    hz, sp = spectrum(x, i)
    band = (hz >= 80) & (hz <= 12000)
    f, e = hz[band], sp[band] ** 2
    peak_i = i + int(np.argmax(env[i: i + int(0.03 * SR)]))
    pk = env[peak_i]
    after = env[peak_i: peak_i + int(win * SR)]
    below = np.flatnonzero(after < pk * 0.1)                       # 20 dB down
    return {"pitch_Hz": float(f[np.argmax(e)]), "ring_ms": float(below[0] / SR * 1000 if len(below) else win * 1000),
            "low_share": float(e[f < 300].sum() / (e.sum() + 1e-12))}


def read_recording(path, name: str = None):
    """(samples, taps, rows) for one recording."""
    name = name or Path(path).name
    x = decode(path)
    on, env, _ = find_taps(x)
    rows = [{"recording": name, "material": material_of(name), "tap": k + 1, **measure(x, i, env)} for k, i in enumerate(on)]
    return x, on, rows


def table(paths: List[str]) -> Tuple[pd.DataFrame, Dict[str, tuple]]:
    """One row per tap over all recordings, and each recording's samples and taps (for the pictures)."""
    rows, audio = [], {}
    for p in paths:
        name = Path(p).name
        x, on, r = read_recording(p, name)
        audio[name] = (x, on)
        rows += r
    df = pd.DataFrame(rows, columns=["recording", "material", "tap", *MEASURES])
    return df, audio


# ----------------------------------------------------------------------------- training and testing
def evaluate(df: pd.DataFrame, model: str, features: List[str] = None):
    """Each test tap's true and predicted material: trained on a random 80 % of the taps, tested on the other 20 %."""
    features = features or list(MEASURES)
    X, y = df[features].values, df.material.values
    idx = np.arange(len(df))
    strat = y if min(pd.Series(y).value_counts()) >= 2 else None
    tr, te = train_test_split(idx, test_size=0.2, random_state=0, stratify=strat)
    out = df.iloc[te][["recording", "material", "tap"]].copy()
    out["predicted"] = MODELS[model]().fit(X[tr], y[tr]).predict(X[te])
    return out


def confusion(res: pd.DataFrame) -> pd.DataFrame:
    labels = sorted(set(res.material) | set(res.predicted))
    m = pd.crosstab(res.material, res.predicted).reindex(index=labels, columns=labels, fill_value=0)
    m.columns = [f"predicted: {l}" for l in m.columns]
    return m.rename_axis("actual").reset_index()


def fit_all(df: pd.DataFrame, model: str):
    return MODELS[model]().fit(df[list(MEASURES)].values, df.material.values)
