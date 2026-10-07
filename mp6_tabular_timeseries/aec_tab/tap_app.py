"""Gradio app for the tap test: upload recordings named by material, get one row per tap, train and test a model, predict a new recording.
Opened from a link (works on a phone, so students can upload straight from where they recorded)."""
import tempfile
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import taps, ui
from .app import fig_png, serve
from .texts import fill, labels

# every title, label and result line of the app; the notebook cell passes its own copy (labels_text)
LABELS = {
    "title": "## Tap Test",
    "make_header": "### 1 · Build the Dataset",
    "make_note": ("After you build the dataset, the app shows it as a table (one row per tap, with the features of each tap) and shows each "
                  "recording with the detected taps marked by red lines."),
    "upload": "recordings (one file per spot, named by material and spot number)",
    "make_button": "Build dataset",
    "summary": "**{taps} taps** in **{recordings} recordings** of **{materials} materials**: {details}",
    "download": "dataset (CSV)",
    "table": "dataset: one row per tap",
    "waves": "recordings with the detected taps (red lines)",
    "wave_title": "{recording}: {taps} taps",
    "train_header": "### 2 · Train and Evaluate",
    "train_note": "The app trains the model on a random 80 % of the taps (training data) and tests it on the other 20 % (test data).",
    "model": "model",
    "models": "straight line | decision tree | neural network",
    "train_button": "Train and evaluate",
    "score": "**Accuracy: {right} of {n} test taps ({accuracy} %)** · {model}",
    "matrix": "confusion matrix (rows: actual; columns: predicted)",
    "new_header": "### 3 · Predict a New Recording",
    "new_file": "new recording (a spot not in the dataset)",
    "new_button": "Predict",
    "new_result": "**{material}**: predicted for {votes} of {n} taps{others}. Model: {model}, trained on all {train} taps of the dataset.",
    "new_table": "model output for each tap",
    "link": "Open the app in a new tab (it also works on a phone): {url}\nThe link stays active while this notebook is running.",
}
L = dict(LABELS)

MAX_WAVES = 12


def _paths(files):
    if not files:
        return []
    files = files if isinstance(files, list) else [files]
    return [f if isinstance(f, str) else getattr(f, "name", str(f)) for f in files]


def _view(df: pd.DataFrame) -> pd.DataFrame:
    v = df.copy()
    for c in taps.MEASURES:
        v[c] = v[c].round(2 if c.endswith("share") else 1)
    return v.rename(columns=taps.MEASURES)


DATASET_COLUMNS = ["recording", "material", "tap", *taps.MEASURES]
NEW_COLUMNS = ["tap", "predicted", *taps.MEASURES]


def _empty(columns) -> pd.DataFrame:
    return _view(pd.DataFrame(columns=columns))


def _model_key(name: str) -> str:
    """The model list shows the wordings of the 'models' label; they map, in order, to the three models."""
    names = [m.strip() for m in L["models"].split("|")]
    return list(taps.MODELS)[names.index(name)] if name in names else list(taps.MODELS)[0]


def _colors(materials):
    cmap = matplotlib.colormaps["tab10"]
    return {m: cmap(k % 10) for k, m in enumerate(sorted(materials))}


def waves_png(audio: dict, df: pd.DataFrame):
    names = list(audio)[:MAX_WAVES]
    col = _colors(df.material.unique())
    fig, axes = plt.subplots(len(names), 1, figsize=(12, 1.5 * len(names)), squeeze=False)
    for ax, name in zip(axes[:, 0], names):
        x, on = audio[name]
        t = np.arange(len(x)) / taps.SR
        ax.plot(t, x, lw=0.3, color=col.get(taps.material_of(name), "k"))
        for i in on:
            ax.axvline(i / taps.SR, color="r", lw=0.6)
        ax.set_title(fill(L["wave_title"], recording=name, taps=len(on)), fontsize=9, loc="left"); ax.tick_params(labelsize=7)
    axes[-1, 0].set_xlabel("seconds")
    fig.tight_layout()
    return fig_png(fig)


def make_table(files):
    paths = _paths(files)
    if not paths:
        return None, "Upload your recordings first.", _empty(DATASET_COLUMNS), None, None
    bad = [Path(p).name for p in paths if not Path(p).suffix.lower() in taps.AUDIO]
    paths = [p for p in paths if Path(p).suffix.lower() in taps.AUDIO]
    try:
        df, audio = taps.table(paths)
    except Exception as e:  # noqa: BLE001
        return None, f"Could not read a recording: {e}", _empty(DATASET_COLUMNS), None, None
    if df.empty:
        return None, "No taps found: tap harder, closer to the phone, in a quieter room.", _empty(DATASET_COLUMNS), None, None
    per = df.groupby(["material", "recording"]).size().reset_index(name="taps")
    details = ", ".join(f"{m} ({g.recording.nunique()} recording{'s' if g.recording.nunique() > 1 else ''}, {len(g)} taps)" for m, g in df.groupby("material"))
    lines = [fill(L["summary"], taps=len(df), recordings=df.recording.nunique(), materials=df.material.nunique(), details=details)]
    few = per[per.taps < 3]
    if len(few):
        lines.append("Fewer than 3 taps found in: " + ", ".join(few.recording) + ".")
    if bad:
        lines.append("Not a sound file, skipped: " + ", ".join(bad) + ".")
    out = Path(tempfile.gettempdir()) / "tap_table.csv"
    _view(df).to_csv(out, index=False)                    # the same column names as the table
    return df, "\n\n".join(lines), _view(df), str(out), waves_png(audio, df)


def train_test(df, model):
    empty = pd.DataFrame(columns=["actual"])
    if df is None or df.empty:
        return "Build the dataset first.", empty
    if df.material.nunique() < 2:
        return "Record at least two materials.", empty
    res = taps.evaluate(df, _model_key(model))
    right = int((res.material == res.predicted).sum())
    score = fill(L["score"], right=right, n=len(res), accuracy=f"{right / len(res) * 100:.0f}", model=model)
    return score, taps.confusion(res)


def predict_new(df, model, file):
    empty = _empty(NEW_COLUMNS)
    if df is None or df.empty:
        return "Build the dataset first.", empty
    paths = _paths(file)
    if not paths:
        return "Upload one recording of a spot that is not in the dataset.", empty
    _, _, rows = taps.read_recording(paths[0])
    if not rows:
        return "No taps found in that recording.", empty
    new = pd.DataFrame(rows)
    m = taps.fit_all(df, _model_key(model))
    new["predicted"] = m.predict(new[list(taps.MEASURES)].values)
    votes = new.predicted.value_counts()
    others = (" (others: " + ", ".join(f"{k} {v}" for k, v in votes.iloc[1:].items()) + ")") if len(votes) > 1 else ""
    text = fill(L["new_result"], material=votes.index[0], votes=votes.iloc[0], n=len(new), others=others, model=model, train=len(df))
    return text, _view(new[NEW_COLUMNS])


def build(labels_text=None):
    import gradio as gr
    L.clear(); L.update(labels(labels_text, LABELS))
    model_names = [m.strip() for m in L["models"].split("|")]
    with gr.Blocks(title="Tap Test") as demo:
        gr.Markdown(L["title"] + "\n" + L["make_header"] + "\n\n" + L["make_note"])
        state = gr.State(None)
        with gr.Row():
            up = gr.File(label=L["upload"], file_count="multiple", type="filepath")
            with gr.Column():
                go = gr.Button(L["make_button"], variant="primary")
                summary = gr.Markdown()
                dl = gr.File(label=L["download"], interactive=False)
        tbl = gr.Dataframe(value=_empty(DATASET_COLUMNS), label=L["table"], interactive=False, wrap=True)
        waves = gr.Image(label=L["waves"], type="pil", interactive=False)
        gr.Markdown(L["train_header"] + "\n\n" + L["train_note"])
        model = gr.Dropdown(model_names, value=model_names[0], label=L["model"])
        run = gr.Button(L["train_button"], variant="primary")
        score = gr.Markdown()
        conf = gr.Dataframe(value=pd.DataFrame(columns=["actual"]), label=L["matrix"], interactive=False)
        gr.Markdown(L["new_header"])
        with gr.Row():
            new = gr.File(label=L["new_file"], file_count="single", type="filepath")
            guess = gr.Button(L["new_button"], variant="primary")
        verdict = gr.Markdown()
        new_tbl = gr.Dataframe(value=_empty(NEW_COLUMNS), label=L["new_table"], interactive=False)

        go.click(make_table, [up], [state, summary, tbl, dl, waves])
        run.click(train_test, [state, model], [score, conf])
        guess.click(predict_new, [state, model, new], [verdict, new_tbl])
    return demo


def launch(share=None, labels_text=None):
    demo = build(labels_text)
    return serve(demo, share, L["link"])
