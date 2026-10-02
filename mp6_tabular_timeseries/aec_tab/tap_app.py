"""Gradio app for the tap test: upload recordings named by material, get one row per tap, look, train, test, blind test.
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
    "upload": "recordings (one file per spot, named by material and spot number)",
    "make_button": "Build dataset",
    "summary": "**{taps} taps** in **{recordings} recordings** of **{materials} materials**: {details}",
    "download": "dataset (CSV)",
    "table": "dataset: one row per tap",
    "waves": "recordings with the detected taps (red lines)",
    "wave_title": "{recording}: {taps} taps",
    "look_header": "### 2 · Explore the Features",
    "spectrum": "mean spectrum of one tap, per material",
    "spectrum_title": "Mean spectrum of one tap, per material",
    "spectrum_x": "frequency (Hz)",
    "spectrum_y": "relative power",
    "x_feature": "x-axis feature",
    "y_feature": "y-axis feature",
    "scatter": "feature scatter plot (one point per tap)",
    "train_header": "### 3 · Train and Evaluate",
    "model": "model",
    "split": "evaluation",
    "train_button": "Train and evaluate",
    "score": "**Accuracy: {right} of {n} test taps ({accuracy} %)** · {model} · {split}",
    "matrix": "confusion matrix (rows: actual; columns: predicted)",
    "per_recording": "results per recording",
    "rules": "rules learned by the decision tree (trained on all taps)",
    "new_header": "### 4 · Predict a New Recording",
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


def spectra_png(audio: dict, df: pd.DataFrame):
    col = _colors(df.material.unique())
    by = {}
    for name, (x, on) in audio.items():
        for i in on:
            hz, sp = taps.spectrum(x, i)
            by.setdefault(taps.material_of(name), []).append(sp ** 2)
    fig, ax = plt.subplots(figsize=(10, 3.6))
    for m, specs in sorted(by.items()):
        s = np.mean(specs, 0); keep = (hz >= 50) & (hz <= 4000)
        ax.semilogy(hz[keep], s[keep] / s[keep].max(), color=col[m], lw=1.2, label=f"{m} ({len(specs)} taps)")
    ax.set_xlabel(L["spectrum_x"]); ax.set_ylabel(L["spectrum_y"]); ax.legend(fontsize=8); ax.set_title(L["spectrum_title"], fontsize=10)
    fig.tight_layout()
    return fig_png(fig)


def scatter_png(df: pd.DataFrame, x_label: str, y_label: str):
    inv = {v: k for k, v in taps.MEASURES.items()}
    xc, yc = inv.get(x_label, "pitch_Hz"), inv.get(y_label, "ring_ms")
    col = _colors(df.material.unique())
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for m, g in df.groupby("material"):
        ax.scatter(g[xc], g[yc], s=28, alpha=0.8, color=col[m], label=m)
    ax.set_xlabel(taps.MEASURES[xc]); ax.set_ylabel(taps.MEASURES[yc]); ax.legend(fontsize=8)
    fig.tight_layout()
    return fig_png(fig)


def make_table(files):
    paths = _paths(files)
    if not paths:
        return None, "Upload your recordings first.", None, None, None, None, None
    bad = [Path(p).name for p in paths if not Path(p).suffix.lower() in taps.AUDIO]
    paths = [p for p in paths if Path(p).suffix.lower() in taps.AUDIO]
    try:
        df, audio = taps.table(paths)
    except Exception as e:  # noqa: BLE001
        return None, f"Could not read a recording: {e}", None, None, None, None, None
    if df.empty:
        return None, "No taps found: tap harder, closer to the phone, in a quieter room.", None, None, None, None, None
    per = df.groupby(["material", "recording"]).size().reset_index(name="taps")
    details = ", ".join(f"{m} ({g.recording.nunique()} recording{'s' if g.recording.nunique() > 1 else ''}, {len(g)} taps)" for m, g in df.groupby("material"))
    lines = [fill(L["summary"], taps=len(df), recordings=df.recording.nunique(), materials=df.material.nunique(), details=details)]
    few = per[per.taps < 3]
    if len(few):
        lines.append("Fewer than 3 taps found in: " + ", ".join(few.recording) + ".")
    if bad:
        lines.append("Not a sound file, skipped: " + ", ".join(bad) + ".")
    out = Path(tempfile.gettempdir()) / "tap_table.csv"
    df.round(3).to_csv(out, index=False)
    return (df, "\n\n".join(lines), _view(df), str(out), waves_png(audio, df), spectra_png(audio, df),
            scatter_png(df, taps.MEASURES["pitch_Hz"], taps.MEASURES["ring_ms"]))


def train_test(df, model, split):
    if df is None or df.empty:
        return "Make the table first.", None, None, ""
    if df.material.nunique() < 2:
        return "Record at least two materials.", None, None, ""
    res = taps.evaluate(df, model, split)
    lines = []
    if taps.SPLITS.get(split) == "spot":
        single = taps.single_recording_materials(df)
        if single:
            lines.append("Not evaluated (only one recording, so no training data for that material): " + ", ".join(single) + ".")
    if res.empty:
        return "\n\n".join(lines + ["Nothing could be evaluated: record at least two spots of each material."]), None, None, ""
    right = int((res.material == res.predicted).sum())
    lines.insert(0, fill(L["score"], right=right, n=len(res), accuracy=f"{right / len(res) * 100:.0f}", model=model, split=split))
    per = (res.assign(right=res.material == res.predicted).groupby(["recording", "material"])
           .agg(taps=("tap", "size"), correct=("right", "sum"),
                most_frequent_prediction=("predicted", lambda s: s.value_counts().index[0])).reset_index())
    per.columns = [c.replace("_", " ") for c in per.columns]
    rules = taps.tree_rules(df) if model.startswith("decision tree") else ""
    return "\n\n".join(lines), taps.confusion(res).reset_index(names=""), per, rules


def predict_new(df, model, file):
    if df is None or df.empty:
        return "Make the table first.", None
    paths = _paths(file)
    if not paths:
        return "Upload one recording of a spot that is not in the dataset.", None
    _, _, rows = taps.read_recording(paths[0])
    if not rows:
        return "No taps found in that recording.", None
    new = pd.DataFrame(rows)
    m = taps.fit_all(df, model)
    new["predicted"] = m.predict(new[list(taps.MEASURES)].values)
    votes = new.predicted.value_counts()
    others = (" (others: " + ", ".join(f"{k} {v}" for k, v in votes.iloc[1:].items()) + ")") if len(votes) > 1 else ""
    text = fill(L["new_result"], material=votes.index[0], votes=votes.iloc[0], n=len(new), others=others, model=model, train=len(df))
    return text, _view(new[["tap", "time_s", "predicted", *taps.MEASURES]])


def build(labels_text=None):
    import gradio as gr
    L.clear(); L.update(labels(labels_text, LABELS))
    names = list(taps.MEASURES.values())
    with gr.Blocks(title="Tap Test") as demo:
        gr.Markdown(L["title"] + "\n" + L["make_header"])
        state = gr.State(None)
        with gr.Row():
            up = gr.File(label=L["upload"], file_count="multiple", type="filepath")
            with gr.Column():
                go = gr.Button(L["make_button"], variant="primary")
                summary = gr.Markdown()
                dl = gr.File(label=L["download"], interactive=False)
        tbl = gr.Dataframe(label=L["table"], interactive=False, wrap=True)
        waves = gr.Image(label=L["waves"], type="pil", interactive=False)
        gr.Markdown(L["look_header"])
        spec = gr.Image(label=L["spectrum"], type="pil", interactive=False)
        with gr.Row():
            xs = gr.Dropdown(names, value=taps.MEASURES["pitch_Hz"], label=L["x_feature"])
            ys = gr.Dropdown(names, value=taps.MEASURES["ring_ms"], label=L["y_feature"])
        sc = gr.Image(label=L["scatter"], type="pil", interactive=False)
        gr.Markdown(L["train_header"])
        with gr.Row():
            model = gr.Dropdown(list(taps.MODELS), value=list(taps.MODELS)[0], label=L["model"])
            split = gr.Radio(list(taps.SPLITS), value=list(taps.SPLITS)[0], label=L["split"])
        run = gr.Button(L["train_button"], variant="primary")
        score = gr.Markdown()
        conf = gr.Dataframe(label=L["matrix"], interactive=False)
        per = gr.Dataframe(label=L["per_recording"], interactive=False)
        rules = gr.Textbox(label=L["rules"], lines=8, interactive=False)
        gr.Markdown(L["new_header"])
        with gr.Row():
            new = gr.File(label=L["new_file"], file_count="single", type="filepath")
            guess = gr.Button(L["new_button"], variant="primary")
        verdict = gr.Markdown()
        new_tbl = gr.Dataframe(label=L["new_table"], interactive=False)

        go.click(make_table, [up], [state, summary, tbl, dl, waves, spec, sc])
        for d in (xs, ys):
            d.change(lambda df, a, b: scatter_png(df, a, b) if df is not None else None, [state, xs, ys], [sc])
        run.click(train_test, [state, model, split], [score, conf, per, rules])
        guess.click(predict_new, [state, model, new], [verdict, new_tbl])
    return demo


def launch(share=None, labels_text=None):
    demo = build(labels_text)
    return serve(demo, share, L["link"])
