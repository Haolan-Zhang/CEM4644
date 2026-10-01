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
        ax.set_title(f"{name}: {len(on)} taps", fontsize=9, loc="left"); ax.tick_params(labelsize=7)
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
    ax.set_xlabel("pitch (Hz)"); ax.set_ylabel("loudness (relative)"); ax.legend(fontsize=8); ax.set_title("average sound of one tap, per material", fontsize=10)
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
    lines = [f"**{len(df)} taps** in **{df.recording.nunique()} recordings** of **{df.material.nunique()} materials**: "
             + ", ".join(f"{m} ({g.recording.nunique()} recording{'s' if g.recording.nunique() > 1 else ''}, {len(g)} taps)" for m, g in df.groupby("material"))]
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
            lines.append("Not tested (only one recording, so the model has nothing to learn it from): " + ", ".join(single) + ".")
    if res.empty:
        return "\n\n".join(lines + ["Nothing could be tested: record at least two spots of each material."]), None, None, ""
    right = int((res.material == res.predicted).sum())
    lines.insert(0, f"**{right} of {len(res)} test taps right ({right / len(res) * 100:.0f} %)**, {model}, {split}.")
    per = (res.assign(right=res.material == res.predicted).groupby(["recording", "material"])
           .agg(taps=("tap", "size"), right=("right", "sum"),
                most_often_called=("predicted", lambda s: s.value_counts().index[0])).reset_index())
    rules = taps.tree_rules(df) if model.startswith("small decision tree") else ""
    return "\n\n".join(lines), taps.confusion(res).reset_index(names=""), per, rules


def blind(df, model, file):
    if df is None or df.empty:
        return "Make the table first.", None
    paths = _paths(file)
    if not paths:
        return "Upload one recording of a spot you did not record before.", None
    _, _, rows = taps.read_recording(paths[0])
    if not rows:
        return "No taps found in that recording.", None
    new = pd.DataFrame(rows)
    m = taps.fit_all(df, model)
    new["predicted"] = m.predict(new[list(taps.MEASURES)].values)
    votes = new.predicted.value_counts()
    text = (f"**{votes.index[0]}**: {votes.iloc[0]} of {len(new)} taps"
            + (" (others: " + ", ".join(f"{k} {v}" for k, v in votes.iloc[1:].items()) + ")" if len(votes) > 1 else "")
            + f". Trained on all {len(df)} taps of your table with {model}.")
    return text, _view(new[["tap", "time_s", "predicted", *taps.MEASURES]])


def build():
    import gradio as gr
    names = list(taps.MEASURES.values())
    with gr.Blocks(title="The tap test") as demo:
        gr.Markdown("## The tap test\n### 1 · Make the table")
        state = gr.State(None)
        with gr.Row():
            up = gr.File(label="your recordings", file_count="multiple", type="filepath")
            with gr.Column():
                go = gr.Button("Make the table", variant="primary")
                summary = gr.Markdown()
                dl = gr.File(label="download the table (CSV)", interactive=False)
        tbl = gr.Dataframe(label="one row per tap", interactive=False, wrap=True)
        waves = gr.Image(label="each recording, with the taps found (red)", type="pil", interactive=False)
        gr.Markdown("### 2 · Look")
        spec = gr.Image(label="the average tap of each material", type="pil", interactive=False)
        with gr.Row():
            xs = gr.Dropdown(names, value=taps.MEASURES["pitch_Hz"], label="across")
            ys = gr.Dropdown(names, value=taps.MEASURES["ring_ms"], label="up")
        sc = gr.Image(label="one dot per tap", type="pil", interactive=False)
        gr.Markdown("### 3 · Train and test")
        with gr.Row():
            model = gr.Dropdown(list(taps.MODELS), value=list(taps.MODELS)[0], label="model")
            split = gr.Radio(list(taps.SPLITS), value=list(taps.SPLITS)[0], label="test on")
        run = gr.Button("Train and test", variant="primary")
        score = gr.Markdown()
        conf = gr.Dataframe(label="truth (rows) against the model (columns)", interactive=False)
        per = gr.Dataframe(label="each recording", interactive=False)
        rules = gr.Textbox(label="the rules the small decision tree learned (from all your taps)", lines=8, interactive=False)
        gr.Markdown("### 4 · Blind test")
        with gr.Row():
            new = gr.File(label="one new recording", file_count="single", type="filepath")
            guess = gr.Button("What is it?", variant="primary")
        verdict = gr.Markdown()
        new_tbl = gr.Dataframe(label="each tap of the new recording", interactive=False)

        go.click(make_table, [up], [state, summary, tbl, dl, waves, spec, sc])
        for d in (xs, ys):
            d.change(lambda df, a, b: scatter_png(df, a, b) if df is not None else None, [state, xs, ys], [sc])
        run.click(train_test, [state, model, split], [score, conf, per, rules])
        guess.click(blind, [state, model, new], [verdict, new_tbl])
    return demo


def launch(share=None):
    return serve(build(), share)
