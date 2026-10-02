"""Gradio app for the students' own CSV: a table (pick the answer column, fit trees, score on held-out rows) or a time
series (a timestamp column and a value column: forecast the last week two ways). Opened from a link, never embedded."""
import contextlib
import io

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from . import ui
from .texts import fill, labels

# every title, label and result line of the app; the notebook cell passes its own copy (labels_text)
LABELS = {
    "title": "## Your Own Data",
    "intro": "Upload a CSV file. **Table:** choose the column to predict; gradient-boosted trees are trained on 80 % of the rows and tested on the other 20 %. **Time series:** choose the time column and the value column; the last period is forecast from the data before it.",
    "upload": "CSV file",
    "mode": "data type",
    "target": "column to predict (table)",
    "time": "time column (time series)",
    "value": "value column (time series)",
    "run_button": "Run",
    "result": "result",
    "numbers": "results",
    "table_plot_title": "Predicted vs. actual ({n_test} test rows)",
    "importance_title": "Permutation importance",
    "table_result": "Trained on {n_train} rows, tested on {n_test} rows not used for training.\nMAE: {mae}   R²: {r2}\nMost important inputs: {top}",
    "series_plot_title": "The last {horizon} values, forecast from the data before them",
    "history": "history",
    "actual": "actual",
    "naive": "seasonal naive (same value one period earlier)",
    "trees": "gradient-boosted trees",
    "series_result": "{n} values, time step {step}. Forecast horizon: the last {horizon} values.\nMAE: seasonal naive {naive_mae}, gradient-boosted trees {trees_mae}.",
    "link": "Open the app in a new tab (it also works on a phone): {url}\nThe link stays active while this notebook is running.",
}
L = dict(LABELS)


def read_csv(file):
    df = pd.read_csv(file.name if hasattr(file, "name") else file)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def fig_png(fig):
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=100, bbox_inches="tight"); plt.close(fig); buf.seek(0)
    from PIL import Image
    return Image.open(buf).convert("RGB")


def fit_table(df: pd.DataFrame, target: str):
    num = df.select_dtypes(include=[np.number])
    if target not in num.columns:
        return None, f"'{target}' is not a numeric column."
    X = num.drop(columns=[target]).dropna(axis=1, how="all"); y = num[target]
    keep = y.notna() & X.notna().all(axis=1)
    X, y = X[keep], y[keep]
    if len(X) < 30 or X.shape[1] == 0:
        return None, f"Need at least 30 complete rows and one numeric input column (have {len(X)} rows, {X.shape[1]} inputs)."
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
    m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, random_state=0).fit(Xtr, ytr)
    p = m.predict(Xte)
    imp = permutation_importance(m, Xte, yte, n_repeats=5, random_state=0).importances_mean
    order = np.argsort(-imp)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].scatter(yte, p, s=12, alpha=0.6, color=ui.BLUE); lim = [min(yte.min(), p.min()), max(yte.max(), p.max())]; ax[0].plot(lim, lim, "k--", lw=1)
    ax[0].set_xlabel(f"actual {target}"); ax[0].set_ylabel(f"predicted {target}"); ax[0].set_title(fill(L["table_plot_title"], n_test=len(Xte)))
    ax[1].barh([X.columns[i] for i in order][::-1][-12:], imp[order][::-1][-12:], color=ui.BLUE); ax[1].set_title(L["importance_title"])
    txt = fill(L["table_result"], n_train=len(Xtr), n_test=len(Xte), mae=f"{mean_absolute_error(yte, p):.3g}", r2=f"{r2_score(yte, p):.3f}",
               top=", ".join(X.columns[i] for i in order[:5]))
    return fig_png(fig), txt


def fit_series(df: pd.DataFrame, time_col: str, value_col: str):
    try:
        t = pd.to_datetime(df[time_col], errors="coerce")
    except Exception as e:  # noqa: BLE001
        return None, f"Could not read '{time_col}' as dates: {e}"
    s = pd.Series(pd.to_numeric(df[value_col], errors="coerce").values, index=t).dropna().sort_index()
    if len(s) < 60:
        return None, f"Need at least 60 timed values (have {len(s)})."
    step = s.index.to_series().diff().median()
    per_week = int(round(pd.Timedelta("7D") / step)) if step and step > pd.Timedelta(0) else 0
    horizon = max(min(per_week, len(s) // 4), 7)
    hist, actual = s.iloc[:-horizon], s.iloc[-horizon:]
    naive = s.shift(horizon).iloc[-horizon:] if per_week and per_week == horizon else pd.Series(hist.iloc[-1], index=actual.index)
    # trees on lags
    lagk = horizon
    d = pd.DataFrame({"y": s}); d["lag"] = d.y.shift(lagk); d["lag2"] = d.y.shift(2 * lagk); d["pos"] = np.arange(len(d)) % max(lagk, 1)
    train = d.iloc[:-horizon].dropna()
    if len(train) < 20:
        tree = naive
    else:
        gb = HistGradientBoostingRegressor(max_iter=200, random_state=0).fit(train[["lag", "lag2", "pos"]], train.y)
        tree = pd.Series(gb.predict(d.iloc[-horizon:][["lag", "lag2", "pos"]].fillna(hist.iloc[-1])), index=actual.index)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(hist.index[-4 * horizon:], hist.values[-4 * horizon:], color=ui.GREY, lw=1, label=L["history"])
    ax.plot(actual.index, actual.values, "k", lw=1.6, label=L["actual"]); ax.plot(actual.index, naive.values, color=ui.GREY, ls="--", label=L["naive"])
    ax.plot(actual.index, tree.values, color=ui.GREEN, label=L["trees"]); ax.legend(fontsize=8); ax.set_title(fill(L["series_plot_title"], horizon=horizon))
    step_txt = f"{step.total_seconds() / 3600:g} h" if step and step < pd.Timedelta("2D") else f"{step.days} day(s)" if step else "?"
    txt = fill(L["series_result"], n=len(s), step=step_txt, horizon=horizon, naive_mae=f"{mean_absolute_error(actual, naive):.3g}",
               trees_mae=f"{mean_absolute_error(actual, tree):.3g}")
    return fig_png(fig), txt


def build(kind=None, labels_text=None):
    import gradio as gr
    L.clear(); L.update(labels(labels_text, LABELS))
    with gr.Blocks(title="Your Own Data") as demo:
        gr.Markdown(L["title"] + "\n" + L["intro"])
        state = gr.State(None)
        with gr.Row():
            with gr.Column(scale=1):
                up = gr.File(label=L["upload"], file_types=[".csv"])
                mode = gr.Radio(["table", "time series"], value=kind or "table", label=L["mode"])
                target = gr.Dropdown([], label=L["target"])
                tcol = gr.Dropdown([], label=L["time"]); vcol = gr.Dropdown([], label=L["value"])
                run = gr.Button(L["run_button"], variant="primary")
            with gr.Column(scale=2):
                img = gr.Image(type="pil", label=L["result"], interactive=False); txt = gr.Textbox(label=L["numbers"], lines=5)

        def on_up(f):
            if f is None:
                return None, gr.update(choices=[]), gr.update(choices=[]), gr.update(choices=[]), ""
            df = read_csv(f); cols = list(df.columns); num = list(df.select_dtypes(include=[np.number]).columns)
            return df, gr.update(choices=num, value=num[-1] if num else None), gr.update(choices=cols, value=cols[0]), gr.update(choices=num, value=num[-1] if num else None), f"{len(df)} rows, {len(cols)} columns: {', '.join(cols[:12])}{' ...' if len(cols) > 12 else ''}"

        def on_run(df, mode_v, target_v, t_v, v_v):
            if df is None:
                return None, "Upload a CSV first."
            return fit_table(df, target_v) if mode_v == "table" else fit_series(df, t_v, v_v)
        up.change(on_up, [up], [state, target, tcol, vcol, txt])
        run.click(on_run, [state, mode, target, tcol, vcol], [img, txt])
    return demo


def launch(share=None, kind=None, labels_text=None):
    demo = build(kind, labels_text)
    return serve(demo, share, L["link"])


def serve(demo, share=None, link=None):
    """Start a Gradio app in the background and print only its public link."""
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        demo.launch(share=True if share is None else share, inline=False, debug=False, quiet=True, show_error=True, prevent_thread_lock=True)
    url = getattr(demo, "share_url", None)
    if url:
        print(fill(link or LABELS["link"], url=url))
    else:
        print(f"The public link could not be created (network hiccup). Run this cell again; the app is running at {getattr(demo, 'local_url', 'the local address')}.")
    return demo
