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
    ax[0].set_xlabel(f"measured {target}"); ax[0].set_ylabel("predicted"); ax[0].set_title(f"{len(Xte)} held-out rows")
    ax[1].barh([X.columns[i] for i in order][::-1][-12:], imp[order][::-1][-12:], color=ui.BLUE); ax[1].set_title("what matters")
    txt = (f"Trained on {len(Xtr)} rows, scored on {len(Xte)} it never saw.\nAverage miss (MAE): {mean_absolute_error(yte, p):.3g}   R²: {r2_score(yte, p):.3f}\n"
           f"Most important inputs: " + ", ".join(X.columns[i] for i in order[:5]))
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
    ax.plot(hist.index[-4 * horizon:], hist.values[-4 * horizon:], color=ui.GREY, lw=1, label="history")
    ax.plot(actual.index, actual.values, "k", lw=1.6, label="actual"); ax.plot(actual.index, naive.values, color=ui.GREY, ls="--", label="same as one period earlier")
    ax.plot(actual.index, tree.values, color=ui.GREEN, label="trees"); ax.legend(fontsize=8); ax.set_title(f"the last {horizon} values, forecast from the data before them")
    step_txt = f"{step.total_seconds() / 3600:g} h" if step and step < pd.Timedelta("2D") else f"{step.days} day(s)" if step else "?"
    txt = (f"{len(s)} values, typical step {step_txt}. Forecast horizon: the last {horizon} values.\n"
           f"Average miss: same-as-earlier {mean_absolute_error(actual, naive):.3g}, trees {mean_absolute_error(actual, tree):.3g}.")
    return fig_png(fig), txt


def build(kind=None):
    import gradio as gr
    with gr.Blocks(title="Your own table or time series") as demo:
        gr.Markdown("## Your own table or time series\nUpload a CSV. **Table**: pick the column to predict; decision trees are fitted on 80 % of the rows and scored on the rest. "
                    "**Time series**: pick the time column and the value column; the last period is forecast from the data before it.")
        state = gr.State(None)
        with gr.Row():
            with gr.Column(scale=1):
                up = gr.File(label="CSV file", file_types=[".csv"])
                mode = gr.Radio(["table", "time series"], value=kind or "table", label="what is it")
                target = gr.Dropdown([], label="column to predict (table)")
                tcol = gr.Dropdown([], label="time column (time series)"); vcol = gr.Dropdown([], label="value column (time series)")
                run = gr.Button("Run", variant="primary")
            with gr.Column(scale=2):
                img = gr.Image(type="pil", label="result", interactive=False); txt = gr.Textbox(label="numbers", lines=5)

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


def launch(share=None, kind=None):
    demo = build(kind)
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        demo.launch(share=True if share is None else share, inline=False, debug=False, quiet=True, show_error=True, prevent_thread_lock=True)
    url = getattr(demo, "share_url", None)
    if url:
        print(f"Open the app in a new tab (works on a phone too): {url}")
        print("The link stays alive while this notebook is running.")
    else:
        print(f"The public link could not be created (network hiccup). Run this cell again; the app is running at {getattr(demo, 'local_url', 'the local address')}.")
    return demo
