"""What each step shows: pictures, short tables, and the few widgets (a guess, sliders, a matching game)."""
import io
from typing import Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import HTML, display

from . import models, series
from .config import TableSpec
from .texts import TEXTS, fill, say, text

BLUE, GREEN, RED, GREY, ORANGE, PURPLE = "#457b9d", "#2a9d8f", "#e63946", "#8d99ae", "#f4a261", "#7b2cbf"


def show(fig, dpi: int = 100):
    from IPython.display import Image
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight"); plt.close(fig)
    display(Image(buf.getvalue()))


def _decimals(col: pd.Series) -> int:
    """As few decimals as the column needs (0 to 2), the same for every row."""
    v = col.dropna().to_numpy(dtype=float)
    for d in (0, 1):
        if np.allclose(v * 10 ** d, np.round(v * 10 ** d), atol=1e-6):
            return d
    return 2


def table(df: pd.DataFrame, max_rows: int = 30):
    view = df.head(max_rows)
    fmt = {c: (lambda v, d=_decimals(view[c]): f"{v:,.{d}f}") for c in view.columns
           if pd.api.types.is_numeric_dtype(view[c]) and not pd.api.types.is_bool_dtype(view[c])}
    display(HTML(view.to_html(index=isinstance(df.index, pd.DatetimeIndex) or df.index.name is not None,
                              formatters=fmt, border=0)))


def confusion(mat: pd.DataFrame):
    """The confusion matrix with its row and column names (Actual / Predicted)."""
    display(HTML(mat.to_html(border=0).replace("<th>", "<th style='text-align:center'>")))


# ----------------------------------------------------------------------------- Part 1: the table
def show_table(lab, rows: int = 10, texts: Optional[dict] = None, scatter: bool = True):
    spec, df = lab.table_spec, lab.df
    G = "show_table"
    output = spec.label(spec.target)
    say(texts, G, "summary_text", n_rows=f"{len(df):,}", rows=spec.rows, n_inputs=len(spec.features), output=output, unit=spec.unit)
    view = df.sample(rows, random_state=1).sort_index(); view.columns = [spec.label(c) for c in view.columns]
    table(view.reset_index(drop=True))
    fig, ax = plt.subplots(1, 2 if scatter else 1, figsize=(11, 3.4) if scatter else (5.5, 3.4), squeeze=False)
    ax = ax[0]
    ax[0].hist(df[spec.target], bins=30, color=BLUE); ax[0].set_xlabel(output); ax[0].set_ylabel(spec.rows)
    ax[0].set_title(fill(text(texts, G, "hist_title_text"), output=output), fontsize=10)
    if not scatter:                          # the histogram only (MP6A homework)
        show(fig); return
    top = spec.whatif[0]
    ax[1].scatter(df[top], df[spec.target], s=8, alpha=0.4, color=BLUE); ax[1].set_xlabel(spec.label(top)); ax[1].set_ylabel(output)
    ax[1].set_title(fill(text(texts, G, "scatter_title_text"), input=spec.label(top).split(" (")[0], output=output.split(" (")[0]), fontsize=10)
    show(fig)


def guess_game(lab, texts: Optional[dict] = None):
    """Five rows without their output: the student picks a level for each, the notebook scores it."""
    import ipywidgets as w
    spec, df = lab.table_spec, lab.df
    _, Xte, _, yte = models.split(df, spec)
    rng = np.random.default_rng(7)
    idx = rng.choice(len(Xte), size=spec.guess_n, replace=False)
    rows = Xte.iloc[idx]; truth = yte.iloc[idx]
    grades = [g[0] for g in spec.grades]
    view = rows.copy(); view.columns = [spec.label(c) for c in view.columns]; view.index = [f"{spec.row_word} {i + 1}" for i in range(len(view))]
    if spec.guess_ratio:
        name, top, bottom = spec.guess_ratio
        view[name] = (rows[top] / rows[bottom]).round(2).values
    table(view)
    picks = [w.Dropdown(options=grades, value=grades[len(grades) // 2], description=f"{spec.row_word} {i + 1}", layout=w.Layout(width="330px")) for i in range(len(rows))]
    btn = w.Button(description=text(texts, "guess", "button_text"), button_style="primary"); out = w.Output()

    def go(_):
        g = [b.value for b in picks]; tg = [spec.grade_of(v) for v in truth.values]
        right = sum(a == b for a, b in zip(g, tg))
        lab.guesses = {"rows": rows.index.tolist(), "guess": g, "truth_grade": tg, "truth": truth.values.tolist(), "right": right}
        with out:
            out.clear_output(wait=True)
            res = pd.DataFrame({"your guess": g, "actual": tg, f"actual ({spec.unit})": [spec.fmt(v) for v in truth.values], "": ["✓" if a == b else "✗" for a, b in zip(g, tg)]}, index=view.index)
            table(res)
            say(texts, "guess", "result_text", right=right, n=len(g), rows=spec.rows)
    btn.on_click(go)
    display(w.VBox([*picks, btn, out]))


# ----------------------------------------------------------------------------- Part 2: regression and classification
REGRESSION_TEXT = TEXTS["regression"]["result_text"]


def md_text(template: str, **values):
    """A result written in the notebook (markdown, one line per line), with the numbers filled in."""
    from IPython.display import Markdown
    s = fill(template, **values)
    display(Markdown("  \n".join(s.strip().split("\n"))))


def _model_name(kind: str) -> str:
    return kind[:1].upper() + kind[1:]


def regression_view(lab, kind: str, result_text: Optional[str] = None, texts: Optional[dict] = None):
    spec, df = lab.table_spec, lab.df
    texts = dict(texts or {}, **({"result_text": result_text} if result_text else {}))
    G = "regression"
    Xtr, Xte, ytr, yte = models.split(df, spec)
    model = models.fit_regressor(kind, Xtr, ytr)
    r = models.score_regression(model, kind, Xte, yte, spec)
    lab.models[("reg", r.kind)] = model; lab.results[("reg", r.kind)] = {"mae": r.mae, "r2": r.r2}
    output = spec.label(spec.target)
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    ax.scatter(r.y_true, r.y_pred, s=14, alpha=0.6, color=BLUE); lim = [min(r.y_true.min(), r.y_pred.min()), max(r.y_true.max(), r.y_pred.max())]
    ax.plot(lim, lim, "k--", lw=1); ax.set_xlabel(f"actual {output}"); ax.set_ylabel(f"predicted {output}")
    import textwrap
    ax.set_title(textwrap.fill(fill(text(texts, G, "plot_title_text"), output=output.split(" (")[0], n_test=len(Xte), rows=spec.rows), 48), fontsize=10)
    show(fig)
    mae = spec.fmt(r.mae) if spec.decimals == 0 else f"{r.mae:.2f}"
    say(texts, G, "result_text", model=_model_name(kind), n_train=len(Xtr), n_test=len(Xte), rows=spec.rows, mae=mae, unit=spec.unit, r2=f"{r.r2:.3f}")
    if lab.guesses:
        p = model.predict(df.loc[lab.guesses["rows"], spec.features])
        m = float(np.mean(np.abs(p - np.array(lab.guesses["truth"]))))
        right = sum(spec.grade_of(v) == g for v, g in zip(p, lab.guesses["truth_grade"]))
        say(texts, G, "guess_text", n=len(p), rows=spec.rows, mae=spec.fmt(m), unit=spec.unit, model_right=right, you_right=lab.guesses["right"])
    w_ = r.worst.copy(); w_.columns = [spec.label(c) if c in spec.features else c for c in w_.columns]
    w_ = w_.rename(columns={"measured": f"actual ({spec.unit})", "predicted": f"predicted ({spec.unit})", "miss": f"error ({spec.unit})"})
    say(texts, G, "worst_text", n=len(r.worst), rows=spec.rows)
    table(w_.reset_index(drop=True))


def classification_view(lab, kind: str, task: str, threshold: float, texts: Optional[dict] = None):
    spec, df = lab.table_spec, lab.df
    Xtr, Xte, ytr, yte = models.split(df, spec)
    output = spec.label(spec.target)
    if task.startswith("grade"):
        G = "classification_grades"
        labels_tr = np.array([spec.grade_of(v) for v in ytr])
        model = models.fit_classifier(kind, Xtr, labels_tr)
        r = models.grades_result(model, kind, Xte, yte, spec)
        say(texts, G, "accuracy_text", model=_model_name(kind), accuracy=f"{r.accuracy * 100:.0f}", n_test=len(Xte), rows=spec.rows)
        say(texts, G, "matrix_text")
        confusion(r.matrix)
        if lab.guesses:
            p = model.predict(df.loc[lab.guesses["rows"], spec.features])
            right = sum(a == b for a, b in zip(p, lab.guesses["truth_grade"]))
            say(texts, G, "guess_text", n=len(p), rows=spec.rows, model_right=right, you_right=lab.guesses["right"])
    else:
        G = "classification_passfail"
        model = models.fit_classifier(kind, Xtr, models.passes(spec, ytr.values, threshold))
        r = models.passfail_result(model, kind, Xte, yte, spec, threshold)
        rule = f"{output.split(' (')[0]} {'≤' if spec.pass_below else '≥'} {spec.fmt(threshold) if spec.decimals == 0 else f'{threshold:g}'} {spec.unit}"
        say(texts, G, "accuracy_text", model=_model_name(kind), accuracy=f"{r.accuracy * 100:.0f}", n_test=len(Xte), rows=spec.rows, rule=rule,
            false_pass=r.false_pass, false_fail=r.false_fail)
        say(texts, G, "matrix_text")
        confusion(r.matrix)
        if r.proba is not None:
            import textwrap
            fig, ax = plt.subplots(figsize=(7, 2.9))
            ax.scatter(yte.values, r.proba, s=12, alpha=0.6, c=np.where(r.y_true, GREEN, RED))
            ax.axvline(threshold, color="k", lw=1, ls="--"); ax.axhline(0.5, color=GREY, lw=1)
            ax.set_xlabel(f"actual {output}"); ax.set_ylabel(fill(text(texts, G, "proba_axis_text")))
            ax.set_title(textwrap.fill(fill(text(texts, G, "proba_title_text"), output=output.split(" (")[0]), 80), fontsize=9)
            show(fig)
    lab.results[("cls", r.kind, r.task)] = {"accuracy": r.accuracy, "false_pass": r.false_pass, "false_fail": r.false_fail}


# ----------------------------------------------------------------------------- Part 3: what the model learned
def importance_view(lab, texts: Optional[dict] = None):
    spec, df = lab.table_spec, lab.df
    Xtr, Xte, ytr, yte = models.split(df, spec)
    model = lab.models.get(("reg", "trees")) or models.fit_regressor("trees", Xtr, ytr)
    lab.models[("reg", "trees")] = model
    imp = models.importance(model, Xte, yte, spec)
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.barh([spec.label(f) for f, _ in imp][::-1], [v for _, v in imp][::-1], color=BLUE)
    ax.set_xlabel(fill(text(texts, "importance", "axis_text"))); ax.set_title(fill(text(texts, "importance", "title_text")), fontsize=10)
    show(fig)
    lab.results["importance"] = imp


def whatif(lab, sample: str = "a typical row", texts: Optional[dict] = None):
    """sample: "a typical <row>" (the median of every column) or "<row> N" (row N of the table).
    Sliders on the chosen inputs; everything else stays at the chosen row's values."""
    import ipywidgets as w
    spec, df = lab.table_spec, lab.df
    Xtr, Xte, ytr, yte = models.split(df, spec)
    model = lab.models.get(("reg", "trees")) or models.fit_regressor("trees", Xtr, ytr)
    lab.models[("reg", "trees")] = model
    base = df.median() if sample.startswith("a typical") else df.loc[int(sample.split()[-1])]
    sliders = {}
    for f in spec.whatif:
        lo, hi = float(df[f].quantile(0.01)), float(df[f].quantile(0.99))
        step = max((hi - lo) / 60, 0.01)
        sliders[f] = w.FloatSlider(value=float(base[f]), min=lo, max=hi, step=step, description=spec.label(f)[:28], continuous_update=False,
                                   style={"description_width": "200px"}, layout=w.Layout(width="520px"), readout_format=".2f" if hi - lo < 5 else ".1f")
    out = w.Output()
    output = spec.label(spec.target).split(" (")[0]

    def render(*_):
        row = base.copy()
        for f, s in sliders.items():
            row[f] = s.value
        pred = models.predict_one(model, row, spec.features)
        with out:
            out.clear_output(wait=True)
            say(texts, "whatif", "result_text", output=output, value=spec.fmt(pred), unit=spec.unit)
    for s in sliders.values():
        s.observe(render, names="value")
    display(w.VBox([*sliders.values(), out])); render()


# ----------------------------------------------------------------------------- Part 4: the meters
def buildings_game(lab, texts: Optional[dict] = None):
    """The four buildings as A-D, a week and a year each, no names."""
    ms = list(lab.meters.values())
    order = lab.game_order
    fig, axes = plt.subplots(len(ms), 2, figsize=(14, 2.6 * len(ms)), gridspec_kw={"width_ratios": [1, 2]})
    for k, i in enumerate(order):
        m = ms[i]; s = m.kwh
        wk = s["2017-03-06":"2017-03-12"]
        axes[k, 0].plot(wk.index, wk.values, color=BLUE, lw=1); axes[k, 0].set_title(fill(text(texts, "buildings", "week_title_text"), letter="ABCD"[k]), fontsize=10)
        axes[k, 1].plot(s.index, s.values, color=BLUE, lw=0.3); axes[k, 1].set_title(fill(text(texts, "buildings", "year_title_text"), letter="ABCD"[k]), fontsize=10)
        for ax in axes[k]:
            ax.set_ylabel("kWh"); ax.tick_params(labelsize=8)
    fig.tight_layout(); show(fig)


def buildings_check(lab, texts: Optional[dict] = None, **answer):
    ms = list(lab.meters.values()); order = lab.game_order
    truth = {"ABCD"[k]: ms[i].use for k, i in enumerate(order)}
    right = 0
    for letter, use in truth.items():
        yours = answer.get(letter.lower(), "?")
        ok = yours == use; right += ok
        m = ms[order["ABCD".index(letter)]]
        say(texts, "buildings_answer", "line_text", letter=letter, answer=yours, use=use, id=m.id, area=f"{m.info['sqft']:,.0f}",
            mean=f"{m.info['mean_kWh']:,}", mark="✓" if ok else "✗")
    say(texts, "buildings_answer", "total_text", right=right)
    lab.results["buildings_game"] = right


ANATOMY_TEXT = TEXTS["anatomy"]["anatomy_text"]


def anatomy(lab, meter_id: str, anatomy_text: Optional[str] = None, texts: Optional[dict] = None):
    texts = dict(texts or {}, **({"anatomy_text": anatomy_text} if anatomy_text else {}))
    G = "anatomy"
    m = lab.meter(meter_id); s = m.kwh
    fig, ax = plt.subplots(3, 1, figsize=(14, 9))
    ax[0].plot(s.index, s.values, lw=0.4, color=BLUE); ax[0].set_title(fill(text(texts, G, "year_title_text"), building=m.label))
    wk = s["2017-03-06":"2017-03-12"]; ax[1].plot(wk.index, wk.values, color=BLUE); ax[1].set_title(fill(text(texts, G, "week_title_text"), building=m.label))
    prof = s.groupby([s.index.dayofweek, s.index.hour]).median().unstack(0)
    for d, name in enumerate(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]):
        ax[2].plot(prof.index, prof[d].values, label=name, color=plt.cm.viridis(d / 6))
    ax[2].legend(ncol=7, fontsize=8); ax[2].set_xlabel("hour of the day"); ax[2].set_title(fill(text(texts, G, "day_title_text"), building=m.label))
    for a in ax:
        a.set_ylabel("kWh")
    fig.tight_layout(); show(fig)
    t, day = m.temp, s.groupby(s.index.hour).median()          # a typical day: the median of each hour over the year
    say(texts, G, "anatomy_text", building=m.label, mean=f"{day.mean():.0f}", maximum=f"{day.max():.0f}",
        minimum=f"{day.min():.0f}", temp_min=f"{t.min():.0f}", temp_max=f"{t.max():.0f}")


# ----------------------------------------------------------------------------- Part 5: next week
def forecast_view(lab, meter_id: str, method: str, texts: Optional[dict] = None):
    G = "forecast"
    m = lab.meter(meter_id)
    hist, actual = series.test_window(m)
    methods = list(series.METHODS) if method.startswith("all") else [method]
    colors = {"naive": GREY, "trees": GREEN, "chronos": RED}
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(actual.index, actual.values, "k", lw=1.8, label=fill(text(texts, G, "actual_text")))
    rows = []
    for meth in methods:
        if series.METHODS.get(meth) == "chronos" and lab.forecaster is None:
            print("Chronos-Bolt is not loaded (Step 0, load_forecaster): skipped."); continue
        f = series.forecast(m, meth, lab.forecaster)
        ax.plot(actual.index, f.median, color=colors[f.method], lw=1.3, label=meth)
        if f.low is not None:
            ax.fill_between(actual.index, f.low, f.high, color=colors[f.method], alpha=0.12)
        rows.append((meth, f"{f.mae:.1f}", f"{f.seconds:.1f} s",
                     f"{f.coverage * 100:.0f}" if f.coverage is not None else ""))
        lab.results[("forecast", m.id, f.method)] = f.mae
    ax.set_title(fill(text(texts, G, "title_text"), building=m.label, week=f"{actual.index[0]:%B %-d, %Y}"), fontsize=11)
    ax.set_ylabel("kWh"); ax.legend(fontsize=8, ncol=4)
    show(fig)
    say(texts, G, "table_text", hours=len(actual))
    table(pd.DataFrame(rows, columns=["method", "MAE (kWh per hour)", "run time", "% hours inside the prediction interval"]).set_index("method"))


# ----------------------------------------------------------------------------- Part 6: odd days
def oddday_view(lab, meter_id: str, threshold: float, texts: Optional[dict] = None):
    G = "odd_days"
    m = lab.meter(meter_id)
    d = series.odd_days(m, threshold)
    fig, ax = plt.subplots(figsize=(14, 3.4))
    ax.bar(d.index, d.deviation_kWh_per_h, width=1, color=np.where(d.flag, RED, BLUE))
    ax.set_ylabel(fill(text(texts, G, "axis_text")))
    ax.set_title(fill(text(texts, G, "title_text"), building=m.label, flagged=int(d.flag.sum()), threshold=f"{threshold:g}"), fontsize=11)
    show(fig)
    fl = d[d.flag].copy()
    say(texts, G, "summary_text", flagged=int(d.flag.sum()), days=len(d), threshold=f"{threshold:g}",
        holidays=int((fl.holiday != "").sum()) if len(fl) else 0)
    if len(fl):
        fl.index = fl.index.strftime("%Y-%m-%d")
        fl = fl[["weekday", "deviation_kWh_per_h", "z", "holiday"]].rename(columns={"deviation_kWh_per_h": "deviation (kWh per hour)", "z": "robust z-score"})
        table(fl, max_rows=40)
    lab.results[("odd_days", m.id, threshold)] = int(d.flag.sum())


# ----------------------------------------------------------------------------- wrap-up
def report_summary(lab):
    spec = lab.table_spec
    model_name = {k: name for name, k in models.KINDS.items()}             # "trees" -> "decision tree", as in the dropdowns
    method_name = {k: name for name, k in series.METHODS.items()}
    print(f"Table: {spec.title}." if lab.part != "series" else f"Time series: {lab.spec.series.title}.")
    for key, v in sorted(lab.results.items(), key=lambda kv: str(kv[0])):
        kind, *rest = key if isinstance(key, tuple) else (key,)
        if kind == "reg":
            print(f"  regression, {model_name[rest[0]]}: MAE {spec.fmt(v['mae'])} {spec.unit}, R² {v['r2']:.3f}")
        elif kind == "cls":
            extra = f", false passes {v['false_pass']}, false fails {v['false_fail']}" if "pass" in rest[1] else ""
            print(f"  classification, {model_name[rest[0]]}, {rest[1]}: {v['accuracy'] * 100:.0f} % right{extra}")
        elif kind == "forecast":
            print(f"  forecast {rest[0]} by {method_name.get(rest[1], rest[1])}: MAE {v:.1f} kWh/h")
        elif kind == "odd_days":
            print(f"  odd days on {rest[0]} at threshold {rest[1]:g}: {v}")
        elif kind == "buildings_game":
            print(f"  which building is which: {v} of 4")
        elif kind == "chat_table":
            print(f"  {rest[0]}: MAE {spec.fmt(v['mae'])} {spec.unit}, within {spec.close_pct:g} % on {v['close']} of {v['n']}")
        elif kind == "chat_forecast":
            print(f"  forecast {rest[0]} by {rest[1]}: MAE {v:.1f} kWh/h")
        elif kind == "chat_odd":
            print(f"  odd days on {rest[0]} by the chat: listed {v['listed']}, found {v['found']} of the rule's {v['flagged']}, added {v['extra']}")
    if lab.guesses:
        print(f"  your own grades in Step 1b: {lab.guesses['right']} of {len(lab.guesses['guess'])} right")
    if not lab.results:
        print("  Nothing run yet: go back to the steps above.")
