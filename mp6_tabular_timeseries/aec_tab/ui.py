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

BLUE, GREEN, RED, GREY, ORANGE, PURPLE = "#457b9d", "#2a9d8f", "#e63946", "#8d99ae", "#f4a261", "#7b2cbf"


def show(fig, dpi: int = 100):
    from IPython.display import Image
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight"); plt.close(fig)
    display(Image(buf.getvalue()))


def table(df: pd.DataFrame, max_rows: int = 30):
    display(HTML(df.head(max_rows).to_html(index=isinstance(df.index, pd.DatetimeIndex) or df.index.name is not None,
                                          float_format=lambda v: f"{v:,.2f}", border=0)))


# ----------------------------------------------------------------------------- Part 1: the table
def show_table(lab, rows: int = 10):
    spec, df = lab.table_spec, lab.df
    print(f"{spec.title}: {len(df):,} {spec.rows} (rows), {len(spec.features)} inputs (columns) and one answer, {spec.label(spec.target)}.")
    view = df.sample(rows, random_state=1).sort_index(); view.columns = [spec.label(c) for c in view.columns]
    table(view.reset_index(drop=True))
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
    ax[0].hist(df[spec.target], bins=30, color=BLUE); ax[0].set_xlabel(spec.label(spec.target)); ax[0].set_ylabel(f"{spec.rows}"); ax[0].set_title("the answer column")
    top = spec.whatif[0]
    ax[1].scatter(df[top], df[spec.target], s=8, alpha=0.4, color=BLUE); ax[1].set_xlabel(spec.label(top)); ax[1].set_ylabel(spec.label(spec.target)); ax[1].set_title("one input against the answer")
    show(fig)


def guess_game(lab):
    """Five rows without their answer: the student picks a grade for each, the notebook scores it."""
    import ipywidgets as w
    spec, df = lab.table_spec, lab.df
    _, Xte, _, yte = models.split(df, spec)
    rng = np.random.default_rng(7)
    idx = rng.choice(len(Xte), size=spec.guess_n, replace=False)
    rows = Xte.iloc[idx]; truth = yte.iloc[idx]
    grades = [g[0] for g in spec.grades]
    view = rows.copy(); view.columns = [spec.label(c) for c in view.columns]; view.index = [f"{spec.row_word} {i + 1}" for i in range(len(view))]
    table(view)
    picks = [w.Dropdown(options=grades, value=grades[len(grades) // 2], description=f"{spec.row_word} {i + 1}", layout=w.Layout(width="330px")) for i in range(len(rows))]
    btn = w.Button(description="Check my guesses", button_style="primary"); out = w.Output()

    def go(_):
        g = [b.value for b in picks]; tg = [spec.grade_of(v) for v in truth.values]
        right = sum(a == b for a, b in zip(g, tg))
        lab.guesses = {"rows": rows.index.tolist(), "guess": g, "truth_grade": tg, "truth": truth.values.tolist(), "right": right}
        with out:
            out.clear_output(wait=True)
            res = pd.DataFrame({"your grade": g, "really": tg, f"measured ({spec.unit})": np.round(truth.values, 1), "": ["✓" if a == b else "✗" for a, b in zip(g, tg)]}, index=view.index)
            table(res)
            print(f"You put {right} of {len(g)} {spec.rows} in the right grade. Step 2a shows what the model does with the same {spec.rows}.")
    btn.on_click(go)
    display(w.VBox([w.HTML(f"Which grade of {spec.target_label} does each {spec.row_word} reach? Pick one per {spec.row_word}, then click the button. "
                           f"The grades: {', '.join(grades)}."), *picks, btn, out]))


# ----------------------------------------------------------------------------- Part 2: regression and classification
def regression_view(lab, kind: str):
    spec, df = lab.table_spec, lab.df
    Xtr, Xte, ytr, yte = models.split(df, spec)
    model = models.fit_regressor(kind, Xtr, ytr)
    r = models.score_regression(model, kind, Xte, yte, spec)
    lab.models[("reg", r.kind)] = model; lab.results[("reg", r.kind)] = {"mae": r.mae, "r2": r.r2}
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.scatter(r.y_true, r.y_pred, s=14, alpha=0.6, color=BLUE); lim = [min(r.y_true.min(), r.y_pred.min()), max(r.y_true.max(), r.y_pred.max())]
    ax.plot(lim, lim, "k--", lw=1); ax.set_xlabel(f"measured {spec.label(spec.target)}"); ax.set_ylabel("predicted"); ax.set_title(f"{len(Xte)} {spec.rows} the model never saw", fontsize=10)
    show(fig)
    print(f"{kind}: trained on {len(Xtr)} {spec.rows}, scored on {len(Xte)} it never saw.")
    print(f"  Average miss (MAE): {r.mae:.2f} {spec.unit}   |   R²: {r.r2:.3f}   (1.0 would be perfect; 0 is no better than always guessing the average)")
    if lab.guesses:
        p = model.predict(df.loc[lab.guesses["rows"], spec.features])
        m = float(np.mean(np.abs(p - np.array(lab.guesses["truth"]))))
        right = sum(spec.grade_of(v) == g for v, g in zip(p, lab.guesses["truth_grade"]))
        print(f"  On the {len(p)} {spec.rows} you graded in Step 1b: the model's numbers are off by {m:.1f} {spec.unit} on average, "
              f"and turned into grades they put {right} of {len(p)} right (you: {lab.guesses['right']}).")
    w_ = r.worst.copy(); w_.columns = [spec.label(c) if c in spec.features else c for c in w_.columns]
    print(f"\nThe {len(r.worst)} worst misses:"); table(w_.reset_index(drop=True))


def classification_view(lab, kind: str, task: str, threshold: float):
    spec, df = lab.table_spec, lab.df
    Xtr, Xte, ytr, yte = models.split(df, spec)
    if task.startswith("grade"):
        labels_tr = np.array([spec.grade_of(v) for v in ytr])
        model = models.fit_classifier(kind, Xtr, labels_tr)
        r = models.grades_result(model, kind, Xte, yte, spec)
        print(f"{kind}, {len(spec.grades)} grades of {spec.label(spec.target)}: {r.accuracy * 100:.0f} % of the {len(Xte)} unseen {spec.rows} put in the right grade.")
        table(r.matrix)
        print("Rows are the truth, columns what the model said; the diagonal is right, everything else is a mistake.")
        if lab.guesses:
            p = model.predict(df.loc[lab.guesses["rows"], spec.features])
            right = sum(a == b for a, b in zip(p, lab.guesses["truth_grade"]))
            print(f"  On the {len(p)} {spec.rows} you graded in Step 1b: the model puts {right} of {len(p)} in the right grade, you put {lab.guesses['right']}.")
    else:
        model = models.fit_classifier(kind, Xtr, ytr.values >= threshold)
        r = models.passfail_result(model, kind, Xte, yte, spec, threshold)
        print(f"{kind}, does the {spec.row_word} reach {threshold:g} {spec.unit}? {r.accuracy * 100:.0f} % of the {len(Xte)} unseen {spec.rows} called right.")
        table(r.matrix)
        print(f"  False passes (called pass, really fails): {r.false_pass}   |   false fails (called fail, really passes): {r.false_fail}")
        if r.proba is not None:
            print(f"  {spec.rows.capitalize()} the model is unsure about (probability of passing between 30 % and 70 %): {r.uncertain}")
            fig, ax = plt.subplots(figsize=(7, 2.8))
            ax.scatter(yte.values, r.proba, s=12, alpha=0.6, c=np.where(r.y_true, GREEN, RED))
            ax.axvline(threshold, color="k", lw=1, ls="--"); ax.axhline(0.5, color=GREY, lw=1)
            ax.set_xlabel(f"measured {spec.label(spec.target)}"); ax.set_ylabel("model's probability of a pass"); ax.set_title("green = really passes, red = really fails", fontsize=9)
            show(fig)
    lab.results[("cls", r.kind, r.task)] = {"accuracy": r.accuracy, "false_pass": r.false_pass, "false_fail": r.false_fail}


# ----------------------------------------------------------------------------- Part 3: what the model learned
def importance_view(lab):
    spec, df = lab.table_spec, lab.df
    Xtr, Xte, ytr, yte = models.split(df, spec)
    model = lab.models.get(("reg", "trees")) or models.fit_regressor("trees", Xtr, ytr)
    lab.models[("reg", "trees")] = model
    imp = models.importance(model, Xte, yte, spec)
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.barh([spec.label(f) for f, _ in imp][::-1], [v for _, v in imp][::-1], color=BLUE)
    ax.set_xlabel("how much the model's accuracy drops when this column is scrambled"); show(fig)
    lab.results["importance"] = imp
    print("The columns the model leans on most are at the top. A column near zero could be removed without the model noticing.")


def whatif(lab, start_from: str = "a typical row"):
    """Sliders on the chosen inputs; everything else stays at the chosen row's values."""
    import ipywidgets as w
    spec, df = lab.table_spec, lab.df
    Xtr, Xte, ytr, yte = models.split(df, spec)
    model = lab.models.get(("reg", "trees")) or models.fit_regressor("trees", Xtr, ytr)
    lab.models[("reg", "trees")] = model
    base = df.median() if start_from.startswith("a typical") else df.loc[int(start_from.split()[-1])]
    sliders = {}
    for f in spec.whatif:
        lo, hi = float(df[f].quantile(0.01)), float(df[f].quantile(0.99))
        step = max((hi - lo) / 60, 0.01)
        sliders[f] = w.FloatSlider(value=float(base[f]), min=lo, max=hi, step=step, description=spec.label(f)[:28], continuous_update=False,
                                   style={"description_width": "200px"}, layout=w.Layout(width="520px"), readout_format=".1f")
    out = w.Output()

    def render(*_):
        row = base.copy()
        for f, s in sliders.items():
            row[f] = s.value
        pred = models.predict_one(model, row, spec.features)
        f0 = spec.whatif[0]
        xs = np.linspace(sliders[f0].min, sliders[f0].max, 40)
        ys = models.whatif_curve(model, row, f0, xs, spec.features)
        with out:
            out.clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(7, 3))
            ax.plot(xs, ys, color=BLUE); ax.scatter([row[f0]], [pred], color=RED, zorder=3)
            ax.set_xlabel(spec.label(f0)); ax.set_ylabel(f"predicted {spec.label(spec.target)}"); ax.set_title(f"predicted {spec.target_label}: {pred:.1f} {spec.unit}", fontsize=11)
            show(fig)
            lo, hi = df[spec.target].min(), df[spec.target].max()
            inside = all(df[f].min() <= row[f] <= df[f].max() for f in spec.whatif)
            print(f"Predicted {spec.target_label}: {pred:.1f} {spec.unit} (the data runs from {lo:.0f} to {hi:.0f})."
                  + ("" if inside else " At least one slider is outside anything the model was trained on: treat this number as a guess."))
    for s in sliders.values():
        s.observe(render, names="value")
    display(w.VBox([*sliders.values(), out])); render()


# ----------------------------------------------------------------------------- Part 4: the meters
def buildings_game(lab):
    """The four buildings as A-D, a week and a year each, no names."""
    ms = list(lab.meters.values())
    order = lab.game_order
    fig, axes = plt.subplots(len(ms), 2, figsize=(14, 2.6 * len(ms)), gridspec_kw={"width_ratios": [1, 2]})
    for k, i in enumerate(order):
        m = ms[i]; s = m.kwh
        wk = s["2017-03-06":"2017-03-12"]
        axes[k, 0].plot(wk.index, wk.values, color=BLUE, lw=1); axes[k, 0].set_title(f"building {'ABCD'[k]}: one week in March (Mon-Sun)", fontsize=10)
        axes[k, 1].plot(s.index, s.values, color=BLUE, lw=0.3); axes[k, 1].set_title(f"building {'ABCD'[k]}: the whole year", fontsize=10)
        for ax in axes[k]:
            ax.set_ylabel("kWh"); ax.tick_params(labelsize=8)
    fig.tight_layout(); show(fig)
    uses = sorted({m.use for m in ms})
    print("The four buildings are, in some order: " + ", ".join(uses) + ". Say which is which in the next cell.")


def buildings_check(lab, **answer):
    ms = list(lab.meters.values()); order = lab.game_order
    truth = {"ABCD"[k]: ms[i].use for k, i in enumerate(order)}
    right = 0
    for letter, use in truth.items():
        yours = answer.get(letter.lower(), "?")
        ok = yours == use; right += ok
        m = ms[order["ABCD".index(letter)]]
        print(f"  building {letter}: you said {yours:15} -> it is the {use} ({m.id}, {m.info['sqm']:,} m², mean {m.info['mean_kWh']} kWh/h)  {'✓' if ok else '✗'}")
    print(f"{right} of 4 right.")
    lab.results["buildings_game"] = right


def anatomy(lab, meter_id: str):
    m = lab.meter(meter_id); s = m.kwh
    fig, ax = plt.subplots(3, 1, figsize=(14, 9))
    ax[0].plot(s.index, s.values, lw=0.4, color=BLUE); ax[0].set_title(f"{m.label}: the year, hour by hour (kWh)")
    wk = s["2017-03-06":"2017-03-12"]; ax[1].plot(wk.index, wk.values, color=BLUE); ax[1].set_title("one week in March, Monday to Sunday")
    prof = s.groupby([s.index.dayofweek, s.index.hour]).median().unstack(0)
    for d, name in enumerate(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]):
        ax[2].plot(prof.index, prof[d].values, label=name, color=plt.cm.viridis(d / 6))
    ax[2].legend(ncol=7, fontsize=8); ax[2].set_xlabel("hour of the day"); ax[2].set_title("the usual day: median of every weekday over the year")
    fig.tight_layout(); show(fig)
    t = m.temp
    print(f"Mean {s.mean():.0f} kWh per hour; the busiest hour of a typical week runs at {prof.max().max():.0f}, the quietest at {prof.min().min():.0f}. "
          f"Site air temperature ran from {t.min():.0f} to {t.max():.0f} °C.")


# ----------------------------------------------------------------------------- Part 5: next week
def forecast_view(lab, meter_id: str, method: str):
    m = lab.meter(meter_id)
    hist, actual = series.test_window(m)
    methods = list(series.METHODS) if method.startswith("all") else [method]
    colors = {"naive": GREY, "trees": GREEN, "chronos": RED}
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(actual.index, actual.values, "k", lw=1.8, label="what really happened")
    rows = []
    for meth in methods:
        if series.METHODS.get(meth) == "chronos" and lab.forecaster is None:
            print("Chronos-Bolt is not loaded (Step 0, load_forecaster): skipped."); continue
        f = series.forecast(m, meth, lab.forecaster)
        ax.plot(actual.index, f.median, color=colors[f.method], lw=1.3, label=meth)
        if f.low is not None:
            ax.fill_between(actual.index, f.low, f.high, color=colors[f.method], alpha=0.12)
        rows.append((meth, f"{f.mae:.1f}", f"{f.mae / actual.mean() * 100:.0f} %", f"{f.seconds:.1f} s", f"{f.coverage * 100:.0f} % of hours inside the band" if f.coverage is not None else ""))
        lab.results[("forecast", m.id, f.method)] = f.mae
    ax.set_title(f"{m.label}: the week of {actual.index[0].date()}, forecast from the data before it", fontsize=11); ax.set_ylabel("kWh"); ax.legend(fontsize=8, ncol=4)
    show(fig)
    table(pd.DataFrame(rows, columns=["method", "average miss (kWh per hour)", "as a share of the mean load", "time", "uncertainty band"]).set_index("method"))


# ----------------------------------------------------------------------------- Part 6: odd days
def oddday_view(lab, meter_id: str, threshold: float):
    m = lab.meter(meter_id)
    d = series.odd_days(m, threshold)
    fig, ax = plt.subplots(figsize=(14, 3.4))
    ax.bar(d.index, d.deviation_kWh_per_h, width=1, color=np.where(d.flag, RED, BLUE))
    ax.set_ylabel("kWh per hour above (+) or below (-) the usual day"); ax.set_title(f"{m.label}: each day against the building's usual pattern; {int(d.flag.sum())} days flagged", fontsize=11)
    show(fig)
    fl = d[d.flag].copy()
    if len(fl):
        fl.index = fl.index.strftime("%Y-%m-%d"); table(fl[["weekday", "deviation_kWh_per_h", "z", "holiday"]], max_rows=40)
    print(f"{int(d.flag.sum())} of {len(d)} days are more than {threshold:g} robust standard deviations from the usual pattern; "
          f"{int((fl.holiday != '').sum()) if len(fl) else 0} of them fall on a public holiday.")
    lab.results[("odd_days", m.id, threshold)] = int(d.flag.sum())


# ----------------------------------------------------------------------------- wrap-up
def report_summary(lab):
    spec = lab.table_spec
    print(f"Table: {spec.title}.")
    for (kind, *rest), v in sorted(lab.results.items(), key=lambda kv: str(kv[0])):
        if kind == "reg":
            print(f"  regression, {rest[0]}: MAE {v['mae']:.2f} {spec.unit}, R² {v['r2']:.3f}")
        elif kind == "cls":
            extra = f", false passes {v['false_pass']}, false fails {v['false_fail']}" if "pass" in rest[1] else ""
            print(f"  classification, {rest[0]}, {rest[1]}: {v['accuracy'] * 100:.0f} % right{extra}")
        elif kind == "forecast":
            print(f"  forecast {rest[0]} by {rest[1]}: MAE {v:.1f} kWh/h")
        elif kind == "odd_days":
            print(f"  odd days on {rest[0]} at threshold {rest[1]:g}: {v}")
        elif kind == "buildings_game":
            print(f"  which building is which: {v} of 4")
    if lab.guesses:
        print(f"  your own grades in Step 1b: {lab.guesses['right']} of {len(lab.guesses['guess'])} right")
    if not lab.results:
        print("  Nothing run yet: go back to the steps above.")
