"""The chat steps: the notebook hands the student files and a prompt for hokie.ai, the student pastes the reply back,
and the notebook scores it on the same rows / the same week as the models of the earlier steps."""
import re
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C
from . import models, series, ui

CHAT_URL = "https://hokie.ai.vt.edu/"
TEST_N = 30
GIVE = {"attach the files": "attach", "paste the data into the prompt": "inline"}

# what the student may ask the chat's analysis tool to train (label in the notebook -> words in the prompt)
TOOL_MODELS = {
    "gradient-boosted trees": "a gradient-boosted tree model (for example scikit-learn's HistGradientBoostingRegressor)",
    "a random forest": "a random forest (for example scikit-learn's RandomForestRegressor)",
    "a straight line": "a linear regression (a straight line, for example scikit-learn's LinearRegression)",
    "a small neural network": "a small neural network (for example scikit-learn's MLPRegressor)",
}

ORANGE = ui.ORANGE


def _article(word: str) -> str:
    return ("an " if word[:1].lower() in "aeiou" else "a ") + word


def _attach(give: str) -> bool:
    return GIVE.get(give, give) == "attach"


def _dir() -> Path:
    d = Path(tempfile.gettempdir()) / "mp6_chat_files"
    d.mkdir(exist_ok=True)
    return d


def _save(name: str, df: pd.DataFrame, index: bool = False) -> Path:
    p = _dir() / name
    df.to_csv(p, index=index)
    return p


def steps_default(n_files: int = 2, attach: bool = True) -> str:
    """The instructions above the prompt box (the notebook cell carries its own copy, which the teacher can edit)."""
    s = "s" if n_files > 1 else ""
    first = (f"1. Download {{files}} with the button{s} below.\n2. Open {{url}} (sign in with your VT account), start a new chat, "
             f"attach the file{s}, paste the prompt, send.\n" if attach else
             "1. The data is already inside the prompt below.\n2. Open {url} (sign in with your VT account), start a new chat, "
             "paste the prompt, send.\n")
    return first + "3. Copy the whole reply and paste it into the second box.\n4. Click *Score*."


def _md_html(text: str) -> str:
    """The little markdown the instructions use: **bold**, *italic*, links, one line per line."""
    import html
    t = html.escape(text.strip())
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", t)
    t = re.sub(r"(https?://[^\s<]+[^\s<.,;)])", r"<a href='\1' target='_blank'>\1</a>", t)
    return "<br>".join(t.split("\n"))


def _box(files: List[Path], prompt: str, what: str, on_score: Callable[[str], None], attach: bool = True, steps: str = None):
    """Download buttons, the prompt to copy, a box for the reply, a Score button (the MP5 chat steps' pattern)."""
    import ipywidgets as w
    from IPython.display import display
    names = " and ".join(f"*{p.name}*" for p in files)
    template = steps or steps_default(len(files), attach)
    try:
        text = template.format(files=names, url=CHAT_URL)
    except (KeyError, IndexError, ValueError) as e:
        text = template + f"\n(the text has a placeholder the notebook does not know: {e})"
    how = _md_html(text)
    rows = []
    if attach:
        for p in files:
            b = w.Button(description=f"Download {p.name}", icon="download", layout=w.Layout(width="auto"))
            o = w.Output()

            def _dl(_, p=p, o=o):
                with o:
                    o.clear_output()
                    try:
                        from google.colab import files as colab_files
                        colab_files.download(str(p))
                    except Exception:  # noqa: BLE001
                        print(f"Not in Colab: the file is at {p}")
            b.on_click(_dl)
            rows.append(w.HBox([b, o]))
    prompt_box = w.Textarea(value=prompt, layout=w.Layout(width="100%", height="170px" if attach else "220px"),
                            description="prompt", style={"description_width": "60px"})
    reply_box = w.Textarea(placeholder="paste the chat's whole reply here (extra words are fine)",
                           layout=w.Layout(width="100%", height="160px"), description="reply", style={"description_width": "60px"})
    btn = w.Button(description=f"Score the reply ({what})", button_style="primary", layout=w.Layout(width="auto"))
    out = w.Output()

    def _go(_):
        with out:
            out.clear_output(wait=True)
            text = reply_box.value.strip()
            if not text:
                print("Paste the reply first."); return
            try:
                on_score(text)
            except Exception as e:  # noqa: BLE001
                print(f"Could not use that reply: {e}")
    btn.on_click(_go)
    display(w.VBox([w.HTML(how), *rows, prompt_box, reply_box, btn, out]))


def _add_run(lab, key, name: str, values) -> Tuple[str, bool]:
    """Keep every scored reply, so a second chat can be compared with the first; the same reply twice is not a new run."""
    runs = lab.chat_runs.setdefault(key, [])
    for r in runs:
        if r["values"] == values:
            return r["name"], False
    if name == "chat":
        full = f"chat {sum(r['name'].startswith('chat ') and 'tool' not in r['name'] for r in runs) + 1}"
    else:
        k = sum(r["name"].startswith(name) for r in runs) + 1
        full = name if k == 1 else f"{name} ({k})"
    runs.append({"name": full, "values": values})
    return full, True


# ============================================================================ the table
def table_files(lab):
    """The training rows (with the answer) and 30 held-out rows (without), the same for everyone."""
    spec = lab.table_spec
    if "table" in lab.chat_cache:
        return lab.chat_cache["table"]
    Xtr, Xte, ytr, yte = models.split(lab.df, spec)
    train = Xtr.copy(); train[spec.target] = ytr.values
    train = train.sort_index().reset_index(drop=True)
    grade = yte.map(spec.grade_of)
    rng = np.random.default_rng(C.SEED)
    names = [g[0] for g in spec.grades]
    per = [TEST_N // len(names) + (i < TEST_N % len(names)) for i in range(len(names))]
    pick = []
    for g, k in zip(names, per):
        idx = grade[grade == g].index.to_numpy()
        pick += list(rng.choice(idx, min(k, len(idx)), replace=False))
    pick = list(rng.permutation(pick))
    idcol = f"{spec.row_word}_id"
    ids = [f"T{i + 1:02d}" for i in range(len(pick))]
    test = Xte.loc[pick].reset_index(drop=True); test.insert(0, idcol, ids)
    truth = pd.Series(yte.loc[pick].values, index=ids)
    trees = lab.models.get(("reg", "trees")) or models.fit_regressor("trees", Xtr, ytr)
    lab.models[("reg", "trees")] = trees
    line = lab.models.get(("reg", "linear")) or models.fit_regressor("linear", Xtr, ytr)
    ref = {"trees (the notebook, Step 2b)": pd.Series(trees.predict(Xte.loc[pick]), index=ids),
           "straight line (the notebook, Step 2b)": pd.Series(line.predict(Xte.loc[pick]), index=ids)}
    d = dict(train=train, test=test, truth=truth, ref=ref, idcol=idcol,
             tr_path=_save(f"{spec.key}_train.csv", train), te_path=_save(f"{spec.key}_test.csv", test))
    lab.chat_cache["table"] = d
    return d


def _table_intro(spec, d, attach: bool) -> str:
    return (f"I am a construction student. {'I attached two files.' if attach else 'The two files are pasted below.'} "
            f"{d['tr_path'].name} has {len(d['train'])} {spec.chat_about}. {d['te_path'].name} has {len(d['test'])} other {spec.rows}, "
            f"T01 to T{len(d['test']):02d}, with the same columns but without {spec.target}.")


def _table_form(spec, d) -> str:
    return (f"Reply in this form, one line per {spec.row_word}, all {len(d['test'])} lines:\n\n"
            f"{d['idcol']}, {spec.target}\nT01, ...\nT02, ...\n...")


def table_prompt(lab, give: str = "attach the files") -> str:
    spec, d = lab.table_spec, table_files(lab)
    attach = _attach(give)
    p = (_table_intro(spec, d, attach) + f"\n\nUsing what the {len(d['train'])} {spec.rows} in {d['tr_path'].name} show, predict the "
         f"{spec.target_label} in {spec.unit} of each of the {len(d['test'])} {spec.rows} in {d['te_path'].name}. " + _table_form(spec, d))
    if not attach:
        p += (f"\n\n--- {d['tr_path'].name} ---\n{d['train'].to_csv(index=False)}\n--- {d['te_path'].name} ---\n"
              f"{d['test'].to_csv(index=False)}")
    return p


def table_tool_prompt(lab, model: str) -> str:
    spec, d = lab.table_spec, table_files(lab)
    return (_table_intro(spec, d, True) + f"\n\nUse your data-analysis tool (run Python code on the files): train "
            f"{TOOL_MODELS.get(model, model)} on {d['tr_path'].name} to predict {spec.target} from the other columns, then use it to "
            f"predict the {spec.target_label} of each of the {len(d['test'])} {spec.rows} in {d['te_path'].name}. " + _table_form(spec, d)
            + "\n\nAfter the list, say in two sentences which model and settings you used and which three columns mattered most.")


def parse_rows(text: str) -> Dict[str, float]:
    """'T01, 38.5' pairs anywhere in the reply (one per line or all on one line; tables, fences and extra words are fine)."""
    out = {}
    for tid, v in re.findall(r"\b(T\d{1,2})\b\s*\**\s*[,;:|=\t ]\s*\**\s*(-?\d+(?:\.\d+)?)", text):
        out.setdefault(f"T{int(tid[1:]):02d}", float(v))
    return out


def table_score(lab, text: str, name: str):
    spec, d = lab.table_spec, table_files(lab)
    got = parse_rows(text)
    ids = list(d["truth"].index)
    have = [i for i in ids if i in got]
    if not have:
        print("No 'T01, number' lines found in the reply: check that you pasted the whole reply."); return
    run, new = _add_run(lab, ("table", spec.key), name, {i: got[i] for i in have})
    if not new:
        print(f"This is the same reply as '{run}': start a new chat for a fresh run.")
    if len(have) < len(ids):
        print(f"The reply has {len(have)} of the {len(ids)} {spec.rows} ({', '.join(i for i in ids if i not in got)} missing); "
              f"only those {len(have)} are scored.")
    full = d["truth"]
    truth = full.reindex(have)

    def row(label, pred: pd.Series):
        """Each forecast on the rows it has (a chat reply may skip some; the notebook's models have all of them)."""
        rows_ = [i for i in full.index if i in pred.index]
        p = pred.reindex(rows_); t_ = full.reindex(rows_); e = (p - t_).abs()
        g = int(sum(spec.grade_of(a) == spec.grade_of(b) for a, b in zip(p, t_)))
        n_ = len(rows_)
        return {"predicted by": label, f"average miss ({spec.unit})": f"{e.mean():.1f}", f"worst miss ({spec.unit})": f"{e.max():.1f}",
                f"within {spec.close_enough:g} {spec.unit}": f"{int((e <= spec.close_enough).sum())} of {n_}",
                "right grade": f"{g} of {n_}"}, float(e.mean()), g
    runs = lab.chat_runs[("table", spec.key)]
    rows = []
    for r in runs:
        rr, mae, g = row(("▶ " if r["name"] == run else "") + r["name"], pd.Series(r["values"]))
        rows.append(rr)
        lab.results[("chat_table", r["name"])] = {"mae": mae, "grades": g, "n": len(r["values"])}
    for label, pred in d["ref"].items():
        rows.append(row(label, pred)[0])
    print(f"Scored on {spec.rows} whose measured {spec.target_label} neither the models nor the chat was given:")
    ui.table(pd.DataFrame(rows))
    if len(have) < len(full):
        same = {label: float((pred.reindex(have) - truth).abs().mean()) for label, pred in d["ref"].items()}
        print(f"On the same {len(have)} {spec.rows} as {run}: " + ", ".join(f"{k.split(' (')[0]} {v:.1f} {spec.unit}" for k, v in same.items()) + ".")

    mine = pd.Series(next(r for r in runs if r["name"] == run)["values"]).reindex(have)
    trees = d["ref"]["trees (the notebook, Step 2b)"].reindex(have)
    fig, ax = plt.subplots(figsize=(5, 4.6))
    ax.scatter(truth, trees, s=26, color=ui.GREEN, alpha=0.8, label="trees (the notebook)")
    ax.scatter(truth, mine, s=26, color=ORANGE, marker="D", alpha=0.9, label=run)
    lim = [min(truth.min(), mine.min()) * 0.95, max(truth.max(), mine.max()) * 1.05]
    ax.plot(lim, lim, "k--", lw=1); ax.set_xlabel(f"measured {spec.label(spec.target)}"); ax.set_ylabel("predicted"); ax.legend(fontsize=8)
    ax.set_title("on the dashed line = exactly right", fontsize=9); ui.show(fig)

    chats = [r for r in runs if r["name"].startswith("chat ") and "tool" not in r["name"]]
    if len(chats) >= 2:
        a, b = pd.Series(chats[0]["values"]), pd.Series(chats[-1]["values"])
        common = [i for i in a.index if i in b.index]
        diff = (a[common] - b[common]).abs()
        same = int((diff < 0.05).sum()); moved = diff[diff >= 0.05]
        print(f"Same prompt, two new chats ({chats[0]['name']} and {chats[-1]['name']}): the same number for {same} of {len(common)} {spec.rows}"
              + (f"; the other {len(moved)} moved by {moved.mean():.1f} {spec.unit} on average, most on {moved.idxmax()} ({moved.max():.1f} {spec.unit})."
                 if len(moved) else "."))
    elif name == "chat":
        print("Now start a second new chat, give it the same prompt, and score that reply too: does the chat give the same numbers twice?")
    worst = (mine - truth).abs().sort_values(ascending=False).index[:5]
    view = d["test"].set_index(d["idcol"]).loc[list(worst)].copy()
    view.columns = [spec.label(c) for c in view.columns]
    view.insert(0, f"{run} ({spec.unit})", mine[worst].round(1).values)
    view.insert(0, f"trees ({spec.unit})", trees[worst].round(1).values)
    view.insert(0, f"measured ({spec.unit})", truth[worst].round(1).values)
    view.insert(0, d["idcol"], list(worst))
    print(f"\nThe {len(worst)} {spec.rows} {run} missed most:"); ui.table(view.reset_index(drop=True))


def _steps(give: str, steps_text, paste_steps_text):
    return steps_text if _attach(give) else paste_steps_text


def chat_table(lab, give: str = "attach the files", steps_text=None, paste_steps_text=None):
    d = table_files(lab)
    _box([d["tr_path"], d["te_path"]], table_prompt(lab, give), "the chat on its own",
         lambda text: table_score(lab, text, "chat"), attach=_attach(give), steps=_steps(give, steps_text, paste_steps_text))


def chat_table_tool(lab, model: str, steps_text=None):
    d = table_files(lab)
    _box([d["tr_path"], d["te_path"]], table_tool_prompt(lab, model), "chat + analysis tool",
         lambda text: table_score(lab, text, f"chat + analysis tool, {model}"), attach=True, steps=steps_text)


# ============================================================================ the time series
def _when(ts: pd.Timestamp) -> str:
    return f"{ts:%A} {ts.day} {ts:%B}"


def series_files(lab, meter_id: str):
    m = lab.meter(meter_id)
    key = ("series", m.id)
    if key in lab.chat_cache:
        return lab.chat_cache[key]
    t0 = pd.Timestamp(C.TEST_START)
    hist = m.df[(m.df.index >= t0 - pd.Timedelta(days=28)) & (m.df.index < t0)].round(2); hist.index.name = "timestamp"
    nxt = m.df[["air_temp_C"]].reindex(pd.date_range(t0, periods=C.HORIZON, freq="h")).interpolate().bfill().ffill().round(1)
    nxt.index.name = "timestamp"
    day = m.kwh.resample("D").sum()
    daily = pd.DataFrame({"weekday": day.index.day_name(), "kWh": day.round(0).values,
                          "mean_air_temp_C": m.temp.resample("D").mean().round(1).values}, index=day.index)
    daily.index.name = "date"; daily.index = daily.index.strftime("%Y-%m-%d")
    d = dict(meter=m, hist=hist, next=nxt, daily=daily,
             hist_path=_save(f"{m.id}_last_4_weeks.csv", hist, index=True),
             next_path=_save(f"{m.id}_next_week_temperature.csv", nxt, index=True),
             daily_path=_save(f"{m.id}_daily_2017.csv", daily, index=True))
    lab.chat_cache[key] = d
    return d


def _series_intro(d, attach: bool) -> str:
    m, h, n = d["meter"], d["hist"], d["next"]
    return (f"I am a construction student. {'I attached two files' if attach else 'The two files are pasted below'} for a building on a "
            f"North American university campus that is used as {_article(m.use)}. {d['hist_path'].name} has its electricity use (kWh) "
            f"and the outdoor air temperature (°C), hour by hour, for the four weeks from {_when(h.index[0])} to {_when(h.index[-1])} "
            f"{h.index[-1].year}. {d['next_path'].name} has the outdoor temperature for the next week, {_when(n.index[0])} to "
            f"{_when(n.index[-1])} {n.index[-1].year} (as a weather forecast would give it).")


def _series_form(d) -> str:
    n = d["next"].index
    return (f"Forecast the building's electricity use for every hour of that next week: {len(n)} values, from "
            f"{n[0]:%Y-%m-%d %H:%M} to {n[-1]:%Y-%m-%d %H:%M}. Reply in this form, all {len(n)} lines:\n\ntimestamp, kWh\n"
            f"{n[0]:%Y-%m-%d %H:%M}, ...\n{n[1]:%Y-%m-%d %H:%M}, ...\n...")


def forecast_prompt(lab, meter_id: str, give: str = "attach the files") -> str:
    d = series_files(lab, meter_id)
    attach = _attach(give)
    p = _series_intro(d, attach) + "\n\n" + _series_form(d)
    if not attach:
        p += f"\n\n--- {d['hist_path'].name} ---\n{d['hist'].to_csv()}\n--- {d['next_path'].name} ---\n{d['next'].to_csv()}"
    return p


def forecast_tool_prompt(lab, meter_id: str, model: str) -> str:
    d = series_files(lab, meter_id)
    return (_series_intro(d, True) + f"\n\nUse your data-analysis tool (run Python code on the files): train {TOOL_MODELS.get(model, model)} "
            f"on the four weeks in {d['hist_path'].name} to predict the electricity use. Choose the inputs yourself (for example the hour "
            f"of the day, the day of the week, the temperature, the use at the same hour one week earlier). Then use it with "
            f"{d['next_path'].name}. " + _series_form(d) + "\n\nAfter the list, say in two sentences which model and which inputs you used.")


def parse_hours(text: str, index: pd.DatetimeIndex) -> pd.Series:
    """'2017-10-16 00:00, 55.2' pairs; failing that, a bare list of exactly as many numbers as there are hours."""
    vals = {}
    for day, hour, v in re.findall(r"(\d{4}-\d{2}-\d{2})[ T](\d{1,2})(?::\d{2}){0,2}\s*\**\s*[,;|\t ]\s*\**\s*(-?\d+(?:\.\d+)?)", text):
        ts = pd.Timestamp(f"{day} {int(hour):02d}:00")
        if ts in index:
            vals.setdefault(ts, float(v))
    if vals:
        return pd.Series(vals).reindex(index)
    nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", text)]
    if len(nums) == len(index):
        return pd.Series(nums, index=index)
    return pd.Series(np.nan, index=index)


def forecast_score(lab, meter_id: str, text: str, name: str):
    d = series_files(lab, meter_id); m = d["meter"]
    _, actual = series.test_window(m)
    pred = parse_hours(text, actual.index)
    n = int(pred.notna().sum())
    if n == 0:
        print("No 'timestamp, kWh' lines found in the reply: check that you pasted the whole reply."); return
    run, new = _add_run(lab, ("forecast", m.id), name, {str(k): v for k, v in pred.dropna().items()})
    if not new:
        print(f"This is the same reply as '{run}': start a new chat for a fresh run.")
    if n < len(actual):
        print(f"The reply has {n} of the {len(actual)} hours; only those are scored (the chat stopped early or skipped hours).")
    ok = (pred.notna() & actual.notna()).values
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(actual.index, actual.values, "k", lw=1.8, label="what really happened")
    rows = []
    colors = {"naive": ui.GREY, "trees": ui.GREEN, "chronos": ui.RED}
    for meth in series.METHODS:
        if series.METHODS[meth] == "chronos" and lab.forecaster is None:
            continue
        f = series.forecast(m, meth, lab.forecaster)
        mae = float(np.mean(np.abs(f.median[ok] - actual.values[ok])))
        ax.plot(actual.index, f.median, color=colors[f.method], lw=1, alpha=0.7, label=meth)
        rows.append((meth + " (the notebook)", f"{mae:.1f}", f"{mae / actual.mean() * 100:.0f} %", f"{len(actual)}"))
    chat_rows = []
    for r in lab.chat_runs[("forecast", m.id)]:
        s = pd.Series({pd.Timestamp(k): v for k, v in r["values"].items()}).reindex(actual.index)
        both = s.notna() & actual.notna()
        mae = float(np.mean(np.abs(s[both] - actual[both])))
        lab.results[("chat_forecast", m.id, r["name"])] = mae
        chat_rows.append((("▶ " if r["name"] == run else "") + r["name"], f"{mae:.1f}", f"{mae / actual.mean() * 100:.0f} %", f"{int(s.notna().sum())}"))
    ax.plot(pred.index, pred.values, color=ORANGE, lw=2, label=run)
    ax.set_title(f"{m.label}: the week of {actual.index[0].date()}", fontsize=11); ax.set_ylabel("kWh"); ax.legend(fontsize=8, ncol=5)
    ui.show(fig)
    ui.table(pd.DataFrame(chat_rows + rows, columns=["forecast", "average miss (kWh per hour)", "as a share of the mean load", "hours given"]))
    print("The chat saw four weeks of history; the notebook's methods saw the whole year up to the same Monday. "
          + ("All are scored on the same hours." if n == len(actual) else f"The notebook's methods are scored on the {int(ok.sum())} hours this reply gave."))


def chat_forecast(lab, meter_id: str, give: str = "attach the files", steps_text=None, paste_steps_text=None):
    d = series_files(lab, meter_id)
    _box([d["hist_path"], d["next_path"]], forecast_prompt(lab, meter_id, give), "the chat on its own",
         lambda text: forecast_score(lab, meter_id, text, "chat"), attach=_attach(give), steps=_steps(give, steps_text, paste_steps_text))


def chat_forecast_tool(lab, meter_id: str, model: str, steps_text=None):
    d = series_files(lab, meter_id)
    _box([d["hist_path"], d["next_path"]], forecast_tool_prompt(lab, meter_id, model), "chat + analysis tool",
         lambda text: forecast_score(lab, meter_id, text, f"chat + analysis tool, {model}"), attach=True, steps=steps_text)


# ---------------------------------------------------------------------------- odd days
def oddday_prompt(lab, meter_id: str, give: str = "attach the files") -> str:
    d = series_files(lab, meter_id); m = d["meter"]
    attach = _attach(give)
    y = d["daily"].index[0][:4]
    p = (f"I am a construction student. {'I attached' if attach else 'Below is'} {d['daily_path'].name}: a building on a North American "
         f"university campus that is used as {_article(m.use)}, with its total electricity use (kWh) and the mean outdoor temperature (°C) "
         f"for every day of {y}.\n\nWhich days do not fit the building's usual pattern? List every such day, say whether use was higher "
         f"or lower than usual, and give the most likely reason. Reply in this form, one line per day:\n\n"
         f"date, higher or lower, reason\n{y}-01-02, lower, ...")
    if not attach:
        p += f"\n\n--- {d['daily_path'].name} ---\n{d['daily'].to_csv()}"
    return p


def parse_days(text: str, year: int) -> Dict[pd.Timestamp, Tuple[str, str]]:
    """Each date in the reply, with 'higher' / 'lower' and the words after it (up to the next date)."""
    hits = list(re.finditer(rf"\b{year}-\d{{2}}-\d{{2}}\b", text))
    out = {}
    for k, h in enumerate(hits):
        seg = text[h.end(): hits[k + 1].start() if k + 1 < len(hits) else len(text)]
        way = re.search(r"\b(higher|lower|high|low|above|below|spike|dip|drop)\b", seg, re.I)
        w_ = "" if not way else ("higher" if way.group(1).lower() in ("higher", "high", "above", "spike") else "lower")
        reason = re.sub(r"^[\s,;:|*\-–]*(higher|lower)?[\s,;:|*\-–]*", "", seg.strip(), flags=re.I).strip().split("\n")[0]
        try:
            out.setdefault(pd.Timestamp(h.group(0)), (w_, reason[:90]))
        except ValueError:
            continue
    return out


def oddday_score(lab, meter_id: str, text: str, threshold: float = 3.5):
    d = series_files(lab, meter_id); m = d["meter"]
    rule = series.odd_days(m, threshold)
    got = parse_days(text, rule.index[0].year)
    got = {k: v for k, v in got.items() if k in rule.index}
    if not got:
        print(f"No dates like {rule.index[0].year}-01-02 found in the reply: check that you pasted the whole reply."); return
    flagged = set(rule.index[rule.flag]); said = set(got)
    both, missed, extra = said & flagged, flagged - said, said - flagged
    hol = {pd.Timestamp(k) for k in C.HOLIDAYS_2017}
    lab.results[("chat_odd", m.id)] = {"listed": len(said), "found": len(both), "missed": len(missed), "extra": len(extra), "flagged": len(flagged)}
    fig, ax = plt.subplots(figsize=(14, 3.4))
    ax.bar(rule.index, rule.deviation_kWh_per_h, width=1, color=np.where(rule.flag, ui.RED, ui.BLUE))
    ys = rule.deviation_kWh_per_h.reindex(sorted(said))
    ax.scatter(ys.index, ys.values, marker="v", s=50, color=ORANGE, zorder=3, label="a day the chat listed")
    ax.set_ylabel("kWh per hour above (+) or below (-) the usual day"); ax.legend(fontsize=8)
    ax.set_title(f"{m.label}: red = flagged by the notebook's rule (threshold {threshold:g}), orange = listed by the chat", fontsize=11)
    ui.show(fig)
    print(f"The chat listed {len(said)} days. The notebook's rule (Step 3a, threshold {threshold:g}) flags {len(flagged)}: "
          f"the chat found {len(both)} of them, missed {len(missed)} and added {len(extra)} the rule does not flag.")
    print(f"Public holidays in the year: {len(hol)}; the chat listed {len(said & hol)} of them, the rule flags {len(flagged & hol)}.")
    rows = []
    for day in sorted(said | flagged):
        w_, why = got.get(day, ("", ""))
        r = rule.loc[day]
        verdict = "found" if day in both else ("missed by the chat" if day in missed else "only the chat")
        rows.append({"date": day.strftime("%Y-%m-%d"), "weekday": r.weekday, "match": verdict,
                     "the rule": (f"{'higher' if r.deviation_kWh_per_h > 0 else 'lower'} (z {r.z:+.1f})" if r.flag else f"not flagged (z {r.z:+.1f})"),
                     "the chat": (w_ or "?") if day in said else "—", "the chat's reason": why, "holiday": r.holiday})
    ui.table(pd.DataFrame(rows), max_rows=80)


def chat_odd_days(lab, meter_id: str, give: str = "attach the files", steps_text=None, paste_steps_text=None):
    d = series_files(lab, meter_id)
    _box([d["daily_path"]], oddday_prompt(lab, meter_id, give), "odd days",
         lambda text: oddday_score(lab, meter_id, text), attach=_attach(give), steps=_steps(give, steps_text, paste_steps_text))
