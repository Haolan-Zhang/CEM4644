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
from .texts import TEXTS, fill, say, text

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
             f"attach the file{s}, and paste the prompt.\n" if attach else
             "1. The data is already inside the provided prompt below.\n2. Open {url} (sign in with your VT account), start a new chat, "
             "and paste the prompt.\n")
    return first + "3. Copy the predictions from HokieAI and paste them into the box below.\n4. Click *Score* to evaluate the predictions."


def _md_html(text: str) -> str:
    """The little markdown the instructions use: **bold**, *italic*, links, one line per line."""
    import html
    t = html.escape(text.strip())
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", t)
    t = re.sub(r"(https?://[^\s<]+[^\s<.,;)])", r"<a href='\1' target='_blank'>\1</a>", t)
    return "<br>".join(t.split("\n"))


def _box(files: List[Path], prompt: str, what: str, on_score: Callable[[str], None], attach: bool = True, steps: str = None,
         reply: str = None):
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
    reply_box = w.Textarea(placeholder=reply or TEXTS["chat_table"]["reply_text"],
                           layout=w.Layout(width="100%", height="160px"), description="reply", style={"description_width": "60px"})
    btn = w.Button(description=what, button_style="primary", layout=w.Layout(width="auto"))
    out = w.Output()

    def _go(_):
        with out:
            out.clear_output(wait=True)
            text = reply_box.value.strip()
            if not text:
                print("Paste HokieAI's output first."); return
            try:
                on_score(text)
            except Exception as e:  # noqa: BLE001
                print(f"Could not read that output: {e}")
    btn.on_click(_go)
    display(w.VBox([w.HTML(how), *rows, prompt_box, reply_box, btn, out]))


CHAT_NAME = "HokieAI"


def _add_run(lab, key, name: str, values) -> Tuple[str, bool]:
    """Keep every scored reply, so a second chat can be compared with the first; the same reply twice is not a new run.
    name: "chat" (the chat on its own: HokieAI, HokieAI (chat 2), ...) or the name of a tool run."""
    runs = lab.chat_runs.setdefault(key, [])
    for r in runs:
        if r["values"] == values:
            return r["name"], False
    plain = name == "chat"
    base = CHAT_NAME if plain else name
    k = sum(r["plain"] for r in runs) + 1 if plain else sum(r["name"].startswith(base) for r in runs) + 1
    full = base if k == 1 else (f"{base} (chat {k})" if plain else f"{base} ({k})")
    runs.append({"name": full, "values": values, "plain": plain})
    return full, True


# ============================================================================ the table
TREES, LINE = "Decision tree", "Straight line"
COMPARE_TEXT = "Comparison of the prediction errors of the two models (Part 1) and HokieAI on the same testing data"
PLOT_TEXT = "Predicted vs. actual plot for the two models (Part 1) and HokieAI on the same testing data"


def paste_prompt_default(spec) -> str:
    """The prompt with the data pasted in (the notebook cell carries its own copy, which the teacher can edit);
    {train_data} and {test_data} are replaced by the two tables."""
    return (spec.paste_prompt.strip() + "\n\nTraining data:\n{train_data}\n\nTesting data:\n{test_data}") if spec.paste_prompt else ""


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
    ref = {TREES: pd.Series(trees.predict(Xte.loc[pick]), index=ids),
           LINE: pd.Series(line.predict(Xte.loc[pick]), index=ids)}
    d = dict(train=train, test=test, truth=truth, ref=ref, idcol=idcol,
             tr_path=_save(f"{spec.key}_train.csv", train), te_path=_save(f"{spec.key}_test.csv", test))
    lab.chat_cache["table"] = d
    return d


def _table_intro(spec, d, attach: bool) -> str:
    return (f"I am a construction student. {'I attached two files.' if attach else 'The two files are pasted below.'} "
            f"{d['tr_path'].name} has {len(d['train'])} {spec.chat_about}. {d['te_path'].name} has {len(d['test'])} other {spec.rows}, "
            f"T01 to T{len(d['test']):02d}, with the same columns but without {spec.target}.")


def _table_form(spec, d) -> str:
    return (f"Reply in this form, one line per {spec.row_word}, all {len(d['test'])} lines, plain numbers without commas:\n\n"
            f"{d['idcol']}, {spec.target}\nT01, ...\nT02, ...\n...")


def table_prompt(lab, give: str = "attach the files", paste_prompt_text=None) -> str:
    spec, d = lab.table_spec, table_files(lab)
    attach = _attach(give)
    template = paste_prompt_text or paste_prompt_default(spec)
    if not attach and template:
        data = dict(train_data=d["train"].to_csv(index=False).strip(), test_data=d["test"].to_csv(index=False).strip())
        try:
            return template.format(**data)
        except (KeyError, IndexError, ValueError):
            return template + f"\n\nTraining data:\n{data['train_data']}\n\nTesting data:\n{data['test_data']}"
    p = (_table_intro(spec, d, attach) + f"\n\nUsing what the {len(d['train'])} {spec.rows} in {d['tr_path'].name} show, predict the "
         f"{spec.target_label} in {spec.unit} of each of the {len(d['test'])} {spec.rows} in {d['te_path'].name}. " + _table_form(spec, d))
    if not attach:
        p += (f"\n\n--- {d['tr_path'].name} ---\n{d['train'].to_csv(index=False)}\n--- {d['te_path'].name} ---\n"
              f"{d['test'].to_csv(index=False)}")
    return p


def tool_prompt_default(spec) -> str:
    """The analysis-tool prompt; {model}, {train_file}, {test_file}, {n_train} and {n_test} are filled in when the cell runs."""
    idcol = f"{spec.row_word}_id"
    return (f"The attached file {{train_file}} contains {{n_train}} {spec.chat_about}. The attached file {{test_file}} contains {{n_test}} other "
            f"{spec.rows} (T01–T{{n_test}}) with the same input columns but without {spec.target}.\n"
            f"Use your data-analysis tool (run Python code on the files) to train {{model}} on {{train_file}} to predict {spec.target} from the "
            f"other columns. Then predict {spec.target} for each {spec.row_word} in {{test_file}}.\n"
            f"Reply in the following format, one line per {spec.row_word} for all {{n_test}} {spec.rows}. Use plain numbers without commas:\n"
            f"{idcol}, {spec.target}\nT01, ...\nT02, ...\n...\n\n"
            "After the list, state in two sentences which model and settings you used and which three input columns mattered most.")


def table_tool_prompt(lab, model: str, tool_prompt_text: str = None) -> str:
    spec, d = lab.table_spec, table_files(lab)
    return fill(tool_prompt_text or tool_prompt_default(spec), model=TOOL_MODELS.get(model, model), train_file=d["tr_path"].name,
                test_file=d["te_path"].name, n_train=len(d["train"]), n_test=len(d["test"]))


def parse_rows(text: str) -> Dict[str, float]:
    """'T01, 5437' pairs anywhere in the reply (one per line or all on one line; tables, fences, extra words and
    thousands separators such as 5,437 are fine)."""
    out = {}
    for tid, v in re.findall(r"\b(T\d{1,2})\b\s*\**\s*[,;:|=\t ]\s*\**\s*(-?(?:\d{1,3}(?:,\d{3})+(?!\d)|\d+)(?:\.\d+)?)", text):
        out.setdefault(f"T{int(tid[1:]):02d}", float(v.replace(",", "")))
    return out


def table_score(lab, reply: str, name: str, texts: dict = None):
    spec, d = lab.table_spec, table_files(lab)
    G = "chat_table"
    got = parse_rows(reply)
    ids = list(d["truth"].index)
    have = [i for i in ids if i in got]
    if not have:
        print("No 'T01, number' lines found: check that you pasted HokieAI's full output."); return
    run, new = _add_run(lab, ("table", spec.key), name, {i: got[i] for i in have})
    if len(have) < len(ids):
        print(f"The output has {len(have)} of the {len(ids)} {spec.rows} ({', '.join(i for i in ids if i not in got)} missing); "
              f"only those {len(have)} are scored.")
    full = d["truth"]
    truth = full.reindex(have)

    def row(label, pred: pd.Series):
        """Each forecast on the rows it has (a chat reply may skip some; the notebook's models have all of them)."""
        rows_ = [i for i in full.index if i in pred.index]
        p = pred.reindex(rows_); t_ = full.reindex(rows_); e = (p - t_).abs()
        close = int((e <= t_.abs() * spec.close_pct / 100).sum())
        n_ = len(rows_)
        return {"predicted by": label, f"MAE ({spec.unit})": spec.fmt(e.mean()), f"max error ({spec.unit})": spec.fmt(e.max()),
                f"within {spec.close_pct:g} %": f"{close} of {n_}"}, float(e.mean()), close
    runs = lab.chat_runs[("table", spec.key)]
    rows = []
    for r in runs:
        rr, mae, close = row(("▶ " if r["name"] == run else "") + r["name"], pd.Series(r["values"]))
        rows.append(rr)
        lab.results[("chat_table", r["name"])] = {"mae": mae, "close": close, "n": len(r["values"])}
    for label, pred in d["ref"].items():
        rows.append(row(label, pred)[0])
    say(texts, G, "compare_text")
    ui.table(pd.DataFrame(rows))
    if len(have) < len(full):
        same = {label: float((pred.reindex(have) - truth).abs().mean()) for label, pred in d["ref"].items()}
        print(f"On the same {len(have)} {spec.rows} as {run}: " + ", ".join(f"{k.split(' (')[0]} {spec.fmt(v)} {spec.unit}" for k, v in same.items()) + ".")

    mine = pd.Series(next(r for r in runs if r["name"] == run)["values"]).reindex(have)
    trees = d["ref"][TREES].reindex(have); line = d["ref"][LINE].reindex(have)
    import textwrap
    fig, ax = plt.subplots(figsize=(5.6, 5))
    ax.scatter(truth, line, s=26, color=ui.BLUE, marker="s", alpha=0.7, label=LINE)
    ax.scatter(truth, trees, s=26, color=ui.GREEN, alpha=0.8, label=TREES)
    ax.scatter(truth, mine, s=26, color=ORANGE, marker="D", alpha=0.9, label=run)
    allv = pd.concat([truth, line, trees, mine])
    lim = [allv.min() * 0.95, allv.max() * 1.05]
    ax.plot(lim, lim, "k--", lw=1); ax.set_xlabel(f"actual {spec.label(spec.target)}"); ax.set_ylabel(f"predicted {spec.label(spec.target)}"); ax.legend(fontsize=8)
    from matplotlib.ticker import StrMethodFormatter
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(StrMethodFormatter("{x:,.%df}" % (0 if spec.decimals == 0 else 1)))
    ax.set_title(textwrap.fill(fill(text(texts, G, "plot_text")), 60), fontsize=9); ui.show(fig)

    chats = [r for r in runs if r["plain"]]
    if len(chats) >= 2:
        a, b = pd.Series(chats[0]["values"]), pd.Series(chats[-1]["values"])
        common = [i for i in a.index if i in b.index]
        diff = (a[common] - b[common]).abs()
        same = int((diff < 0.05).sum()); moved = diff[diff >= 0.05]
        say(texts, G, "repeat_text", first=chats[0]["name"], second=chats[-1]["name"], same=same, n=len(common), rows=spec.rows, moved=len(moved),
            mean_diff=spec.fmt(moved.mean()) if len(moved) else "0", unit=spec.unit,
            largest_id=moved.idxmax() if len(moved) else "–", largest=spec.fmt(moved.max()) if len(moved) else "0")
    worst = (mine - truth).abs().sort_values(ascending=False).index[:5]
    view = d["test"].set_index(d["idcol"]).loc[list(worst)].copy()
    view.columns = [spec.label(c) for c in view.columns]
    view.insert(0, f"{run} ({spec.unit})", mine[worst].round(spec.decimals).values)
    view.insert(0, f"{TREES} ({spec.unit})", trees[worst].round(spec.decimals).values)
    view.insert(0, f"actual ({spec.unit})", truth[worst].round(spec.decimals).values)
    view.insert(0, d["idcol"], list(worst))
    say(texts, G, "worst_text", n=len(worst), rows=spec.rows, run=run); ui.table(view.reset_index(drop=True))


def _steps(give: str, steps_text, paste_steps_text):
    return steps_text if _attach(give) else paste_steps_text


def _texts(texts, **named):
    return dict(texts or {}, **{k: v for k, v in named.items() if v})


def chat_table(lab, give: str = "attach the files", steps_text=None, paste_steps_text=None, paste_prompt_text=None,
               compare_text=None, plot_text=None, texts=None):
    t = _texts(texts, compare_text=compare_text, plot_text=plot_text)
    d = table_files(lab)
    _box([d["tr_path"], d["te_path"]], table_prompt(lab, give, paste_prompt_text), text(t, "chat_table", "button_text"),
         lambda reply: table_score(lab, reply, "chat", t), attach=_attach(give),
         steps=_steps(give, steps_text, paste_steps_text), reply=text(t, "chat_table", "reply_text"))


def chat_table_tool(lab, model: str, steps_text=None, compare_text=None, plot_text=None, tool_prompt_text=None, texts=None):
    t = _texts(texts, compare_text=compare_text, plot_text=plot_text)
    d = table_files(lab)
    _box([d["tr_path"], d["te_path"]], table_tool_prompt(lab, model, tool_prompt_text), text(t, "chat_table", "button_text"),
         lambda reply: table_score(lab, reply, f"{CHAT_NAME} + analysis tool, {model}", t), attach=True, steps=steps_text,
         reply=text(t, "chat_table", "reply_text"))


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
    nxt = m.df[["air_temp_F"]].reindex(pd.date_range(t0, periods=C.HORIZON, freq="h")).interpolate().bfill().ffill().round(1)
    nxt.index.name = "timestamp"
    day = m.kwh.resample("D").sum()
    daily = pd.DataFrame({"weekday": day.index.day_name(), "kWh": day.round(0).values,
                          "mean_air_temp_F": m.temp.resample("D").mean().round(1).values}, index=day.index)
    daily.index.name = "date"; daily.index = daily.index.strftime("%Y-%m-%d")
    d = dict(meter=m, hist=hist, next=nxt, daily=daily,
             hist_path=_save(f"{m.id}_last_4_weeks.csv", hist, index=True),
             next_path=_save(f"{m.id}_next_week_temperature.csv", nxt, index=True),
             daily_path=_save(f"{m.id}_daily_2017.csv", daily, index=True))
    lab.chat_cache[key] = d
    return d


FORECAST_PROMPT = (
    "Below are two datasets for a building on a North American university campus used as {use}. The first contains the building's "
    "hourly electricity use (kWh) and outdoor air temperature (°F) for the four weeks from {start} to {end}. The second contains the "
    "outdoor air temperature for the following week, {next_start} to {next_end} (as a weather forecast would provide it).\n"
    "Forecast the building's electricity use for every hour of that week: {hours} values, from {first_hour} to {last_hour}.\n"
    "Reply in the following format, one line per hour for all {hours} hours. Use plain numbers without commas:\n"
    "timestamp, kWh\n{first_hour}, ...\n{second_hour}, ...\n...\n\n"
    "Hourly electricity use and temperature:\n{history_data}\n\nNext week's temperature:\n{temperature_data}")
FORECAST_TOOL_PROMPT = (
    "The attached file {history_file} contains the hourly electricity use (kWh) and outdoor air temperature (°F) of a building on a "
    "North American university campus used as {use}, for the four weeks from {start} to {end}. The attached file {temperature_file} "
    "contains the outdoor air temperature for the following week, {next_start} to {next_end}.\n"
    "Use your data-analysis tool (run Python code on the files) to train {model} on {history_file} to predict hourly electricity use. "
    "Choose the input features yourself (for example hour of day, day of week, temperature, and the use at the same hour one week "
    "earlier). Then forecast every hour of the following week with {temperature_file}: {hours} values, from {first_hour} to {last_hour}.\n"
    "Reply in the following format, one line per hour for all {hours} hours. Use plain numbers without commas:\n"
    "timestamp, kWh\n{first_hour}, ...\n{second_hour}, ...\n...")
ODD_DAYS_PROMPT = (
    "Below is a dataset for a building on a North American university campus used as {use}: its total daily electricity use (kWh) "
    "and mean outdoor air temperature (°F) for every day of {year}.\n"
    "Identify the anomalous days: days whose electricity use does not fit the building's usual pattern. For each, state whether use "
    "was higher or lower than usual and give the most likely reason.\n"
    "Reply in the following format, one line per day:\n"
    "date, higher or lower, reason\n{year}-01-02, lower, ...\n\n"
    "Daily data:\n{daily_data}")


def _series_values(d, model: str = "") -> dict:
    m, h, n = d["meter"], d["hist"], d["next"]
    day = lambda ts: f"{ts:%A}, {ts:%B} {ts.day}, {ts.year}"
    return dict(use=_article(m.use), start=day(h.index[0]), end=day(h.index[-1]), next_start=day(n.index[0]), next_end=day(n.index[-1]),
                hours=len(n), first_hour=f"{n.index[0]:%Y-%m-%d %H:%M}", second_hour=f"{n.index[1]:%Y-%m-%d %H:%M}",
                last_hour=f"{n.index[-1]:%Y-%m-%d %H:%M}", history_file=d["hist_path"].name, temperature_file=d["next_path"].name,
                history_data=d["hist"].to_csv().strip(), temperature_data=d["next"].to_csv().strip(),
                daily_data=d["daily"].to_csv().strip(), year=d["daily"].index[0][:4], model=TOOL_MODELS.get(model, model))


def forecast_prompt(lab, meter_id: str, give: str = "paste the data into the prompt", paste_prompt_text: str = None) -> str:
    return fill(paste_prompt_text or FORECAST_PROMPT, **_series_values(series_files(lab, meter_id)))


def forecast_tool_prompt(lab, meter_id: str, model: str, tool_prompt_text: str = None) -> str:
    return fill(tool_prompt_text or FORECAST_TOOL_PROMPT, **_series_values(series_files(lab, meter_id), model))


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


def forecast_score(lab, meter_id: str, reply: str, name: str, texts: dict = None):
    G = "chat_forecast"
    d = series_files(lab, meter_id); m = d["meter"]
    _, actual = series.test_window(m)
    pred = parse_hours(reply, actual.index)
    n = int(pred.notna().sum())
    if n == 0:
        print("No 'timestamp, kWh' lines found: check that you pasted HokieAI's full output."); return
    run, new = _add_run(lab, ("forecast", m.id), name, {str(k): v for k, v in pred.dropna().items()})
    if n < len(actual):
        print(f"The output has {n} of the {len(actual)} hours; only those are scored.")
    ok = (pred.notna() & actual.notna()).values
    week = f"{actual.index[0]:%B %-d, %Y}"
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(actual.index, actual.values, "k", lw=1.8, label=fill(text(texts, "forecast", "actual_text")))
    rows = []
    colors = {"naive": ui.GREY, "trees": ui.GREEN, "chronos": ui.RED}
    for meth in series.METHODS:
        if series.METHODS[meth] == "chronos" and lab.forecaster is None:
            continue
        f = series.forecast(m, meth, lab.forecaster)
        mae = float(np.mean(np.abs(f.median[ok] - actual.values[ok])))
        ax.plot(actual.index, f.median, color=colors[f.method], lw=1, alpha=0.7, label=meth)
        rows.append((meth, f"{mae:.1f}", f"{mae / actual.mean() * 100:.0f} %", f"{len(actual)}"))
    chat_rows = []
    for r in lab.chat_runs[("forecast", m.id)]:
        s = pd.Series({pd.Timestamp(k): v for k, v in r["values"].items()}).reindex(actual.index)
        both = s.notna() & actual.notna()
        mae = float(np.mean(np.abs(s[both] - actual[both])))
        lab.results[("chat_forecast", m.id, r["name"])] = mae
        chat_rows.append((("▶ " if r["name"] == run else "") + r["name"], f"{mae:.1f}", f"{mae / actual.mean() * 100:.0f} %", f"{int(s.notna().sum())}"))
    ax.plot(pred.index, pred.values, color=ORANGE, lw=2, label=run)
    ax.set_title(fill(text(texts, G, "plot_text"), building=m.label, week=week), fontsize=11); ax.set_ylabel("kWh"); ax.legend(fontsize=8, ncol=5)
    ui.show(fig)
    say(texts, G, "compare_text")
    ui.table(pd.DataFrame(chat_rows + rows, columns=["forecast", "MAE (kWh per hour)", "MAE / mean load", "hours scored"]))
    say(texts, G, "note_text", week=week)


def chat_forecast(lab, meter_id: str, give: str = "paste the data into the prompt", steps_text=None, paste_steps_text=None,
                  paste_prompt_text=None, texts=None):
    d = series_files(lab, meter_id)
    t = texts or {}
    _box([d["hist_path"], d["next_path"]], forecast_prompt(lab, meter_id, give, paste_prompt_text), text(t, "chat_forecast", "button_text"),
         lambda reply: forecast_score(lab, meter_id, reply, "chat", t), attach=False,
         steps=paste_steps_text or (steps_text if not _attach(give) else None) or steps_default(2, False), reply=text(t, "chat_forecast", "reply_text"))


def chat_forecast_tool(lab, meter_id: str, model: str, steps_text=None, tool_prompt_text=None, texts=None):
    d = series_files(lab, meter_id)
    t = texts or {}
    _box([d["hist_path"], d["next_path"]], forecast_tool_prompt(lab, meter_id, model, tool_prompt_text), text(t, "chat_forecast", "button_text"),
         lambda reply: forecast_score(lab, meter_id, reply, f"{CHAT_NAME} + analysis tool, {model}", t), attach=True, steps=steps_text,
         reply=text(t, "chat_forecast", "reply_text"))


# ---------------------------------------------------------------------------- odd days
def oddday_prompt(lab, meter_id: str, give: str = "paste the data into the prompt", paste_prompt_text: str = None) -> str:
    return fill(paste_prompt_text or ODD_DAYS_PROMPT, **_series_values(series_files(lab, meter_id)))


def parse_days(text_: str, year: int) -> Dict[pd.Timestamp, Tuple[str, str]]:
    """Each date in the reply, with 'higher' / 'lower' and the words after it (up to the next date)."""
    hits = list(re.finditer(rf"\b{year}-\d{{2}}-\d{{2}}\b", text_))
    out = {}
    for k, h in enumerate(hits):
        seg = text_[h.end(): hits[k + 1].start() if k + 1 < len(hits) else len(text_)]
        way = re.search(r"\b(higher|lower|high|low|above|below|spike|dip|drop)\b", seg, re.I)
        w_ = "" if not way else ("higher" if way.group(1).lower() in ("higher", "high", "above", "spike") else "lower")
        reason = re.sub(r"^[\s,;:|*\-–]*(higher|lower)?[\s,;:|*\-–]*", "", seg.strip(), flags=re.I).strip().split("\n")[0]
        try:
            out.setdefault(pd.Timestamp(h.group(0)), (w_, reason[:90]))
        except ValueError:
            continue
    return out


def oddday_score(lab, meter_id: str, reply: str, threshold: float = 3.5, texts: dict = None):
    G = "chat_odd_days"
    d = series_files(lab, meter_id); m = d["meter"]
    rule = series.odd_days(m, threshold)
    got = parse_days(reply, rule.index[0].year)
    got = {k: v for k, v in got.items() if k in rule.index}
    if not got:
        print(f"No dates like {rule.index[0].year}-01-02 found: check that you pasted HokieAI's full output."); return
    flagged = set(rule.index[rule.flag]); said = set(got)
    both, missed, extra = said & flagged, flagged - said, said - flagged
    hol = {pd.Timestamp(k) for k in C.HOLIDAYS_2017}
    lab.results[("chat_odd", m.id)] = {"listed": len(said), "found": len(both), "missed": len(missed), "extra": len(extra), "flagged": len(flagged)}
    fig, ax = plt.subplots(figsize=(14, 3.4))
    ax.bar(rule.index, rule.deviation_kWh_per_h, width=1, color=np.where(rule.flag, ui.RED, ui.BLUE))
    ys = rule.deviation_kWh_per_h.reindex(sorted(said))
    ax.scatter(ys.index, ys.values, marker="v", s=50, color=ORANGE, zorder=3, label=CHAT_NAME)
    ax.set_ylabel(fill(text(texts, "odd_days", "axis_text"))); ax.legend(fontsize=8)
    ax.set_title(fill(text(texts, G, "plot_text"), building=m.label, threshold=f"{threshold:g}"), fontsize=11)
    ui.show(fig)
    say(texts, G, "summary_text", listed=len(said), threshold=f"{threshold:g}", flagged=len(flagged), found=len(both), missed=len(missed), extra=len(extra))
    say(texts, G, "holidays_text", holidays=len(hol), chat_holidays=len(said & hol), rule_holidays=len(flagged & hol))
    rows = []
    for day in sorted(said | flagged):
        w_, why = got.get(day, ("", ""))
        r = rule.loc[day]
        verdict = "both" if day in both else ("rule only" if day in missed else f"{CHAT_NAME} only")
        rows.append({"date": day.strftime("%Y-%m-%d"), "weekday": r.weekday, "flagged by": verdict,
                     "z-score": f"{r.z:+.1f}", f"{CHAT_NAME}": (w_ or "?") if day in said else "—",
                     f"{CHAT_NAME}'s reason": why, "holiday": r.holiday})
    ui.table(pd.DataFrame(rows), max_rows=80)


def chat_odd_days(lab, meter_id: str, give: str = "paste the data into the prompt", steps_text=None, paste_steps_text=None,
                  paste_prompt_text=None, texts=None):
    d = series_files(lab, meter_id)
    t = texts or {}
    _box([d["daily_path"]], oddday_prompt(lab, meter_id, give, paste_prompt_text), text(t, "chat_odd_days", "button_text"),
         lambda reply: oddday_score(lab, meter_id, reply, texts=t), attach=False,
         steps=paste_steps_text or (steps_text if not _attach(give) else None) or steps_default(1, False), reply=text(t, "chat_odd_days", "reply_text"))
