"""Default wordings of every result the MP6 notebooks show.

Each notebook cell carries its own copy (name_text = \"\"\"...\"\"\", editable in Colab); these are only the fallbacks.
Words in {braces} are filled in when the cell runs; **bold** and *italic* work; each line is shown as its own line.
"""
from typing import Dict, Optional

TEXTS: Dict[str, Dict[str, str]] = {
    # ------------------------------------------------------------------ MP6A: the table
    "show_table": {
        "summary_text": "The table has **{n_rows}** {rows} (rows), **{n_inputs}** input variables, and one output variable: **{output}**.",
        "hist_title_text": "Distribution of {output}",
        "scatter_title_text": "{input} vs. {output}",
    },
    "guess": {
        "button_text": "Submit",
        "result_text": "You classified **{right} of {n}** {rows} into the correct level.",
    },
    "classification_grades": {
        "accuracy_text": "**{model}:** {accuracy} % of the {n_test} test {rows} classified into the correct level.",
        "matrix_text": "Confusion matrix (rows: actual level; columns: predicted level):",
        "guess_text": "On the {n} {rows} from Step 1a, the model was correct on **{model_right}** and you were correct on **{you_right}**.",
    },
    "classification_passfail": {
        "accuracy_text": "**{model}:** {accuracy} % of the {n_test} test {rows} classified correctly (pass: {rule}).",
        "matrix_text": "Confusion matrix (rows: actual; columns: predicted):",
        "proba_title_text": "Predicted probability of passing vs. actual {output} (green: actual pass; red: actual fail)",
        "proba_axis_text": "predicted probability of passing",
    },
    "regression": {
        "result_text": "**{model}:**\nTrained on **{n_train}** {rows}, tested on **{n_test}** it never saw.\n**MAE:** {mae} {unit}\n**R²:** {r2}",
        "plot_title_text": "Predicted vs. actual {output} for the {n_test} test {rows}",
        "worst_text": "The {n} test {rows} with the largest errors:",
        "guess_text": "On the {n} {rows} from Step 1a: MAE {mae} {unit}; converted to levels, the model was correct on **{model_right}** and you on **{you_right}**.",
    },
    "importance": {
        "title_text": "Input importance (decision tree model)",
        "axis_text": "permutation importance (decrease in R² when the input is shuffled)",
    },
    "whatif": {
        "result_text": "Predicted {output}: **{value} {unit}**",
    },
    "chat_table": {
        "button_text": "Score HokieAI predictions",
        "reply_text": "Paste HokieAI's full output here.",
        "compare_text": "Prediction errors of the two models (Part 1) and HokieAI on the same test data:",
        "plot_text": "Predicted vs. actual values for the two models (Part 1) and HokieAI on the same test data",
        "worst_text": "The {n} test {rows} with the largest {run} errors:",
        "repeat_text": "Same prompt, two new chats ({first} and {second}): identical predictions for {same} of {n} {rows}; the other {moved} differ by {mean_diff} {unit} on average (largest: {largest_id}, {largest} {unit}).",
    },
    # ------------------------------------------------------------------ MP6B: the time series
    "buildings": {
        "week_title_text": "Building {letter}: one week in March 2017 (Monday–Sunday)",
        "year_title_text": "Building {letter}: all of 2017",
    },
    "buildings_answer": {
        "line_text": "Building {letter}: you answered **{answer}**; it is the **{use}** ({id}, {area} ft², mean {mean} kWh per hour). {mark}",
        "total_text": "**{right} of 4** correct.",
    },
    "anatomy": {
        "anatomy_text": "Typical day (the median of each hour over 2017): mean **{mean}** kWh per hour, maximum **{maximum}**, minimum **{minimum}**.\nSite air temperature ranged from {temp_min} to {temp_max} °F.",
        "year_title_text": "{building}: hourly electricity use in 2017 (kWh)",
        "week_title_text": "One week in March 2017 (Monday–Sunday)",
        "day_title_text": "Typical day: median hourly use for each weekday over 2017",
    },
    "forecast": {
        "title_text": "{building}: forecasts for the week of {week}",
        "actual_text": "actual",
        "table_text": "Forecast errors over the {hours} hours of the test week:",
    },
    "odd_days": {
        "title_text": "{building}: daily deviation from the typical weekly pattern ({flagged} days flagged)",
        "axis_text": "deviation from typical (kWh per hour)",
        "summary_text": "**{flagged} of {days}** days have a robust z-score beyond ±{threshold}; {holidays} of them are public holidays.",
    },
    "chat_forecast": {
        "button_text": "Score HokieAI predictions",
        "reply_text": "Paste HokieAI's full output here.",
        "plot_text": "{building}: HokieAI and the three methods of Step 2a for the week of {week}",
        "compare_text": "Forecast errors over the same hours:",
        "note_text": "HokieAI received four weeks of history; the methods of Step 2a were trained on all data before {week}.",
    },
    "chat_odd_days": {
        "button_text": "Score HokieAI predictions",
        "reply_text": "Paste HokieAI's full output here.",
        "plot_text": "{building}: days flagged by the z-score rule (red) and listed by HokieAI (orange)",
        "summary_text": "HokieAI listed **{listed}** days. The z-score rule of Step 3a (threshold {threshold}) flags **{flagged}**: HokieAI found {found}, missed {missed}, and added {extra}.",
        "holidays_text": "Public holidays in 2017: {holidays}; HokieAI listed {chat_holidays}, the rule flags {rule_holidays}.",
    },
}


def text(texts: Optional[dict], group: str, key: str) -> str:
    """The notebook's wording if the cell passed one, else the default."""
    v = (texts or {}).get(key)
    return v if v else TEXTS[group][key]


def fill(template: str, **values) -> str:
    try:
        return template.format(**values)
    except (KeyError, IndexError, ValueError) as e:
        return template + f"  [unknown placeholder: {e}]"


def say(texts: Optional[dict], group: str, key: str, **values):
    """Show one wording as markdown, one line per line, with the numbers filled in."""
    from IPython.display import Markdown, display
    s = fill(text(texts, group, key), **values)
    display(Markdown("  \n".join(s.strip().split("\n"))))


def labels(block: Optional[str], defaults: Dict[str, str]) -> Dict[str, str]:
    """An app's labels from a 'key: wording' block written in the notebook (one per line), over the defaults."""
    out = dict(defaults)
    for line in (block or "").splitlines():
        if ":" in line and not line.lstrip().startswith("#"):
            k, v = line.split(":", 1)
            if k.strip() in out and v.strip():
                out[k.strip()] = v.strip().replace("\\n", "\n")
    return out


def block(defaults: Dict[str, str]) -> str:
    """The defaults as a 'key: wording' block for a notebook cell."""
    return "\n".join(f"{k}: {v}".replace("\n", "\\n") for k, v in defaults.items())
