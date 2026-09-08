"""Whole-test-set evaluation, confusion matrices, error browsing and the
decision-threshold explorer."""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

from .data import ImageSet
from .models import Classifier
from . import ui


@dataclass
class EvalResult:
    classes: List[str]
    paths: List[Path]
    y_true: np.ndarray
    probs: np.ndarray
    model_name: str = "model"

    @property
    def y_pred(self) -> np.ndarray:
        return self.probs.argmax(1)

    @property
    def accuracy(self) -> float:
        return float((self.y_pred == self.y_true).mean()) if len(self.y_true) else float("nan")

    def confusion(self) -> np.ndarray:
        k = len(self.classes)
        m = np.zeros((k, k), dtype=int)
        for t, p in zip(self.y_true, self.y_pred):
            m[t, p] += 1
        return m

    def per_class_accuracy(self):
        m = self.confusion()
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(m.sum(1) > 0, np.diag(m) / m.sum(1), np.nan)


def evaluate_set(clf: Classifier, image_set: ImageSet, max_images: Optional[int] = None, seed: int = 0,
                 progress: bool = True) -> EvalResult:
    ts = image_set.stratified(total=max_images, seed=seed) if max_images and max_images < len(image_set) else image_set
    probs = clf.predict_paths(ts.paths(), progress=progress)
    return EvalResult(list(ts.pretty_classes), ts.paths(), np.array(ts.labels()), probs, clf.name)


# --------------------------------------------------------------------------- plots
def plot_confusion(res: EvalResult, title: Optional[str] = None, ax=None):
    import matplotlib.pyplot as plt
    m = res.confusion()
    k = len(res.classes)
    if ax is None:
        fig, ax = plt.subplots(figsize=(1.1 * k + 3, 1.0 * k + 2.2))
    else:
        fig = ax.figure
    ax.imshow(m, cmap="Blues")
    ax.set_xticks(range(k)); ax.set_yticks(range(k))
    ax.set_xticklabels(res.classes, rotation=35, ha="right"); ax.set_yticklabels(res.classes)
    ax.set_xlabel("what the model said"); ax.set_ylabel("what it really is")
    vmax = m.max() if m.max() else 1
    for i in range(k):
        for j in range(k):
            ax.text(j, i, str(m[i, j]), ha="center", va="center",
                    color="white" if m[i, j] > 0.5 * vmax else "black",
                    fontweight="bold" if i == j else "normal")
    ax.set_title(title or f"{res.model_name}: {res.accuracy * 100:.1f}% correct on {len(res.y_true)} test photos")
    fig.tight_layout()
    return fig


def summary_text(res: EvalResult) -> str:
    lines = [f"Accuracy: {res.accuracy * 100:.1f}% of {len(res.y_true)} unseen test photos classified correctly."]
    pca = res.per_class_accuracy()
    m = res.confusion()
    for i, c in enumerate(res.classes):
        if np.isnan(pca[i]):
            continue
        wrong = [(res.classes[j], m[i, j]) for j in range(len(res.classes)) if j != i and m[i, j] > 0]
        wrong.sort(key=lambda t: -t[1])
        extra = f" (most often mistaken for: {', '.join(f'{n} x{v}' for n, v in wrong[:2])})" if wrong else ""
        lines.append(f"  - {c}: {pca[i] * 100:.0f}% correct{extra}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- widgets
def error_explorer(res: EvalResult, max_show: int = 12):
    """Dropdowns: true class / predicted class -> grid of matching test photos."""
    import ipywidgets as w
    from IPython.display import display

    opts = ["any"] + list(res.classes)
    true_dd = w.Dropdown(options=opts, value="any", description="really is:", style={"description_width": "initial"})
    pred_dd = w.Dropdown(options=opts, value="any", description="model said:", style={"description_width": "initial"})
    only_wrong = w.Checkbox(value=True, description="show only mistakes")
    out = w.Output()

    def render(*_):
        with out:
            out.clear_output(wait=True)
            idx = np.arange(len(res.y_true))
            if true_dd.value != "any":
                idx = idx[res.y_true[idx] == res.classes.index(true_dd.value)]
            if pred_dd.value != "any":
                idx = idx[res.y_pred[idx] == res.classes.index(pred_dd.value)]
            if only_wrong.value:
                idx = idx[res.y_pred[idx] != res.y_true[idx]]
            n_total = len(idx)
            # most confident first: they are the most interesting mistakes
            conf = res.probs[idx, res.y_pred[idx]] if len(idx) else np.array([])
            idx = idx[np.argsort(-conf)][:max_show]
            if not len(idx):
                if only_wrong.value and not (res.y_pred != res.y_true).any():
                    print("The model made NO mistakes on these test photos. Untick 'show only mistakes' to browse them, "
                          "and go to Part 3 to find out where it does fail.")
                else:
                    print("No photos match this combination.")
                return
            print(f"{n_total} photo(s) match; showing the {len(idx)} where the model was most confident.")
            images = [Image.open(res.paths[i]).convert("RGB") for i in idx]
            titles, colors = [], []
            for i in idx:
                t, p = res.classes[res.y_true[i]], res.classes[res.y_pred[i]]
                titles.append(f"really: {t}\nmodel: {p} ({res.probs[i, res.y_pred[i]] * 100:.0f}%)")
                colors.append(ui.GREEN if t == p else ui.RED)
            ui.show(ui.image_grid(images, titles, colors=colors, ncols=4))

    for x in (true_dd, pred_dd, only_wrong):
        x.observe(render, names="value")
    display(w.VBox([w.HBox([true_dd, pred_dd, only_wrong]), out]))
    render()


def threshold_explorer(res: EvalResult, positive: str):
    """Binary only. Slider on the decision threshold -> counts of misses / false alarms."""
    import ipywidgets as w
    from IPython.display import display
    import matplotlib.pyplot as plt

    pi = res.classes.index(positive)
    score = res.probs[:, pi]
    is_pos = res.y_true == pi
    slider = w.FloatSlider(value=0.5, min=0.02, max=0.98, step=0.02, description="threshold",
                           continuous_update=False, readout_format=".2f")
    out = w.Output()

    def render(*_):
        t = slider.value
        flag = score >= t
        tp, fn = int((flag & is_pos).sum()), int((~flag & is_pos).sum())
        fp, tn = int((flag & ~is_pos).sum()), int((~flag & ~is_pos).sum())
        acc = (tp + tn) / len(score)
        with out:
            out.clear_output(wait=True)
            print(f"Rule: call it '{positive}' when the model's '{positive}' confidence is at least {t * 100:.0f}%.")
            print(f"  Accuracy: {acc * 100:.1f}%")
            print(f"  Missed {positive}s (model said no, but it was one): {fn}  <- dangerous for an inspector")
            print(f"  False alarms (model said {positive}, but it was not): {fp}  <- wastes follow-up work")
            print(f"  Correct: {tp} {positive}s caught, {tn} correct all-clears")
            fig, ax = plt.subplots(figsize=(7, 2.8))
            bins = np.linspace(0, 1, 26)
            ax.hist(score[~is_pos], bins=bins, alpha=0.7, color=ui.BLUE, label=f"really '{res.classes[1 - pi]}'")
            ax.hist(score[is_pos], bins=bins, alpha=0.7, color=ui.RED, label=f"really '{positive}'")
            ax.axvline(t, color="black", ls="--", lw=2, label="threshold")
            ax.set_xlabel(f"model confidence that the photo shows '{positive}'"); ax.set_ylabel("number of test photos")
            ax.set_yscale("log"); ax.legend(loc="upper center", fontsize=8)
            fig.tight_layout(); ui.show(fig)

    slider.observe(render, names="value")
    display(w.VBox([slider, out]))
    render()
