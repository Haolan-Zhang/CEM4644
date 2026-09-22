"""Regression and classification on the table: the same split, two kinds of model, scored on rows the model never saw."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import confusion_matrix, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import SEED, TEST_SIZE, TableSpec

KINDS = {"a straight line (linear regression)": "linear", "decision trees (gradient boosting)": "trees"}


def split(df: pd.DataFrame, spec: TableSpec):
    X, y = df[spec.features], df[spec.target]
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)


def kind_of(choice: str) -> str:
    return KINDS.get(choice, choice)


def fit_regressor(kind: str, X, y):
    if kind_of(kind) == "linear":
        return LinearRegression().fit(X, y)
    return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, random_state=0).fit(X, y)


def fit_classifier(kind: str, X, labels):
    if kind_of(kind) == "linear":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)).fit(X, labels)
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, random_state=0).fit(X, labels)


@dataclass
class RegResult:
    kind: str
    y_true: np.ndarray
    y_pred: np.ndarray
    mae: float
    r2: float
    worst: pd.DataFrame                                  # the rows with the largest misses


def score_regression(model, kind: str, Xte, yte, spec: TableSpec, n_worst: int = 5) -> RegResult:
    p = model.predict(Xte)
    err = pd.Series(p - yte.values, index=Xte.index)
    worst = Xte.copy(); worst["measured"] = yte.values; worst["predicted"] = np.round(p, 1); worst["miss"] = np.round(err.values, 1)
    worst = worst.reindex(err.abs().sort_values(ascending=False).index[:n_worst])
    return RegResult(kind_of(kind), yte.values, p, float(mean_absolute_error(yte, p)), float(r2_score(yte, p)), worst)


@dataclass
class ClsResult:
    kind: str
    task: str                                            # "grades" or "pass/fail at X"
    labels: List[str]
    y_true: np.ndarray
    y_pred: np.ndarray
    proba: Optional[np.ndarray]                          # probability of "pass" (binary only)
    accuracy: float
    matrix: pd.DataFrame
    false_pass: int = 0
    false_fail: int = 0
    uncertain: int = 0


def grades_result(model, kind, Xte, yte, spec: TableSpec) -> ClsResult:
    labels = [g[0] for g in spec.grades]
    true = np.array([spec.grade_of(v) for v in yte]); pred = model.predict(Xte)
    m = confusion_matrix(true, pred, labels=labels)
    mat = pd.DataFrame(m, index=[f"really {l}" for l in labels], columns=[f"called {l.split(' (')[0]}" for l in labels])
    return ClsResult(kind_of(kind), "grades", labels, true, pred, None, float((true == pred).mean()), mat)


def passfail_result(model, kind, Xte, yte, spec: TableSpec, threshold: float) -> ClsResult:
    true = (yte.values >= threshold); pred = model.predict(Xte).astype(bool)
    proba = model.predict_proba(Xte)[:, list(model.classes_).index(True)] if hasattr(model, "predict_proba") else None
    m = confusion_matrix(true, pred, labels=[True, False])
    mat = pd.DataFrame(m, index=["really passes", "really fails"], columns=["called pass", "called fail"])
    fp = int(((~true) & pred).sum()); fn = int((true & (~pred)).sum())
    unc = int(((proba > 0.3) & (proba < 0.7)).sum()) if proba is not None else 0
    return ClsResult(kind_of(kind), f"pass/fail at {threshold:g} {spec.unit}", ["pass", "fail"], true, pred, proba,
                     float((true == pred).mean()), mat, fp, fn, unc)


def importance(model, Xte, yte, spec: TableSpec) -> List[tuple]:
    imp = permutation_importance(model, Xte, yte, n_repeats=10, random_state=0).importances_mean
    order = np.argsort(-imp)
    return [(spec.features[i], float(imp[i])) for i in order]


def whatif_curve(model, base: pd.Series, feature: str, values: Sequence[float], features: List[str]) -> List[float]:
    rows = []
    for v in values:
        r = base.copy(); r[feature] = v; rows.append(r[features])
    return [float(x) for x in model.predict(pd.DataFrame(rows))]


def predict_one(model, row: pd.Series, features: List[str]) -> float:
    return float(model.predict(pd.DataFrame([row[features]]))[0])
