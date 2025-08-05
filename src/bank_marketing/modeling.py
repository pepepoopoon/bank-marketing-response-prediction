"""Marketing-response models, calibration, and metrics."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES

MODEL_BUNDLE_VERSION = 1


def validate_model_bundle(bundle: object) -> dict[str, object]:
    """Reject incomplete or schema-incompatible model artifacts before scoring."""
    if not isinstance(bundle, dict):
        raise ValueError("model bundle must be a dictionary")
    required = {
        "bundle_version",
        "model",
        "model_name",
        "threshold",
        "budget_fraction",
        "features",
    }
    missing = sorted(required.difference(bundle))
    if missing:
        raise ValueError(f"model bundle is missing keys: {missing}")
    if bundle["bundle_version"] != MODEL_BUNDLE_VERSION:
        raise ValueError(f"unsupported model bundle version: {bundle['bundle_version']}")
    if bundle["features"] != FEATURES:
        raise ValueError("model bundle feature contract does not match this package")
    if not hasattr(bundle["model"], "predict_proba"):
        raise ValueError("model bundle does not provide predict_proba")
    budget_fraction = float(bundle["budget_fraction"])
    if not 0 < budget_fraction <= 1:
        raise ValueError("model bundle budget_fraction must be in (0, 1]")
    threshold = float(bundle["threshold"])
    if not np.isfinite(threshold):
        raise ValueError("model bundle threshold must be finite")
    return bundle


def preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric, NUMERIC_FEATURES), ("categorical", categorical, CATEGORICAL_FEATURES)]
    )


def _pipeline(estimator: object) -> Pipeline:
    return Pipeline([("preprocess", preprocessor()), ("model", estimator)])


def candidate_models(seed: int) -> dict[str, object]:
    random_forest = RandomForestClassifier(
        n_estimators=80,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=seed,
        n_jobs=1,
    )
    calibrated_forest = CalibratedClassifierCV(
        estimator=_pipeline(random_forest), method="sigmoid", cv=3
    )
    return {
        "dummy": _pipeline(DummyClassifier(strategy="prior")),
        "logistic_regression": _pipeline(LogisticRegression(max_iter=1_000, random_state=seed)),
        "weighted_logistic_regression": _pipeline(
            LogisticRegression(max_iter=1_000, class_weight="balanced", random_state=seed)
        ),
        "random_forest": _pipeline(random_forest),
        "gradient_boosting": _pipeline(GradientBoostingClassifier(random_state=seed)),
        "calibrated_random_forest": calibrated_forest,
    }


def select_for_budget(scores: np.ndarray, budget_fraction: float) -> tuple[np.ndarray, float]:
    """Select an exact top-score quota with stable row-order tie-breaking."""
    if not 0 < budget_fraction <= 1:
        raise ValueError("budget_fraction must be in (0, 1]")
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or not len(values):
        raise ValueError("scores must be a non-empty one-dimensional array")
    if not np.isfinite(values).all():
        raise ValueError("scores must contain only finite values")
    count = max(1, math.ceil(len(values) * budget_fraction))
    ranked = np.argsort(-values, kind="stable")
    selected = np.zeros(len(values), dtype=bool)
    selected[ranked[:count]] = True
    threshold = float(values[ranked[count - 1]])
    return selected, threshold


def threshold_for_budget(scores: np.ndarray, budget_fraction: float) -> float:
    """Return the cutoff for compatibility; ties require ``select_for_budget``."""
    return select_for_budget(scores, budget_fraction)[1]


def classification_metrics(
    truth: pd.Series | np.ndarray,
    scores: np.ndarray,
    threshold: float,
    *,
    selected: np.ndarray | None = None,
) -> dict[str, object]:
    labels = (
        (scores >= threshold).astype(int) if selected is None else np.asarray(selected, dtype=int)
    )
    if labels.shape != np.asarray(scores).shape:
        raise ValueError("selected must have the same shape as scores")
    return {
        "pr_auc": float(average_precision_score(truth, scores)),
        "roc_auc": float(roc_auc_score(truth, scores)),
        "brier_score": float(brier_score_loss(truth, scores)),
        "precision": float(precision_score(truth, labels, zero_division=0)),
        "recall": float(recall_score(truth, labels, zero_division=0)),
        "f1": float(f1_score(truth, labels, zero_division=0)),
        "recall_at_budget": float(recall_score(truth, labels, zero_division=0)),
        "selected_count": int(labels.sum()),
        "selected_fraction": float(labels.mean()),
        "threshold": float(threshold),
        "confusion_matrix": confusion_matrix(truth, labels, labels=[0, 1]).tolist(),
    }
