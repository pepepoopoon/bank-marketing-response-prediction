"""UCI Bank Marketing schema and chronological split."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

TARGET = "y"
NUMERIC_FEATURES = [
    "age",
    "campaign",
    "pdays",
    "previous",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
]
CATEGORICAL_FEATURES = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "poutcome",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
LEAKAGE_COLUMNS = ["duration"]


class SchemaError(ValueError):
    """Raised when marketing data violates the pre-contact contract."""


def _binary_target(series: pd.Series) -> pd.Series:
    values = series.astype(str).str.strip().str.lower()
    mapped = values.map({"yes": 1, "no": 0, "1": 1, "0": 0, "true": 1, "false": 0})
    if mapped.isna().any():
        bad = sorted(values[mapped.isna()].unique().tolist())
        raise SchemaError(f"unsupported target values: {bad[:5]}")
    return mapped.astype(int)


def validate_frame(frame: pd.DataFrame, *, require_target: bool = True) -> pd.DataFrame:
    required = FEATURES + ([TARGET] if require_target else [])
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise SchemaError(f"missing columns: {missing}")
    if require_target and len(frame) < 50:
        raise SchemaError("at least 50 ordered rows are required")
    clean = frame.copy()
    for column in NUMERIC_FEATURES:
        converted = pd.to_numeric(clean[column], errors="coerce")
        if converted.isna().sum() > clean[column].isna().sum():
            raise SchemaError(f"{column} contains non-numeric values")
        clean[column] = converted
    if ((clean["age"].dropna() < 16) | (clean["age"].dropna() > 100)).any():
        raise SchemaError("age must be between 16 and 100")
    if (clean[["campaign", "previous"]].dropna() < 0).any().any():
        raise SchemaError("contact counts cannot be negative")
    if require_target:
        clean[TARGET] = _binary_target(clean[TARGET])
        if clean[TARGET].nunique() != 2:
            raise SchemaError("target must contain both classes")
    return clean


def load_data(path: str | Path, *, require_target: bool = True) -> pd.DataFrame:
    frame = pd.read_csv(path, sep=None, engine="python")
    return validate_frame(frame, require_target=require_target)


def split_data(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    first = int(len(frame) * 0.60)
    second = int(len(frame) * 0.80)
    train = frame.iloc[:first].copy()
    validation = frame.iloc[first:second].copy()
    test = frame.iloc[second:].copy()
    for name, split in [("train", train), ("validation", validation), ("test", test)]:
        if split[TARGET].nunique() != 2:
            raise SchemaError(f"{name} split does not contain both target classes")
    return (
        train.reset_index(drop=True),
        validation.reset_index(drop=True),
        test.reset_index(drop=True),
    )
