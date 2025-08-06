"""Воспроизводимые локальные эксперименты на синтетических временных данных."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data import FEATURES, TARGET, split_data, validate_frame
from .generate_smoke_data import generate_smoke_frame
from .modeling import candidate_models, classification_metrics, select_for_budget

RESULT_SCHEMA_VERSION = 1
BASELINE_MODEL = "dummy"
ABLATION_GROUPS = {
    "none": (),
    "demographics": ("age", "job", "marital", "education"),
    "credit_profile": ("default", "housing", "loan"),
    "contact_context": ("contact", "month", "day_of_week", "campaign"),
    "contact_history": ("pdays", "previous", "poutcome"),
    "macroeconomics": (
        "emp.var.rate",
        "cons.price.idx",
        "cons.conf.idx",
        "euribor3m",
        "nr.employed",
    ),
}
SEGMENT_COLUMNS = ("contact", "job", "month")
DATA_QUALITY_SCENARIOS = ("clean", "missing_numeric", "unseen_category", "mixed_missingness")


@dataclass(frozen=True)
class ExperimentConfig:
    """Полная конфигурация одного запуска."""

    experiment_id: str
    rows: int = 420
    data_seed: int = 20250719
    model_seed: int = 20250719
    budget_fraction: float = 0.15
    ablation: str = "none"
    data_quality: str = "clean"

    def validate(self) -> None:
        if not self.experiment_id.strip():
            raise ValueError("experiment_id не может быть пустым")
        if self.rows < 90:
            raise ValueError("Для хронологического эксперимента требуется не менее 90 строк")
        if not 0 < self.budget_fraction <= 1:
            raise ValueError("budget_fraction должен быть в диапазоне (0, 1]")
        if self.ablation not in ABLATION_GROUPS:
            raise ValueError(f"Неизвестная группа абляции: {self.ablation}")
        if self.data_quality not in DATA_QUALITY_SCENARIOS:
            raise ValueError(f"Неизвестный сценарий качества данных: {self.data_quality}")


def _frame_fingerprint(frame: pd.DataFrame) -> str:
    payload = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _apply_data_quality_scenario(
    frame: pd.DataFrame, scenario: str
) -> tuple[pd.DataFrame, dict[str, object]]:
    changed = frame.copy()
    test_start = int(len(changed) * 0.8)
    test_index = changed.index[test_start:]
    if scenario == "missing_numeric":
        changed.loc[test_index[::3], "euribor3m"] = np.nan
    elif scenario == "unseen_category":
        changed.loc[test_index, "job"] = "future_remote_role"
    elif scenario == "mixed_missingness":
        changed.loc[test_index[::2], "education"] = pd.NA
        changed.loc[test_index[1::3], "cons.conf.idx"] = np.nan
    elif scenario != "clean":
        raise ValueError(f"Неизвестный сценарий качества данных: {scenario}")
    report = {
        "scenario": scenario,
        "test_rows": len(test_index),
        "missing_test_cells": {
            column: int(changed.loc[test_index, column].isna().sum())
            for column in FEATURES
            if changed.loc[test_index, column].isna().any()
        },
        "unseen_job_rows": int(changed.loc[test_index, "job"].eq("future_remote_role").sum()),
    }
    return validate_frame(changed), report


def _evaluate_model(
    model: object,
    frame: pd.DataFrame,
    features: list[str],
    budget_fraction: float,
) -> dict[str, object]:
    probability = model.predict_proba(frame[features])[:, 1]
    selected, threshold = select_for_budget(probability, budget_fraction)
    return classification_metrics(frame[TARGET], probability, threshold, selected=selected)


def _temporal_window_report(
    model: object,
    frame: pd.DataFrame,
    features: list[str],
    budget_fraction: float,
) -> dict[str, float | int]:
    probability = model.predict_proba(frame[features])[:, 1]
    selected, _ = select_for_budget(probability, budget_fraction)
    truth = frame[TARGET].to_numpy(dtype=int)
    positives = int(truth.sum())
    return {
        "rows": len(frame),
        "positive_rate": float(truth.mean()),
        "mean_score": float(probability.mean()),
        "brier_score": float(np.mean((truth - probability) ** 2)),
        "selected_count": int(selected.sum()),
        "precision_at_budget": float(truth[selected].mean()),
        "recall_at_budget": float(truth[selected].sum() / positives) if positives else 0.0,
    }


def _temporal_diagnostics(
    model: object,
    test: pd.DataFrame,
    features: list[str],
    budget_fraction: float,
) -> dict[str, object]:
    midpoint = len(test) // 2
    early = _temporal_window_report(model, test.iloc[:midpoint], features, budget_fraction)
    late = _temporal_window_report(model, test.iloc[midpoint:], features, budget_fraction)
    return {
        "early_test_window": early,
        "late_test_window": late,
        "late_minus_early": {
            metric: float(late[metric]) - float(early[metric])
            for metric in (
                "positive_rate",
                "mean_score",
                "brier_score",
                "precision_at_budget",
                "recall_at_budget",
            )
        },
    }


def _segment_diagnostics(
    model: object,
    test: pd.DataFrame,
    features: list[str],
    budget_fraction: float,
) -> dict[str, object]:
    probability = model.predict_proba(test[features])[:, 1]
    selected, _ = select_for_budget(probability, budget_fraction)
    work = test.assign(_score=probability, _selected=selected.astype(int))
    diagnostics: dict[str, object] = {}
    for column in SEGMENT_COLUMNS:
        groups: dict[str, dict[str, float | int]] = {}
        for value, group in work.groupby(column, observed=True, sort=True):
            positives = int(group[TARGET].sum())
            selected_positives = int(group.loc[group["_selected"].eq(1), TARGET].sum())
            groups[str(value)] = {
                "rows": len(group),
                "positive_rate": float(group[TARGET].mean()),
                "mean_score": float(group["_score"].mean()),
                "selected_rate": float(group["_selected"].mean()),
                "precision_at_budget": (
                    float(selected_positives / group["_selected"].sum())
                    if group["_selected"].sum()
                    else 0.0
                ),
                "recall_at_budget": (
                    float(selected_positives / positives) if positives else 0.0
                ),
            }
        selected_rates = [float(report["selected_rate"]) for report in groups.values()]
        diagnostics[column] = {
            "groups": groups,
            "selected_rate_max_gap": max(selected_rates) - min(selected_rates),
        }
    return diagnostics


def run_experiment(config: ExperimentConfig) -> dict[str, object]:
    """Обучить кандидатов и один раз оценить выбранную модель на будущем holdout."""
    config.validate()
    generated = validate_frame(generate_smoke_frame(config.rows, config.data_seed))
    frame, quality_report = _apply_data_quality_scenario(generated, config.data_quality)
    train, validation, test = split_data(frame)
    removed_features = list(ABLATION_GROUPS[config.ablation])
    features = [column for column in FEATURES if column not in removed_features]
    validation_metrics: dict[str, dict[str, object]] = {}
    fitted: dict[str, object] = {}
    candidate_failures: dict[str, str] = {}
    for name, model in candidate_models(config.model_seed, features).items():
        try:
            model.fit(train[features], train[TARGET])
        except ValueError as error:
            candidate_failures[name] = str(error)
            continue
        validation_metrics[name] = _evaluate_model(
            model, validation, features, config.budget_fraction
        )
        fitted[name] = model

    if not fitted:
        raise ValueError("Ни один кандидат не обучился на заданной конфигурации")

    eligible_models = [name for name in validation_metrics if name != BASELINE_MODEL]
    if not eligible_models:
        raise ValueError("Ни одна обучаемая модель не доступна для сравнения с baseline")
    selected_model = max(
        eligible_models,
        key=lambda name: float(validation_metrics[name]["pr_auc"]),
    )
    test_metrics = _evaluate_model(
        fitted[selected_model], test, features, config.budget_fraction
    )
    baseline_metrics = _evaluate_model(
        fitted[BASELINE_MODEL], test, features, config.budget_fraction
    )
    comparison = {
        "pr_auc_delta": float(test_metrics["pr_auc"]) - float(baseline_metrics["pr_auc"]),
        "recall_at_budget_delta": float(test_metrics["recall_at_budget"])
        - float(baseline_metrics["recall_at_budget"]),
        "brier_score_reduction": float(baseline_metrics["brier_score"])
        - float(test_metrics["brier_score"]),
    }
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "experiment_id": config.experiment_id,
        "data_mode": "synthetic_smoke",
        "config": asdict(config),
        "dataset": {
            "sha256": _frame_fingerprint(frame),
            "rows": len(frame),
            "split_rows": {
                "train": len(train),
                "validation": len(validation),
                "test": len(test),
            },
            "positive_rate": {
                "train": float(train[TARGET].mean()),
                "validation": float(validation[TARGET].mean()),
                "test": float(test[TARGET].mean()),
            },
            "quality_scenario": quality_report,
        },
        "feature_set": {
            "ablation": config.ablation,
            "features": features,
            "removed_features": removed_features,
        },
        "selected_model": selected_model,
        "validation_candidates": validation_metrics,
        "candidate_failures": candidate_failures,
        "test": test_metrics,
        "baseline_test": baseline_metrics,
        "comparison_to_baseline": comparison,
        "temporal_diagnostics": _temporal_diagnostics(
            fitted[selected_model], test, features, config.budget_fraction
        ),
        "segment_diagnostics": _segment_diagnostics(
            fitted[selected_model], test, features, config.budget_fraction
        ),
    }
    return result


def write_result(result: dict[str, object], output: str | Path) -> None:
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"Результат уже существует: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=420)
    parser.add_argument("--data-seed", type=int, default=20250719)
    parser.add_argument("--model-seed", type=int, default=20250719)
    parser.add_argument("--budget-fraction", type=float, default=0.15)
    parser.add_argument("--ablation", choices=sorted(ABLATION_GROUPS), default="none")
    parser.add_argument(
        "--data-quality", choices=DATA_QUALITY_SCENARIOS, default="clean"
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = ExperimentConfig(
        experiment_id=args.experiment_id,
        rows=args.rows,
        data_seed=args.data_seed,
        model_seed=args.model_seed,
        budget_fraction=args.budget_fraction,
        ablation=args.ablation,
        data_quality=args.data_quality,
    )
    result = run_experiment(config)
    write_result(result, args.output)
    print(json.dumps(result["test"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
