"""Воспроизводимые локальные эксперименты на синтетических временных данных."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from .data import FEATURES, TARGET, split_data, validate_frame
from .generate_smoke_data import generate_smoke_frame
from .modeling import candidate_models, classification_metrics, select_for_budget

RESULT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ExperimentConfig:
    """Полная конфигурация одного запуска."""

    experiment_id: str
    rows: int = 420
    data_seed: int = 20250719
    model_seed: int = 20250719
    budget_fraction: float = 0.15

    def validate(self) -> None:
        if not self.experiment_id.strip():
            raise ValueError("experiment_id не может быть пустым")
        if self.rows < 90:
            raise ValueError("Для хронологического эксперимента требуется не менее 90 строк")
        if not 0 < self.budget_fraction <= 1:
            raise ValueError("budget_fraction должен быть в диапазоне (0, 1]")


def _frame_fingerprint(frame: pd.DataFrame) -> str:
    payload = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _evaluate_model(
    model: object, frame: pd.DataFrame, budget_fraction: float
) -> dict[str, object]:
    probability = model.predict_proba(frame[FEATURES])[:, 1]
    selected, threshold = select_for_budget(probability, budget_fraction)
    return classification_metrics(frame[TARGET], probability, threshold, selected=selected)


def run_experiment(config: ExperimentConfig) -> dict[str, object]:
    """Обучить кандидатов и один раз оценить выбранную модель на будущем holdout."""
    config.validate()
    frame = validate_frame(generate_smoke_frame(config.rows, config.data_seed))
    train, validation, test = split_data(frame)
    validation_metrics: dict[str, dict[str, object]] = {}
    fitted: dict[str, object] = {}
    candidate_failures: dict[str, str] = {}
    for name, model in candidate_models(config.model_seed).items():
        try:
            model.fit(train[FEATURES], train[TARGET])
        except ValueError as error:
            candidate_failures[name] = str(error)
            continue
        validation_metrics[name] = _evaluate_model(model, validation, config.budget_fraction)
        fitted[name] = model

    if not fitted:
        raise ValueError("Ни один кандидат не обучился на заданной конфигурации")

    selected_model = max(
        validation_metrics,
        key=lambda name: float(validation_metrics[name]["pr_auc"]),
    )
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
        },
        "selected_model": selected_model,
        "validation_candidates": validation_metrics,
        "candidate_failures": candidate_failures,
        "test": _evaluate_model(fitted[selected_model], test, config.budget_fraction),
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
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = ExperimentConfig(
        experiment_id=args.experiment_id,
        rows=args.rows,
        data_seed=args.data_seed,
        model_seed=args.model_seed,
        budget_fraction=args.budget_fraction,
    )
    result = run_experiment(config)
    write_result(result, args.output)
    print(json.dumps(result["test"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
