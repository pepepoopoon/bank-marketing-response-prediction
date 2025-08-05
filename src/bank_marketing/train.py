"""Train marketing-response candidates without touching the future test window."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from .data import FEATURES, TARGET, load_data, split_data
from .modeling import (
    MODEL_BUNDLE_VERSION,
    candidate_models,
    classification_metrics,
    select_for_budget,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, default=Path("artifacts/model.joblib"))
    parser.add_argument("--report", type=Path, default=Path("reports/validation_metrics.json"))
    parser.add_argument("--seed", type=int, default=20250719)
    parser.add_argument("--budget-fraction", type=float, default=0.15)
    args = parser.parse_args(argv)
    train, validation, _ = split_data(load_data(args.data))
    results: dict[str, dict[str, object]] = {}
    fitted = {}
    thresholds = {}
    for name, model in candidate_models(args.seed).items():
        model.fit(train[FEATURES], train[TARGET])
        scores = model.predict_proba(validation[FEATURES])[:, 1]
        selected, threshold = select_for_budget(scores, args.budget_fraction)
        results[name] = classification_metrics(
            validation[TARGET], scores, threshold, selected=selected
        )
        fitted[name] = model
        thresholds[name] = threshold
    best_name = max(results, key=lambda name: float(results[name]["pr_auc"]))
    artifact = {
        "bundle_version": MODEL_BUNDLE_VERSION,
        "model": fitted[best_name],
        "model_name": best_name,
        "threshold": thresholds[best_name],
        "budget_fraction": args.budget_fraction,
        "features": FEATURES,
    }
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.artifact)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps({"selected_model": best_name, "models": results}, indent=2), encoding="utf-8"
    )
    print(f"selected {best_name}; artifact written to {args.artifact}")


if __name__ == "__main__":
    main()
