"""Rank future contacts using a trained marketing artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from .data import load_data
from .modeling import select_for_budget, validate_model_bundle


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, default=Path("artifacts/model.joblib"))
    parser.add_argument("--output", type=Path, default=Path("reports/predictions.csv"))
    args = parser.parse_args(argv)
    artifact = validate_model_bundle(joblib.load(args.artifact))
    frame = load_data(args.data, require_target=False)
    scores = artifact["model"].predict_proba(frame[artifact["features"]])[:, 1]
    selected, _ = select_for_budget(scores, float(artifact["budget_fraction"]))
    output = pd.DataFrame(
        {
            "row_id": frame.index,
            "response_probability": scores,
            "selected_for_contact": selected,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"predictions written to {args.output}")


if __name__ == "__main__":
    main()
