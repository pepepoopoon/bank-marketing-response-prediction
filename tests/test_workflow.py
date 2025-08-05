from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import joblib
import pandas as pd

from bank_marketing.data import FEATURES, SchemaError, split_data, validate_frame
from bank_marketing.evaluate import main as evaluate
from bank_marketing.generate_smoke_data import generate_smoke_frame
from bank_marketing.modeling import select_for_budget
from bank_marketing.predict import main as predict
from bank_marketing.train import main as train


class BankMarketingWorkflowTest(unittest.TestCase):
    def test_duration_is_not_a_feature_and_split_is_ordered(self) -> None:
        frame = validate_frame(generate_smoke_frame(120))
        pd.testing.assert_frame_equal(generate_smoke_frame(120), generate_smoke_frame(120))
        self.assertNotIn("duration", FEATURES)
        train_frame, validation, test = split_data(frame)
        self.assertEqual(len(frame), len(train_frame) + len(validation) + len(test))
        with self.assertRaises(SchemaError):
            validate_frame(frame.drop(columns=["age"]))

    def test_exact_budget_is_preserved_when_scores_are_tied(self) -> None:
        selected, threshold = select_for_budget(pd.Series([0.25] * 20).to_numpy(), 0.15)

        self.assertEqual(int(selected.sum()), 3)
        self.assertEqual(selected.nonzero()[0].tolist(), [0, 1, 2])
        self.assertEqual(threshold, 0.25)

    def test_end_to_end_without_downloads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "smoke.csv"
            artifact = root / "model.joblib"
            validation = root / "validation.json"
            metrics = root / "metrics.json"
            errors = root / "errors.csv"
            predictions = root / "predictions.csv"
            frame = generate_smoke_frame(180)
            frame.to_csv(data_path, index=False)
            train(
                ["--data", str(data_path), "--artifact", str(artifact), "--report", str(validation)]
            )
            evaluate(
                [
                    "--data",
                    str(data_path),
                    "--artifact",
                    str(artifact),
                    "--metrics",
                    str(metrics),
                    "--errors",
                    str(errors),
                ]
            )
            inference = root / "inference.csv"
            frame.drop(columns=["y"]).head(8).to_csv(inference, index=False)
            predict(
                [
                    "--data",
                    str(inference),
                    "--artifact",
                    str(artifact),
                    "--output",
                    str(predictions),
                ]
            )
            self.assertIn("brier_score", json.loads(metrics.read_text(encoding="utf-8")))
            prediction_frame = pd.read_csv(predictions)
            self.assertEqual(len(prediction_frame), 8)
            self.assertEqual(int(prediction_frame["selected_for_contact"].sum()), 2)

            incompatible_artifact = root / "incompatible.joblib"
            bundle = joblib.load(artifact)
            bundle["features"] = bundle["features"][:-1]
            joblib.dump(bundle, incompatible_artifact)
            with self.assertRaisesRegex(ValueError, "feature contract"):
                predict(
                    [
                        "--data",
                        str(inference),
                        "--artifact",
                        str(incompatible_artifact),
                        "--output",
                        str(root / "unsafe_predictions.csv"),
                    ]
                )


if __name__ == "__main__":
    unittest.main()
