from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from bank_marketing.experiments import ExperimentConfig, run_experiment, write_result


class ExperimentRunnerTest(unittest.TestCase):
    def test_run_is_reproducible_and_writes_real_metrics(self) -> None:
        config = ExperimentConfig(
            experiment_id="reproducible-baseline",
            rows=120,
            data_seed=17,
            model_seed=23,
            budget_fraction=0.15,
        )
        first = run_experiment(config)
        second = run_experiment(config)

        self.assertEqual(first, second)
        self.assertEqual(
            first["dataset"]["split_rows"],
            {"train": 72, "validation": 24, "test": 24},
        )
        self.assertGreaterEqual(first["test"]["pr_auc"], 0.0)
        self.assertLessEqual(first["test"]["pr_auc"], 1.0)
        self.assertNotEqual(first["selected_model"], "dummy")
        self.assertEqual(
            first["comparison_to_baseline"]["pr_auc_delta"],
            first["test"]["pr_auc"] - first["baseline_test"]["pr_auc"],
        )
        temporal = first["temporal_diagnostics"]
        self.assertEqual(
            temporal["early_test_window"]["rows"] + temporal["late_test_window"]["rows"],
            24,
        )
        self.assertIn("mean_score", temporal["late_minus_early"])

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "result.json"
            write_result(first, destination)
            stored = json.loads(destination.read_text(encoding="utf-8"))
            self.assertEqual(stored["dataset"]["sha256"], first["dataset"]["sha256"])
            with self.assertRaises(FileExistsError):
                write_result(first, destination)

    def test_invalid_configuration_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            run_experiment(ExperimentConfig(experiment_id="bad", rows=89))

    def test_feature_ablation_removes_only_selected_group(self) -> None:
        result = run_experiment(
            ExperimentConfig(
                experiment_id="macro-ablation",
                rows=120,
                data_seed=17,
                model_seed=23,
                ablation="macroeconomics",
            )
        )

        self.assertEqual(result["feature_set"]["ablation"], "macroeconomics")
        self.assertNotIn("euribor3m", result["feature_set"]["features"])
        self.assertIn("campaign", result["feature_set"]["features"])
        self.assertGreaterEqual(result["test"]["pr_auc"], 0.0)


if __name__ == "__main__":
    unittest.main()
