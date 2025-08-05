"""Create deterministic synthetic chronological marketing data for smoke tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate_smoke_frame(rows: int = 420, seed: int = 20250719) -> pd.DataFrame:
    if rows < 90:
        raise ValueError("rows must be at least 90")
    rng = np.random.default_rng(seed)
    progress = np.linspace(0, 1, rows)
    jobs = rng.choice(
        ["admin.", "blue-collar", "technician", "services", "management", "retired"], rows
    )
    poutcome = rng.choice(["nonexistent", "failure", "success"], rows, p=[0.82, 0.13, 0.05])
    campaign = np.maximum(1, rng.poisson(1.6, rows))
    euribor = 4.8 - 3.5 * progress + rng.normal(0, 0.12, rows)
    logit = (
        -2.8
        + 1.7 * (poutcome == "success")
        + 0.5 * (jobs == "retired")
        - 0.22 * campaign
        - 0.25 * euribor
        + 0.7 * progress
    )
    probability = 1 / (1 + np.exp(-logit))
    target = rng.binomial(1, probability)
    boundaries = [
        (0, int(rows * 0.6)),
        (int(rows * 0.6), int(rows * 0.8)),
        (int(rows * 0.8), rows),
    ]
    for start, stop in boundaries:
        target[start] = 1
        target[min(start + 1, stop - 1)] = 0
    ordered_months = np.array(
        ["mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
    )
    month_index = np.minimum((progress * len(ordered_months)).astype(int), len(ordered_months) - 1)
    return pd.DataFrame(
        {
            "age": rng.integers(18, 86, rows),
            "job": jobs,
            "marital": rng.choice(["single", "married", "divorced"], rows),
            "education": rng.choice(
                ["basic.9y", "high.school", "university.degree", "unknown"], rows
            ),
            "default": rng.choice(["no", "unknown", "yes"], rows, p=[0.78, 0.21, 0.01]),
            "housing": rng.choice(["yes", "no", "unknown"], rows, p=[0.53, 0.44, 0.03]),
            "loan": rng.choice(["yes", "no", "unknown"], rows, p=[0.15, 0.82, 0.03]),
            "contact": rng.choice(["cellular", "telephone"], rows, p=[0.67, 0.33]),
            "month": ordered_months[month_index],
            "day_of_week": rng.choice(["mon", "tue", "wed", "thu", "fri"], rows),
            "campaign": campaign,
            "pdays": rng.choice([999, 3, 6, 10], rows, p=[0.82, 0.05, 0.08, 0.05]),
            "previous": rng.poisson(0.3, rows),
            "poutcome": poutcome,
            "emp.var.rate": 1.4 - 3.2 * progress + rng.normal(0, 0.08, rows),
            "cons.price.idx": 94.1 - 1.5 * progress + rng.normal(0, 0.1, rows),
            "cons.conf.idx": -35 - 12 * progress + rng.normal(0, 1.5, rows),
            "euribor3m": euribor,
            "nr.employed": 5_220 - 250 * progress + rng.normal(0, 12, rows),
            "duration": rng.integers(15, 900, rows),
            "y": np.where(target == 1, "yes", "no"),
        }
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/smoke.csv"))
    parser.add_argument("--rows", type=int, default=420)
    parser.add_argument("--seed", type=int, default=20250719)
    args = parser.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    generate_smoke_frame(args.rows, args.seed).to_csv(args.output, index=False)
    print(f"synthetic smoke data written to {args.output}")


if __name__ == "__main__":
    main()
