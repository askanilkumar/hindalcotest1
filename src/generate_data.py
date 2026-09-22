"""Generate deterministic synthetic machine-failure data."""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "machine_data.csv"
RANDOM_SEED = 42
FEATURES = [
    "temperature",
    "vibration",
    "pressure",
    "operating_hours",
    "load_percentage",
]


def generate_dataset(output_path: Path = DATA_PATH) -> pd.DataFrame:
    """Generate and save 2,000 records with physically plausible ranges."""
    rng = np.random.default_rng(RANDOM_SEED)
    records = 2_000
    temperature = rng.normal(75, 12, records).clip(35, 120)
    vibration = rng.normal(4.5, 1.6, records).clip(0.5, 12)
    pressure = rng.normal(100, 8, records).clip(70, 130)
    operating_hours = rng.uniform(100, 20_000, records)
    load_percentage = rng.normal(70, 15, records).clip(20, 100)

    risk_score = (
        -7.0
        + 0.055 * (temperature - 70)
        + 0.50 * (vibration - 4)
        + 0.00016 * operating_hours
        + 0.045 * (load_percentage - 65)
        + 0.008 * np.abs(pressure - 100)
    )
    failure_probability = 1 / (1 + np.exp(-risk_score))
    machine_failure = rng.binomial(1, failure_probability)

    dataset = pd.DataFrame(
        {
            "temperature": temperature.round(2),
            "vibration": vibration.round(2),
            "pressure": pressure.round(2),
            "operating_hours": operating_hours.round(2),
            "load_percentage": load_percentage.round(2),
            "machine_failure": machine_failure,
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic machine data.")
    parser.add_argument(
        "--output_path",
        type=Path,
        default=DATA_PATH,
        help="Path for the generated machine_data.csv file.",
    )
    args = parser.parse_args()
    data = generate_dataset(args.output_path)
    print(f"Generated {len(data)} records at {args.output_path}")