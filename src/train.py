"""Train and evaluate the machine-failure classifier."""

from pathlib import Path
import sys
import argparse

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_data import DATA_PATH, FEATURES, generate_dataset  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "machine_failure_model.pkl"


def train_model(data_path: Path = DATA_PATH, model_path: Path = MODEL_PATH):
    if not data_path.exists():
        generate_dataset(data_path)
    dataset = pd.read_csv(data_path)
    x_train, x_test, y_train, y_test = train_test_split(
        dataset[FEATURES],
        dataset["machine_failure"],
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=dataset["machine_failure"],
    )
    model = RandomForestClassifier(
        n_estimators=200, random_state=RANDOM_STATE, class_weight="balanced"
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1 Score": f1_score(y_test, predictions, zero_division=0),
        "Confusion Matrix": confusion_matrix(y_test, predictions),
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return model, metrics


RANDOM_STATE = 42


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the machine-failure model.")
    parser.add_argument(
        "--data_path",
        type=Path,
        default=DATA_PATH,
        help="Path to the machine_data.csv input file.",
    )
    parser.add_argument(
        "--model_dir",
        type=Path,
        default=MODEL_PATH.parent,
        help="Directory where machine_failure_model.pkl will be saved.",
    )
    args = parser.parse_args()
    _, evaluation = train_model(args.data_path, args.model_dir / "machine_failure_model.pkl")
    for name, value in evaluation.items():
        print(f"{name}:\n{value}" if name == "Confusion Matrix" else f"{name}: {value:.4f}")
    print(f"Model saved to {args.model_dir / 'machine_failure_model.pkl'}")