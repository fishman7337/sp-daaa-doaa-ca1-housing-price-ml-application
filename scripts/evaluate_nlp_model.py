"""Evaluate the deployed NLP model against the committed held-out split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate(model_path: Path, features_path: Path, targets_path: Path) -> dict[str, int | float]:
    """Compute regression metrics for a serialized Keras model.

    Args:
        model_path: Keras model artifact to evaluate.
        features_path: NumPy array containing tokenized test features.
        targets_path: NumPy array containing test-set prices.

    Returns:
        Sample count, error metrics, and observed prediction ranges.

    """
    model = tf.keras.models.load_model(model_path, compile=False)
    features = np.load(features_path)
    targets = np.load(targets_path)
    predictions = model.predict(features, batch_size=128, verbose=0).reshape(-1)

    return {
        "samples": int(len(targets)),
        "mae_usd": float(mean_absolute_error(targets, predictions)),
        "rmse_usd": float(mean_squared_error(targets, predictions) ** 0.5),
        "r2": float(r2_score(targets, predictions)),
        "target_min_usd": float(np.min(targets)),
        "target_max_usd": float(np.max(targets)),
        "prediction_min_usd": float(np.min(predictions)),
        "prediction_max_usd": float(np.max(predictions)),
    }


def main() -> int:
    """Parse command-line paths, run evaluation, and emit JSON metrics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/best_rnn_model_full.keras"))
    parser.add_argument("--features", type=Path, default=Path("nlp_data/processed/X_test_seq.npy"))
    parser.add_argument("--targets", type=Path, default=Path("nlp_data/processed/y_test.npy"))
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    args = parser.parse_args()

    metrics = evaluate(args.model, args.features, args.targets)
    rendered = json.dumps(metrics, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{rendered}\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
