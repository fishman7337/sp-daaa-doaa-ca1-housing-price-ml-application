"""Compatibility tests for the model artifacts shipped with the application."""

from __future__ import annotations

from pathlib import Path

import joblib
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"


def test_tabular_artifact_loads() -> None:
    model = joblib.load(MODEL_DIR / "best_regressor.joblib")

    assert callable(getattr(model, "predict", None))


def test_keras_artifacts_load_with_expected_shapes() -> None:
    expected_shapes = {
        "best_rnn_model_full.keras": ((None, 154), (None, 1)),
        "tuned_cnn_best.keras": ((None, 224, 224, 3), (None, 1)),
    }

    for filename, (input_shape, output_shape) in expected_shapes.items():
        model = tf.keras.models.load_model(MODEL_DIR / filename, compile=False)
        assert model.input_shape == input_shape
        assert model.output_shape == output_shape
        tf.keras.backend.clear_session()
