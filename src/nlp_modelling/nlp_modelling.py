"""NLP modelling pipeline for DOAA housing price regression.

This module implements the end-to-end *modelling* pipeline for the NLP
component of the DOAA CA1 project. It assumes that the data preparation
pipeline (`nlp_data_pipeline.py`) has already been run and that the
following artefacts exist in `PROCESSED_DIR`:

* `nlp_sequences.npz` – containing:
    - X_train_seq, X_val_seq, X_test_seq
    - y_train, y_val, y_test
* `vocabulary.txt` – one token per line, representing the vocabulary.

Key responsibilities:
    * Load tokenised sequences and targets from disk.
    * Infer vocabulary size and sequence length.
    * Build a baseline RNN regression model.
    * Build a BiLSTM+BiGRU hypermodel for Keras Tuner.
    * Run Bayesian optimisation to tune the hypermodel.
    * Train both baseline and tuned models with callbacks.
    * Evaluate both models on the test set.
    * Produce a comparison table of test metrics.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import keras_tuner as kt
import numpy as np
import pandas as pd
import tensorflow as tf

# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------

PROCESSED_DIR: str = (
    "/content/drive/MyDrive/Colab Notebooks/DOAA/nlp_data_processed"
)

MODEL_ROOT_DIR: str = (
    "/content/drive/MyDrive/Colab Notebooks/DOAA/nlp_models"
)

os.makedirs(MODEL_ROOT_DIR, exist_ok=True)

RANDOM_STATE: int = 42

# Training configuration.
BASELINE_EPOCHS: int = 30
TUNER_MAX_TRIALS: int = 100
TUNER_EPOCHS: int = 40
TUNED_FINAL_EPOCHS: int = 80

BASELINE_MODEL_NAME: str = "baseline_bilstm"
TUNED_MODEL_NAME: str = "tuned_bilstm_bigru"

print(f"[INFO] Processed NLP data directory: {PROCESSED_DIR}")
print(f"[INFO] NLP model artefacts root:     {MODEL_ROOT_DIR}")
print("[INFO] TensorFlow version:", tf.__version__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SequenceData:
    """Container for tokenised sequences and targets.

    Attributes:
        X_train: Training sequences (shape: [n_train, seq_len]).
        X_val: Validation sequences (shape: [n_val, seq_len]).
        X_test: Test sequences (shape: [n_test, seq_len]).
        y_train: Training targets (shape: [n_train]).
        y_val: Validation targets (shape: [n_val]).
        y_test: Test targets (shape: [n_test]).
        vocab_size: Size of the vocabulary inferred from data or file.
        seq_length: Sequence length (time steps) for each sample.
    """

    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    vocab_size: int
    seq_length: int


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def configure_seeds(seed: int = RANDOM_STATE) -> None:
    """Configure random seeds for reproducibility.

    Args:
        seed: Integer random seed for NumPy and TensorFlow.
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)


def root_mean_squared_error(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
) -> tf.Tensor:
    """Compute the Root Mean Squared Error (RMSE) metric.

    Args:
        y_true: Ground-truth target values.
        y_pred: Predicted target values.

    Returns:
        Scalar tensor representing the RMSE.
    """
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_sequences_and_targets(
    processed_dir: str = PROCESSED_DIR,
) -> SequenceData:
    """Load tokenised sequences and targets from compressed NPZ.

    This function expects a file named `nlp_sequences.npz` in
    `processed_dir`, produced by the `nlp_data_pipeline.py` module.

    The NPZ must contain the following arrays:
        * X_train_seq, X_val_seq, X_test_seq
        * y_train, y_val, y_test

    The vocabulary size is inferred as `max_token_id + 1` across all
    splits. The sequence length is inferred from the shape of
    `X_train_seq`.

    Args:
        processed_dir: Directory containing `nlp_sequences.npz`.

    Returns:
        SequenceData dataclass instance with arrays, vocab size, and
        sequence length.

    Raises:
        FileNotFoundError: If `nlp_sequences.npz` is not found.
    """
    npz_path = os.path.join(processed_dir, "nlp_sequences.npz")
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"Sequences NPZ not found at: {npz_path}")

    data = np.load(npz_path)

    X_train = data["X_train_seq"]
    X_val = data["X_val_seq"]
    X_test = data["X_test_seq"]
    y_train = data["y_train"]
    y_val = data["y_val"]
    y_test = data["y_test"]

    # Infer sequence length and vocab size from sequences.
    seq_length = X_train.shape[1]
    max_token = int(
        max(
            X_train.max(initial=0),
            X_val.max(initial=0),
            X_test.max(initial=0),
        )
    )
    vocab_size = max_token + 1

    print("[INFO] Sequences and targets loaded.")
    print("[INFO] Shapes:")
    print("  X_train:", X_train.shape, "y_train:", y_train.shape)
    print("  X_val:  ", X_val.shape, "y_val:  ", y_val.shape)
    print("  X_test: ", X_test.shape, "y_test: ", y_test.shape)
    print("[INFO] Inferred vocab_size:", vocab_size)
    print("[INFO] Sequence length:", seq_length)

    return SequenceData(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train.astype("float32"),
        y_val=y_val.astype("float32"),
        y_test=y_test.astype("float32"),
        vocab_size=vocab_size,
        seq_length=seq_length,
    )


# ---------------------------------------------------------------------------
# Model architectures
# ---------------------------------------------------------------------------


def build_baseline_rnn(
    vocab_size: int,
    seq_length: int,
    embedding_dim: int = 128,
    rnn_units: int = 64,
) -> tf.keras.Model:
    """Build and compile a baseline BiLSTM regression model.

    Architecture:
        * Embedding layer.
        * Bidirectional LSTM.
        * Dense(64, relu) + Dropout(0.3).
        * Dense(1, linear) for regression.

    Args:
        vocab_size: Size of the vocabulary (max token ID + 1).
        seq_length: Input sequence length (number of time steps).
        embedding_dim: Dimensionality of the token embeddings.
        rnn_units: Number of LSTM units in the Bidirectional layer.

    Returns:
        A compiled Keras Model with MSE loss, MAE, and RMSE metrics.
    """
    inputs = tf.keras.Input(
        shape=(seq_length,),
        dtype="int32",
        name="token_ids",
    )

    x = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        name="embedding",
    )(inputs)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            rnn_units,
            return_sequences=False,
            name="lstm",
        ),
        name="bilstm",
    )(x)

    x = tf.keras.layers.Dense(
        64,
        activation="relu",
        name="dense_64",
    )(x)
    x = tf.keras.layers.Dropout(
        0.3,
        name="dropout_0_3",
    )(x)

    outputs = tf.keras.layers.Dense(
        1,
        activation="linear",
        name="price_output",
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name=BASELINE_MODEL_NAME,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
            root_mean_squared_error,
        ],
    )

    return model


def make_rnn_hypermodel(
    vocab_size: int,
    seq_length: int,
):
    """Create a hypermodel builder function for Keras Tuner.

    The returned callable accepts a `HyperParameters` object and returns
    a compiled Keras model. This design allows the hypermodel to be
    parameterised by `vocab_size` and `seq_length`.

    Search space (illustrative):
        * Embedding dimension.
        * BiLSTM units.
        * BiGRU units.
        * Optional second BiGRU layer.
        * Dense layer width.
        * Dropout rate.
        * Adam learning rate.

    Args:
        vocab_size: Size of the vocabulary.
        seq_length: Fixed sequence length.

    Returns:
        A function `hypermodel(hp: kt.HyperParameters) -> tf.keras.Model`.
    """

    def hypermodel(hp: kt.HyperParameters) -> tf.keras.Model:
        """Build a BiLSTM+BiGRU regression model for tuning."""
        embedding_dim = hp.Choice(
            "embedding_dim",
            values=[64, 128, 192, 256],
        )

        lstm_units = hp.Int(
            "lstm_units",
            min_value=64,
            max_value=256,
            step=64,
        )

        gru_units = hp.Int(
            "gru_units",
            min_value=64,
            max_value=256,
            step=64,
        )

        use_second_gru = hp.Boolean("use_second_gru")

        dense_units = hp.Int(
            "dense_units",
            min_value=64,
            max_value=256,
            step=64,
        )

        dropout_rate = hp.Float(
            "dropout_rate",
            min_value=0.1,
            max_value=0.5,
            step=0.1,
        )

        learning_rate = hp.Float(
            "learning_rate",
            min_value=1e-4,
            max_value=5e-3,
            sampling="log",
        )

        inputs = tf.keras.Input(
            shape=(seq_length,),
            dtype="int32",
            name="token_ids",
        )

        x = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            name="embedding",
        )(inputs)

        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                lstm_units,
                return_sequences=True,
                name="lstm",
            ),
            name="bilstm",
        )(x)

        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(
                gru_units,
                return_sequences=use_second_gru,
                name="gru_1",
            ),
            name="bigru_1",
        )(x)

        if use_second_gru:
            x = tf.keras.layers.Bidirectional(
                tf.keras.layers.GRU(
                    gru_units // 2,
                    return_sequences=False,
                    name="gru_2",
                ),
                name="bigru_2",
            )(x)

        x = tf.keras.layers.Dense(
            dense_units,
            activation="relu",
            name="dense_1",
        )(x)
        x = tf.keras.layers.Dropout(
            dropout_rate,
            name="dropout",
        )(x)

        outputs = tf.keras.layers.Dense(
            1,
            activation="linear",
            name="price_output",
        )(x)

        model = tf.keras.Model(
            inputs=inputs,
            outputs=outputs,
            name=TUNED_MODEL_NAME,
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="mse",
            metrics=[
                tf.keras.metrics.MeanAbsoluteError(name="mae"),
                root_mean_squared_error,
            ],
        )

        return model

    return hypermodel


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------


def train_model_with_callbacks(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    out_dir: str,
    model_name: str,
    max_epochs: int,
) -> tf.keras.callbacks.History:
    """Train a model with EarlyStopping, ReduceLROnPlateau, and checkpoint.

    Args:
        model: Compiled Keras model to train.
        X_train: Training sequences.
        y_train: Training targets.
        X_val: Validation sequences.
        y_val: Validation targets.
        out_dir: Directory to store checkpoints and plots.
        model_name: Short name used in file naming.
        max_epochs: Maximum number of training epochs.

    Returns:
        Keras History object with training and validation metrics.
    """
    os.makedirs(out_dir, exist_ok=True)

    checkpoint_path = os.path.join(
        out_dir,
        f"{model_name}_best.weights.h5",
    )

    early_stop_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
    )

    reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1,
    )

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=max_epochs,
        batch_size=64,
        callbacks=[early_stop_cb, reduce_lr_cb, checkpoint_cb],
        verbose=1,
    )

    return history


def save_history_and_plots(
    history: tf.keras.callbacks.History,
    out_dir: str,
    prefix: str,
) -> None:
    """Save training history as JSON and generate metric plots.

    Args:
        history: Keras History object from `model.fit`.
        out_dir: Directory to store outputs.
        prefix: Prefix for filenames (e.g. "baseline", "tuned").
    """
    os.makedirs(out_dir, exist_ok=True)

    history_dict = history.history
    history_path = os.path.join(out_dir, f"{prefix}_history.json")

    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history_dict, f, indent=2)

    print(f"[INFO] Saved {prefix} history → {history_path}")

    def _plot(metric_name: str, ylabel: str) -> None:
        """Plot a single metric and save to disk."""
        values = history_dict.get(metric_name)
        val_values = history_dict.get(f"val_{metric_name}")

        if values is None or val_values is None:
            print(
                f"[WARN] Metric '{metric_name}' not found in history. "
                "Skipping plot."
            )
            return

        import matplotlib.pyplot as plt  # Local import for plotting.

        epochs = range(1, len(values) + 1)
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, values, label=f"Train {metric_name}")
        plt.plot(epochs, val_values, label=f"Val {metric_name}")
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(f"{prefix} – {metric_name}")
        plt.legend()
        plt.tight_layout()

        plot_path = os.path.join(out_dir, f"{prefix}_{metric_name}.png")
        plt.savefig(plot_path)
        plt.close()

        print(f"[INFO] Saved {prefix} {metric_name} plot → {plot_path}")

    _plot("loss", "MSE loss")
    _plot("mae", "Mean Absolute Error")
    _plot("root_mean_squared_error", "Root Mean Squared Error")


def evaluate_model(
    model: tf.keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    """Evaluate a model on the test set.

    Args:
        model: Trained Keras model.
        X_test: Test sequences.
        y_test: Test targets.

    Returns:
        Dictionary mapping metric names to scalar values.
    """
    results = model.evaluate(X_test, y_test, verbose=1)
    metrics_dict: Dict[str, float] = {
        name: float(value)
        for name, value in zip(model.metrics_names, results)
    }

    print("[INFO] Test metrics:")
    for name, value in metrics_dict.items():
        print(f"  {name}: {value:.4f}")

    return metrics_dict


# ---------------------------------------------------------------------------
# Results summary
# ---------------------------------------------------------------------------


def build_results_summary(
    baseline_model: tf.keras.Model,
    baseline_test_metrics: List[float],
    tuned_model: tf.keras.Model,
    tuned_test_metrics: List[float],
    tuned_model_name: str = TUNED_MODEL_NAME,
) -> pd.DataFrame:
    """Construct a summary table comparing baseline and tuned models.

    Args:
        baseline_model: Trained baseline Keras model.
        baseline_test_metrics: List of metric values returned from
            `baseline_model.evaluate(...)` on the test set.
        tuned_model: Trained tuned Keras model.
        tuned_test_metrics: List of metric values returned from
            `tuned_model.evaluate(...)` on the test set.
        tuned_model_name: Display name for the tuned model in the summary.

    Returns:
        DataFrame with one row per model and columns corresponding to
        metric names (e.g. 'loss', 'mae', 'root_mean_squared_error').

    Raises:
        ValueError: If the metric name lists of the two models differ.
    """
    baseline_names = baseline_model.metrics_names
    tuned_names = tuned_model.metrics_names

    if baseline_names != tuned_names:
        raise ValueError(
            "Baseline and tuned models must have identical metric "
            "names to build a comparison summary."
        )

    rows = []

    baseline_row = {"model": BASELINE_MODEL_NAME}
    baseline_row.update(
        {name: float(value) for name, value in zip(baseline_names, baseline_test_metrics)}
    )
    rows.append(baseline_row)

    tuned_row = {"model": tuned_model_name}
    tuned_row.update(
        {name: float(value) for name, value in zip(tuned_names, tuned_test_metrics)}
    )
    rows.append(tuned_row)

    summary_df = pd.DataFrame(rows)
    return summary_df


def save_results_summary(
    summary_df: pd.DataFrame,
    out_dir: str,
    filename: str = "results_summary.csv",
) -> None:
    """Save a comparison summary DataFrame to CSV.

    Args:
        summary_df: DataFrame produced by `build_results_summary`.
        out_dir: Directory to store the CSV file.
        filename: Name of the CSV file to write.
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    summary_df.to_csv(path, index=False)
    print(f"[INFO] Saved results summary → {path}")


# ---------------------------------------------------------------------------
# Orchestrators
# ---------------------------------------------------------------------------


def run_baseline_pipeline(
    processed_dir: str = PROCESSED_DIR,
) -> Dict[str, object]:
    """Run the baseline RNN training and evaluation pipeline.

    Args:
        processed_dir: Directory containing NLP sequence artefacts.

    Returns:
        Dictionary containing:
            * "sequence_data": SequenceData instance.
            * "baseline_model": trained baseline model.
            * "baseline_history": Keras History object.
            * "baseline_metrics": dict of test metrics.
            * "baseline_dir": directory where artefacts are stored.
    """
    configure_seeds(RANDOM_STATE)

    seq_data = load_sequences_and_targets(processed_dir)
    baseline_dir = os.path.join(MODEL_ROOT_DIR, BASELINE_MODEL_NAME)

    model = build_baseline_rnn(
        vocab_size=seq_data.vocab_size,
        seq_length=seq_data.seq_length,
    )

    model.summary(print_fn=lambda line: print("[BASELINE] " + line))

    history = train_model_with_callbacks(
        model=model,
        X_train=seq_data.X_train,
        y_train=seq_data.y_train,
        X_val=seq_data.X_val,
        y_val=seq_data.y_val,
        out_dir=baseline_dir,
        model_name=BASELINE_MODEL_NAME,
        max_epochs=BASELINE_EPOCHS,
    )

    save_history_and_plots(
        history=history,
        out_dir=baseline_dir,
        prefix="baseline",
    )

    metrics_dict = evaluate_model(
        model=model,
        X_test=seq_data.X_test,
        y_test=seq_data.y_test,
    )

    return {
        "sequence_data": seq_data,
        "baseline_model": model,
        "baseline_history": history,
        "baseline_metrics": metrics_dict,
        "baseline_dir": baseline_dir,
    }


def run_tuned_pipeline(
    processed_dir: str = PROCESSED_DIR,
    max_trials: int = TUNER_MAX_TRIALS,
) -> Dict[str, object]:
    """Run Keras Tuner + tuned training and evaluation pipeline.

    This pipeline:
        1. Loads sequences and targets.
        2. Configures a BayesianOptimization tuner for the hypermodel.
        3. Runs `tuner.search` with EarlyStopping.
        4. Retrieves and saves best hyperparameters.
        5. Builds the tuned model and trains it with callbacks.
        6. Evaluates tuned model on the test set.

    Args:
        processed_dir: Directory containing NLP sequence artefacts.
        max_trials: Maximum number of hyperparameter trials.

    Returns:
        Dictionary containing:
            * "sequence_data": SequenceData instance.
            * "tuned_model": trained tuned model.
            * "tuned_history": Keras History object.
            * "tuned_metrics": dict of test metrics.
            * "tuner": Keras Tuner object.
            * "tuned_dir": directory where artefacts are stored.
    """
    configure_seeds(RANDOM_STATE)

    seq_data = load_sequences_and_targets(processed_dir)

    tuner_dir = os.path.join(MODEL_ROOT_DIR, "tuner_nlp_rnn")
    os.makedirs(tuner_dir, exist_ok=True)

    hypermodel_fn = make_rnn_hypermodel(
        vocab_size=seq_data.vocab_size,
        seq_length=seq_data.seq_length,
    )

    tuner = kt.BayesianOptimization(
        hypermodel=hypermodel_fn,
        objective=kt.Objective("val_root_mean_squared_error", "min"),
        max_trials=max_trials,
        directory=tuner_dir,
        project_name="nlp_price_rnn",
        overwrite=False,
    )

    print("[INFO] Tuner search space summary:")
    tuner.search_space_summary()

    tuner_early_stop_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=False,
    )

    print("[INFO] Starting hyperparameter search...")
    tuner.search(
        seq_data.X_train,
        seq_data.y_train,
        validation_data=(seq_data.X_val, seq_data.y_val),
        epochs=TUNER_EPOCHS,
        batch_size=64,
        callbacks=[tuner_early_stop_cb],
        verbose=1,
    )

    print("[INFO] Tuning completed. Best trials summary:")
    tuner.results_summary()

    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

    # Persist best hyperparameters.
    best_hp_path = os.path.join(tuner_dir, "best_hyperparameters.json")
    with open(best_hp_path, "w", encoding="utf-8") as f:
        json.dump(best_hp.values, f, indent=2)
    print(f"[INFO] Saved best hyperparameters → {best_hp_path}")

    tuned_dir = os.path.join(MODEL_ROOT_DIR, TUNED_MODEL_NAME)
    os.makedirs(tuned_dir, exist_ok=True)

    tuned_model = hypermodel_fn(best_hp)
    tuned_model.summary(print_fn=lambda line: print("[TUNED] " + line))

    tuned_history = train_model_with_callbacks(
        model=tuned_model,
        X_train=seq_data.X_train,
        y_train=seq_data.y_train,
        X_val=seq_data.X_val,
        y_val=seq_data.y_val,
        out_dir=tuned_dir,
        model_name=TUNED_MODEL_NAME,
        max_epochs=TUNED_FINAL_EPOCHS,
    )

    save_history_and_plots(
        history=tuned_history,
        out_dir=tuned_dir,
        prefix="tuned",
    )

    tuned_metrics = evaluate_model(
        model=tuned_model,
        X_test=seq_data.X_test,
        y_test=seq_data.y_test,
    )

    # Also persist tuned test metrics as JSON.
    metrics_path = os.path.join(tuned_dir, "tuned_test_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(tuned_metrics, f, indent=2)
    print(f"[INFO] Saved tuned test metrics → {metrics_path}")

    return {
        "sequence_data": seq_data,
        "tuned_model": tuned_model,
        "tuned_history": tuned_history,
        "tuned_metrics": tuned_metrics,
        "tuner": tuner,
        "tuned_dir": tuned_dir,
    }


def run_full_nlp_modelling_pipeline(
    processed_dir: str = PROCESSED_DIR,
) -> Dict[str, object]:
    """Run baseline and tuned NLP modelling pipelines and compare results.

    Args:
        processed_dir: Directory containing NLP sequence artefacts.

    Returns:
        Dictionary containing combined artefacts:
            * "baseline": outputs from `run_baseline_pipeline`.
            * "tuned": outputs from `run_tuned_pipeline`.
            * "results_summary": DataFrame comparing test metrics.
    """
    baseline_outputs = run_baseline_pipeline(processed_dir)
    tuned_outputs = run_tuned_pipeline(processed_dir)

    # Re-evaluate both models to get raw metric lists for summary builder.
    seq_data = baseline_outputs["sequence_data"]

    baseline_model = baseline_outputs["baseline_model"]
    tuned_model = tuned_outputs["tuned_model"]

    baseline_metrics_list = baseline_model.evaluate(
        seq_data.X_test,
        seq_data.y_test,
        verbose=0,
    )
    tuned_metrics_list = tuned_model.evaluate(
        seq_data.X_test,
        seq_data.y_test,
        verbose=0,
    )

    results_summary = build_results_summary(
        baseline_model=baseline_model,
        baseline_test_metrics=baseline_metrics_list,
        tuned_model=tuned_model,
        tuned_test_metrics=tuned_metrics_list,
    )

    comparison_dir = os.path.join(MODEL_ROOT_DIR, "comparison")
    save_results_summary(results_summary, comparison_dir)

    return {
        "baseline": baseline_outputs,
        "tuned": tuned_outputs,
        "results_summary": results_summary,
    }


if __name__ == "__main__":
    artefacts = run_full_nlp_modelling_pipeline()
    print("[INFO] NLP modelling pipeline completed.")
    print("[INFO] Results summary:")
    print(artefacts["results_summary"])
