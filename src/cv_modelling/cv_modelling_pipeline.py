"""Computer vision modelling pipeline for DOAA CV house price regression.

This module implements an end-to-end modelling workflow for predicting
house prices from images using TensorFlow and Keras. It assumes that a
prior data-preparation step has exported processed splits in the
following structure:

    <PROCESSED_DATA_DIR>/
        train/
            images/
                *.jpg / *.png
            labels.csv   # columns: filename, price_usd
        val/
            images/
                *.jpg / *.png
            labels.csv
        test/
            images/
                *.jpg / *.png
            labels.csv

The pipeline covers:

* Loading labels and attaching full image paths for each split.
* Building high-performance ``tf.data.Dataset`` objects.
* Defining a simple baseline CNN model.
* Defining a ResNet-style CNN hypermodel for Keras Tuner.
* Training the baseline model with callbacks.
* Running hyperparameter tuning using Bayesian optimisation.
* Training the best hyperparameter configuration.
* Saving training histories and learning curves.
* Evaluating the final tuned model on the held-out test set and
  persisting metrics.

The code is written to be PEP 8 and PEP 257 compliant, with
Google-style docstrings.
"""

from __future__ import annotations

import json
import os
from typing import Literal

import keras_tuner as kt
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image

# =============================================================================
# Global configuration
# =============================================================================

RANDOM_STATE: int = 42

# Image and batching configuration
IMG_HEIGHT: int = 224
IMG_WIDTH: int = 224
BATCH_SIZE: int = 32

# NOTE: Adjust these paths to match your environment if needed.
PROCESSED_DATA_DIR: str = "/content/drive/MyDrive/Colab Notebooks/DOAA/cv_data_processed"

MODEL_ROOT_DIR: str = "/content/drive/MyDrive/Colab Notebooks/DOAA/cv_models"

# Subdirectories for different model artefacts
BASELINE_DIR: str = os.path.join(MODEL_ROOT_DIR, "baseline_cnn")
TUNED_DIR: str = os.path.join(MODEL_ROOT_DIR, "tuned_cnn")
TUNER_DIR: str = os.path.join(TUNED_DIR, "tuner")

# Ensure all directories exist
for _dir in (MODEL_ROOT_DIR, BASELINE_DIR, TUNED_DIR, TUNER_DIR):
    os.makedirs(_dir, exist_ok=True)

print(f"[INFO] Processed data directory: {PROCESSED_DATA_DIR}")
print(f"[INFO] Model artefacts directory: {MODEL_ROOT_DIR}")
print("[INFO] TensorFlow version:", tf.__version__)


# =============================================================================
# Split loading utilities
# =============================================================================


def load_split_labels(
    base_dir: str,
    split_name: Literal["train", "val", "test"],
    target_col: str = "price_usd",
) -> pd.DataFrame:
    """Load labels for a given split and attach full image paths.

    This function expects a directory layout such as::

        base_dir/
            train/
                images/
                    *.jpg / *.png
                labels.csv
            val/
                ...
            test/
                ...

    Where each ``labels.csv`` contains at least:

    * ``filename``: image filename.
    * ``price_usd``: numeric regression target.

    Args:
        base_dir: Root directory of the processed dataset containing split
            subdirectories (for example, ``train/``, ``val/``, ``test/``).
        split_name: Name of the split to load. Must be one of
            ``"train"``, ``"val"``, or ``"test"``.
        target_col: Name of the target column in ``labels.csv``. Defaults
            to ``"price_usd"``.

    Returns:
        A DataFrame with columns ``filename``, ``price_usd``, and
        ``image_path``, filtered to only rows where the image file exists.

    Raises:
        FileNotFoundError: If the labels CSV does not exist.
        KeyError: If required columns are missing in the CSV.

    """
    split_dir = os.path.join(base_dir, split_name)
    images_dir = os.path.join(split_dir, "images")
    labels_path = os.path.join(split_dir, "labels.csv")

    if not os.path.isfile(labels_path):
        raise FileNotFoundError(f"labels.csv not found for split '{split_name}' at {labels_path}")

    df_labels = pd.read_csv(labels_path)

    required_cols = {"filename", target_col}
    missing_cols = required_cols.difference(df_labels.columns)
    if missing_cols:
        raise KeyError(f"Missing expected columns in labels.csv: {sorted(missing_cols)}")

    df_labels["image_path"] = df_labels["filename"].apply(
        lambda name: os.path.join(images_dir, str(name))
    )

    # Keep only rows where the corresponding image file exists.
    df_labels = df_labels[df_labels["image_path"].apply(os.path.isfile)]

    print(f"[INFO] Loaded {len(df_labels)} samples for split '{split_name}' from {labels_path}")

    return df_labels.reset_index(drop=True)


def load_all_splits(
    base_dir: str = PROCESSED_DATA_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, validation, and test splits.

    Args:
        base_dir: Root directory of processed CV data containing
            ``train/``, ``val/``, and ``test/`` subdirectories.

    Returns:
        A tuple of three DataFrames: ``(df_train, df_val, df_test)``.

    """
    df_train = load_split_labels(base_dir, "train")
    df_val = load_split_labels(base_dir, "val")
    df_test = load_split_labels(base_dir, "test")

    print("\n[INFO] Sample of training labels:")
    print(df_train.head())

    return df_train, df_val, df_test


# =============================================================================
# tf.data pipeline utilities
# =============================================================================


def load_and_preprocess_image(
    image_path: tf.Tensor,
    img_height: int,
    img_width: int,
) -> tf.Tensor:
    """Load an image from a file path and apply robust preprocessing.

    This function uses Pillow via a ``tf.numpy_function`` wrapper to safely
    handle corrupted or truncated image files.

    Behaviour:
        * Reads the image from disk.
        * Converts to RGB.
        * Resizes to ``(img_height, img_width)``.
        * Scales pixel values to ``[0.0, 1.0]``.

    On error (for example, unreadable file), the function returns a black
    image of the requested size instead of crashing the input pipeline.

    Args:
        image_path: Tensor containing the filesystem path to an image.
        img_height: Target image height for resizing.
        img_width: Target image width for resizing.

    Returns:
        A float32 image tensor of shape ``(img_height, img_width, 3)`` with
        values in ``[0.0, 1.0]``.

    """

    def _py_load_image(path_bytes: bytes) -> np.ndarray:
        path_str = path_bytes.decode("utf-8")
        try:
            with Image.open(path_str) as img:
                img = img.convert("RGB")
                img = img.resize((img_width, img_height))
                arr = np.asarray(img, dtype="float32") / 255.0
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Failed to load image '{path_str}': {exc}")
            arr = np.zeros((img_height, img_width, 3), dtype="float32")
        return arr

    image = tf.numpy_function(
        func=_py_load_image,
        inp=[image_path],
        Tout=tf.float32,
    )
    image.set_shape((img_height, img_width, 3))
    return image


def make_dataset_from_df(
    df: pd.DataFrame,
    img_height: int,
    img_width: int,
    batch_size: int,
    shuffle: bool = False,
) -> tf.data.Dataset:
    """Create a ``tf.data.Dataset`` from a labels DataFrame.

    Args:
        df: DataFrame containing ``image_path`` and target column
            ``price_usd``.
        img_height: Target image height for resizing.
        img_width: Target image width for resizing.
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle the dataset at each epoch.

    Returns:
        A ``tf.data.Dataset`` yielding batches of ``(image, target)``.

    """
    required_cols = {"image_path", "price_usd"}
    if not required_cols.issubset(df.columns):
        raise KeyError("DataFrame must contain 'image_path' and 'price_usd' columns.")

    image_paths = df["image_path"].astype(str).to_numpy()
    targets = df["price_usd"].to_numpy(dtype=np.float32)

    ds = tf.data.Dataset.from_tensor_slices((image_paths, targets))

    def _map_fn(
        path: tf.Tensor,
        target: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        image = load_and_preprocess_image(path, img_height, img_width)
        return image, target

    autotune = tf.data.AUTOTUNE
    ds = ds.map(_map_fn, num_parallel_calls=autotune)
    ds = ds.cache()

    if shuffle:
        buffer_size = max(len(image_paths), batch_size * 4)
        ds = ds.shuffle(
            buffer_size=buffer_size,
            reshuffle_each_iteration=True,
        )

    ds = ds.batch(batch_size)
    ds = ds.prefetch(autotune)
    return ds


def build_datasets_from_splits(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    img_height: int = IMG_HEIGHT,
    img_width: int = IMG_WIDTH,
    batch_size: int = BATCH_SIZE,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, int, int, int]:
    """Build train, validation, and test datasets from split DataFrames.

    Args:
        df_train: Training split labels DataFrame.
        df_val: Validation split labels DataFrame.
        df_test: Test split labels DataFrame.
        img_height: Image height in pixels.
        img_width: Image width in pixels.
        batch_size: Batch size.

    Returns:
        A tuple consisting of:
            * ``train_ds``
            * ``val_ds``
            * ``test_ds``
            * ``steps_per_epoch``
            * ``val_steps``
            * ``test_steps``

    """
    train_ds = make_dataset_from_df(
        df=df_train,
        img_height=img_height,
        img_width=img_width,
        batch_size=batch_size,
        shuffle=True,
    )

    val_ds = make_dataset_from_df(
        df=df_val,
        img_height=img_height,
        img_width=img_width,
        batch_size=batch_size,
        shuffle=False,
    )

    test_ds = make_dataset_from_df(
        df=df_test,
        img_height=img_height,
        img_width=img_width,
        batch_size=batch_size,
        shuffle=False,
    )

    steps_per_epoch = int(np.ceil(len(df_train) / batch_size))
    val_steps = int(np.ceil(len(df_val) / batch_size))
    test_steps = int(np.ceil(len(df_test) / batch_size))

    sample_images, sample_targets = next(iter(train_ds))
    print("[INFO] Sample batch shapes:")
    print("  images:", sample_images.shape)
    print("  targets:", sample_targets.shape)

    return train_ds, val_ds, test_ds, steps_per_epoch, val_steps, test_steps


# =============================================================================
# Metrics and baseline CNN definition
# =============================================================================


@tf.keras.utils.register_keras_serializable(package="doaa_cv")
def root_mean_squared_error(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
) -> tf.Tensor:
    """Compute the Root Mean Squared Error (RMSE) metric.

    Args:
        y_true: Ground-truth target tensor.
        y_pred: Predicted target tensor.

    Returns:
        A scalar tensor representing the RMSE between ``y_true`` and
        ``y_pred``.

    """
    error = y_pred - y_true
    mse = tf.reduce_mean(tf.square(error))
    rmse = tf.sqrt(mse)
    return rmse


def build_baseline_cnn(
    img_height: int = IMG_HEIGHT,
    img_width: int = IMG_WIDTH,
) -> tf.keras.Model:
    """Build a simple baseline CNN regression model.

    Architecture:
        * Conv2D(32) + MaxPool
        * Conv2D(64) + MaxPool
        * Conv2D(128) + MaxPool
        * Flatten
        * Dense(256, ReLU) + Dropout(0.3)
        * Dense(1, linear)

    The model is compiled with:
        * MSE loss
        * MAE metric
        * RMSE metric (custom)

    Args:
        img_height: Input image height.
        img_width: Input image width.

    Returns:
        A compiled Keras Model configured for regression.

    """
    inputs = tf.keras.Input(
        shape=(img_height, img_width, 3),
        name="image_input",
    )

    x = tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        activation="relu",
        padding="same",
        name="conv_32",
    )(inputs)
    x = tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2),
        name="pool_32",
    )(x)

    x = tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation="relu",
        padding="same",
        name="conv_64",
    )(x)
    x = tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2),
        name="pool_64",
    )(x)

    x = tf.keras.layers.Conv2D(
        filters=128,
        kernel_size=(3, 3),
        activation="relu",
        padding="same",
        name="conv_128",
    )(x)
    x = tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2),
        name="pool_128",
    )(x)

    x = tf.keras.layers.Flatten(name="flatten")(x)
    x = tf.keras.layers.Dense(
        256,
        activation="relu",
        name="dense_256",
    )(x)
    x = tf.keras.layers.Dropout(
        rate=0.3,
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
        name="baseline_cnn",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
            root_mean_squared_error,
        ],
    )

    model.summary(print_fn=lambda line: print("[BASELINE] " + line))

    return model


# =============================================================================
# Baseline training and history plotting
# =============================================================================

BASELINE_CHECKPOINT_PATH: str = os.path.join(
    BASELINE_DIR,
    "baseline_cnn_best.keras",
)


def get_baseline_callbacks(
    checkpoint_path: str = BASELINE_CHECKPOINT_PATH,
) -> list[tf.keras.callbacks.Callback]:
    """Create callbacks for baseline CNN training.

    The callbacks include:

    * EarlyStopping on validation loss with patience.
    * ReduceLROnPlateau on validation loss.
    * ModelCheckpoint to save the best model.

    Args:
        checkpoint_path: File path for saving the best model.

    Returns:
        A list of configured Keras callbacks.

    """
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    early_stop_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True,
    )

    reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=4,
        min_delta=0.0,
        verbose=1,
    )

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
    )

    return [early_stop_cb, reduce_lr_cb, checkpoint_cb]


def train_baseline_cnn(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    epochs: int = 30,
) -> tf.keras.callbacks.History:
    """Train the baseline CNN model.

    Args:
        model: Compiled baseline CNN model.
        train_ds: Training dataset of (image, target) batches.
        val_ds: Validation dataset of (image, target) batches.
        epochs: Maximum number of training epochs.

    Returns:
        A Keras History object containing the training and validation
        metrics for each epoch.

    """
    callbacks = get_baseline_callbacks(checkpoint_path=BASELINE_CHECKPOINT_PATH)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
    )
    return history


def save_history_and_plots(
    history: tf.keras.callbacks.History,
    out_dir: str,
    prefix: str,
) -> None:
    """Save training history to JSON and generate learning curve plots.

    This function:

    * Serialises the ``history.history`` dictionary to a JSON file.
    * Generates and saves line plots for key metrics over epochs.

    Args:
        history: Keras History object returned by ``model.fit``.
        out_dir: Directory to save JSON and PNG files.
        prefix: Prefix for the output filenames (for example,
            ``"baseline"`` or ``"tuned_cnn"``).

    """
    os.makedirs(out_dir, exist_ok=True)

    history_dict = history.history
    history_path = os.path.join(out_dir, f"{prefix}_history.json")
    with open(history_path, "w", encoding="utf-8") as fp:
        json.dump(history_dict, fp, indent=2)
    print(f"[INFO] Saved training history → {history_path}")

    def _plot_metric(metric_name: str, ylabel: str) -> None:
        """Plot a single metric over epochs."""
        if metric_name not in history_dict:
            return

        values = history_dict[metric_name]
        val_key = f"val_{metric_name}"
        val_values = history_dict.get(val_key)

        epochs_range = range(1, len(values) + 1)

        plt.figure(figsize=(8, 5))
        plt.plot(epochs_range, values, label=f"Train {metric_name}")
        if val_values is not None:
            plt.plot(epochs_range, val_values, label=f"Val {metric_name}")
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(f"{prefix}: {metric_name} over epochs")
        plt.legend()
        plot_path = os.path.join(out_dir, f"{prefix}_{metric_name}.png")
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()
        print(f"[INFO] Saved plot → {plot_path}")

    _plot_metric("loss", "MSE Loss")
    _plot_metric("mae", "Mean Absolute Error")
    _plot_metric("root_mean_squared_error", "Root Mean Squared Error")


# =============================================================================
# ResNet-style hypermodel and Keras Tuner
# =============================================================================


def build_cnn_hypermodel(hp: kt.HyperParameters) -> tf.keras.Model:
    """Build a powerful ResNet-style CNN regression model for tuning.

    The search space includes:

    * Number of residual stages.
    * Number of blocks per stage.
    * Base number of filters.
    * L2 weight regularisation strength.
    * Dense head width and optional second dense layer.
    * Dropout rate.
    * Adam learning rate.

    The architecture is inspired by modern ResNet/VGG-style backbones:
    stacked convolutional stages with residual connections, followed by
    global average pooling and a dense regression head.

    Args:
        hp: Keras HyperParameters object used by Keras Tuner.

    Returns:
        A compiled Keras Model ready for tuning.

    """
    weight_decay = hp.Float(
        "l2_weight",
        min_value=1e-6,
        max_value=1e-3,
        sampling="log",
    )
    base_filters = hp.Int(
        "base_filters",
        min_value=32,
        max_value=64,
        step=32,
    )
    num_stages = hp.Int(
        "num_stages",
        min_value=2,
        max_value=4,
        step=1,
    )
    blocks_per_stage = hp.Int(
        "blocks_per_stage",
        min_value=1,
        max_value=3,
        step=1,
    )

    reg = tf.keras.regularizers.l2(weight_decay)

    inputs = tf.keras.Input(
        shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        name="image_input",
    )

    x = tf.keras.layers.Conv2D(
        filters=base_filters,
        kernel_size=(3, 3),
        strides=1,
        padding="same",
        use_bias=False,
        kernel_regularizer=reg,
        name="stem_conv",
    )(inputs)
    x = tf.keras.layers.BatchNormalization(name="stem_bn")(x)
    x = tf.keras.layers.Activation("relu", name="stem_relu")(x)
    x = tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2),
        name="stem_pool",
    )(x)

    def residual_block(
        block_input: tf.Tensor,
        filters: int,
        stride: int,
        block_name: str,
    ) -> tf.Tensor:
        """Build a single residual block with optional downsampling."""
        shortcut = block_input

        # First conv
        x_rb = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=(3, 3),
            strides=stride,
            padding="same",
            use_bias=False,
            kernel_regularizer=reg,
            name=f"{block_name}_conv1",
        )(block_input)
        x_rb = tf.keras.layers.BatchNormalization(
            name=f"{block_name}_bn1",
        )(x_rb)
        x_rb = tf.keras.layers.Activation(
            "relu",
            name=f"{block_name}_relu1",
        )(x_rb)

        # Second conv
        x_rb = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=(3, 3),
            strides=1,
            padding="same",
            use_bias=False,
            kernel_regularizer=reg,
            name=f"{block_name}_conv2",
        )(x_rb)
        x_rb = tf.keras.layers.BatchNormalization(
            name=f"{block_name}_bn2",
        )(x_rb)

        # Projection for shape match if needed
        if stride != 1 or shortcut.shape[-1] != filters:
            shortcut = tf.keras.layers.Conv2D(
                filters=filters,
                kernel_size=(1, 1),
                strides=stride,
                padding="same",
                use_bias=False,
                kernel_regularizer=reg,
                name=f"{block_name}_proj_conv",
            )(shortcut)
            shortcut = tf.keras.layers.BatchNormalization(
                name=f"{block_name}_proj_bn",
            )(shortcut)

        x_rb = tf.keras.layers.Add(name=f"{block_name}_add")([x_rb, shortcut])
        x_rb = tf.keras.layers.Activation(
            "relu",
            name=f"{block_name}_out",
        )(x_rb)
        return x_rb

    # Residual stages
    x_stage = x
    for stage_idx in range(num_stages):
        filters = base_filters * (2**stage_idx)
        for block_idx in range(blocks_per_stage):
            stride = 2 if block_idx == 0 and stage_idx > 0 else 1
            x_stage = residual_block(
                x_stage,
                filters=filters,
                stride=stride,
                block_name=f"stage{stage_idx + 1}_block{block_idx + 1}",
            )
    x = x_stage

    x = tf.keras.layers.GlobalAveragePooling2D(
        name="global_avg_pool",
    )(x)

    dense_units = hp.Int(
        "dense_units",
        min_value=128,
        max_value=512,
        step=128,
    )
    x = tf.keras.layers.Dense(
        dense_units,
        activation="relu",
        kernel_regularizer=reg,
        name="dense_1",
    )(x)

    if hp.Boolean("use_second_dense"):
        dense_units_2 = hp.Int(
            "dense_units_2",
            min_value=64,
            max_value=256,
            step=64,
        )
        x = tf.keras.layers.Dense(
            dense_units_2,
            activation="relu",
            kernel_regularizer=reg,
            name="dense_2",
        )(x)

    dropout_rate = hp.Float(
        "dropout_rate",
        min_value=0.2,
        max_value=0.6,
        step=0.1,
    )
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
        name="resnet_style_cnn",
    )

    learning_rate = hp.Float(
        "learning_rate",
        min_value=1e-4,
        max_value=5e-3,
        sampling="log",
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


def run_hyperparameter_tuning(
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    max_trials: int = 20,
    executions_per_trial: int = 1,
    epochs: int = 20,
) -> kt.BayesianOptimization:
    """Run Bayesian optimisation for the ResNet-style CNN hypermodel.

    Args:
        train_ds: Training dataset.
        val_ds: Validation dataset.
        max_trials: Maximum number of hyperparameter configurations to try.
        executions_per_trial: Number of executions per trial for robustness.
        epochs: Maximum number of epochs for each trial.

    Returns:
        A configured and already-searched Keras Tuner instance.

    """
    tuner = kt.BayesianOptimization(
        hypermodel=build_cnn_hypermodel,
        objective=kt.Objective(
            "val_root_mean_squared_error",
            direction="min",
        ),
        max_trials=max_trials,
        executions_per_trial=executions_per_trial,
        directory=TUNER_DIR,
        project_name="cv_cnn_tuning",
        overwrite=True,
    )

    early_stop_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_root_mean_squared_error",
        patience=5,
        restore_best_weights=True,
    )

    print("[INFO] Starting hyperparameter search...")
    tuner.search(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[early_stop_cb],
    )

    print("[INFO] Tuning completed. Best models summary:")
    tuner.results_summary()
    return tuner


def get_and_save_best_hyperparameters(
    tuner_obj: kt.engine.tuner.Tuner,
    out_dir: str = TUNER_DIR,
    num_trials: int = 1,
    filename: str = "best_hyperparameters.json",
) -> kt.HyperParameters:
    """Retrieve the best hyperparameters from a tuner and save them to JSON.

    Args:
        tuner_obj: A configured and run Keras Tuner instance.
        out_dir: Directory where the best hyperparameters JSON will be saved.
        num_trials: Number of top trials to consider; the first is returned.
        filename: Name of the JSON file to write.

    Returns:
        The best HyperParameters object from the tuner.

    """
    os.makedirs(out_dir, exist_ok=True)

    best_hps = tuner_obj.get_best_hyperparameters(num_trials=num_trials)
    best_hp = best_hps[0]

    print("[INFO] Best hyperparameters found:")
    for name, value in best_hp.values.items():
        print(f"  {name}: {value}")

    best_hp_path = os.path.join(out_dir, filename)
    with open(best_hp_path, "w", encoding="utf-8") as fp:
        json.dump(best_hp.values, fp, indent=2)

    print(f"[INFO] Best hyperparameters saved to: {best_hp_path}")
    return best_hp


def train_best_hyperparameter_model(
    best_hp: kt.HyperParameters,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    out_dir: str = TUNED_DIR,
    epochs: int = 40,
) -> tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """Build and train the best hyperparameter CNN model.

    Args:
        best_hp: HyperParameters object returned by
            :func:`get_and_save_best_hyperparameters`.
        train_ds: Training dataset.
        val_ds: Validation dataset.
        out_dir: Directory where tuned model and artefacts will be stored.
        epochs: Maximum number of training epochs.

    Returns:
        A tuple of (trained_model, history).

    """
    os.makedirs(out_dir, exist_ok=True)
    checkpoint_path = os.path.join(out_dir, "tuned_cnn_best.keras")

    model = build_cnn_hypermodel(best_hp)
    model.summary(print_fn=lambda line: print("[TUNED] " + line))

    early_stop_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_root_mean_squared_error",
        patience=10,
        restore_best_weights=True,
    )
    reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_root_mean_squared_error",
        factor=0.5,
        patience=5,
        min_delta=0.0,
        verbose=1,
    )
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_root_mean_squared_error",
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[early_stop_cb, reduce_lr_cb, checkpoint_cb],
    )

    return model, history


# =============================================================================
# Final evaluation on test set
# =============================================================================


def evaluate_and_save_metrics(
    model: tf.keras.Model,
    test_dataset: tf.data.Dataset,
    out_dir: str,
    filename: str = "test_metrics.json",
) -> dict[str, float]:
    """Evaluate a trained model on the test set and save metrics to JSON.

    Args:
        model: Trained Keras model to be evaluated.
        test_dataset: Test dataset of (image, target) batches.
        out_dir: Directory where the metrics JSON file will be written.
        filename: Name of the JSON file for metrics.

    Returns:
        A dictionary mapping metric names to their numeric values on the
        test set.

    """
    os.makedirs(out_dir, exist_ok=True)

    results = model.evaluate(test_dataset, return_dict=True)
    metrics_dict = {k: float(v) for k, v in results.items()}

    metrics_path = os.path.join(out_dir, filename)
    with open(metrics_path, "w", encoding="utf-8") as fp:
        json.dump(metrics_dict, fp, indent=2)

    print(f"[INFO] Saved test metrics → {metrics_path}")
    print("[INFO] Test metrics:")
    for name, value in metrics_dict.items():
        print(f"  {name}: {value:.4f}")

    return metrics_dict


# =============================================================================
# Orchestrator
# =============================================================================


def run_full_cv_modelling_pipeline(
    processed_data_dir: str = PROCESSED_DATA_DIR,
    model_root_dir: str = MODEL_ROOT_DIR,
) -> dict[str, object]:
    """Run the full CV modelling pipeline: baseline, tuning, and evaluation.

    This high-level function:

    * Loads train/val/test splits from ``processed_data_dir``.
    * Builds ``tf.data`` datasets.
    * Trains the baseline CNN and saves its history and plots.
    * Runs Keras Tuner to search for a better CNN configuration.
    * Trains the best hyperparameter model and saves its artefacts.
    * Evaluates the tuned model on the test set and saves metrics.

    Args:
        processed_data_dir: Root directory containing processed CV data.
        model_root_dir: Root directory for model artefacts. Currently
            used only for logging; folder paths are derived from module
            constants.

    Returns:
        A dictionary containing key artefacts such as histories, tuner,
        and evaluation metrics.

    """
    print("\n[INFO] === Loading processed CV splits ===")
    df_train, df_val, df_test = load_all_splits(base_dir=processed_data_dir)

    print("\n[INFO] === Building tf.data datasets ===")
    train_ds, val_ds, test_ds, _, _, _ = build_datasets_from_splits(
        df_train=df_train,
        df_val=df_val,
        df_test=df_test,
    )

    print("\n[INFO] === Training baseline CNN ===")
    baseline_model = build_baseline_cnn()
    baseline_history = train_baseline_cnn(
        model=baseline_model,
        train_ds=train_ds,
        val_ds=val_ds,
        epochs=30,
    )
    save_history_and_plots(
        history=baseline_history,
        out_dir=BASELINE_DIR,
        prefix="baseline",
    )

    print("\n[INFO] === Hyperparameter tuning (ResNet-style CNN) ===")
    tuner = run_hyperparameter_tuning(
        train_ds=train_ds,
        val_ds=val_ds,
        max_trials=20,
        executions_per_trial=1,
        epochs=20,
    )
    best_hp = get_and_save_best_hyperparameters(
        tuner_obj=tuner,
        out_dir=TUNER_DIR,
        num_trials=1,
    )

    print("\n[INFO] === Training best tuned CNN ===")
    tuned_model, tuned_history = train_best_hyperparameter_model(
        best_hp=best_hp,
        train_ds=train_ds,
        val_ds=val_ds,
        out_dir=TUNED_DIR,
        epochs=40,
    )
    save_history_and_plots(
        history=tuned_history,
        out_dir=TUNED_DIR,
        prefix="tuned_cnn",
    )

    print("\n[INFO] === Final evaluation on held-out test set ===")
    test_metrics = evaluate_and_save_metrics(
        model=tuned_model,
        test_dataset=test_ds,
        out_dir=TUNED_DIR,
    )

    artefacts: dict[str, object] = {
        "df_train": df_train,
        "df_val": df_val,
        "df_test": df_test,
        "train_ds": train_ds,
        "val_ds": val_ds,
        "test_ds": test_ds,
        "baseline_model": baseline_model,
        "baseline_history": baseline_history,
        "tuner": tuner,
        "best_hp": best_hp,
        "tuned_model": tuned_model,
        "tuned_history": tuned_history,
        "test_metrics": test_metrics,
        "model_root_dir": model_root_dir,
    }
    return artefacts


if __name__ == "__main__":
    ARTEFACTS = run_full_cv_modelling_pipeline()
    print("\n[INFO] CV modelling pipeline completed.")
    print("[INFO] Test metrics:", ARTEFACTS["test_metrics"])
