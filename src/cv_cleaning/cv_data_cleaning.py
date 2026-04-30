"""Computer vision data preparation and dataset pipeline for DOAA CV modelling.

This module implements a complete, reusable data pipeline for an
image-based house price regression task (Austin housing dataset).

The pipeline is designed to be MLOps-friendly and Colab-compatible, and
covers the following responsibilities:

* (Optional) Unzip the raw Kaggle archive into a working directory.
* Load the CSV metadata and attach fully qualified image paths.
* Perform basic cleaning and winsorisation of the target variable.
* Split the cleaned data into train / validation / test sets.
* Apply a log1p + z-normalisation transform to the target.
* Build high-performance ``tf.data.Dataset`` objects for each split.
* Inspect a sample batch for sanity checks.
* Export splits to disk in ``images/`` + ``labels.csv`` format.
* Summarise row counts at each pipeline stage.

The implementation aims to comply with PEP 8, PEP 257, and uses
Google-style docstrings throughout.
"""

from __future__ import annotations

import os
import shutil
import zipfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Default paths (override these when importing the module if needed)
# ---------------------------------------------------------------------------

RAW_ZIP_PATH: str = (
    "/content/drive/MyDrive/Colab Notebooks/DOAA/cv_data_raw/archive (8).zip"
)
EXTRACT_DIR: str = "/content/austin_cv_data"

CSV_FILENAME: str = "austinHousingData.csv"
IMAGE_SUBDIR: str = "homeImages"

PROCESSED_DIR: str = (
    "/content/drive/MyDrive/Colab Notebooks/DOAA/cv_data_processed"
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SplitArrays:
    """Container for image paths and target arrays for dataset splits.

    Attributes:
        x_train: Array of filesystem paths to training images.
        y_train: Array of training targets (normalised log-prices).
        x_val: Array of filesystem paths to validation images.
        y_val: Array of validation targets.
        x_test: Array of filesystem paths to test images.
        y_test: Array of test targets.
        price_mean: Mean of log1p(price_usd) on the training split.
        price_std: Standard deviation of log1p(price_usd) on the
            training split.
    """

    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    price_mean: float
    price_std: float


@dataclass
class TfDatasets:
    """TensorFlow ``tf.data.Dataset`` containers for each split.

    Attributes:
        train_ds: Dataset yielding (image, target) batches for training.
        val_ds: Dataset yielding (image, target) batches for validation.
        test_ds: Dataset yielding (image, target) batches for evaluation.
        steps_per_epoch: Number of batches in one training epoch.
        val_steps: Number of batches in one validation epoch.
        test_steps: Number of batches in the test evaluation loop.
    """

    train_ds: tf.data.Dataset
    val_ds: tf.data.Dataset
    test_ds: tf.data.Dataset
    steps_per_epoch: int
    val_steps: int
    test_steps: int


# ---------------------------------------------------------------------------
# Data extraction utilities
# ---------------------------------------------------------------------------


def unzip_raw_data(raw_zip_path: str, extract_dir: str) -> None:
    """Unzip a raw Kaggle archive into the specified directory.

    If the directory already exists, it is removed before extraction to
    ensure a clean state.

    Args:
        raw_zip_path: Path to the zipped dataset archive.
        extract_dir: Destination directory for the unzipped contents.

    Raises:
        FileNotFoundError: If ``raw_zip_path`` does not exist.
    """
    if not os.path.isfile(raw_zip_path):
        raise FileNotFoundError(f"Zip file not found: {raw_zip_path}")

    if os.path.isdir(extract_dir):
        shutil.rmtree(extract_dir)

    os.makedirs(extract_dir, exist_ok=True)

    with zipfile.ZipFile(raw_zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    print(f"[INFO] Extracted raw data to: {extract_dir}")


# ---------------------------------------------------------------------------
# Metadata loading and cleaning
# ---------------------------------------------------------------------------


def load_metadata(csv_path: str, image_root: str) -> pd.DataFrame:
    """Load ``austinHousingData.csv`` and attach full image paths.

    The CSV is expected to contain at least the following columns:

    * ``latestPrice``: numeric house price (assumed to be in USD).
    * ``homeImage``: image filename for the listing.

    This function:

    * Loads the CSV.
    * Keeps only the price and image filename columns.
    * Renames the columns to ``price_usd`` and ``image_filename``.
    * Builds a full ``image_path`` by joining ``image_root`` with the
      filename.
    * Drops any rows whose image files do not exist on disk.

    Args:
        csv_path: Path to the ``austinHousingData.csv`` file.
        image_root: Directory containing the house image files.

    Returns:
        A cleaned DataFrame with columns ``price_usd`` and ``image_path``.

    Raises:
        FileNotFoundError: If the CSV file cannot be found.
        KeyError: If required columns are missing from the CSV.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required_cols = {"latestPrice", "homeImage"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise KeyError(f"Missing required columns in CSV: {missing}")

    df = df.loc[:, ["latestPrice", "homeImage"]].copy()
    df = df.rename(
        columns={
            "latestPrice": "price_usd",
            "homeImage": "image_filename",
        }
    )

    df["image_filename"] = df["image_filename"].astype(str)
    df["image_path"] = df["image_filename"].apply(
        lambda name: os.path.join(image_root, name)
    )

    df["image_exists"] = df["image_path"].apply(os.path.isfile)
    df = df[df["image_exists"]].drop(
        columns=["image_exists", "image_filename"]
    )
    df = df.reset_index(drop=True)

    print(
        "[INFO] Cleaned metadata with images: "
        f"{df.shape[0]} rows × {df.shape[1]} columns"
    )
    return df


def plot_price_boxplot(df: pd.DataFrame, price_col: str = "price_usd") -> None:
    """Plot a boxplot of property prices to visualise distribution.

    Args:
        df: Cleaned DataFrame containing the target price column.
        price_col: Name of the price column to plot.

    Raises:
        KeyError: If the specified price column is not present.
    """
    if price_col not in df.columns:
        raise KeyError(f"Column {price_col!r} not found in DataFrame.")

    plt.figure(figsize=(10, 5))
    plt.boxplot(df[price_col], vert=False, showfliers=True)
    plt.title("Boxplot of Property Prices (USD)")
    plt.xlabel("Price (USD)")
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.show()

    print(
        "[INFO] Boxplot displayed. "
        f"Min: {df[price_col].min():,.2f}, "
        f"Median: {df[price_col].median():,.2f}, "
        f"Max: {df[price_col].max():,.2f}"
    )


def clean_and_filter(
    df: pd.DataFrame,
    min_price_usd: Optional[float] = None,
    max_price_usd: Optional[float] = None,
    winsor_upper: float = 0.75 * 1e7,
) -> pd.DataFrame:
    """Clean the dataset and filter valid rows for CV regression.

    This function performs a minimal but robust cleaning pipeline for the
    computer vision pricing dataset. It:

    * Validates the presence of required columns.
    * Drops rows with missing or non-positive prices and missing image paths.
    * Optionally filters rows by a specified USD price range.
    * Winsorises extreme prices above ``winsor_upper`` by capping them at
      that value.
    * Ensures that image paths exist on disk.
    * Retains only the core columns: ``image_path`` and ``price_usd``.

    Args:
        df: Input DataFrame containing at minimum ``price_usd`` and
            ``image_path`` columns.
        min_price_usd: Optional lower bound for the USD price filter.
        max_price_usd: Optional upper bound for the USD price filter.
        winsor_upper: Upper cap for winsorisation of ``price_usd``.

    Returns:
        Cleaned DataFrame with only valid rows and the required columns.

    Raises:
        KeyError: If required columns are missing from ``df``.
    """
    required_cols = {"price_usd", "image_path"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    df_out = df.copy()
    df_out = df_out.dropna(subset=["price_usd", "image_path"])

    df_out["price_usd"] = pd.to_numeric(df_out["price_usd"], errors="coerce")
    df_out = df_out.dropna(subset=["price_usd"])
    df_out = df_out[df_out["price_usd"] > 0]

    if min_price_usd is not None:
        df_out = df_out[df_out["price_usd"] >= float(min_price_usd)]
    if max_price_usd is not None:
        df_out = df_out[df_out["price_usd"] <= float(max_price_usd)]

    original_max = df_out["price_usd"].max()
    df_out["price_usd"] = df_out["price_usd"].clip(upper=float(winsor_upper))
    capped_max = df_out["price_usd"].max()

    df_out = df_out[df_out["image_path"].apply(os.path.isfile)]

    keep_cols = ["image_path", "price_usd"]
    df_out = df_out[keep_cols].reset_index(drop=True)

    print(f"[INFO] Cleaned dataset shape: {df_out.shape}")
    print(
        f"[INFO] Winsorisation applied at {winsor_upper:,.0f} USD. "
        f"Original max: {original_max:,.2f}, "
        f"Capped max: {capped_max:,.2f}"
    )
    print("[INFO] All non-essential feature columns dropped.")
    return df_out


# ---------------------------------------------------------------------------
# Train/validation/test splitting and target normalisation
# ---------------------------------------------------------------------------


def create_splits(
    df: pd.DataFrame,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
) -> SplitArrays:
    """Create train, validation, and test splits and normalise targets.

    This function partitions the cleaned dataset into train / validation /
    test splits and applies a log1p-based normalisation to the target
    prices. The normalisation statistics are computed only from the
    training split and then reused across validation and test splits.

    The DataFrame must contain at least:

    * ``image_path``: full path to each image file.
    * ``price_usd``: numeric house price in USD.

    Args:
        df: Cleaned DataFrame containing ``image_path`` and ``price_usd``.
        train_size: Proportion of data allocated to the training split.
        val_size: Proportion of data allocated to the validation split.
        test_size: Proportion of data allocated to the test split.

    Returns:
        SplitArrays instance with image paths and normalised targets.

    Raises:
        KeyError: If required columns are missing.
        ValueError: If ``train_size + val_size + test_size`` is not 1.0
            within a small numerical tolerance.
    """
    required_cols = {"image_path", "price_usd"}
    missing_cols = required_cols.difference(df.columns)
    if missing_cols:
        raise KeyError(f"Missing required columns: {missing_cols}")

    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError(
            "train_size + val_size + test_size must sum to 1.0."
        )

    x_all = df["image_path"].to_numpy()
    y_all = df["price_usd"].to_numpy(dtype=np.float32)

    x_temp, x_test, y_temp, y_test = train_test_split(
        x_all,
        y_all,
        test_size=test_size,
        random_state=42,
        shuffle=True,
    )

    relative_val_size = val_size / (train_size + val_size)
    x_train, x_val, y_train, y_val = train_test_split(
        x_temp,
        y_temp,
        test_size=relative_val_size,
        random_state=42,
        shuffle=True,
    )

    y_train_log = np.log1p(y_train)
    price_mean = float(y_train_log.mean())
    price_std = float(y_train_log.std(ddof=0))

    def _normalise(y_in: np.ndarray) -> np.ndarray:
        """Apply log1p + z-normalisation using training statistics."""
        y_log = np.log1p(y_in)
        return (y_log - price_mean) / (price_std + 1e-8)

    y_train_norm = _normalise(y_train)
    y_val_norm = _normalise(y_val)
    y_test_norm = _normalise(y_test)

    print(
        "[INFO] Split sizes – "
        f"train: {len(x_train)}, val: {len(x_val)}, test: {len(x_test)}"
    )

    return SplitArrays(
        x_train=x_train,
        y_train=y_train_norm,
        x_val=x_val,
        y_val=y_val_norm,
        x_test=x_test,
        y_test=y_test_norm,
        price_mean=price_mean,
        price_std=price_std,
    )


# ---------------------------------------------------------------------------
# TensorFlow dataset creation
# ---------------------------------------------------------------------------


def load_and_preprocess_image(
    image_path: tf.Tensor,
    img_height: int,
    img_width: int,
) -> tf.Tensor:
    """Load an image file and apply robust preprocessing.

    This function reads an image from disk using Pillow via a
    ``tf.numpy_function`` wrapper to avoid crashes from corrupted or
    invalid files.

    Behaviour:
        * On success:
            * Loads image as RGB.
            * Resizes to (img_height, img_width).
            * Scales pixel values to [0.0, 1.0].
        * On failure (for example, invalid or missing image):
            * Returns an array of zeros with the requested shape.

    Args:
        image_path: Tensor containing the path to the image file.
        img_height: Target image height in pixels.
        img_width: Target image width in pixels.

    Returns:
        Float32 image tensor of shape (img_height, img_width, 3).
    """

    def _py_load_image(path_bytes: bytes) -> np.ndarray:
        path_str = path_bytes.decode("utf-8")
        try:
            with Image.open(path_str) as img:
                img = img.convert("RGB")
                img = img.resize((img_width, img_height))
                arr = np.asarray(img, dtype=np.float32) / 255.0
        except Exception:
            arr = np.zeros(
                (img_height, img_width, 3),
                dtype=np.float32,
            )
        return arr

    image = tf.numpy_function(
        func=_py_load_image,
        inp=[image_path],
        Tout=tf.float32,
    )
    image.set_shape((img_height, img_width, 3))
    return image


def make_tf_datasets(
    splits: SplitArrays,
    img_height: int = 224,
    img_width: int = 224,
    batch_size: int = 32,
) -> TfDatasets:
    """Build TensorFlow datasets for training, validation, and testing.

    Args:
        splits: SplitArrays instance produced by :func:`create_splits`.
        img_height: Target image height for resizing.
        img_width: Target image width for resizing.
        batch_size: Batch size for all datasets.

    Returns:
        TfDatasets instance containing tf.data pipelines and step counts.
    """
    autotune = tf.data.AUTOTUNE

    aug_layer = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.02),
            tf.keras.layers.RandomZoom(0.1),
        ]
    )

    def _build_dataset(
        x_paths: np.ndarray,
        y_targets: np.ndarray,
        shuffle: bool,
        augment: bool,
    ) -> tf.data.Dataset:
        ds = tf.data.Dataset.from_tensor_slices((x_paths, y_targets))

        def _map_fn(
            path: tf.Tensor,
            target: tf.Tensor,
        ) -> Tuple[tf.Tensor, tf.Tensor]:
            image = load_and_preprocess_image(path, img_height, img_width)
            if augment:
                image = aug_layer(image, training=True)
            return image, target

        ds = ds.map(_map_fn, num_parallel_calls=autotune)
        ds = ds.cache()

        if shuffle:
            buffer_size = max(len(x_paths), batch_size * 4)
            ds = ds.shuffle(
                buffer_size=buffer_size,
                reshuffle_each_iteration=True,
            )

        ds = ds.batch(batch_size)
        ds = ds.prefetch(autotune)
        return ds

    train_ds = _build_dataset(
        splits.x_train,
        splits.y_train,
        shuffle=True,
        augment=True,
    )
    val_ds = _build_dataset(
        splits.x_val,
        splits.y_val,
        shuffle=False,
        augment=False,
    )
    test_ds = _build_dataset(
        splits.x_test,
        splits.y_test,
        shuffle=False,
        augment=False,
    )

    steps_per_epoch = int(np.ceil(len(splits.x_train) / batch_size))
    val_steps = int(np.ceil(len(splits.x_val) / batch_size))
    test_steps = int(np.ceil(len(splits.x_test) / batch_size))

    print(
        "[INFO] Dataset steps – "
        f"train: {steps_per_epoch}, val: {val_steps}, test: {test_steps}"
    )

    return TfDatasets(
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        steps_per_epoch=steps_per_epoch,
        val_steps=val_steps,
        test_steps=test_steps,
    )


def inspect_one_batch(train_dataset: tf.data.Dataset) -> None:
    """Inspect shapes and statistics from a single training batch.

    This utility retrieves the first batch from ``train_dataset`` and
    prints:

    * Image tensor shape.
    * Target tensor shape.
    * Mean and standard deviation of the normalised targets.

    Args:
        train_dataset: ``tf.data.Dataset`` that yields (image, target)
            batches.
    """
    for images, targets in train_dataset.take(1):
        sample_images = images
        sample_targets = targets
        break
    else:
        print("[WARN] Training dataset appears to be empty.")
        return

    print("[INFO] Sample batch shapes:")
    print(f"  images:  {sample_images.shape}")
    print(f"  targets: {sample_targets.shape}")

    target_mean = float(tf.reduce_mean(sample_targets).numpy())
    target_std = float(tf.math.reduce_std(sample_targets).numpy())

    print("\n[INFO] Sample target statistics (normalised log-prices):")
    print(f"  mean: {target_mean:.4f}")
    print(f"  std:  {target_std:.4f}")


# ---------------------------------------------------------------------------
# Summaries and export helpers
# ---------------------------------------------------------------------------


def summarise_dataframe_stages(
    df_meta_raw: pd.DataFrame,
    df_with_images: pd.DataFrame,
    df_clean: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise row counts across the key preparation stages.

    Args:
        df_meta_raw: Raw metadata DataFrame loaded from CSV.
        df_with_images: DataFrame after attaching image paths.
        df_clean: Final cleaned DataFrame used for splitting.

    Returns:
        DataFrame with one row per stage and the corresponding row count.
    """
    stages = [
        "raw_metadata",
        "with_images",
        "cleaned_for_cv",
    ]
    counts = [
        len(df_meta_raw),
        len(df_with_images),
        len(df_clean),
    ]
    return pd.DataFrame({"stage": stages, "num_rows": counts})


def summarise_splits(splits: SplitArrays) -> pd.DataFrame:
    """Summarise sample counts for each in-memory split.

    Args:
        splits: SplitArrays instance from :func:`create_splits`.

    Returns:
        DataFrame with counts for train / val / test splits.
    """
    split_names = ["train", "val", "test"]
    counts = [
        len(splits.x_train),
        len(splits.x_val),
        len(splits.x_test),
    ]
    return pd.DataFrame({"split": split_names, "num_samples": counts})


def build_price_lookup(df_clean: pd.DataFrame) -> Dict[str, float]:
    """Create a lookup mapping from image path to ``price_usd``.

    Args:
        df_clean: Cleaned DataFrame with ``image_path`` and ``price_usd``.

    Returns:
        Dictionary mapping full image path to price in USD.
    """
    if not {"image_path", "price_usd"}.issubset(df_clean.columns):
        raise KeyError(
            "df_clean must contain 'image_path' and 'price_usd' columns."
        )
    return dict(zip(df_clean["image_path"], df_clean["price_usd"]))


def prepare_split_dir(processed_dir: str, split_name: str) -> str:
    """Prepare (reset) the directory for a given split.

    This function removes any existing directory for the split and
    recreates the ``images`` subdirectory.

    Args:
        processed_dir: Root directory for processed CV data.
        split_name: Name of the split (for example, ``\"train\"``).

    Returns:
        Path to the split directory.
    """
    split_dir = os.path.join(processed_dir, split_name)

    if os.path.isdir(split_dir):
        shutil.rmtree(split_dir)
        print(f"[INFO] Existing split directory removed: {split_dir}")

    os.makedirs(split_dir, exist_ok=True)
    os.makedirs(os.path.join(split_dir, "images"), exist_ok=True)
    return split_dir


def export_split(
    x_paths: Sequence[str],
    split_name: str,
    price_lookup: Dict[str, float],
    processed_dir: str,
) -> None:
    """Export a dataset split into ``images/`` + ``labels.csv`` format.

    Args:
        x_paths: Paths to the original images included in this split.
        split_name: Name of the split (train / val / test).
        price_lookup: Mapping from image path to true ``price_usd``.
        processed_dir: Root directory for processed CV data.
    """
    split_dir = prepare_split_dir(processed_dir, split_name)
    images_dir = os.path.join(split_dir, "images")

    filenames: List[str] = []
    labels: List[float] = []

    for src_path in x_paths:
        src_path_str = str(src_path)
        if src_path_str not in price_lookup:
            continue

        filename = os.path.basename(src_path_str)
        dst_path = os.path.join(images_dir, filename)

        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copy2(src_path_str, dst_path)

        filenames.append(filename)
        labels.append(price_lookup[src_path_str])

    labels_df = pd.DataFrame(
        {
            "filename": filenames,
            "price_usd": labels,
        }
    )
    labels_path = os.path.join(split_dir, "labels.csv")
    labels_df.to_csv(labels_path, index=False)

    print(
        f"[INFO] Exported {split_name}: {len(filenames)} images → {images_dir}"
    )
    print(f"[INFO] Labels (price_usd) written to: {labels_path}")


def summarise_exported_splits(base_dir: str) -> pd.DataFrame:
    """Summarise counts of exported images and labels per split.

    Args:
        base_dir: Base directory containing ``train``, ``val``, and
            ``test`` subdirectories.

    Returns:
        DataFrame with columns ``split``, ``num_images_on_disk``, and
        ``num_labels_in_csv``.
    """
    split_names = ["train", "val", "test"]
    split_list: List[str] = []
    image_counts: List[int] = []
    label_counts: List[int] = []

    for split_name in split_names:
        split_dir = os.path.join(base_dir, split_name)
        images_dir = os.path.join(split_dir, "images")
        labels_path = os.path.join(split_dir, "labels.csv")

        num_images = 0
        if os.path.isdir(images_dir):
            num_images = sum(
                1
                for name in os.listdir(images_dir)
                if name.lower().endswith((".jpg", ".jpeg", ".png"))
            )

        num_labels = 0
        if os.path.isfile(labels_path):
            labels_df = pd.read_csv(labels_path)
            num_labels = len(labels_df)

        split_list.append(split_name)
        image_counts.append(num_images)
        label_counts.append(num_labels)

    summary_df = pd.DataFrame(
        {
            "split": split_list,
            "num_images_on_disk": image_counts,
            "num_labels_in_csv": label_counts,
        }
    )
    return summary_df


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_full_cv_data_pipeline(
    raw_zip_path: str = RAW_ZIP_PATH,
    extract_dir: str = EXTRACT_DIR,
    processed_dir: str = PROCESSED_DIR,
) -> Dict[str, object]:
    """Run the full CV data preparation and export pipeline.

    This function is intended to be called from a notebook or as a
    one-off script entry point. It:

    * Unzips the raw archive (if present).
    * Loads metadata and attaches image paths.
    * Cleans the data for CV modelling.
    * Splits into train / val / test with target normalisation.
    * Builds TensorFlow datasets.
    * Exports images and labels for each split to disk.
    * Returns key artefacts for interactive analysis.

    Args:
        raw_zip_path: Path to the Kaggle archive to unzip.
        extract_dir: Directory where the archive will be extracted.
        processed_dir: Root directory where processed splits are written.

    Returns:
        Dictionary of artefacts including DataFrames, split arrays,
        datasets, and summary tables.
    """
    unzip_raw_data(raw_zip_path=raw_zip_path, extract_dir=extract_dir)

    base_dir = extract_dir
    csv_path = os.path.join(base_dir, CSV_FILENAME)
    image_root = os.path.join(base_dir, IMAGE_SUBDIR)

    print(f"[INFO] BASE_DIR:   {base_dir}")
    print(f"[INFO] CSV_PATH:   {csv_path}")
    print(f"[INFO] IMAGE_ROOT: {image_root}")

    df_meta_raw = pd.read_csv(csv_path)
    df_with_images = load_metadata(csv_path=csv_path, image_root=image_root)
    df_clean = clean_and_filter(df_with_images)

    splits = create_splits(df_clean)
    datasets = make_tf_datasets(splits)

    stage_summary = summarise_dataframe_stages(
        df_meta_raw=df_meta_raw,
        df_with_images=df_with_images,
        df_clean=df_clean,
    )
    split_summary = summarise_splits(splits)

    os.makedirs(processed_dir, exist_ok=True)
    price_lookup = build_price_lookup(df_clean)

    export_split(
        x_paths=splits.x_train,
        split_name="train",
        price_lookup=price_lookup,
        processed_dir=processed_dir,
    )
    export_split(
        x_paths=splits.x_val,
        split_name="val",
        price_lookup=price_lookup,
        processed_dir=processed_dir,
    )
    export_split(
        x_paths=splits.x_test,
        split_name="test",
        price_lookup=price_lookup,
        processed_dir=processed_dir,
    )

    export_summary = summarise_exported_splits(processed_dir)

    return {
        "df_meta_raw": df_meta_raw,
        "df_with_images": df_with_images,
        "df_clean": df_clean,
        "splits": splits,
        "datasets": datasets,
        "stage_summary": stage_summary,
        "split_summary": split_summary,
        "export_summary": export_summary,
    }


if __name__ == "__main__":
    artefacts = run_full_cv_data_pipeline()
    print("[INFO] CV data pipeline completed.")
    print(artefacts["export_summary"])
