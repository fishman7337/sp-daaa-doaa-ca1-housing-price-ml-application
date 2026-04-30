"""NLP data preparation pipeline for DOAA CA1 (housing descriptions).

This module implements the end-to-end NLP data pipeline for the
housing price project:

* Loading the housing dataset.
* Cleaning and normalising the description text.
* Explicitly removing price mentions from descriptions to prevent
  target leakage.
* Splitting the data into train/validation/test sets.
* Tokenising and padding sequences for RNN models.
* Exporting all key artefacts (cleaned CSV, sequences, targets,
  vocabulary) to disk for downstream modelling.
"""

from __future__ import annotations

import os
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------

RAW_PATH: str = (
    "/content/drive/MyDrive/Colab Notebooks/DOAA/"
    "nlp_data_raw/austinHousingData.csv"
)

PROCESSED_DIR: str = (
    "/content/drive/MyDrive/Colab Notebooks/DOAA/"
    "nlp_data_processed"
)

TEXT_COLUMN_RAW: str = "description"
PRICE_COLUMN_RAW: str = "latestPrice"

RANDOM_STATE: int = 42

MAX_TOKENS: int = 20_000
MAX_SEQUENCE_LENGTH: int = 300

TRAIN_SIZE: float = 0.7
VAL_SIZE: float = 0.15
TEST_SIZE: float = 0.15

# Price-removal regexes for target leakage prevention.
CURRENCY_PATTERN = re.compile(
    r"(\$|usd|dollar[s]?)\s*\d[\d,]*(\.\d+)?",
    flags=re.IGNORECASE,
)
PURE_PRICE_PATTERN = re.compile(
    r"\b\d{5,}([\.,]\d+)?\b",  # large numbers that look like prices
    flags=re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TextSplits:
    """Container for raw text and target splits.

    Attributes:
        X_train_text: Training descriptions (cleaned).
        X_val_text: Validation descriptions (cleaned).
        X_test_text: Test descriptions (cleaned).
        y_train: Training targets (prices in USD).
        y_val: Validation targets.
        y_test: Test targets.
    """

    X_train_text: np.ndarray
    X_val_text: np.ndarray
    X_test_text: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray


@dataclass
class SequenceSplits:
    """Container for tokenised and padded sequence splits.

    Attributes:
        X_train_seq: Training sequences (int-encoded, padded).
        X_val_seq: Validation sequences.
        X_test_seq: Test sequences.
        y_train: Training targets (matching sequences).
        y_val: Validation targets.
        y_test: Test targets.
    """

    X_train_seq: np.ndarray
    X_val_seq: np.ndarray
    X_test_seq: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def configure_random_seeds(seed: int = RANDOM_STATE) -> None:
    """Configure random seeds for reproducibility.

    Args:
        seed: Integer random seed to set for Python, NumPy, and TensorFlow.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ---------------------------------------------------------------------------
# Loading and inspection
# ---------------------------------------------------------------------------


def load_raw_dataset(path: str) -> pd.DataFrame:
    """Load the raw housing dataset from CSV.

    This function expects the CSV to contain at least the following
    columns:

    * ``latestPrice`` – numeric price in USD.
    * ``description`` – free-text property description.

    Args:
        path: Filesystem path to the raw CSV file.

    Returns:
        Loaded DataFrame.

    Raises:
        FileNotFoundError: If the file cannot be found at ``path``.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Raw data not found at: {path}")

    df = pd.read_csv(path)

    print("[INFO] Raw dataset loaded successfully.")
    print("[INFO] Shape:", df.shape)
    print("[INFO] Columns:", df.columns.tolist())

    return df


def plot_description_length_distribution(
    df: pd.DataFrame,
    text_col: str = TEXT_COLUMN_RAW,
    max_length: int = 1_000,
) -> None:
    """Plot a histogram of description lengths (in characters).

    Args:
        df: DataFrame containing the raw text column.
        text_col: Name of the column containing the descriptions.
        max_length: Maximum length to display on the x-axis to avoid
            extremely long tails swamping the distribution.
    """
    if text_col not in df.columns:
        raise KeyError(f"Column {text_col!r} not in DataFrame.")

    lengths = df[text_col].fillna("").astype(str).str.len()

    plt.figure(figsize=(8, 5))
    clipped = lengths.clip(upper=max_length)
    plt.hist(clipped, bins=50, edgecolor="black", alpha=0.7)
    plt.xlabel("Description length (characters, clipped)")
    plt.ylabel("Frequency")
    plt.title("Distribution of description lengths")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

    print("[INFO] Description length statistics:")
    print("  mean:", float(lengths.mean()))
    print("  std: ", float(lengths.std()))
    print("  min: ", int(lengths.min()))
    print("  max: ", int(lengths.max()))


# ---------------------------------------------------------------------------
# Text cleaning and leakage prevention
# ---------------------------------------------------------------------------


def _remove_price_mentions(text: str) -> str:
    """Remove explicit price mentions from description text.

    This function targets patterns that look like prices, for example:

        * ``$350000``, ``$ 350,000``, ``usd 450000``, ``450000 dollars``.
        * Large numeric values without symbols (e.g. ``650000``), which
          often represent prices.

    Args:
        text: Raw description text.

    Returns:
        Text with likely price mentions removed.
    """
    text_no_currency = CURRENCY_PATTERN.sub(" ", text)
    text_no_pure_price = PURE_PRICE_PATTERN.sub(" ", text_no_currency)
    return text_no_pure_price


def clean_description_text(text: str) -> str:
    """Clean and normalise a single description string.

    Cleaning steps:
        * Convert to lowercase.
        * Remove price mentions to avoid target leakage.
        * Remove HTML tags.
        * Replace non-letter characters with spaces.
        * Collapse multiple spaces.
        * Strip leading/trailing whitespace.

    Args:
        text: Raw description string (may be NaN or empty).

    Returns:
        Cleaned description string suitable for tokenisation.
    """
    if not isinstance(text, str):
        text = "" if text is None else str(text)

    text = text.lower()
    text = _remove_price_mentions(text)

    # Remove simple HTML tags.
    text = re.sub(r"<[^>]+>", " ", text)

    # Replace anything that is not a letter or basic punctuation with space.
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    # Collapse multiple spaces.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def build_nlp_dataframe(
    df_raw: pd.DataFrame,
    text_col: str = TEXT_COLUMN_RAW,
    price_col: str = PRICE_COLUMN_RAW,
) -> pd.DataFrame:
    """Build the cleaned NLP DataFrame for modelling.

    The resulting DataFrame will contain:

        * ``description_raw`` – the original description text.
        * ``description_clean`` – cleaned, leakage-free text.
        * ``price_usd`` – numeric house price in USD.

    Rows with missing descriptions or missing / non-positive prices are
    dropped.

    Args:
        df_raw: Raw dataset loaded from CSV.
        text_col: Name of the raw description column.
        price_col: Name of the raw price column.

    Returns:
        Cleaned NLP DataFrame with the three columns described above.
    """
    if text_col not in df_raw.columns:
        raise KeyError(f"Text column {text_col!r} not found in DataFrame.")
    if price_col not in df_raw.columns:
        raise KeyError(f"Price column {price_col!r} not found in DataFrame.")

    df = df_raw[[text_col, price_col]].copy()

    df = df.rename(
        columns={
            text_col: "description_raw",
            price_col: "price_usd",
        }
    )

    df = df.dropna(subset=["description_raw", "price_usd"])
    df["price_usd"] = pd.to_numeric(df["price_usd"], errors="coerce")
    df = df.dropna(subset=["price_usd"])
    df = df[df["price_usd"] > 0]

    df["description_clean"] = df["description_raw"].apply(clean_description_text)

    df = df.reset_index(drop=True)

    print("[INFO] NLP DataFrame built.")
    print("[INFO] Shape:", df.shape)
    print(df[["description_clean", "price_usd"]].head())

    return df


# ---------------------------------------------------------------------------
# Train/validation/test splitting
# ---------------------------------------------------------------------------


def create_text_splits(
    df_nlp: pd.DataFrame,
    text_col: str = "description_clean",
    target_col: str = "price_usd",
    train_size: float = TRAIN_SIZE,
    val_size: float = VAL_SIZE,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> TextSplits:
    """Create train/validation/test splits for text and targets.

    The three proportions must sum to 1.0 within a small tolerance.

    Args:
        df_nlp: Cleaned NLP DataFrame.
        text_col: Column containing cleaned text descriptions.
        target_col: Column containing numeric price targets.
        train_size: Proportion allocated to the training split.
        val_size: Proportion allocated to the validation split.
        test_size: Proportion allocated to the test split.
        random_state: Seed for reproducible splits.

    Returns:
        TextSplits dataclass with arrays for each split.

    Raises:
        ValueError: If the split proportions do not sum to 1.0.
    """
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("train_size + val_size + test_size must sum to 1.0.")

    if text_col not in df_nlp.columns:
        raise KeyError(f"Text column {text_col!r} not found in df_nlp.")
    if target_col not in df_nlp.columns:
        raise KeyError(f"Target column {target_col!r} not found in df_nlp.")

    X_all = df_nlp[text_col].astype(str).to_numpy()
    y_all = df_nlp[target_col].to_numpy(dtype=np.float32)

    X_temp, X_test, y_temp, y_test = train_test_split(
        X_all,
        y_all,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )

    relative_val_size = val_size / (train_size + val_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=relative_val_size,
        random_state=random_state,
        shuffle=True,
    )

    print(
        "[INFO] Text splits created – "
        f"train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}"
    )

    return TextSplits(
        X_train_text=X_train,
        X_val_text=X_val,
        X_test_text=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )


# ---------------------------------------------------------------------------
# Tokenisation and sequence building
# ---------------------------------------------------------------------------


def build_text_vectorizer(
    max_tokens: int = MAX_TOKENS,
    output_sequence_length: int = MAX_SEQUENCE_LENGTH,
) -> tf.keras.layers.TextVectorization:
    """Create an unadapted TextVectorization layer.

    Args:
        max_tokens: Maximum vocabulary size.
        output_sequence_length: Fixed length to pad / truncate sequences.

    Returns:
        A TextVectorization layer ready to be adapted on training text.
    """
    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=max_tokens,
        output_mode="int",
        output_sequence_length=output_sequence_length,
        standardize=None,  # we already standardise in clean_description_text
    )
    return vectorizer


def adapt_vectorizer(
    vectorizer: tf.keras.layers.TextVectorization,
    X_train_text: np.ndarray,
) -> None:
    """Adapt a TextVectorization layer on training text.

    Args:
        vectorizer: TextVectorization layer to be adapted.
        X_train_text: Array of cleaned training descriptions.
    """
    print("[INFO] Adapting TextVectorization layer on training text...")
    vectorizer.adapt(X_train_text)
    print("[INFO] TextVectorization adaptation complete.")
    print("[INFO] Vocabulary size:", len(vectorizer.get_vocabulary()))


def sequences_from_vectorizer(
    vectorizer: tf.keras.layers.TextVectorization,
    splits: TextSplits,
) -> SequenceSplits:
    """Convert raw text splits into integer sequences using a vectoriser.

    Args:
        vectorizer: Adapted TextVectorization layer.
        splits: TextSplits dataclass containing train/val/test text and
            targets.

    Returns:
        SequenceSplits dataclass with NumPy arrays of sequences and
        matching target arrays.
    """
    def _to_seq(text_array: np.ndarray) -> np.ndarray:
        """Vectorise and convert to NumPy array."""
        ds = tf.constant(text_array)
        seq = vectorizer(ds)
        return seq.numpy()

    X_train_seq = _to_seq(splits.X_train_text)
    X_val_seq = _to_seq(splits.X_val_text)
    X_test_seq = _to_seq(splits.X_test_text)

    print("[INFO] Sequence shapes:")
    print("  X_train_seq:", X_train_seq.shape)
    print("  X_val_seq:  ", X_val_seq.shape)
    print("  X_test_seq: ", X_test_seq.shape)

    return SequenceSplits(
        X_train_seq=X_train_seq,
        X_val_seq=X_val_seq,
        X_test_seq=X_test_seq,
        y_train=splits.y_train.astype(np.float32),
        y_val=splits.y_val.astype(np.float32),
        y_test=splits.y_test.astype(np.float32),
    )


# ---------------------------------------------------------------------------
# Export utilities
# ---------------------------------------------------------------------------


def export_nlp_artifacts(
    df: pd.DataFrame,
    processed_dir: str,
    X_train_seq: np.ndarray,
    X_val_seq: np.ndarray,
    X_test_seq: np.ndarray,
    y_train_array: np.ndarray,
    y_val_array: np.ndarray,
    y_test_array: np.ndarray,
    vectorizer: tf.keras.layers.TextVectorization | None = None,
) -> None:
    """Export cleaned NLP artefacts to disk.

    This function writes:

        * ``nlp_dataset.csv`` – cleaned DataFrame with text and price.
        * ``nlp_sequences.npz`` – NumPy arrays for sequences and targets.
        * ``vocabulary.txt`` – one token per line (if vectoriser provided).

    Args:
        df: Cleaned NLP DataFrame.
        processed_dir: Root directory where artefacts will be saved.
        X_train_seq: Training sequences array.
        X_val_seq: Validation sequences array.
        X_test_seq: Test sequences array.
        y_train_array: Training targets.
        y_val_array: Validation targets.
        y_test_array: Test targets.
        vectorizer: Optional adapted TextVectorization layer whose
            vocabulary will be saved as ``vocabulary.txt``.
    """
    os.makedirs(processed_dir, exist_ok=True)

    dataset_path = os.path.join(processed_dir, "nlp_dataset.csv")
    df.to_csv(dataset_path, index=False)
    print(f"[INFO] Saved cleaned NLP dataset → {dataset_path}")

    npz_path = os.path.join(processed_dir, "nlp_sequences.npz")
    np.savez_compressed(
        npz_path,
        X_train_seq=X_train_seq,
        X_val_seq=X_val_seq,
        X_test_seq=X_test_seq,
        y_train=y_train_array,
        y_val=y_val_array,
        y_test=y_test_array,
    )
    print(f"[INFO] Saved sequences and targets → {npz_path}")

    if vectorizer is not None:
        vocab = vectorizer.get_vocabulary()
        vocab_path = os.path.join(processed_dir, "vocabulary.txt")
        with open(vocab_path, "w", encoding="utf-8") as f:
            for token in vocab:
                f.write(f"{token}\n")
        print(f"[INFO] Saved vocabulary ({len(vocab)} tokens) → {vocab_path}")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_full_nlp_data_pipeline(
    raw_path: str = RAW_PATH,
    processed_dir: str = PROCESSED_DIR,
) -> Dict[str, object]:
    """Run the full NLP data preparation and sequence-building pipeline.

    Steps performed:
        1. Configure random seeds for reproducibility.
        2. Load the raw dataset from CSV.
        3. Build the cleaned NLP DataFrame with leakage-free text.
        4. Optionally inspect description length distribution.
        5. Split into train/validation/test text and targets.
        6. Build and adapt a TextVectorization layer.
        7. Convert text splits into integer sequences.
        8. Export cleaned DataFrame, sequences, targets, and vocabulary.

    Args:
        raw_path: Filesystem path to the raw housing CSV.
        processed_dir: Destination directory for processed artefacts.

    Returns:
        Dictionary containing the main artefacts:
            - "df_raw": raw DataFrame.
            - "df_nlp": cleaned NLP DataFrame.
            - "text_splits": TextSplits dataclass.
            - "vectorizer": adapted TextVectorization layer.
            - "sequence_splits": SequenceSplits dataclass.
    """
    configure_random_seeds(RANDOM_STATE)

    df_raw = load_raw_dataset(raw_path)

    # Optional: EDA on description length.
    plot_description_length_distribution(df_raw, text_col=TEXT_COLUMN_RAW)

    df_nlp = build_nlp_dataframe(
        df_raw=df_raw,
        text_col=TEXT_COLUMN_RAW,
        price_col=PRICE_COLUMN_RAW,
    )

    text_splits = create_text_splits(df_nlp)

    vectorizer = build_text_vectorizer(
        max_tokens=MAX_TOKENS,
        output_sequence_length=MAX_SEQUENCE_LENGTH,
    )
    adapt_vectorizer(vectorizer, text_splits.X_train_text)

    sequence_splits = sequences_from_vectorizer(vectorizer, text_splits)

    export_nlp_artifacts(
        df=df_nlp,
        processed_dir=processed_dir,
        X_train_seq=sequence_splits.X_train_seq,
        X_val_seq=sequence_splits.X_val_seq,
        X_test_seq=sequence_splits.X_test_seq,
        y_train_array=sequence_splits.y_train,
        y_val_array=sequence_splits.y_val,
        y_test_array=sequence_splits.y_test,
        vectorizer=vectorizer,
    )

    return {
        "df_raw": df_raw,
        "df_nlp": df_nlp,
        "text_splits": text_splits,
        "vectorizer": vectorizer,
        "sequence_splits": sequence_splits,
    }


if __name__ == "__main__":
    artefacts = run_full_nlp_data_pipeline()
    print("[INFO] NLP data pipeline completed.")
    print(
        "[INFO] Train/val/test sizes (seq):",
        len(artefacts["sequence_splits"].X_train_seq),
        len(artefacts["sequence_splits"].X_val_seq),
        len(artefacts["sequence_splits"].X_test_seq),
    )
