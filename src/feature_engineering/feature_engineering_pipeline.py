"""End-to-end feature engineering pipeline for USA real estate pricing.

This module performs the following high-level steps:

    1) Load the cleaned real-estate dataset from CSV.
    2) Drop intermediate cleaning columns that must not enter modelling.
    3) Engineer leakage-free features (no usage of the target).
    4) Evaluate the statistical impact of engineered features.
    5) Build a numeric feature matrix for modelling.
    6) Run univariate feature screening (F-test + mutual information).
    7) Decide retention between engineered features and their parent columns.
    8) Build the final modelling feature matrix, handling one-hot and
       redundant unit conversions.
    9) Export final datasets (X, y, combined) in both Parquet and CSV.

The pipeline is intended for regression modelling with ``price`` as the
supervised target and assumes that the input dataset has already passed
through a robust cleaning pipeline.
"""

from __future__ import annotations

import os
import warnings
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

TARGET_COL: str = "price"
RANDOM_STATE: int = 42
MAX_SAMPLE_FOR_STATS: int = 200_000

CLEANED_DATA_PATH: str = (
    "/content/drive/MyDrive/Colab Notebooks/DOAA/"
    "data_processed/usa_real_estate_clean.csv"
)


def _safe_div(n: pd.Series, d: pd.Series) -> pd.Series:
    """Perform element-wise safe division between two Series.

    Zeros in the denominator are replaced with NaN before division to avoid
    division-by-zero warnings and infinite values.

    Args:
        n: Numerator series.
        d: Denominator series.

    Returns:
        A pandas Series containing the result of the safe division.
    """
    d_safe = d.replace(0, np.nan)
    return n / d_safe


def engineer_features_minimal(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """Engineer leakage-free features for the real-estate dataset.

    This function creates a minimal, interpretable set of engineered features
    from the raw columns. It does not use the target column in any computation
    to strictly avoid target leakage. The function also returns a parent-map
    dictionary that records which raw columns were used to derive each
    engineered feature, supporting downstream transparency and governance.

    Args:
        df: Input cleaned DataFrame containing at least the core real-estate
            features (e.g., ``bed``, ``bath``, ``house_size``, ``acre_lot``,
            ``city``, ``state``, ``status``) where available.
        target_col: Name of the target column. The target is preserved in the
            returned DataFrame but never used to engineer features.

    Returns:
        A tuple of:
            df_fe: DataFrame with engineered features added on top of the
                original columns.
            parent_map: Mapping from engineered feature names to a list of
                original parent columns used to derive them.
    """
    del target_col  # explicit to show we do not use it for leakage

    df_fe = df.copy()
    parent_map: Dict[str, List[str]] = {}

    # --- Basic presence flags ---
    has_h = "house_size" in df_fe.columns
    has_bed = "bed" in df_fe.columns
    has_bath = "bath" in df_fe.columns
    has_acre = "acre_lot" in df_fe.columns

    # 1) Total Rooms
    if has_bed and has_bath:
        df_fe["total_rooms"] = df_fe["bed"] + df_fe["bath"]
        parent_map["total_rooms"] = ["bed", "bath"]

    # 2) Ratio Features
    if has_bed and has_bath:
        df_fe["bed_per_bath"] = _safe_div(df_fe["bed"], df_fe["bath"])
        df_fe["bath_per_bed"] = _safe_div(df_fe["bath"], df_fe["bed"])
        parent_map["bed_per_bath"] = ["bed", "bath"]
        parent_map["bath_per_bed"] = ["bed", "bath"]

    if has_h and has_bed:
        df_fe["house_size_per_bed"] = _safe_div(
            df_fe["house_size"],
            df_fe["bed"],
        )
        parent_map["house_size_per_bed"] = ["house_size", "bed"]

    if has_h and has_bath:
        df_fe["house_size_per_bath"] = _safe_div(
            df_fe["house_size"],
            df_fe["bath"],
        )
        parent_map["house_size_per_bath"] = ["house_size", "bath"]

    if has_h and has_bed and has_bath:
        total_rooms = df_fe["bed"] + df_fe["bath"]
        df_fe["house_size_per_room"] = _safe_div(
            df_fe["house_size"],
            total_rooms,
        )
        parent_map["house_size_per_room"] = ["house_size", "bed", "bath"]

    # 3) Lot Features
    if has_acre:
        df_fe["lot_sqft"] = df_fe["acre_lot"] * 43_560
        parent_map["lot_sqft"] = ["acre_lot"]

        if has_h:
            df_fe["building_coverage_ratio"] = _safe_div(
                df_fe["house_size"],
                df_fe["lot_sqft"],
            )
            parent_map["building_coverage_ratio"] = ["house_size", "acre_lot"]

    # 4) Log1p transforms
    for col in ["house_size", "bed", "bath", "acre_lot"]:
        if col in df_fe.columns:
            df_fe[f"{col}_log1p"] = np.log1p(df_fe[col].clip(lower=0))
            parent_map[f"{col}_log1p"] = [col]

    # 5) City/State Density
    if "city" in df_fe.columns:
        df_fe["city_clean"] = (
            df_fe["city"].astype(str).str.lower().str.strip()
        )
        df_fe["city_listing_count"] = (
            df_fe.groupby("city_clean")["city_clean"].transform("count")
        )
        parent_map["city_listing_count"] = ["city"]

    if "state" in df_fe.columns:
        df_fe["state_clean"] = (
            df_fe["state"].astype(str).str.lower().str.strip()
        )
        df_fe["state_listing_count"] = (
            df_fe.groupby("state_clean")["state_clean"].transform("count")
        )
        parent_map["state_listing_count"] = ["state"]

    # 6) Status Encoding — sold / for_sale / ready_to_build
    if "status" in df_fe.columns:
        df_fe["status_clean"] = (
            df_fe["status"].astype(str).str.lower().str.strip()
        )

        df_fe["status_is_for_sale"] = (
            df_fe["status_clean"] == "for_sale"
        ).astype(int)
        df_fe["status_is_sold"] = (
            df_fe["status_clean"] == "sold"
        ).astype(int)
        df_fe["status_is_ready_to_build"] = (
            df_fe["status_clean"] == "ready_to_build"
        ).astype(int)

        parent_map["status_is_for_sale"] = ["status"]
        parent_map["status_is_sold"] = ["status"]
        parent_map["status_is_ready_to_build"] = ["status"]

    return df_fe, parent_map


def run_minimal_feature_engineering(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """Apply minimal feature engineering and log shape changes.

    This helper wraps :func:`engineer_features_minimal` to:

      * Apply feature engineering on the cleaned dataset.
      * Print pre- and post-engineering shapes.
      * Display the engineered features alongside their parent columns.

    Args:
        df: Cleaned input DataFrame prior to feature engineering.

    Returns:
        A tuple of:
            df_fe: DataFrame with engineered features added.
            parent_map: Mapping from engineered feature names to their
                originating raw columns.
    """
    df_fe, parent_map = engineer_features_minimal(df)

    print(f"[INFO] Shape before feature engineering: {df.shape}")
    print(f"[INFO] Shape after  feature engineering: {df_fe.shape}")
    print("\n[INFO] Engineered features and their parent columns:")
    for feature_name, parents in parent_map.items():
        print(f"  {feature_name}: {parents}")

    return df_fe, parent_map


def evaluate_engineered_features(
    df: pd.DataFrame,
    target_col: str,
    engineered_cols: List[str],
    max_sample: int = MAX_SAMPLE_FOR_STATS,
) -> pd.DataFrame:
    """Evaluate engineered features using correlation and mutual information.

    This function:
      * Cleans the dataset of infinite and missing values.
      * Optionally subsamples rows to a maximum size for efficiency.
      * Computes Pearson correlation (r and p-value) between each engineered
        feature and the target.
      * Computes mutual information (MI) between each engineered feature and
        the target in a single batch call.

    Args:
        df: Input DataFrame containing the target column and engineered
            feature columns.
        target_col: Name of the numeric target column for regression.
        engineered_cols: List of engineered feature names to evaluate.
        max_sample: Maximum number of rows to use for statistical analysis.
            If the cleaned dataset exceeds this size, a random sample is
            drawn using the global RANDOM_STATE.

    Returns:
        A DataFrame where each row corresponds to one engineered feature and
        includes:

            * ``feature``: Feature name.
            * ``pearson_r``: Pearson correlation coefficient with the target.
            * ``pearson_p``: Two-sided p-value for the Pearson test.
            * ``mutual_info``: Mutual information score with the target.

        The output is sorted in descending order of ``mutual_info``.
    """
    df_eval = df[[target_col] + engineered_cols].copy()
    df_eval = df_eval.replace([np.inf, -np.inf], np.nan).dropna()

    if len(df_eval) > max_sample:
        df_eval = df_eval.sample(max_sample, random_state=RANDOM_STATE)

    y = df_eval[target_col].values
    summary_rows: List[Dict[str, float]] = []

    for col in engineered_cols:
        x = df_eval[col].values
        r_val, p_val = stats.pearsonr(x, y)
        summary_rows.append(
            {
                "feature": col,
                "pearson_r": r_val,
                "pearson_p": p_val,
            }
        )

    X = df_eval[engineered_cols].values
    mi_vals = mutual_info_regression(
        X,
        y,
        random_state=RANDOM_STATE,
    )

    for row, mi in zip(summary_rows, mi_vals):
        row["mutual_info"] = mi

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values("mutual_info", ascending=False)

    return summary_df


def build_feature_matrix(
    df: pd.DataFrame,
    target_col: str,
    engineered_cols: List[str],
) -> Tuple[pd.DataFrame, pd.Series]:
    """Construct the numeric feature matrix for modelling.

    This function consolidates raw numeric features and engineered features
    into a unified feature matrix suitable for statistical selection or
    downstream modelling. It excludes the target column from ``X`` and
    performs light cleaning on numeric fields (handling infinities and
    imputing missing values).

    Args:
        df: Input DataFrame containing raw and engineered features.
        target_col: Name of the supervised learning target variable.
        engineered_cols: List of engineered feature names that must be
            retained in the candidate matrix even if they are not naturally
            numeric columns within the dataset.

    Returns:
        A tuple containing:

            * X_candidates: A DataFrame of numeric candidate features.
            * y: The target Series aligned to ``X_candidates``.
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in num_cols:
        num_cols.remove(target_col)

    for col in engineered_cols:
        if col not in num_cols:
            num_cols.append(col)

    X_candidates = df[num_cols].copy()
    y = df[target_col].copy()

    X_candidates = X_candidates.replace([np.inf, -np.inf], np.nan)
    X_candidates = X_candidates.fillna(
        X_candidates.median(numeric_only=True),
    )

    return X_candidates, y


def run_univariate_feature_screening(
    X: pd.DataFrame,
    y: pd.Series,
    max_sample: int = MAX_SAMPLE_FOR_STATS,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Run univariate feature screening using F-test and mutual information.

    This function:
      * Optionally subsamples the data for speed.
      * Standardises features for F-test stability.
      * Computes F-statistics and p-values via ``f_regression``.
      * Computes mutual information scores via ``mutual_info_regression``.
      * Returns a ranking of features sorted by mutual information.

    Args:
        X: Candidate numeric feature matrix.
        y: Target series aligned with ``X``.
        max_sample: Maximum number of rows to use for screening. If the
            dataset exceeds this size, a sample is drawn using
            ``train_size=max_sample``.
        random_state: Random seed for reproducible sampling.

    Returns:
        A DataFrame where each row represents a feature and includes:

            * ``feature``: Feature name.
            * ``F_stat``: F-statistic from the linear regression F-test.
            * ``F_pvalue``: Corresponding p-value for the F-test.
            * ``mutual_info``: Mutual information between the feature and
              the target.

        The DataFrame is sorted in descending order of ``mutual_info``.
    """
    if len(X) > max_sample:
        X_fs, _, y_fs, _ = train_test_split(
            X,
            y,
            train_size=max_sample,
            random_state=random_state,
        )
    else:
        X_fs, y_fs = X.copy(), y.copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_fs)

    F_vals, p_vals = f_regression(X_scaled, y_fs)

    mi_vals_all = mutual_info_regression(
        X_fs.values,
        y_fs.values,
        random_state=random_state,
    )

    fs_summary = pd.DataFrame(
        {
            "feature": X_fs.columns,
            "F_stat": F_vals,
            "F_pvalue": p_vals,
            "mutual_info": mi_vals_all,
        }
    ).sort_values("mutual_info", ascending=False)

    return fs_summary


def decide_feature_retention(
    fs_table: pd.DataFrame,
    parent_map: Dict[str, List[str]],
) -> Tuple[List[str], List[str]]:
    """Resolve which features to keep, comparing engineered vs parent features.

    This function uses mutual information scores from ``fs_table`` and the
    lineage information in ``parent_map`` to decide whether to keep an
    engineered feature, its parents, or both. The core heuristic is:

        * If an engineered feature's mutual information is greater than or
          equal to all of its parents' mutual information scores, the
          engineered feature is preferred and the parents are dropped.
        * Otherwise, the parents are preferred and the engineered feature is
          dropped.

    Features not present in ``parent_map`` are left untouched and follow the
    feature selection table as-is.

    Args:
        fs_table: Feature selection summary with at least the following
            columns:

                * ``feature``
                * ``mutual_info``

            F-statistics and p-values may also be present but are not used
            in this decision logic.
        parent_map: Mapping from engineered feature name to a list of its
            parent raw column names.

    Returns:
        A tuple of:

            * keep_features: Sorted list of feature names to retain.
            * drop_features: Sorted list of feature names to drop.
    """
    mi_lookup = fs_table.set_index("feature")["mutual_info"].to_dict()
    all_features = set(fs_table["feature"].tolist())

    drop_features: List[str] = []
    prefer_engineered: List[str] = []

    for eng_feat, parents in parent_map.items():
        if eng_feat not in all_features:
            continue

        parent_in_fs = [parent for parent in parents if parent in all_features]
        if not parent_in_fs:
            continue

        eng_mi = mi_lookup.get(eng_feat, 0.0)
        parent_mis = [mi_lookup.get(parent, 0.0) for parent in parent_in_fs]

        if all(eng_mi >= mi for mi in parent_mis):
            drop_features.extend(parent_in_fs)
            prefer_engineered.append(eng_feat)
        else:
            drop_features.append(eng_feat)

    keep_features = sorted(list(all_features.difference(drop_features)))
    drop_features = sorted(list(set(drop_features)))

    return keep_features, drop_features


def build_final_feature_matrix(
    X_candidates: pd.DataFrame,
    y: pd.Series,
    keep_features: List[str],
) -> Tuple[pd.DataFrame, pd.Series]:
    """Construct the final feature matrix for modelling.

    This function:

      * Filters the candidate matrix down to the selected features.
      * Handles one-hot-encoded status columns by dropping one category to
        avoid the dummy-variable trap.
      * Drops redundant unit-conversion features (e.g., ``lot_sqft`` when
        ``acre_lot`` and/or ``acre_lot_log1p`` are present).

    Args:
        X_candidates: Candidate feature matrix prior to final pruning.
        y: Target series aligned with ``X_candidates``.
        keep_features: List of feature names that passed the retention
            decision step.

    Returns:
        A tuple of:

            * X_final: Final feature matrix for modelling.
            * y_final: Target series (unchanged, but aligned to ``X_final``).
    """
    X_final = X_candidates[keep_features].copy()
    y_final = y.copy()

    status_cols = [
        col for col in X_final.columns if col.startswith("status_is_")
    ]

    if len(status_cols) > 1:
        status_cols_sorted = sorted(status_cols)
        col_to_drop = status_cols_sorted[0]
        print(
            "[INFO] Dropping one-hot category to avoid dummy trap: "
            f"{col_to_drop}",
        )
        X_final = X_final.drop(columns=[col_to_drop])

    redundant_groups = {
        "acre_lot": ["lot_sqft"],
    }

    for base, redundant_list in redundant_groups.items():
        log_col = f"{base}_log1p"

        for redundant_col in redundant_list:
            if redundant_col in X_final.columns:
                if base in X_final.columns or log_col in X_final.columns:
                    print(
                        "[INFO] Dropping redundant unit-converted feature: "
                        f"{redundant_col}",
                    )
                    X_final = X_final.drop(columns=[redundant_col])

    print(f"[INFO] Final X shape: {X_final.shape}")
    print(f"[INFO] y length: {len(y_final)}")

    return X_final, y_final


def export_final_datasets(
    X_final: pd.DataFrame,
    y_final: pd.Series,
    output_dir: str = "data_processed",
) -> Dict[str, str]:
    """Export the final modelling datasets in both Parquet and CSV formats.

    This function:

      * Ensures the export directory exists.
      * Applies a timestamp to filenames for reproducibility and versioning.
      * Saves:

            - ``X_final`` (features only)
            - ``y_final`` (target only)
            - ``combined`` (features + target)

        in both Parquet and CSV formats.

    Args:
        X_final: Final feature matrix post-selection and redundancy removal.
        y_final: Target series aligned with ``X_final``.
        output_dir: Directory where exported files will be written.

    Returns:
        A dictionary containing the filepaths of all exported files.
    """
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    X_parquet_path = os.path.join(output_dir, f"X_final_{timestamp}.parquet")
    y_parquet_path = os.path.join(output_dir, f"y_final_{timestamp}.parquet")
    combined_parquet_path = os.path.join(
        output_dir,
        f"final_dataset_{timestamp}.parquet",
    )

    X_csv_path = os.path.join(output_dir, f"X_final_{timestamp}.csv")
    y_csv_path = os.path.join(output_dir, f"y_final_{timestamp}.csv")
    combined_csv_path = os.path.join(
        output_dir,
        f"final_dataset_{timestamp}.csv",
    )

    df_final = X_final.copy()
    df_final[TARGET_COL] = y_final

    X_final.to_parquet(X_parquet_path, index=False)
    y_final.to_frame().to_parquet(y_parquet_path, index=False)
    df_final.to_parquet(combined_parquet_path, index=False)

    X_final.to_csv(X_csv_path, index=False)
    y_final.to_frame().to_csv(y_csv_path, index=False)
    df_final.to_csv(combined_csv_path, index=False)

    return {
        "X_parquet": X_parquet_path,
        "y_parquet": y_parquet_path,
        "combined_parquet": combined_parquet_path,
        "X_csv": X_csv_path,
        "y_csv": y_csv_path,
        "combined_csv": combined_csv_path,
    }


def load_and_trim_clean_dataset(
    path: str,
    target_col: str = TARGET_COL,
) -> pd.DataFrame:
    """Load the cleaned dataset and drop intermediate cleaning columns.

    Args:
        path: Filepath to the cleaned CSV dataset.
        target_col: Name of the target column expected in the dataset.

    Returns:
        A trimmed DataFrame, suitable as input to the feature engineering
        pipeline.
    """
    df_clean = pd.read_csv(path)

    assert target_col in df_clean.columns, (
        f"Target column '{target_col}' not found in the cleaned dataset."
    )

    cols_to_drop = [
        "brokered_by",
        "street",
        "zip_code",
        "prev_sold_date",
        "brokered_by_was_missing",
        "bed_was_missing",
        "acre_lot_was_missing",
        "street_was_missing",
        "city_was_missing",
        "state_was_missing",
        "zip_code_was_missing",
        "house_size_was_missing",
        "bath_was_missing",
        "prev_sold_date_was_missing",
        "price_log",
        "house_size_log",
        "bed_log",
        "bath_log",
        "acre_lot_log",
        "price__is_uni_outlier",
        "house_size__is_uni_outlier",
        "bed__is_uni_outlier",
        "bath__is_uni_outlier",
        "acre_lot__is_uni_outlier",
        "is_multivar_outlier",
        "price_peer_z",
        "is_price_context_outlier",
    ]

    existing_drop_cols = [col for col in cols_to_drop if col in df_clean.columns]
    df_clean = df_clean.drop(columns=existing_drop_cols)

    print(f"[INFO] Cleaned dataset loaded. Shape: {df_clean.shape}")

    return df_clean


def main() -> None:
    """Run the end-to-end feature engineering and export pipeline."""
    df_clean = load_and_trim_clean_dataset(CLEANED_DATA_PATH, TARGET_COL)

    df_fe, parent_map = run_minimal_feature_engineering(df_clean)

    engineered_cols = list(parent_map.keys())

    impact_df = evaluate_engineered_features(
        df_fe,
        target_col=TARGET_COL,
        engineered_cols=engineered_cols,
    )
    print("\n[INFO] Top 10 engineered features by mutual information:")
    print(impact_df.head(10))

    X_candidates, y = build_feature_matrix(
        df_fe,
        TARGET_COL,
        engineered_cols,
    )
    print(
        f"[INFO] Candidate feature matrix shape: {X_candidates.shape}",
    )

    fs_summary = run_univariate_feature_screening(
        X_candidates,
        y,
        max_sample=MAX_SAMPLE_FOR_STATS,
        random_state=RANDOM_STATE,
    )

    keep_feats, drop_feats = decide_feature_retention(
        fs_summary,
        parent_map,
    )

    print("\n[INFO] Features to DROP (raw or engineered):")
    print(drop_feats)
    print("\n[INFO] Features to KEEP (first 40):")
    print(keep_feats[:40])
    print(
        f"\n[INFO] Total kept: {len(keep_feats)}  |  "
        f"Total dropped: {len(drop_feats)}",
    )

    X_final, y_final = build_final_feature_matrix(
        X_candidates,
        y,
        keep_features=keep_feats,
    )

    export_paths = export_final_datasets(
        X_final,
        y_final,
        output_dir="data_processed",
    )

    print("\n[INFO] Export complete. Files written:")
    for label, path in export_paths.items():
        print(f"  - {label}: {path}")


if __name__ == "__main__":
    main()
