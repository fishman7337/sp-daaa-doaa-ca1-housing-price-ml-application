"""Data cleaning pipeline for USA real estate dataset.

This module implements a reusable, production-oriented cleaning pipeline for the
USA real estate dataset used in the DOAA project. It focuses on:

* Robust handling of the target column ``price`` (parsing, validation, and
  strict removal of rows with invalid prices).
* Structured imputation strategies for numeric, categorical, and datetime
  features.
* Optional univariate and multivariate outlier flagging and row-removal.
* PEP 8, PEP 257, and Google-style docstring compliance.
* Logging-based diagnostics instead of print statements (MLOps-friendly).

Typical usage example
---------------------

    import pandas as pd
    from data_cleaning_pipeline import CleaningConfig, run_cleaning_pipeline

    df_raw = pd.read_csv("usa_housing_raw.csv")
    config = CleaningConfig(
        remove_univariate_outliers=False,
        remove_multivar_outliers=False,
        cap_univariate_outliers=True,
    )
    df_clean, meta = run_cleaning_pipeline(df_raw, config=config)

"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import ExtraTreesRegressor, IsolationForest
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())

TARGET_COL = "price"


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class CleaningConfig:
    """Configuration for the real estate data cleaning pipeline.

    Attributes:
        remove_univariate_outliers: Whether to drop rows flagged as univariate
            outliers on selected numeric columns.
        remove_multivar_outliers: Whether to drop rows flagged as multivariate
            outliers by IsolationForest.
        cap_univariate_outliers: Whether to clip univariate outliers (IQR-based)
            instead of leaving them unchanged.
        iforest_contamination: Expected fraction of multivariate outliers for
            IsolationForest.

    """

    remove_univariate_outliers: bool = False
    remove_multivar_outliers: bool = False
    cap_univariate_outliers: bool = True
    iforest_contamination: float = 0.01


# =============================================================================
# Helper utilities
# =============================================================================


def drop_duplicates_safely(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop duplicate rows from a DataFrame.

    Args:
        df: Input DataFrame.

    Returns:
        A tuple of:
            * A new DataFrame with duplicate rows removed.
            * The number of duplicate rows removed.

    """
    n_before = len(df)
    df_dedup = df.drop_duplicates()
    n_removed = n_before - len(df_dedup)
    LOGGER.info("Dropped %d duplicate rows.", n_removed)
    return df_dedup, n_removed


def summarize_structure(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise structure, missingness, and cardinality for each column.

    Args:
        df: Input DataFrame.

    Returns:
        A DataFrame indexed by column name with the following columns:
            * dtype: String representation of the dtype.
            * n_missing: Number of missing values.
            * pct_missing: Percentage of missing values (0–100).
            * n_unique: Number of unique non-null values.

    """
    records: list[dict[str, Any]] = []
    n_rows = len(df)

    for col in df.columns:
        s = df[col]
        n_missing = int(s.isna().sum())
        pct_missing = float((n_missing / n_rows) * 100.0) if n_rows > 0 else 0.0
        n_unique = int(s.nunique(dropna=True))
        records.append(
            {
                "column": col,
                "dtype": str(s.dtype),
                "n_missing": n_missing,
                "pct_missing": pct_missing,
                "n_unique": n_unique,
            }
        )

    struct = pd.DataFrame.from_records(records).set_index("column")
    return struct


def detect_column_types(df: pd.DataFrame) -> dict[str, list[str]]:
    """Infer basic column types (numeric, categorical, datetime).

    Args:
        df: Input DataFrame.

    Returns:
        A dictionary with keys:
            * "numeric": List of numeric columns.
            * "categorical": List of categorical-like columns.
            * "datetime": List of datetime-like columns.

    """
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    datetime_cols = df.select_dtypes(
        include=["datetime64[ns]", "datetime64[ns, tz]"]
    ).columns.tolist()

    # Treat any non-numeric, non-datetime column as categorical.
    categorical_cols = [c for c in df.columns if c not in numeric_cols + datetime_cols]

    return {
        "numeric": numeric_cols,
        "categorical": categorical_cols,
        "datetime": datetime_cols,
    }


def choose_imputation_strategy(
    col_name: str,
    dtype: str,
    pct_missing: float,
    n_unique: int,
) -> str:
    """Select an imputation strategy for a single column.

    The heuristic is intentionally simple and deterministic to remain
    explainable and easy to maintain in a production setting.

    Args:
        col_name: Name of the column.
        dtype: String representation of the column dtype.
        pct_missing: Percentage of missing values (0–100).
        n_unique: Number of unique non-null values.

    Returns:
        A strategy string from:
            * "skip": Do not impute this column.
            * "median": Median imputation (numeric).
            * "iterative_et": IterativeImputer + ExtraTrees (numeric).
            * "knn": KNN imputation (numeric).
            * "mode": Mode imputation (categorical).
            * "unknown_token": Fill with "unknown".
            * "unknown_date": Treat date-like as categorical "unknown".
            * "drop_column": Drop column from the dataset.

    """
    col_lower = col_name.lower()

    # Special rule: target price is never imputed or dropped by strategy.
    if col_lower == TARGET_COL:
        return "skip"

    # Handle by dtype group.
    if "float" in dtype or "int" in dtype:
        if pct_missing == 0.0:
            return "skip"
        if pct_missing <= 5.0:
            return "median"
        if pct_missing <= 30.0:
            return "iterative_et"
        if pct_missing <= 50.0:
            return "knn"
        return "drop_column"

    if "datetime" in dtype:
        if pct_missing == 0.0:
            return "skip"
        if pct_missing <= 40.0:
            return "unknown_date"
        return "drop_column"

    # Categorical / object-like
    if pct_missing == 0.0:
        return "skip"
    if pct_missing <= 40.0:
        return "mode"
    return "unknown_token"


def fit_iterative_imputer_et(
    df: pd.DataFrame,
    numeric_cols: list[str],
    random_state: int = 42,
) -> IterativeImputer:
    """Fit an IterativeImputer with ExtraTreesRegressor on selected columns.

    Args:
        df: Input DataFrame containing the numeric columns.
        numeric_cols: List of numeric column names to impute.
        random_state: Random seed for reproducibility.

    Returns:
        A fitted IterativeImputer instance.

    """
    x = df[numeric_cols].copy()
    x = x.replace([np.inf, -np.inf], np.nan)
    estimator = ExtraTreesRegressor(
        n_estimators=50,
        max_depth=None,
        n_jobs=-1,
        random_state=random_state,
    )
    imputer = IterativeImputer(
        estimator=estimator,
        max_iter=10,
        random_state=random_state,
        sample_posterior=False,
        initial_strategy="median",
    )
    imputer.fit(x)
    return imputer


def apply_iterative_imputer_et(
    df: pd.DataFrame,
    numeric_cols: list[str],
    imputer: IterativeImputer,
) -> pd.DataFrame:
    """Apply a fitted IterativeImputer to selected numeric columns.

    Args:
        df: Input DataFrame.
        numeric_cols: List of numeric columns to impute.
        imputer: Fitted IterativeImputer.

    Returns:
        A new DataFrame with imputed numeric columns.

    """
    df_out = df.copy()
    x = df_out[numeric_cols].copy()
    x = x.replace([np.inf, -np.inf], np.nan)
    x_imputed = imputer.transform(x)
    df_out[numeric_cols] = x_imputed
    return df_out


def choose_univariate_method(series: pd.Series) -> str:
    """Choose a univariate outlier detection method based on distribution.

    Args:
        series: Numeric Series.

    Returns:
        One of "zscore", "mad", or "iqr".

    """
    x = pd.to_numeric(series, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan).dropna()
    if x.empty:
        return "iqr"

    skew_val = float(stats.skew(x))
    kurt_val = float(stats.kurtosis(x, fisher=False))

    if abs(skew_val) < 0.5 and abs(kurt_val - 3.0) < 1.0:
        return "zscore"
    if abs(skew_val) > 1.5 or kurt_val > 5.0:
        return "mad"
    return "iqr"


def flag_univariate_outliers(series: pd.Series, method: str = "iqr") -> pd.Series:
    """Flag univariate outliers in a Series.

    Args:
        series: Numeric Series.
        method: Outlier detection method: "zscore", "mad", or "iqr".

    Returns:
        A boolean Series where True indicates an outlier.

    """
    x = pd.to_numeric(series, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan)

    if method == "zscore":
        z = (x - x.mean()) / x.std(ddof=0)
        return z.abs() > 3.0

    if method == "mad":
        median = x.median()
        mad = (x - median).abs().median()
        if mad == 0 or np.isnan(mad):
            return pd.Series(False, index=series.index)
        modified_z = 0.6745 * (x - median) / mad
        return modified_z.abs() > 3.5

    # Default: IQR
    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    iqr = q3 - q1
    if pd.isna(iqr) or iqr == 0:
        return pd.Series(False, index=series.index)
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return (x < lower) | (x > upper)


def detect_multivariate_outliers_iforest(
    df: pd.DataFrame,
    numeric_cols: list[str],
    contamination: float = 0.01,
    random_state: int = 42,
) -> pd.Series:
    """Detect multivariate outliers using IsolationForest.

    Args:
        df: Input DataFrame.
        numeric_cols: List of numeric columns to consider.
        contamination: Expected fraction of outliers in the data.
        random_state: Random seed for reproducibility.

    Returns:
        A boolean Series where True indicates an outlier.

    """
    x = df[numeric_cols].copy()
    x = x.replace([np.inf, -np.inf], np.nan)
    x = x.fillna(x.median(numeric_only=True))

    iforest = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    preds = iforest.fit_predict(x)
    # In IsolationForest: -1 = outlier, 1 = inlier
    flags = preds == -1
    return pd.Series(flags, index=df.index)


# =============================================================================
# Core cleaning pipeline
# =============================================================================


def clean_dataset(
    df: pd.DataFrame,
    remove_univariate_outliers: bool = False,
    remove_multivar_outliers: bool = False,
    cap_univariate_outliers: bool = True,
    iforest_contamination: float = 0.01,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Clean the USA real estate dataset using advanced methods.

    The cleaning pipeline performs the following high-level steps:

        0) Drop duplicate rows.
        0.25) Clean the target column ``price`` into a numeric dtype.
        0.5) Drop rows with missing/invalid ``price``.
        1) Derive an imputation strategy per column.
        2) Drop columns marked as ``drop_column`` (except ``price``).
        3) Create missingness indicator flags.
        4) Normalise dtypes (dates, categoricals, numerics).
        5) Apply numeric imputation (median / Iterative + ExtraTrees / KNN).
        6) Apply categorical and date imputation (mode / "unknown").
        7) Detect and flag univariate and multivariate outliers on key
           numeric columns.
        8) Remove impossible/corrupted values and optionally drop outliers.
        9) Enforce discrete counts for ``bed`` and ``bath``.

    Args:
        df: Raw input dataset.
        remove_univariate_outliers: If True, drop rows flagged as univariate
            outliers on the selected numeric columns.
        remove_multivar_outliers: If True, drop rows flagged as multivariate
            outliers by IsolationForest.
        cap_univariate_outliers: If True, clip univariate outliers instead of
            leaving them unchanged.
        iforest_contamination: Expected fraction of multivariate outliers for
            IsolationForest.

    Returns:
        A tuple of:
            * cleaned: Cleaned DataFrame.
            * meta: Dictionary containing strategies, diagnostics, and summary
              statistics for the cleaning process.

    """
    meta: dict[str, dict[str, Any]] = {}

    # --- 0) Drop duplicates upfront ---
    df_dedup, n_dupes_removed = drop_duplicates_safely(df)
    meta["duplicates"] = {
        "removed": int(n_dupes_removed),
        "rows_before": int(len(df)),
        "rows_after": int(len(df_dedup)),
    }

    cleaned = df_dedup.copy()
    n_rows_before = len(cleaned)

    # --- 0.25) Clean target 'price' into a proper numeric column ---
    if TARGET_COL not in cleaned.columns:
        raise KeyError(f"Target column '{TARGET_COL}' not found in input dataset.")

    cleaned[TARGET_COL] = (
        cleaned[TARGET_COL]
        .astype("string")
        .str.strip()
        .str.replace(r"[^\d\.\-]", "", regex=True)
        .replace("", np.nan)
        .astype("float64")
    )

    # --- 0.5) Target enforcement: drop rows with missing price, keep column ---
    n_missing_price_initial = int(cleaned[TARGET_COL].isna().sum())
    if n_missing_price_initial > 0:
        LOGGER.info(
            "Dropping %d rows with missing %s after initial price cleaning.",
            n_missing_price_initial,
            TARGET_COL,
        )
        cleaned = cleaned.loc[cleaned[TARGET_COL].notna()].copy()
    meta["target_price_missing_rows_dropped_initial"] = {"count": n_missing_price_initial}

    # --- Type & missingness info (after dropping missing price) ---
    struct = summarize_structure(cleaned)
    col_types = detect_column_types(cleaned)
    numeric_cols = col_types["numeric"]
    categorical_cols = col_types["categorical"]
    datetime_cols = col_types["datetime"]

    # --- 1) Imputation plan (from struct) ---
    plan_rows: list[dict[str, Any]] = []
    for col, row in struct.iterrows():
        strategy = choose_imputation_strategy(
            col_name=col,
            dtype=row["dtype"],
            pct_missing=float(row["pct_missing"]),
            n_unique=int(row["n_unique"]),
        )
        plan_rows.append(
            {
                "column": col,
                "dtype": row["dtype"],
                "pct_missing": float(row["pct_missing"]),
                "n_missing": int(row["n_missing"]),
                "strategy": strategy,
            }
        )

    impute_plan = pd.DataFrame(plan_rows).set_index("column")
    meta["impute_plan"] = impute_plan.to_dict(orient="index")

    # --- 2) Drop columns marked as 'drop_column' (never drop TARGET_COL) ---
    drop_cols = impute_plan[impute_plan["strategy"] == "drop_column"].index.tolist()
    if TARGET_COL in drop_cols:
        drop_cols = [c for c in drop_cols if c != TARGET_COL]

    if drop_cols:
        LOGGER.info("Dropping %d columns with strategy 'drop_column'.", len(drop_cols))
        cleaned = cleaned.drop(columns=drop_cols)
        meta["dropped_columns"] = {
            "columns": drop_cols,
            "reason": "imputation_strategy=drop_column",
        }
        numeric_cols = [c for c in numeric_cols if c not in drop_cols]
        categorical_cols = [c for c in categorical_cols if c not in drop_cols]
        datetime_cols = [c for c in datetime_cols if c not in drop_cols]

    # --- 2.5) Date columns with 'unknown_date' → treat as categorical ---
    unknown_date_cols = impute_plan[impute_plan["strategy"] == "unknown_date"].index.tolist()
    unknown_date_cols = [c for c in unknown_date_cols if c in cleaned.columns]

    datetime_cols = [c for c in datetime_cols if c not in unknown_date_cols]
    categorical_cols = categorical_cols + unknown_date_cols

    # --- 3) Missingness flags ---
    struct_after_drop = summarize_structure(cleaned)
    for col in cleaned.columns:
        if struct_after_drop.loc[col, "n_missing"] > 0:
            flag_col = f"{col}_was_missing"
            cleaned[flag_col] = cleaned[col].isna().astype("int8")
    meta["missingness_flags"] = {
        "columns": [c for c in cleaned.columns if c.endswith("_was_missing")]
    }

    # --- 4) Normalise dtypes ---

    # Datetime columns
    for col in datetime_cols:
        cleaned[col] = pd.to_datetime(
            cleaned[col],
            errors="coerce",
            infer_datetime_format=True,
        )

    # Categorical → string (includes unknown_date_cols)
    for col in categorical_cols:
        cleaned[col] = cleaned[col].astype("string")

    # Numeric coercion
    for col in numeric_cols:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    # --- 5) Numeric block imputation ---

    numeric_advanced = [
        col
        for col in numeric_cols
        if col in impute_plan.index
        and impute_plan.loc[col, "strategy"] == "iterative_et"
        and impute_plan.loc[col, "n_missing"] > 0
    ]
    numeric_knn = [
        col
        for col in numeric_cols
        if col in impute_plan.index
        and impute_plan.loc[col, "strategy"] == "knn"
        and impute_plan.loc[col, "n_missing"] > 0
    ]
    numeric_simple = [
        col
        for col in numeric_cols
        if col in impute_plan.index
        and impute_plan.loc[col, "strategy"] == "median"
        and impute_plan.loc[col, "n_missing"] > 0
    ]

    # Ensure target is never imputed even if plan mislabels it.
    for col_list in (numeric_advanced, numeric_knn, numeric_simple):
        if TARGET_COL in col_list:
            col_list.remove(TARGET_COL)

    # 5a) Simple median
    meta["imputation_numeric_simple"] = {}
    for col in numeric_simple:
        median_val = cleaned[col].median()
        cleaned[col] = cleaned[col].fillna(median_val)
        meta["imputation_numeric_simple"][col] = {
            "strategy": "median",
            "value": float(median_val),
        }

    # 5b) IterativeImputer + ExtraTrees
    if numeric_advanced:
        LOGGER.info("IterativeImputer on columns: %s", numeric_advanced)
        imp_et = fit_iterative_imputer_et(cleaned, numeric_advanced)
        cleaned = apply_iterative_imputer_et(
            cleaned,
            numeric_cols=numeric_advanced,
            imputer=imp_et,
        )
        meta["imputation_numeric_iterative_et"] = {
            "columns": numeric_advanced,
            "estimator": "ExtraTreesRegressor",
        }

    # 5c) KNNImputer
    if numeric_knn:
        LOGGER.info("KNNImputer on columns: %s", numeric_knn)
        knn = KNNImputer(n_neighbors=5, weights="distance")
        x_knn = cleaned[numeric_knn].copy()
        x_knn = x_knn.replace([np.inf, -np.inf], np.nan)
        x_knn_imputed = knn.fit_transform(x_knn)
        cleaned[numeric_knn] = x_knn_imputed
        meta["imputation_numeric_knn"] = {
            "columns": numeric_knn,
            "n_neighbors": 5,
        }

    # --- 6) Categorical + date imputation ---
    meta["imputation_categorical"] = {}
    for col in categorical_cols:
        if col not in impute_plan.index:
            continue

        strat = impute_plan.loc[col, "strategy"]

        if strat == "skip":
            continue

        if strat == "mode":
            mode_val = cleaned[col].mode(dropna=True)
            fill_val = mode_val.iloc[0] if not mode_val.empty else "unknown"
            cleaned[col] = cleaned[col].fillna(fill_val)
            meta["imputation_categorical"][col] = {
                "strategy": "mode",
                "value": str(fill_val),
            }
        elif strat in {"unknown_token", "unknown_date"}:
            cleaned[col] = cleaned[col].fillna("unknown")
            meta["imputation_categorical"][col] = {
                "strategy": strat,
                "value": "unknown",
            }

    # --- 7) Advanced outlier handling on key numeric columns ---

    target_outlier_cols = [
        c for c in ["price", "house_size", "bed", "bath", "acre_lot"] if c in numeric_cols
    ]

    meta["log_transform"] = {}
    meta["univariate_outliers"] = {}
    meta["multivariate_outliers"] = {}

    # 7.1) Log-transform heavily skewed columns (keep original)
    for col in target_outlier_cols:
        series = cleaned[col]
        skew_val = series.skew()
        if abs(float(skew_val)) > 1.0:
            cleaned[f"{col}_log"] = np.log1p(series.clip(lower=0))
            meta["log_transform"][col] = float(skew_val)

    # 7.2) Univariate outlier flagging
    for col in target_outlier_cols:
        series = cleaned[col]
        method = choose_univariate_method(series)
        flags = flag_univariate_outliers(series, method=method)
        cleaned[f"{col}__is_uni_outlier"] = flags.astype("int8")
        outlier_rate = float(flags.mean() * 100.0)
        meta["univariate_outliers"][col] = {
            "method": method,
            "outlier_rate_pct": outlier_rate,
        }

        # Optional clipping (IQR-based)
        if cap_univariate_outliers and flags.any():
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            cleaned[col] = series.clip(lower=lower, upper=upper)

    # 7.3) Multivariate outliers via IsolationForest
    iforest_cols = [
        c for c in ["price", "house_size", "bed", "bath", "acre_lot"] if c in numeric_cols
    ]

    if iforest_cols:
        multi_flags = detect_multivariate_outliers_iforest(
            cleaned,
            numeric_cols=iforest_cols,
            contamination=iforest_contamination,
        )
        cleaned["is_multivar_outlier"] = multi_flags.astype("int8")
        meta["multivariate_outliers"] = {
            "contamination": float(iforest_contamination),
            "columns": iforest_cols,
            "outlier_rate_pct": float(multi_flags.mean() * 100.0),
        }

    # --- 8) Remove impossible / corrupted values + optional outlier row drops ---

    impossible_mask = pd.Series(False, index=cleaned.index)

    if "price" in cleaned.columns:
        impossible_mask |= cleaned["price"] < 0
    if "house_size" in cleaned.columns:
        impossible_mask |= cleaned["house_size"] <= 0
    if "bed" in cleaned.columns:
        impossible_mask |= cleaned["bed"] < 0
    if "bath" in cleaned.columns:
        impossible_mask |= cleaned["bath"] < 0
    if "acre_lot" in cleaned.columns:
        impossible_mask |= cleaned["acre_lot"] < 0

    n_impossible = int(impossible_mask.sum())
    if n_impossible > 0:
        LOGGER.info("Removing %d rows with impossible values.", n_impossible)
        cleaned = cleaned.loc[~impossible_mask].copy()
    meta["rows_removed_impossible"] = {"count": n_impossible}

    # 8.2) Optional: row drops based on univariate flags
    if remove_univariate_outliers and target_outlier_cols:
        uni_flag_cols = [
            f"{c}__is_uni_outlier"
            for c in target_outlier_cols
            if f"{c}__is_uni_outlier" in cleaned.columns
        ]
        uni_flags = cleaned[uni_flag_cols].max(axis=1).astype(bool)
        mask_keep_uni = ~uni_flags
        n_drop_uni = int((~mask_keep_uni).sum())
        if n_drop_uni > 0:
            LOGGER.info("Removing %d rows due to univariate outliers.", n_drop_uni)
            cleaned = cleaned.loc[mask_keep_uni].copy()
        meta["rows_removed_univariate"] = {"count": n_drop_uni}

    # 8.3) Optional: row drops based on multivariate flags
    if remove_multivar_outliers and iforest_cols and "is_multivar_outlier" in cleaned.columns:
        multi_flags_aligned = cleaned["is_multivar_outlier"] == 1
        mask_keep_multi = ~multi_flags_aligned
        n_drop_multi = int((~mask_keep_multi).sum())
        if n_drop_multi > 0:
            LOGGER.info("Removing %d rows due to multivariate outliers.", n_drop_multi)
            cleaned = cleaned.loc[mask_keep_multi].copy()
        meta["rows_removed_multivariate"] = {"count": n_drop_multi}

    # --- 9) Enforce discrete counts for bed and bath ---
    for col in ["bed", "bath"]:
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].round(0).clip(lower=0).astype("int16")

    # Final sanity check: price must have no NaNs.
    if TARGET_COL in cleaned.columns:
        n_price_nan = int(cleaned[TARGET_COL].isna().sum())
        if n_price_nan > 0:
            raise RuntimeError(
                f"Invariant broken: {TARGET_COL} has {n_price_nan} NaNs "
                "at the end of clean_dataset()."
            )

    LOGGER.info(
        "Cleaning finished. Rows before: %d — after: %d.",
        n_rows_before,
        len(cleaned),
    )

    return cleaned, meta


# =============================================================================
# Public entrypoint for MLOps-style usage
# =============================================================================


def run_cleaning_pipeline(
    df: pd.DataFrame,
    config: CleaningConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Run the full cleaning pipeline using a CleaningConfig.

    This function is the main entrypoint to integrate the cleaner into an
    MLOps workflow (e.g., orchestrated jobs, feature pipelines, or training
    pipelines).

    Args:
        df: Raw input DataFrame.
        config: Cleaning configuration. If None, default values are used.

    Returns:
        A tuple of:
            * cleaned: Cleaned DataFrame.
            * meta: Metadata dictionary with diagnostics and decisions.

    """
    if config is None:
        config = CleaningConfig()

    cleaned, meta = clean_dataset(
        df=df,
        remove_univariate_outliers=config.remove_univariate_outliers,
        remove_multivar_outliers=config.remove_multivar_outliers,
        cap_univariate_outliers=config.cap_univariate_outliers,
        iforest_contamination=config.iforest_contamination,
    )
    meta["config"] = {
        "remove_univariate_outliers": config.remove_univariate_outliers,
        "remove_multivar_outliers": config.remove_multivar_outliers,
        "cap_univariate_outliers": config.cap_univariate_outliers,
        "iforest_contamination": config.iforest_contamination,
    }
    return cleaned, meta
