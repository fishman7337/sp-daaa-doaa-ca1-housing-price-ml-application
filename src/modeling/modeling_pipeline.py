"""Regression modelling pipeline for USA real estate data.

This module implements an end-to-end regression workflow including:
    * Deterministic data loading and train–test splitting.
    * Shared preprocessing via ColumnTransformer.
    * A library of baseline and advanced regressors.
    * Cross-validation with RMSE/MAE/R² metrics.
    * Random search hyperparameter tuning for the top models.
    * Final model selection, learning-curve analysis, and metric export.

The code is structured as importable functions so it can be reused in
notebooks, scripts, or APIs while remaining readable and testable.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from skopt import BayesSearchCV
from skopt.space import Categorical, Integer, Real
from sklearn.base import RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    ElasticNet,
    HuberRegressor,
    Lasso,
    LinearRegression,
    Ridge,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV,
    cross_validate,
    learning_curve,
    train_test_split,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVR, SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


# # 05: Modelling Pipeline (Regression on USA Real Estate)
#
# This script implements a complete regression modelling pipeline on the
# cleaned and feature-engineered USA real estate dataset. It follows an
# MLOps-oriented structure:
#
# - Deterministic data loading and splitting.
# - Shared preprocessing with `ColumnTransformer`.
# - Multiple baseline and advanced models.
# - Bayesian hyperparameter optimisation for strong candidates.
# - Model selection based on cross-validated RMSE.
# - Final evaluation on a held-out test set using MAE, RMSE and R².
# - Learning curve analysis for underfitting vs overfitting diagnosis.
# - Persisting the best pipeline and metrics to disk.
#
# ---------------------------------------------------------------------------
# 1. Import libraries and define global configuration
# ---------------------------------------------------------------------------

# Global configuration constants
# ---------------------------------------------------------------------------
TARGET_COL: str = "price"
RANDOM_STATE: int = 42
N_SPLITS: int = 5
TEST_SIZE: float = 0.2

MODEL_DIR: str = "/content/drive/MyDrive/Colab Notebooks/DOAA/models"
METRICS_PATH: str = os.path.join(MODEL_DIR, "best_model_metrics.csv")
BEST_MODEL_PATH: str = os.path.join(MODEL_DIR, "best_regressor.joblib")

sns.set(style="whitegrid")

# The modelling environment is now configured with a consistent random seed and
# standard paths for saving artefacts. All subsequent steps will reuse these
# constants to keep the workflow reproducible and MLOps-aligned.

# ---------------------------------------------------------------------------
# 2. Load feature-engineered dataset
# ---------------------------------------------------------------------------


"""Load the engineered dataset from Parquet for regression modelling.

This logical section:
    * Loads the final processed dataset stored in Google Drive (or local disk).
    * Validates the presence of the target column.
    * Prints the dataset shape for a quick sanity check.
"""


def load_engineered_parquet(path: str, target_col: str) -> pd.DataFrame:
    """Load a Parquet dataset and verify the target column exists.

    Args:
        path: Absolute path to the Parquet file.
        target_col: Name of the required target column.

    Returns:
        A pandas DataFrame containing the dataset.

    Raises:
        FileNotFoundError: If the Parquet file cannot be located.
        KeyError: If the target column is not present in the dataset.
    """
    try:
        df_loaded = pd.read_parquet(path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Dataset file not found at path: {path!r}"
        ) from exc

    if target_col not in df_loaded.columns:
        raise KeyError(
            f"Target column '{target_col}' not found in dataset. "
            f"Available columns: {list(df_loaded.columns)}"
        )

    print(f"Dataset loaded successfully. Shape: {df_loaded.shape}")
    return df_loaded


# ---------------------------------------------------------------------------
# Execute: Load engineered dataset
# ---------------------------------------------------------------------------
ENGINEERED_CSV_PATH = (
    "/content/drive/MyDrive/Colab Notebooks/DOAA/"
    "data_processed/final_dataset_20251116_0720.parquet"
)

df = load_engineered_parquet(
    path=ENGINEERED_CSV_PATH,
    target_col=TARGET_COL,
)

print(df.head())

# The loaded dataset will now feed into the feature typing and data splitting
# steps. The expectation is that:
# - Target column "price" is present and numeric.
# - Engineered features are clean and ready for modelling.

# ---------------------------------------------------------------------------
# 3. Feature typing and train–test split
# ---------------------------------------------------------------------------

"""Feature type inference with structured DataFrame output.

This section:
    * Infers numeric vs categorical features (excluding the target).
    * Skips identifier-like columns (e.g. containing "id").
    * Returns a neatly formatted DataFrame summarising feature metadata.
"""


def infer_feature_metadata(
    df_in: pd.DataFrame,
    target_col: str,
) -> pd.DataFrame:
    """Infer feature metadata for modelling, excluding the target column.

    Args:
        df_in: Dataset containing features and target.
        target_col: Name of the target column to exclude.

    Returns:
        A pandas DataFrame with columns:
            - feature: Name of the feature column.
            - feature_type: "numeric" or "categorical".
            - dtype: Pandas dtype of the column.
            - n_unique: Number of unique non-null values.
            - n_missing: Number of missing entries.
            - pct_missing: Percentage of missing entries.
    """
    feature_df = df_in.drop(columns=[target_col])

    global numeric_cols

    numeric_cols = (
        feature_df.select_dtypes(include=[np.number]).columns.tolist()
    )

    global categorical_cols

    categorical_cols = [
        col
        for col in feature_df.columns
        if col not in numeric_cols and not col.lower().startswith("id")
    ]

    records = []

    for col in feature_df.columns:
        if col in numeric_cols:
            ftype = "numeric"
        elif col in categorical_cols:
            ftype = "categorical"
        else:
            # Skip ID-like or unidentified columns
            continue

        n_missing = feature_df[col].isna().sum()
        n_unique = feature_df[col].nunique(dropna=True)
        pct_missing = (n_missing / len(feature_df)) * 100

        records.append(
            {
                "feature": col,
                "feature_type": ftype,
                "dtype": str(feature_df[col].dtype),
                "n_unique": n_unique,
                "n_missing": n_missing,
                "pct_missing": pct_missing,
            }
        )

    feature_metadata_df = pd.DataFrame(records).sort_values(
        by=["feature_type", "feature"]
    )

    print(
        "Inferred feature types:"
        f"\nNumeric: {numeric_cols}\nCategorical: {categorical_cols}"
    )

    return feature_metadata_df


# ---------------------------------------------------------------------------
# Execute and display feature metadata
# ---------------------------------------------------------------------------
feature_metadata_df = infer_feature_metadata(
    df_in=df,
    target_col=TARGET_COL,
)

print(feature_metadata_df)

# The lists displayed above summarise the modelling feature space:
#
# - `numeric_features` will feed into the numeric preprocessing branch.
# - `categorical_features` will be handled by the categorical branch
#   (imputation + one-hot encoding).
#
# This explicit separation supports a clean ColumnTransformer definition and
# reduces the risk of accidentally leaking target information.

# ---------------------------------------------------------------------------
# 3.2 Train–test split
# ---------------------------------------------------------------------------

"""Train–test split utilities.

This section:
    * Splits the dataset into train and test portions.
    * Ensures that feature/target matrices are shaped consistently.
    * Provides a helper for summarising the shapes.
"""


def split_features_and_target(
    df_in: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
    target_col: str,
    test_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split a DataFrame into train and test sets.

    Args:
        df_in: Input DataFrame containing features and target.
        numeric_cols: List of numeric feature names.
        categorical_cols: List of categorical feature names.
        target_col: Name of the target column.
        test_size: Fraction of data to allocate to the test set.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    feature_cols = numeric_cols + categorical_cols

    X = df_in[feature_cols].copy()
    y = df_in[target_col].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    return X_train, X_test, y_train, y_test


def summarise_train_test_shapes(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> pd.DataFrame:
    """Summarise the shapes of the train and test splits.

    Args:
        X_train: Training features.
        X_test: Test features.
        y_train: Training target.
        y_test: Test target.

    Returns:
        DataFrame summarising the row and column counts for each split.
    """
    summary = pd.DataFrame(
        [
            {
                "set": "X_train",
                "n_rows": X_train.shape[0],
                "n_cols": X_train.shape[1],
            },
            {
                "set": "X_test",
                "n_rows": X_test.shape[0],
                "n_cols": X_test.shape[1],
            },
            {
                "set": "y_train",
                "n_rows": y_train.shape[0],
                "n_cols": 1,
            },
            {
                "set": "y_test",
                "n_rows": y_test.shape[0],
                "n_cols": 1,
            },
        ]
    )

    return summary


# ---------------------------------------------------------------------------
# Execute: Split + summary
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = split_features_and_target(
    df_in=df,
    numeric_cols=numeric_cols,
    categorical_cols=categorical_cols,
    target_col=TARGET_COL,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
)

split_summary_df = summarise_train_test_shapes(
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
)

print(split_summary_df)

# The logged shapes confirm an 80/20 train–test split (based on TEST_SIZE).
# All model selection, cross-validation and hyperparameter optimisation steps
# will now use only the training data (X_train, y_train), preserving the test
# set for final evaluation.

# ---------------------------------------------------------------------------
# 4. Preprocessing and model candidates
# ---------------------------------------------------------------------------

"""Preprocessing pipelines for numeric and categorical features.

This module defines:
    * Numeric transformer: median imputation + standard scaling.
    * Categorical transformer: most-frequent imputation + one-hot encoding.
    * Combined ColumnTransformer for unified preprocessing.

These transformers are used in downstream modelling pipelines to ensure that
the same preprocessing logic is applied consistently in training and inference.
"""


def build_preprocessor(
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> ColumnTransformer:
    """Build a ColumnTransformer for numeric and categorical features.

    Args:
        numeric_cols: List of numeric feature names.
        categorical_cols: List of categorical feature names.

    Returns:
        Configured ColumnTransformer handling numeric and categorical branches.
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    return preprocessor


# ---------------------------------------------------------------------------
# Execute: Build preprocessor
# ---------------------------------------------------------------------------
preprocessor = build_preprocessor(
    numeric_cols=numeric_cols,
    categorical_cols=categorical_cols,
)

print(preprocessor)

# The `preprocessor` object now encapsulates all transformation logic required
# to convert the raw input features into a fully numeric, scaled design matrix.
# Because it is part of the final Pipeline, this logic is consistently applied
# during training, cross-validation and inference.

# ---------------------------------------------------------------------------
# 4.2 Define baseline and candidate regression models
# ---------------------------------------------------------------------------

"""Candidate regression models with shared preprocessing.

This section defines:
    * A helper to wrap any regressor into a Pipeline with preprocessing.
    * A dictionary of model candidates spanning:
        - Linear baselines
        - Regularised linear models
        - Tree ensembles
        - Gradient boosting (sklearn / XGB / LGBM / CatBoost)
        - kNN
        - MLP
        - Decision trees
        - ExtraTrees
        - AdaBoost
        - HistGradientBoosting
"""


def make_pipeline(model: RegressorMixin) -> Pipeline:
    """Create a modelling pipeline with shared preprocessing.

    Args:
        model: A scikit-learn compatible regressor.

    Returns:
        Pipeline combining preprocessing and the regressor.
    """
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )


# Candidate models registry
model_candidates: Dict[str, Pipeline] = {
    # ------------------- Baseline -------------------
    "dummy_median": make_pipeline(
        DummyRegressor(strategy="median"),
    ),
    "linear": make_pipeline(
        LinearRegression(),
    ),
    "ridge": make_pipeline(
        Ridge(alpha=10.0, random_state=RANDOM_STATE),
    ),
    "lasso": make_pipeline(
        Lasso(alpha=0.001, random_state=RANDOM_STATE),
    ),
    "elastic_net": make_pipeline(
        ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=RANDOM_STATE),
    ),
    "huber": make_pipeline(
        HuberRegressor(
            epsilon=1.35,
        ),
    ),
    # ------------------- Trees and ensembles -------------------
    "dt": make_pipeline(
        DecisionTreeRegressor(
            random_state=RANDOM_STATE,
        ),
    ),
    "rf": make_pipeline(
        RandomForestRegressor(
            n_estimators=500,
            max_depth=20,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    ),
    "extra_trees": make_pipeline(
        ExtraTreesRegressor(
            n_estimators=600,
            max_depth=25,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    ),
    # ------------------- kNN -------------------
    "knn": make_pipeline(
        KNeighborsRegressor(
            n_neighbors=10,
            weights="distance",
        ),
    ),
    # ------------------- SVM -------------------
    "svr_rbf": make_pipeline(
        SVR(
            kernel="rbf",
            C=10.0,
            epsilon=0.1,
        ),
    ),
    "svr_linear": make_pipeline(
        LinearSVR(
            C=1.0,
            random_state=RANDOM_STATE,
        ),
    ),
    # ------------------- MLP -------------------
    "mlp": make_pipeline(
        MLPRegressor(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            random_state=RANDOM_STATE,
            max_iter=500,
        ),
    ),
    # ------------------- Boosting (sklearn) -------------------
    "gbr": make_pipeline(
        GradientBoostingRegressor(random_state=RANDOM_STATE),
    ),
    "hist_gbr": make_pipeline(
        HistGradientBoostingRegressor(random_state=RANDOM_STATE),
    ),
    "ada": make_pipeline(
        AdaBoostRegressor(random_state=RANDOM_STATE),
    ),
    # ------------------- External boosting (XGB, LGBM, CatBoost) -------------------
    "xgb": make_pipeline(
        XGBRegressor(
            n_estimators=800,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    ),
    "lgbm": make_pipeline(
        LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    ),
    "catboost": make_pipeline(
        CatBoostRegressor(
            depth=8,
            learning_rate=0.05,
            iterations=1000,
            loss_function="RMSE",
            random_state=RANDOM_STATE,
            verbose=False,
        )
    ),
}

# ---------------------------------------------------------------------------
# 5. Cross-validation evaluation for all models
# ---------------------------------------------------------------------------

"""Cross-validation utilities for model comparison.

This section:
    * Defines a scoring dictionary (MAE, RMSE, R²).
    * Implements a helper to run cross_validate for each model.
    * Returns a tidy DataFrame summarising mean/std metrics.
"""

SCORING: Dict[str, str] = {
    "mae": "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error",
    "r2": "r2",
}


def evaluate_models_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    models: Dict[str, Pipeline],
    cv_strategy: KFold,
    scoring: Dict[str, str],
    n_jobs_cv: int = 1,
) -> pd.DataFrame:
    """Run cross-validation for a dictionary of model pipelines.

    For each model, this function:
        * Executes cross-validation using the provided strategy and scoring.
        * Uses a configurable n_jobs_cv to control parallelism and avoid
          out-of-memory issues on constrained environments.
        * Converts MAE and RMSE from negative (sklearn convention) to
          positive values.

    Args:
        X_train: Training feature matrix.
        y_train: Training target vector.
        models: Dictionary mapping model names to Pipeline instances.
        cv_strategy: KFold instance for cross-validation.
        scoring: Dictionary of scoring metrics.
        n_jobs_cv: Number of jobs for cross_validate.

    Returns:
        DataFrame summarising mean and standard deviation of metrics per model.
    """
    rows = []

    for name, model in models.items():
        print(f"[CV] Evaluating model: {name}")
        cv_results = cross_validate(
            model,
            X_train,
            y_train,
            scoring=scoring,
            cv=cv_strategy,
            n_jobs=n_jobs_cv,
            return_train_score=False,
        )

        mae_scores = -cv_results["test_mae"]
        rmse_scores = -cv_results["test_rmse"]
        r2_scores = cv_results["test_r2"]

        rows.append(
            {
                "model": name,
                "mae_mean": mae_scores.mean(),
                "mae_std": mae_scores.std(),
                "rmse_mean": rmse_scores.mean(),
                "rmse_std": rmse_scores.std(),
                "r2_mean": r2_scores.mean(),
                "r2_std": r2_scores.std(),
            }
        )

    cv_summary = pd.DataFrame(rows).sort_values(
        by="rmse_mean",
        ascending=True,
    )

    return cv_summary.reset_index(drop=True)


# Cross-validation strategy
cv = KFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

# ---------------------------------------------------------------------------
# Execute: cross-validation and summary table
# ---------------------------------------------------------------------------
cv_summary = evaluate_models_cv(
    X_train=X_train,
    y_train=y_train,
    models=model_candidates,
    cv_strategy=cv,
    scoring=SCORING,
    n_jobs_cv=1,
)

print(cv_summary)

# The cross-validation summary ranks all candidate models by mean RMSE.
# The dummy_median baseline indicates how well a naive “always predict the
# median price” strategy performs. Any production-quality model should beat
# this baseline comfortably.

# ---------------------------------------------------------------------------
# 6. Random search tuning for top models
# ---------------------------------------------------------------------------

"""Random search hyperparameter tuning for top models.

This script:
    * Selects the TOP-2 models from a CV summary based on RMSE.
    * Defines random search spaces for supported models in model_candidates,
      including:
        - Linear family (ridge, lasso, elastic_net, huber)
        - Tree-based (dt, rf, extra_trees)
        - Boosting (gbr, hist_gbr, ada, xgb, lgbm, catboost)
        - Nearest neighbours (knn)
    * For models without a search space (e.g. dummy_median, linear),
      the base pipeline is reused without tuning.
    * Runs RandomizedSearchCV on a tuning subset to control runtime.
    * Refits the best hyperparameters on the FULL training set.
    * Combines the tuned TOP-2 models into a VotingRegressor ensemble.
    * Returns tuned pipelines (including the ensemble) and a tuning summary.
"""

# Search spaces for randomised tuning
search_spaces: Dict[str, Dict[str, object]] = {
    # Linear models
    "ridge": {
        "model__alpha": Real(1e-3, 1e3, prior="log-uniform"),
    },
    "lasso": {
        "model__alpha": Real(1e-4, 1.0, prior="log-uniform"),
    },
    "elastic_net": {
        "model__alpha": Real(1e-4, 1.0, prior="log-uniform"),
        "model__l1_ratio": Real(0.1, 0.9),
    },
    "huber": {
        "model__alpha": Real(1e-4, 10.0, prior="log-uniform"),
        "model__epsilon": Real(1.1, 2.0),
    },
    # Trees and ensembles
    "dt": {
        "model__max_depth": Integer(3, 30),
        "model__min_samples_split": Integer(2, 20),
        "model__min_samples_leaf": Integer(1, 10),
    },
    "rf": {
        "model__n_estimators": Integer(200, 1000),
        "model__max_depth": Integer(5, 40),
        "model__min_samples_split": Integer(2, 20),
        "model__min_samples_leaf": Integer(1, 10),
        "model__max_features": Real(0.3, 1.0),
    },
    "extra_trees": {
        "model__n_estimators": Integer(200, 1200),
        "model__max_depth": Integer(5, 40),
        "model__min_samples_split": Integer(2, 20),
        "model__min_samples_leaf": Integer(1, 10),
        "model__max_features": Real(0.3, 1.0),
    },
    # Gradient boosting (sklearn)
    "gbr": {
        "model__n_estimators": Integer(300, 1000),
        "model__learning_rate": Real(1e-2, 0.2, prior="log-uniform"),
        "model__max_depth": Integer(2, 6),
        "model__subsample": Real(0.6, 1.0),
        "model__min_samples_split": Integer(2, 20),
        "model__min_samples_leaf": Integer(1, 10),
    },
    "hist_gbr": {
        "model__learning_rate": Real(1e-2, 0.3, prior="log-uniform"),
        "model__max_depth": Integer(3, 16),
        "model__max_leaf_nodes": Integer(15, 120),
        "model__min_samples_leaf": Integer(20, 100),
        "model__l2_regularization": Real(1e-4, 10.0, prior="log-uniform"),
    },
    # kNN
    "knn": {
        "model__n_neighbors": Integer(3, 50),
        "model__weights": Categorical(["uniform", "distance"]),
        "model__p": Integer(1, 2),
    },
    # XGBoost
    "xgb": {
        "model__n_estimators": Integer(400, 1500),
        "model__max_depth": Integer(3, 12),
        "model__learning_rate": Real(1e-2, 0.3, prior="log-uniform"),
        "model__subsample": Real(0.5, 1.0),
        "model__colsample_bytree": Real(0.5, 1.0),
    },
    # LightGBM
    "lgbm": {
        "model__n_estimators": Integer(400, 1500),
        "model__learning_rate": Real(1e-2, 0.3, prior="log-uniform"),
        "model__num_leaves": Integer(31, 512),
        "model__subsample": Real(0.5, 1.0),
        "model__colsample_bytree": Real(0.5, 1.0),
    },
    # CatBoost
    "catboost": {
        "model__depth": Integer(4, 10),
        "model__learning_rate": Real(1e-2, 0.3, prior="log-uniform"),
        "model__iterations": Integer(400, 1500),
    },
}


def get_best_from_cv_summary(
    cv_summary: pd.DataFrame,
    top_n: int = 2,
) -> List[str]:
    """Get the top-N model names by RMSE."""

    if "model" not in cv_summary.columns or "rmse_mean" not in cv_summary.columns:
        raise KeyError(
            "cv_summary must contain 'model' and 'rmse_mean' columns."
        )

    return cv_summary.sort_values(by="rmse_mean", ascending=True)[
        "model"
    ].head(top_n).tolist()


def make_random_search_checkpoint_callback(
    model_name: str,
    checkpoint_dir: str,
):
    """Create callback to log random search progress and save checkpoints.

    Args:
        model_name: Name of the model being tuned.
        checkpoint_dir: Directory to store checkpoints.

    Returns:
        A callback function that can be invoked manually after fitting.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    iter_state = {"i": 0}

    log_path = os.path.join(
        checkpoint_dir,
        f"{model_name}_tuning_log.csv",
    )

    def _callback(
        estimator: RandomizedSearchCV,
        search_space: Dict[str, object],
    ) -> None:
        """Inner callback to persist the best estimator after tuning.

        Args:
            estimator: Fitted RandomizedSearchCV instance.
            search_space: Parameter distributions for this model.
        """
        iter_state["i"] += 1
        i = iter_state["i"]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        best_estimator = estimator.best_estimator_
        best_params = estimator.best_params_
        best_score = -estimator.best_score_

        model_path = os.path.join(
            checkpoint_dir,
            f"{model_name}_best_iter{i:03d}_{timestamp}.pkl",
        )
        joblib.dump(best_estimator, model_path)

        print(
            f"[CHECKPOINT] {model_name}: iter={i} saved → {model_path}"
        )

        record = {
            "timestamp": timestamp,
            "iteration": i,
            "best_rmse": best_score,
            "best_params": best_params,
        }
        df_log = (
            pd.DataFrame([record])
            if not os.path.exists(log_path)
            else pd.concat(
                [pd.read_csv(log_path), pd.DataFrame([record])],
                ignore_index=True,
            )
        )
        df_log.to_csv(log_path, index=False)
        print(f"[CHECKPOINT] Log updated → {log_path}")

    return _callback


def random_tune_top_models(
    top_models: List[str],
    model_registry: Dict[str, Pipeline],
    search_spaces: Dict[str, Dict[str, object]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = RANDOM_STATE,
    n_iter: int = 30,
    tune_fraction: float = 0.5,
    cv_strategy: Optional[KFold] = None,
    checkpoint_dir: str = os.path.join(MODEL_DIR, "tuning_checkpoints"),
) -> Tuple[Dict[str, Pipeline], pd.DataFrame]:
    """Randomised hyperparameter tuning for the top-N models.

    Args:
        top_models: List of model names selected from the CV summary.
        model_registry: Dictionary of all candidate model pipelines.
        search_spaces: Dictionary mapping model names to search spaces.
        X_train: Training feature matrix.
        y_train: Training target vector.
        random_state: Random seed for the randomised search.
        n_iter: Number of RandomizedSearchCV iterations.
        tune_fraction: Fraction of training data used for tuning.
        cv_strategy: KFold strategy; defaults to a standard KFold if None.
        checkpoint_dir: Directory to store checkpoints and logs.

    Returns:
        Tuple of:
            - Dictionary of tuned pipelines (including the ensemble).
            - DataFrame summarising the tuning results.
    """
    if cv_strategy is None:
        cv_strategy = KFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=random_state,
        )

    tuned_pipelines: Dict[str, Pipeline] = {}
    tuning_records: List[Dict[str, object]] = []

    # Subsample for tuning to control runtime
    X_tune, _, y_tune, _ = train_test_split(
        X_train,
        y_train,
        test_size=(1.0 - tune_fraction),
        random_state=random_state,
    )

    for name in top_models:
        if name not in model_registry:
            print(f"[WARN] Model '{name}' not found in registry. Skipping.")
            continue

        base_pipe = model_registry[name]

        if name not in search_spaces:
            print(
                f"[INFO] No search space defined for '{name}', "
                "reusing base pipeline without tuning."
            )
            tuned_pipelines[f"{name}_tuned"] = clone(base_pipe).fit(
                X_train,
                y_train,
            )
            continue

        print(f"\n=== RANDOM SEARCH tuning for: {name} ===")

        callback = make_random_search_checkpoint_callback(
            model_name=name,
            checkpoint_dir=checkpoint_dir,
        )

        rand_cv = RandomizedSearchCV(
            estimator=base_pipe,
            param_distributions=search_spaces[name],
            n_iter=n_iter,
            scoring="neg_root_mean_squared_error",
            cv=cv_strategy,
            random_state=random_state,
            n_jobs=-1,
            verbose=2,
            return_train_score=True,
        )

        rand_cv.fit(X_tune, y_tune)

        callback(rand_cv, search_spaces[name])

        best_rmse = -rand_cv.best_score_
        best_params = rand_cv.best_params_

        print(f"[RESULT] {name} → Best RMSE (subset): {best_rmse:.4f}")
        print("[RESULT] Best Params:", best_params)

        tuned_model = clone(rand_cv.best_estimator_)
        tuned_model.fit(X_train, y_train)

        tuned_pipelines[f"{name}_tuned"] = tuned_model

        tuning_records.append(
            {
                "model": name,
                "best_rmse_subset": best_rmse,
                "best_params": best_params,
            }
        )

    # Build VotingRegressor ensemble for the tuned top-2
    tuned_top2 = list(tuned_pipelines.keys())[:2]
    estimators_for_voting = [
        (name, tuned_pipelines[name]) for name in tuned_top2
    ]

    voting_reg = VotingRegressor(
        estimators=estimators_for_voting,
        n_jobs=-1,
    )

    voting_reg.fit(X_train, y_train)
    tuned_pipelines["voting_top2_tuned"] = voting_reg

    tuning_summary = pd.DataFrame(tuning_records)
    summary_path = os.path.join(MODEL_DIR, "tuning_summary.csv")
    tuning_summary.to_csv(summary_path, index=False)
    print(f"[CHECKPOINT] Summary saved → {summary_path}")

    return tuned_pipelines, tuning_summary


# ---------------------------------------------------------------------------
# Execute: random search tuning for top models
# ---------------------------------------------------------------------------
top_models = get_best_from_cv_summary(
    cv_summary=cv_summary,
    top_n=2,
)
print("Top-2 models for Random Search:", top_models)

tuned_pipelines, tuning_summary = random_tune_top_models(
    top_models=top_models,
    model_registry=model_candidates,
    search_spaces=search_spaces,
    X_train=X_train,
    y_train=y_train,
    random_state=RANDOM_STATE,
    n_iter=30,
    tune_fraction=0.5,
    cv_strategy=cv,
    checkpoint_dir=os.path.join(MODEL_DIR, "tuning_checkpoints"),
)

print(tuning_summary)

# ---------------------------------------------------------------------------
# 7. Final model selection and evaluation
# ---------------------------------------------------------------------------

"""Final model selection: baseline vs tuned models.

This module:
    * Combines baseline and tuned model pipelines into a single registry.
    * Re-runs cross-validation to compare all models.
    * Selects the best model by mean RMSE.
"""


def build_all_final_models(
    base_models: Dict[str, Pipeline],
    tuned_models: Dict[str, Pipeline],
) -> Dict[str, Pipeline]:
    """Combine baseline, tuned, and ensemble models into a unified registry.

    Tuned models are appended with the suffix "_tuned". The VotingRegressor
    (if present) is treated exactly like a tuned model.

    Args:
        base_models: Dictionary of baseline model pipelines.
        tuned_models: Dictionary of tuned model pipelines (including ensemble).

    Returns:
        Combined dictionary of all models.
    """
    all_models = base_models.copy()

    for name, model in tuned_models.items():
        all_models[name] = model

    return all_models


def evaluate_final_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    models: Dict[str, Pipeline],
    cv_strategy: KFold,
    scoring: Dict[str, str],
    n_jobs_cv: int = 1,
) -> pd.DataFrame:
    """Evaluate baseline + tuned models using cross-validation.

    Args:
        X_train: Training feature matrix.
        y_train: Training target vector.
        models: Dictionary of all candidate pipelines.
        cv_strategy: KFold strategy.
        scoring: Dictionary of scoring metrics.
        n_jobs_cv: Number of parallel jobs.

    Returns:
        DataFrame summarising metrics across all models.
    """
    return evaluate_models_cv(
        X_train=X_train,
        y_train=y_train,
        models=models,
        cv_strategy=cv_strategy,
        scoring=scoring,
        n_jobs_cv=n_jobs_cv,
    )


# ---------------------------------------------------------------------------
# Execute: combine baseline + tuned + VotingRegressor into leaderboard
# ---------------------------------------------------------------------------
all_final_models: Dict[str, Pipeline] = build_all_final_models(
    base_models=model_candidates,
    tuned_models=tuned_pipelines,
)

selection_df = evaluate_final_models(
    X_train=X_train,
    y_train=y_train,
    models=all_final_models,
    cv_strategy=cv,
    scoring=SCORING,
    n_jobs_cv=1,
)

print(selection_df)

# ---------------------------------------------------------------------------
# 7.1 Best model training and test-set evaluation
# ---------------------------------------------------------------------------

"""Final model selection, training, and test-set evaluation.

This module:
    * Selects the best model based on cross-validation metrics.
    * Fits the best model on the full training split.
    * Evaluates performance on the held-out test set.
    * Works for both single models and ensembles (e.g. VotingRegressor).
"""

EstimatorType = Union[Pipeline, RegressorMixin]


def evaluate_best_model(
    selection_df: pd.DataFrame,
    model_registry: Dict[str, object],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[str, object, pd.DataFrame]:
    """Retrieve, train, and evaluate the best model from CV results.

    Args:
        selection_df: DataFrame of CV metrics for all models.
        model_registry: Dictionary mapping model names to fitted/unfitted
            estimators.
        X_train: Training features.
        y_train: Training target.
        X_test: Test features.
        y_test: Test target.

    Returns:
        Tuple containing:
            - best_model_name: Name of the best model.
            - best_estimator: Fitted estimator object.
            - summary_df: DataFrame with test metrics.
    """
    # 1) Select the best model
    best_model_name = selection_df.iloc[0]["model"]
    print(f"[INFO] Best model selected: {best_model_name}")

    if best_model_name not in model_registry:
        raise KeyError(f"'{best_model_name}' not found in model_registry.")

    best_estimator = model_registry[best_model_name]

    # 2) Fit on full training data
    best_estimator.fit(X_train, y_train)

    # 3) Evaluate on test
    y_pred = best_estimator.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    summary_df = pd.DataFrame(
        [
            {
                "model": best_model_name,
                "mae_test": mae,
                "rmse_test": rmse,
                "r2_test": r2,
            }
        ]
    )

    print("[TEST] Best model performance on held-out test set:")
    print(summary_df)

    return best_model_name, best_estimator, summary_df


# ---------------------------------------------------------------------------
# Execute: best model training and test-set evaluation
# ---------------------------------------------------------------------------
best_model_name, best_estimator, test_summary_df = evaluate_best_model(
    selection_df=selection_df,
    model_registry=all_final_models,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
)

print(test_summary_df)

# ---------------------------------------------------------------------------
# 7.2 Learning curve analysis for the final model
# ---------------------------------------------------------------------------

"""Learning curve generation and summary table for the final model.

This module:
    * Computes a learning curve using R² as the scoring metric.
    * Supports all estimator types (Pipeline, XGB, LGBM, CatBoost, etc.).
    * Returns both raw curve data and a plotting helper.
"""


def compute_learning_curve(
    model: EstimatorType,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_strategy: KFold,
    train_sizes: Optional[np.ndarray] = None,
    scoring: str = "r2",
) -> pd.DataFrame:
    """Compute learning curve for a given estimator.

    Args:
        model: Estimator for which to compute the learning curve.
        X_train: Training features.
        y_train: Training target.
        cv_strategy: KFold strategy for learning_curve.
        train_sizes: Optional array of train sizes; defaults to 10 points.
        scoring: Scoring metric; defaults to "r2".

    Returns:
        DataFrame with train_size, train_r2_mean, train_r2_std,
        val_r2_mean, val_r2_std.
    """
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 10)

    sizes_abs, train_scores, val_scores = learning_curve(
        model,
        X_train,
        y_train,
        train_sizes=train_sizes,
        cv=cv_strategy,
        scoring=scoring,
        n_jobs=-1,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    lc_df = pd.DataFrame(
        {
            "train_size": sizes_abs,
            "train_r2_mean": train_mean,
            "train_r2_std": train_std,
            "val_r2_mean": val_mean,
            "val_r2_std": val_std,
        }
    )

    return lc_df


def plot_learning_curve(
    lc_df: pd.DataFrame,
    model_name: str,
) -> None:
    """Plot learning curve for any final model (including voting ensembles).

    Args:
        lc_df: DataFrame returned by compute_learning_curve.
        model_name: Name of the model (for plot title).
    """
    plt.figure(figsize=(8, 5))
    plt.plot(
        lc_df["train_size"],
        lc_df["train_r2_mean"],
        "o-",
        label="Training R²",
    )
    plt.plot(
        lc_df["train_size"],
        lc_df["val_r2_mean"],
        "o-",
        label="Validation R²",
    )
    plt.fill_between(
        lc_df["train_size"],
        lc_df["train_r2_mean"] - lc_df["train_r2_std"],
        lc_df["train_r2_mean"] + lc_df["train_r2_std"],
        alpha=0.2,
    )
    plt.fill_between(
        lc_df["train_size"],
        lc_df["val_r2_mean"] - lc_df["val_r2_std"],
        lc_df["val_r2_mean"] + lc_df["val_r2_std"],
        alpha=0.2,
    )
    plt.xlabel("Training set size")
    plt.ylabel("R² score")
    plt.title(f"Learning Curve – {model_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Execute: learning curve for best model
# ---------------------------------------------------------------------------
learning_curve_df = compute_learning_curve(
    model=best_estimator,
    X_train=X_train,
    y_train=y_train,
    cv_strategy=cv,
)

print(learning_curve_df)

plot_learning_curve(
    lc_df=learning_curve_df,
    model_name=best_model_name,
)

# ---------------------------------------------------------------------------
# 7.3 Persist best model and metrics to disk
# ---------------------------------------------------------------------------

"""Model export utilities for saving the final trained estimator and metrics.

This module:
    * Ensures the model directory exists.
    * Serialises the best trained estimator (Pipeline, VotingRegressor, etc.)
      using joblib.
    * Computes test-set evaluation metrics directly inside this function.
    * Stores test-set metrics into a CSV.
    * Returns a summary DataFrame for notebook or script display.
"""


def save_final_model_and_metrics(
    model: EstimatorType,
    model_name: str,
    model_dir: str,
    model_path: str,
    metrics_path: str,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """Persist the best model and its test-set metrics to disk.

    Args:
        model: Fitted estimator object.
        model_name: Name of the final model.
        model_dir: Directory where model artefacts will be stored.
        model_path: Full path to the serialized model file.
        metrics_path: Full path to the CSV metrics file.
        X_test: Test features.
        y_test: Test target.

    Returns:
        DataFrame summarising the test-set metrics.
    """
    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(model, model_path)
    print(f"[EXPORT] Model saved to: {model_path}")

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    metrics_df = pd.DataFrame(
        [
            {
                "model": model_name,
                "mae_test": mae,
                "rmse_test": rmse,
                "r2_test": r2,
            }
        ]
    )

    metrics_df.to_csv(metrics_path, index=False)
    print(f"[EXPORT] Metrics saved to: {metrics_path}")

    return metrics_df


# ---------------------------------------------------------------------------
# Execute: save model + metrics
# ---------------------------------------------------------------------------
final_metrics_df = save_final_model_and_metrics(
    model=best_estimator,
    model_name=best_model_name,
    model_dir=MODEL_DIR,
    model_path=BEST_MODEL_PATH,
    metrics_path=METRICS_PATH,
    X_test=X_test,
    y_test=y_test,
)

print(final_metrics_df)

# The BEST_MODEL_PATH and METRICS_PATH files are now available in the models
# directory. These persisted artefacts form the bridge between this modelling
# pipeline and later stages such as API serving, monitoring and deployment
# documentation.
