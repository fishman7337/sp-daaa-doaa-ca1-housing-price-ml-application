"""Model service orchestrating tabular, NLP, and CNN predictors."""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .preprocessing import engineer_minimal_from_payload, load_listing_counts

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def _require_tensorflow():
    """Import TensorFlow only when a TensorFlow-backed modality is used."""
    try:
        import tensorflow as tf
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "TensorFlow is required for the NLP and image models. Install the "
            "pinned dependencies from requirements.txt and keep NumPy below 2.0 "
            "for this project."
        ) from exc
    return tf


def _keras_text_tools():
    """Return Keras text helpers from the active TensorFlow installation."""
    tf = _require_tensorflow()
    return (
        tf.keras.preprocessing.text.Tokenizer,
        tf.keras.preprocessing.text.tokenizer_from_json,
        tf.keras.preprocessing.sequence.pad_sequences,
    )


class ModelService:
    """Service responsible for training and inference of all models."""

    def __init__(
        self,
        model_dir: Path,
        upload_dir: Path,
        nlp_processed_csv: Path = Path("nlp_data/processed/nlp_text_cleaned.csv"),
    ) -> None:
        """Initialize artifact paths and lazy model state.

        Args:
            model_dir: Directory containing deployed model artifacts.
            upload_dir: Directory used for uploaded property images.
            nlp_processed_csv: Cleaned listing text used to rebuild a tokenizer if needed.
        """
        self.model_dir = Path(model_dir)
        self.upload_dir = Path(upload_dir)
        self.nlp_processed_csv = nlp_processed_csv
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.listing_counts = load_listing_counts()

        self.tabular_path = self.model_dir / "best_regressor.joblib"
        self.cnn_path = self.model_dir / "tuned_cnn_best.keras"
        self.nlp_model_path = self.model_dir / "best_rnn_model_full.keras"
        self.nlp_tokenizer_path = self.model_dir / "nlp_tokenizer.json"
        self._tabular_model = None
        self._cnn_model = None
        self._nlp_model = None
        self._tokenizer = None

    # ------------------------------------------------------------------
    # Tabular model
    # ------------------------------------------------------------------
    def _load_tabular_model(self):
        if self._tabular_model is None:
            if not self.tabular_path.exists():
                raise FileNotFoundError(
                    "Tabular model not found. Train/export it via 05_modelling.ipynb "
                    "or src/modeling/modeling_pipeline.py, then place "
                    "best_regressor.joblib in models/."
                )
            self._tabular_model = joblib.load(self.tabular_path)
        return self._tabular_model

    def predict_tabular(self, structured_payload: dict[str, object]) -> float:
        """Predict price using tabular data."""
        model = self._load_tabular_model()
        df_row = engineer_minimal_from_payload(
            structured_payload,
            counts=self.listing_counts,
        )
        return float(model.predict(df_row)[0])

    # ------------------------------------------------------------------
    # NLP model
    # ------------------------------------------------------------------
    def _train_nlp_model(self) -> None:
        """Train a Bidirectional RNN on cleaned descriptions."""
        tf = _require_tensorflow()
        Tokenizer, _, pad_sequences = _keras_text_tools()

        if not self.nlp_processed_csv.exists():
            raise FileNotFoundError(f"Processed NLP CSV not found at {self.nlp_processed_csv}.")

        data = pd.read_csv(self.nlp_processed_csv)
        text_col = "description_clean" if "description_clean" in data else "description"
        data[text_col] = data[text_col].fillna("unknown").astype(str)
        data["price"] = pd.to_numeric(data["price"], errors="coerce")
        data = data.dropna(subset=["price"])

        texts = data[text_col].tolist()
        targets = data["price"].astype("float32").values

        max_words = 12000
        max_len = 220

        tokenizer = Tokenizer(num_words=max_words, oov_token="<UNK>")
        tokenizer.fit_on_texts(texts)
        sequences = tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")

        X_train, X_val, y_train, y_val = train_test_split(
            padded,
            targets,
            test_size=0.15,
            random_state=42,
        )

        inputs = tf.keras.Input(shape=(max_len,), name="text_input")
        x = tf.keras.layers.Embedding(max_words, 64)(inputs)
        x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, return_sequences=True))(x)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(64, activation="relu")(x)
        outputs = tf.keras.layers.Dense(1, name="price")(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="nlp_regressor")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss="mse",
            metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
        )

        model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=2,
            batch_size=64,
            verbose=0,
        )

        model.save(self.nlp_model_path)
        self.nlp_tokenizer_path.write_text(tokenizer.to_json(), encoding="utf-8")

    def _load_tokenizer(self):
        if self._tokenizer is None:
            if not self.nlp_tokenizer_path.exists():
                self._train_nlp_model()
            _, tokenizer_from_json, _ = _keras_text_tools()
            tokenizer_json = self.nlp_tokenizer_path.read_text(encoding="utf-8")
            self._tokenizer = tokenizer_from_json(tokenizer_json)
        return self._tokenizer

    def _load_nlp_model(self):
        if self._nlp_model is None:
            tf = _require_tensorflow()
            if not self.nlp_model_path.exists():
                self._train_nlp_model()
            self._nlp_model = tf.keras.models.load_model(self.nlp_model_path)
        return self._nlp_model

    def predict_nlp(self, description: str) -> float:
        """Predict price from a textual description."""
        if not description:
            raise ValueError("Description is empty.")
        tokenizer = self._load_tokenizer()
        model = self._load_nlp_model()
        _, _, pad_sequences = _keras_text_tools()

        seq = tokenizer.texts_to_sequences([description])
        padded = pad_sequences(seq, maxlen=model.input_shape[1], padding="post", truncating="post")
        return float(model.predict(padded, verbose=0)[0][0])

    # ------------------------------------------------------------------
    # CNN model
    # ------------------------------------------------------------------
    def _default_price_anchor(self) -> float:
        """Return a stable median-like price anchor for synthetic CNN fallback data."""
        candidates = [
            self.model_dir.parent / "data" / "usa_real_estate_price_histogram.csv",
            Path("data/usa_real_estate_price_histogram.csv"),
        ]
        for path in candidates:
            if not path.exists():
                continue
            try:
                df = pd.read_csv(path)
            except Exception:
                continue
            if "price" not in df.columns:
                continue
            prices = pd.to_numeric(df["price"], errors="coerce").dropna()
            prices = prices[prices > 0]
            if not prices.empty:
                return float(prices.median())
        return 450_000.0

    def _train_cnn_model(self) -> None:
        """Train a small CNN on synthetic brightness data."""
        tf = _require_tensorflow()
        tf.random.set_seed(42)
        med_price = self._default_price_anchor()

        def _make_sample(brightness: float, n: int = 64) -> np.ndarray:
            arr = np.ones((n, n, 3), dtype="float32") * brightness
            noise = np.random.normal(0, 0.05, arr.shape).astype("float32")
            return np.clip(arr + noise, 0.0, 1.0)

        brightness_levels = np.linspace(0.15, 0.95, 96)
        images = np.stack([_make_sample(b) for b in brightness_levels])
        labels = med_price * (0.85 + 0.3 * brightness_levels)

        inputs = tf.keras.Input(shape=(64, 64, 3), name="image")
        x = tf.keras.layers.Conv2D(16, 3, activation="relu")(inputs)
        x = tf.keras.layers.MaxPool2D()(x)
        x = tf.keras.layers.Conv2D(32, 3, activation="relu")(x)
        x = tf.keras.layers.MaxPool2D()(x)
        x = tf.keras.layers.Conv2D(64, 3, activation="relu")(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(64, activation="relu")(x)
        outputs = tf.keras.layers.Dense(1)(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="cnn_regressor")
        model.compile(optimizer="adam", loss="mse")
        model.fit(images, labels, epochs=6, batch_size=16, verbose=0, validation_split=0.1)
        model.save(self.cnn_path)

    @staticmethod
    def _rmse(y_true, y_pred):
        tf = _require_tensorflow()
        return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))

    def _load_cnn_model(self):
        if self._cnn_model is None:
            tf = _require_tensorflow()
            if not self.cnn_path.exists():
                self._train_cnn_model()
            custom_objs = {
                "root_mean_squared_error": tf.keras.metrics.RootMeanSquaredError(
                    name="root_mean_squared_error"
                ),
                "rmse": tf.keras.metrics.RootMeanSquaredError(name="rmse"),
                "function": self._rmse,
            }
            try:
                self._cnn_model = tf.keras.models.load_model(
                    self.cnn_path,
                    custom_objects=custom_objs,
                    compile=False,
                )
            except Exception:
                self._cnn_model = tf.keras.models.load_model(self.cnn_path, compile=False)
        return self._cnn_model

    @staticmethod
    def _get_cnn_target_size(model) -> tuple[int, int]:
        """Infer target size from the model input; default to 224x224 if unknown."""
        try:
            shape = model.input_shape
            if isinstance(shape, list | tuple) and len(shape) == 4:
                h, w = shape[1], shape[2]
                if h and w and h > 0 and w > 0:
                    return (int(h), int(w))
        except Exception:
            pass
        return (224, 224)

    def predict_image(self, image_path: Path) -> float:
        """Predict price using the CNN model from an uploaded image."""
        tf = _require_tensorflow()
        model = self._load_cnn_model()
        target_size = self._get_cnn_target_size(model)
        img = tf.keras.utils.load_img(image_path, target_size=target_size)
        arr = tf.keras.utils.img_to_array(img) / 255.0
        arr = np.expand_dims(arr, axis=0)
        return float(model.predict(arr, verbose=0)[0][0])

    # ------------------------------------------------------------------
    # Ensemble
    # ------------------------------------------------------------------
    def predict(
        self,
        structured_payload: dict[str, object] | None = None,
        description: str | None = None,
        image_paths: list[Path] | None = None,
    ) -> tuple[dict[str, float], float]:
        """Run available modalities and return per-model + ensemble prices."""
        predictions: dict[str, float] = {}
        if structured_payload:
            try:
                predictions["tabular"] = float(self.predict_tabular(structured_payload))
            except FileNotFoundError:
                pass
        if description and description.strip():
            predictions["nlp"] = float(self.predict_nlp(description.strip()))
        if image_paths:
            image_preds = []
            for path in image_paths:
                if path and Path(path).exists():
                    image_preds.append(self.predict_image(Path(path)))
            if image_preds:
                predictions["cnn"] = float(np.nanmean(image_preds))

        if not predictions:
            raise ValueError("No inputs supplied for prediction.")

        final_price = float(np.nanmean(list(predictions.values())))
        return predictions, final_price


__all__ = ["ModelService"]
