"""
model_registry.py

Loads every ML artifact ONCE (classifier weights, TF-IDF vectorizers,
regression pipeline) and exposes them via a single ModelRegistry
object, meant to be instantiated once at FastAPI startup (not
per-request -- reloading a .pth or .joblib file on every request
would add unnecessary latency and is unnecessary given these
artifacts don't change between requests).

Deliberately does NOT depend on df_train / parquet training data at
all: RiskClassifier's input dimensions (TITLE_INPUT_DIM,
DESC_INPUT_DIM) are derived directly from the already-fitted
TF-IDF vectorizers' vocabulary sizes, which is all inference needs.
The pos_weight / BCEWithLogitsLoss machinery in the training-time
model_setup() is irrelevant here -- that's a training-loss concern,
not something a forward pass needs.

RiskClassifier itself is imported from src/classification/train.py
(its actual home) rather than duplicated here, to avoid the same
architecture-drift risk that staging_marts_sql.py avoids for the SQL
logic -- if the architecture changes in train.py, this module picks
it up automatically instead of silently going stale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import torch

# Adjust these imports to match your actual project layout.
from config.config_script import (
    MODEL_CLASSIFICATION_PATH,
    MODEL_REGRESSION_PATH,
    BRANCH_HIDDEN_DIM,
    HEAD_HIDDEN_DIM,
    DROPOUT_RATE,
    DEVICE,
    feature_cols,
    numeric_features,
    categorical_features,
    boolean_features,
    CONFORMAL_ARTIFACT_PATH
)
from src.classification.preprocessor import TextVectorizer  # adjust path if preprocessor.py lives elsewhere
from src.classification.train import RiskClassifier


# ============================================================
# Registry
# ============================================================

@dataclass
class ModelRegistry:
    classifier: RiskClassifier
    title_vectorizer: Any  # fitted sklearn TfidfVectorizer
    desc_vectorizer: Any   # fitted sklearn TfidfVectorizer
    regression_pipeline: Any  # fitted sklearn Pipeline (ColumnTransformer + RobustScaler + TransformedTargetRegressor)
    device: torch.device

    # Exposed so the assembly step (next module) can build the
    # regressor input DataFrame with exactly the right columns, in
    # any order, matching config_script.py's feature_cols.
    numeric_features: list[str]
    categorical_features: list[str]
    boolean_features: list[str]

    conformal_quantile: float | None = None
    conformal_confidence_level: float | None = None

    def classify(self, title_vec: torch.Tensor, desc_vec: torch.Tensor) -> torch.Tensor:
        """Runs a forward pass and returns raw logits (NOT
        probabilities -- apply torch.sigmoid() and a threshold
        separately, since threshold policy belongs in the endpoint
        layer, not the registry)."""
        self.classifier.eval()
        with torch.no_grad():
            return self.classifier(title_vec, desc_vec)

    def predict_price(self, features_df) -> float:
        """Runs the regression pipeline on a single-row DataFrame
        (columns matching feature_cols) and returns the point price
        prediction. Conformal interval is not yet wired in here --
        deferred, per product decision (2026-09)."""
        prediction = np.asarray(self.regression_pipeline.predict(features_df))
        if prediction.size != 1:
            raise ValueError(
                f"Expected one price prediction, got shape {prediction.shape}."
            )
        return float(prediction.reshape(-1)[0])
    
    def predict_price_with_interval(
        self, features_df,
    ) -> tuple[float, float | None, float | None]:
        """Returns (point_prediction, lower_bound, upper_bound).
 
        lower_bound/upper_bound are None if no conformal artifact was
        loaded at startup (see ModelRegistry.conformal_quantile) --
        callers should treat that as "point prediction only, no
        interval available" rather than an error, since the endpoint
        should still degrade gracefully if calibration hasn't been
        run yet.
 
        Interval is computed by applying the fixed log1p-scale
        conformal quantile symmetrically around the point prediction,
        then mapping back with expm1 -- matching exactly how the
        quantile was derived in the calibration notebook (same log1p
        transform, same symmetric application).
        """
        point_prediction = self.predict_price(features_df)
 
        if self.conformal_quantile is None:
            return point_prediction, None, None
 
        log_pred = np.log1p(point_prediction)
        lower = float(np.expm1(log_pred - self.conformal_quantile))
        upper = float(np.expm1(log_pred + self.conformal_quantile))
        return point_prediction, lower, upper


def load_model_registry() -> ModelRegistry:
    """Loads all artifacts from disk. Call this ONCE at FastAPI
    startup (e.g. in a lifespan handler / on_event("startup")), and
    store the returned ModelRegistry on app.state -- do not call this
    per-request.
    """
    # --- TF-IDF vectorizers ---
    vectorizer = TextVectorizer.load()
    title_vectorizer = vectorizer.title_vectorizer
    desc_vectorizer = vectorizer.desc_vectorizer
 
    title_input_dim = len(title_vectorizer.vocabulary_)
    desc_input_dim = len(desc_vectorizer.vocabulary_)
 
    # --- Classifier ---
    device = torch.device(DEVICE)
    classifier = RiskClassifier(
        title_input_dim=title_input_dim,
        desc_input_dim=desc_input_dim,
        branch_hidden_dim=BRANCH_HIDDEN_DIM,
        head_hidden_dim=HEAD_HIDDEN_DIM,
        dropout_rate=DROPOUT_RATE,
    )
    checkpoint = torch.load(MODEL_CLASSIFICATION_PATH, map_location=device)
    # checkpoint may be a raw state_dict or a dict wrapping it under
    # "model_state_dict" (the training code in Alvin's original
    # message saves it the latter way) -- handle both.
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    classifier.load_state_dict(state_dict)
    classifier.to(device)
    classifier.eval()
 
    # --- Regression pipeline (LightGBM wrapped in sklearn Pipeline) ---
    regression_pipeline = joblib.load(MODEL_REGRESSION_PATH)
 
    # --- Conformal prediction artifact (optional) ---
    # Missing file is NOT fatal -- the API should still serve point
    # predictions even if calibration hasn't been run yet. Any other
    # load error (corrupted file, wrong format) IS surfaced, since
    # that suggests something genuinely wrong rather than "not yet
    # created".
    conformal_quantile = None
    conformal_confidence_level = None

    if CONFORMAL_ARTIFACT_PATH.exists():
        conformal_artifact = joblib.load(CONFORMAL_ARTIFACT_PATH)
        conformal_quantile = conformal_artifact["quantile"]
        conformal_confidence_level = conformal_artifact["confidence_level"]
 
    return ModelRegistry(
        classifier=classifier,
        title_vectorizer=title_vectorizer,
        desc_vectorizer=desc_vectorizer,
        regression_pipeline=regression_pipeline,
        device=device,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        boolean_features=boolean_features,
        conformal_quantile=conformal_quantile,
        conformal_confidence_level=conformal_confidence_level,
    )