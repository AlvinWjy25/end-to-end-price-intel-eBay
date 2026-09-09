"""
assembly.py

Takes the engineered feature dict (from staging_marts_sql.py, i.e.
one row of int_ebay_listing_risk_analysis-equivalent output) and
assembles it into the two shapes the models actually need:

  - build_classifier_inputs(): title_vec, desc_vec tensors for
    RiskClassifier.forward()
  - build_regressor_dataframe(): a single-row pandas DataFrame with
    exactly config_script.py's feature_cols, for
    regression_pipeline.predict()

Kept separate from the FastAPI route handlers so both can be unit
tested without spinning up the app or hitting the DB/eBay API.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import torch

from src.pipeline.encoders import (
    encode_condition,
    encode_volume_tier,
    group_seller_location,
)
from src.pipeline.model_registry import ModelRegistry


def build_classifier_inputs(
    features: dict[str, Any],
    registry: ModelRegistry,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transforms title/description text through the fitted TF-IDF
    vectorizers and returns dense tensors shaped (1, vocab_size),
    ready for RiskClassifier.forward(title_vec, desc_vec).

    NOTE: title/description are transformed here, NOT re-derived from
    features['title'] via any extra cleaning -- staging_marts_sql.py
    already applies the same trim()/cleaning the training pipeline
    used, so no additional preprocessing should happen here (doing so
    would risk yet another training-serving mismatch).
    """
    title = features.get("title") or ""
    description = features.get("description") or ""

    title_sparse = registry.title_vectorizer.transform([title])
    desc_sparse = registry.desc_vectorizer.transform([description])

    title_tensor = torch.tensor(title_sparse.toarray(), dtype=torch.float32).to(registry.device)
    desc_tensor = torch.tensor(desc_sparse.toarray(), dtype=torch.float32).to(registry.device)

    return title_tensor, desc_tensor


def build_regressor_dataframe(
    features: dict[str, Any],
    registry: ModelRegistry,
) -> pd.DataFrame:
    """Assembles a single-row DataFrame with exactly the columns in
    config_script.py's feature_cols (numeric + categorical +
    boolean), applying the manual encoders (condition_encoded,
    volume_tier_encoded, seller_location_grouped) that live outside
    the saved regression Pipeline.

    Raises ValueError if any required column ends up None -- this is
    intentional: silently feeding the regressor a None/NaN for a
    required feature would produce a prediction that looks valid but
    is built on a missing signal. The caller should catch this and
    turn it into an explicit "cannot price this item" response rather
    than a misleading number.
    """
    row: dict[str, Any] = {}

    # --- numeric_features ---
    # title_length, title_word_count, volume_count, text_risk_score,
    # total_bonus_count come straight from the SQL output.
    for col in ("title_length", "title_word_count", "volume_count",
                "text_risk_score", "total_bonus_count"):
        row[col] = features.get(col)

    row["condition_encoded"] = encode_condition(features.get("condition"))
    row["volume_tier_encoded"] = encode_volume_tier(features.get("volume_count"))

    # --- categorical_features ---
    row["currency"] = features.get("currency")
    row["seller_location_grouped"] = group_seller_location(features.get("seller_location"))

    # --- boolean_features ---
    for col in ("is_boxset", "is_special_edition", "boxset_side_story_edition_included",
                "standalone_side_story_edition", "is_first_print", "has_signature",
                "has_merch", "has_paper_extra"):
        row[col] = features.get(col)

    all_cols = registry.numeric_features + registry.categorical_features + registry.boolean_features
    missing = [c for c in all_cols if row.get(c) is None]
    if missing:
        raise ValueError(
            f"Cannot build regressor input -- missing/unencodable feature(s): {missing}. "
            "This usually means an unrecognized 'condition' value, a missing "
            "currency/seller_location from the eBay API response, or a gap "
            "in staging_marts_sql.py's output for this item."
        )

    df = pd.DataFrame([row], columns=all_cols)
    return df
