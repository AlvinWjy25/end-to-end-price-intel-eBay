"""
encoders.py

Manual categorical/numeric encoders for features that are NOT wrapped
inside the saved regression Pipeline (final_regression.joblib). The
saved Pipeline only wraps a ColumnTransformer (OneHotEncoder) for
`currency` / `seller_location_grouped` + RobustScaler +
TransformedTargetRegressor -- it does NOT encode `condition` or
bucket `volume_count` into `volume_tier_encoded`. Those are handled
manually here, matching whatever logic was used in the training
notebook.

USER: fill in CONDITION_ENCODING below to match EXACTLY what was
used at training time (check the notebook -- likely a LabelEncoder or
a manual dict). If the mapping here doesn't match training, the
regressor will silently receive wrong-but-valid looking numbers and
produce plausible-but-wrong predictions -- this kind of mismatch
won't raise an error, so it's worth double-checking against the
notebook's actual encoder output (e.g. print
`condition_encoder.classes_` or equivalent) rather than guessing.

encode_volume_tier() and group_seller_location() are NOT
placeholders -- both are already fully specified (confirmed by
Alvin, 2026-09) and require no further editing.
"""

from __future__ import annotations

from typing import Optional

CONDITION_ENCODING: dict[str, int] = {
    'Acceptable': 1,
    'Used': 3,
    'Good': 3,
    'Very Good': 4,
    'Like New': 5,
    'New': 6,
    'Brand New': 7,
    'unknown': 3  
}

# ============================================================
# Lookup functions
# ============================================================

def encode_condition(condition: Optional[str]) -> Optional[int]:
    """Returns the encoded value for `condition`, or None if the
    value is missing or wasn't seen during training (CONDITION_ENCODING
    not yet filled in, or a genuinely unseen category).

    Caller (the FastAPI endpoint) should treat a None return as a
    signal to either reject the prediction or flag it as
    low-confidence -- silently defaulting to some arbitrary encoded
    value would feed the regressor a fabricated feature.
    """
    if condition is None:
        return None
    return CONDITION_ENCODING.get(condition)


def encode_volume_tier(volume_count: Optional[int]) -> Optional[int]:
    """Buckets volume_count into a tier, matching the exact thresholds
    used at training time:

        volume_count == 1        -> 1
        2 <= volume_count <= 5   -> 2
        6 <= volume_count <= 15  -> 3
        volume_count >= 16       -> 4

    NOTE: despite the "tier" naming overlap with volume_confidence
    ("low"/"medium"/"high", the extraction-reliability signal from
    int_ebay_listing_risk_analysis.sql), this is a DIFFERENT feature
    -- a bucketing of volume_count (how many volumes the listing
    contains), not of extraction confidence. Confirmed by Alvin,
    2026-09. volume_confidence is not fed into the regressor at all;
    only this count-based tier is (see config_script.py's
    numeric_features list).

    Returns None if volume_count is missing, so the caller can decide
    whether to reject/flag the prediction rather than silently
    bucketing a null count.
    """
    if volume_count is None:
        return None
    if volume_count == 1:
        return 1
    elif 2 <= volume_count <= 5:
        return 2
    elif 6 <= volume_count <= 15:
        return 3
    else:
        return 4


# ============================================================
# seller_location_grouped
# ============================================================
#
# NOT a placeholder -- this is a fixed lookup table, not a per-request
# computation. In training, seller_location_grouped was derived via a
# threshold on the full training set's value_counts():
#
#   location_counts = df['seller_location'].value_counts()
#   rare_locations = location_counts[location_counts < 30].index
#   df['seller_location_grouped'] = df['seller_location'].replace(rare_locations, 'Other')
#
# That threshold can't be recomputed from a single real-time row (a
# single row has no distribution to threshold against), so the
# resulting category assignment is hardcoded here instead, based on
# the training set's actual value_counts (confirmed by Alvin,
# 2026-09):
#
#   US 1153  JP 268  MY 71  GB 45  AU 42   <- all >= 30, kept as-is
#   CA 3     DE 1                          <- < 30, grouped to "Other"
#
# If the training data is refreshed and re-thresholded later, this
# set must be regenerated to match (otherwise a location that used to
# map to itself could silently start mapping to "Other", or vice
# versa, without any error being raised).
_SELLER_LOCATION_KEPT_AS_IS = {"US", "JP", "MY", "GB", "AU"}


def group_seller_location(seller_location: Optional[str]) -> Optional[str]:
    """Maps a raw seller_location (e.g. "US") to seller_location_grouped
    ("US" or "Other"), matching the >=30-count threshold used at
    training time. Returns None if seller_location itself is missing
    (distinct from "Other", which means "present but rare/unseen").
    """
    if seller_location is None:
        return None
    if seller_location in _SELLER_LOCATION_KEPT_AS_IS:
        return seller_location
    return "Other"


def is_encoding_configured() -> bool:
    """Sanity check the FastAPI startup can call to fail fast if
    User forgot to fill in CONDITION_ENCODING above, rather than
    silently serving predictions built on None-valued features.
    (encode_volume_tier() and group_seller_location() need no such
    check -- both are fixed logic/lookup tables, already complete.)
    """
    return bool(CONDITION_ENCODING)