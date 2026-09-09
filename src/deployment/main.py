"""
main.py

FastAPI service exposing POST /predict: paste an eBay listing URL,
get back a risk classification and (if the listing passes the risk
gate) a price prediction.

Pipeline:
    1. Parse item_id from the URL (url_parser.py)
    2. Fetch item detail from eBay Browse API (reusing ingest.py's
       get_access_token / get_item_detail / parse_item)
    3. Insert into raw.prediction_requests + run the same
       staging+intermediate SQL transform used for training data
       (staging_marts_sql.py via db.py) -- guarantees feature parity
    4. Run RiskClassifier (TF-IDF + MLP) for risk_category
    5. Gate: High Risk blocks the price prediction by default, unless
       override_risk_gate=true is passed (with a disclaimer attached)
    6. Assemble regressor features (assembly.py, using encoders.py
       for condition_encoded / volume_tier_encoded /
       seller_location_grouped) and run the regression pipeline

All ML/DB artifacts are loaded ONCE at startup via ModelRegistry
(model_registry.py) and stored on app.state -- not reloaded per
request.

Sync (not async) throughout, per project decision (2026-09): this is
a low-traffic portfolio-project endpoint, the eBay client already
uses `requests` (sync), and FastAPI's threadpool is sufficient at
this scale. See prior discussion for the async migration path if
traffic ever demands it.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.deployment.url_parser import parse_item_id, UnsupportedEbayUrlError
from src.deployment.ebay_token_manager import EbayTokenManager
from src.deployment.legacy_id_resolver import resolve_legacy_item_id
from src.pipeline.db import get_connection, store_and_engineer_features, insert_prediction_result
from src.pipeline.model_registry import load_model_registry
from src.pipeline.assembly import build_classifier_inputs, build_regressor_dataframe
from src.pipeline.encoders import is_encoding_configured

# Reused directly from the existing ingestion module -- see
# project note: same eBay client, no separate re-implementation.
from src.ingestion.ingest import (  # adjust import path to match actual package layout
    create_retry_session,
    get_item_detail,
    parse_item,
)


logger = logging.getLogger("price_intel_api")
logging.basicConfig(level=logging.INFO)

# Free-text tag identifying which model artifacts produced a given
# prediction (written into raw.prediction_results.model_version).
# Bump this manually whenever final_classification.pth or
# final_regression.joblib is retrained/replaced, so past predictions
# stay traceable to the artifacts that generated them.
MODEL_VERSION = "classifier_v1+regressor_v1+conformal_90"


# ============================================================
# Startup / shutdown: load models + eBay session once
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not is_encoding_configured():
        raise RuntimeError(
            "encoders.CONDITION_ENCODING is empty -- fill it in before "
            "starting the API (see src/pipeline/encoders.py). Refusing "
            "to start rather than silently serving predictions built on "
            "unencoded features."
        )

    logger.info("Loading model registry (classifier, vectorizers, regression pipeline)...")
    app.state.registry = load_model_registry()
    logger.info("Model registry loaded.")

    logger.info("Setting up eBay token manager...")
    app.state.ebay_session = create_retry_session()
    app.state.ebay_token_manager = EbayTokenManager()
    # Fetch the first token now so startup fails fast if credentials
    # are wrong, rather than on the first incoming request.
    app.state.ebay_token_manager.get_token()
    logger.info("eBay access token obtained.")

    yield

    # No explicit teardown needed: DB connections are opened/closed
    # per-request (see db.get_connection), and the eBay session/token
    # don't hold resources that need explicit release.


app = FastAPI(title="Light Novel Price Intelligence API", lifespan=lifespan)


# ============================================================
# Request / response schemas
# ============================================================

class PredictRequest(BaseModel):
    ebay_url: str = Field(..., description="Full eBay listing URL, e.g. https://www.ebay.com/itm/358790904803")
    override_risk_gate: bool = Field(
        False,
        description="If true, still attempt a price prediction even when the "
                    "classifier flags this listing as High Risk. The response "
                    "will include a reliability disclaimer.",
    )


class RiskAssessment(BaseModel):
    risk_category_model: str          # classifier's own prediction
    risk_category_rule_based: str     # from staging_marts_sql.py (text_risk_score_v2 rule)
    classifier_confidence: float      # sigmoid output, 0-1 (probability of High Risk)
    text_risk_score: int
    price_risk_score: int
    total_risk_score: int


class ListingFeatures(BaseModel):
    title: str
    condition: Optional[str]
    price: Optional[float]
    currency: Optional[str]
    seller_location: Optional[str]
    is_boxset: bool
    volume_number: Optional[float]
    volume_number_end: Optional[float]
    volume_confidence: str
    volume_count: int
    price_per_volume: Optional[float]
    is_ambigous_bulk_pricing: bool
    is_special_edition: bool
    is_first_print: bool
    has_signature: bool
    has_merch: bool
    has_paper_extra: bool
    total_bonus_count: int
    is_out_of_scope: Optional[bool] = None
    out_of_scope_reason: Optional[str] = None


class PredictResponse(BaseModel):
    item_id: str
    legacy_item_id: str
    ebay_url: str
    risk: RiskAssessment
    features: ListingFeatures
    predicted_price: Optional[float] = None
    price_interval_lower: Optional[float] = None
    price_interval_upper: Optional[float] = None
    price_interval_confidence: Optional[float] = None
    prediction_blocked: bool = False
    prediction_disclaimer: Optional[str] = None
    message: str


# ============================================================
# Endpoint
# ============================================================

def _write_prediction_result(
    request_id: str,
    risk: RiskAssessment,
    prediction_blocked: bool,
    override_risk_gate_used: bool,
    predicted_price: Optional[float],
    lower: Optional[float],
    upper: Optional[float],
) -> None:
    """Writes the model's output back to raw.prediction_results.

    Deliberately swallows and logs (rather than raises) any failure
    here: a logging/audit-trail write failing should not prevent the
    user from receiving a prediction they already waited for. This is
    an intentional trade-off -- it means prediction_results can in
    principle miss rows if the DB write fails, but that's preferable
    to a 500 error on an otherwise-successful prediction.
    """
    try:
        with get_connection() as conn:
            insert_prediction_result(conn, {
                "request_id": request_id,
                "model_version": MODEL_VERSION,
                "risk_category_model": risk.risk_category_model,
                "risk_category_rule_based": risk.risk_category_rule_based,
                "classifier_confidence": risk.classifier_confidence,
                "text_risk_score": risk.text_risk_score,
                "price_risk_score": risk.price_risk_score,
                "total_risk_score": risk.total_risk_score,
                "prediction_blocked": prediction_blocked,
                "override_risk_gate_used": override_risk_gate_used,
                "predicted_price": predicted_price,
                "price_interval_lower": lower,
                "price_interval_upper": upper,
                "price_interval_confidence": (upper is not None and lower is not None
                                               and app.state.registry.conformal_confidence_level) or None,
            })
    except Exception:
        logger.exception(
            "Failed to write prediction_results row for request_id=%s "
            "(prediction still returned to caller successfully)", request_id,
        )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    registry = app.state.registry

    # --- 1. Parse item_id from URL ---
    try:
        legacy_item_id = parse_item_id(request.ebay_url)
    except UnsupportedEbayUrlError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # --- 2. Fetch from eBay Browse API ---
    ebay_token = app.state.ebay_token_manager.get_token()

    # url_parser extracts the LEGACY item ID (the plain 12-digit
    # number visible in a listing's browser URL). getItem, however,
    # requires the RESTful item ID format ("v1|{legacy_id}|0"), so
    # that conversion happens first via get_item_by_legacy_id. This
    # step is unique to the URL-based /predict flow -- ingest.py's
    # batch search flow never hits this, since Browse API search
    # results already return RESTful IDs directly.
    item_id = resolve_legacy_item_id(app.state.ebay_session, ebay_token, legacy_item_id)
    if item_id is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Could not resolve eBay legacy item ID {legacy_item_id} to a "
                "RESTful item ID (the listing may be delisted, invalid, or "
                "there was an eBay API error)."
            ),
        )

    raw_item = get_item_detail(app.state.ebay_session, ebay_token, item_id)
    if raw_item is None:
        raise HTTPException(
            status_code=404,
            detail=f"eBay item {item_id} could not be retrieved (delisted, invalid ID, or API error).",
        )

    parsed_tuple = parse_item(raw_item)
    # parse_item() returns a tuple shaped for direct DB insertion
    # (see ingest.py); rebuild it as a dict here for downstream code.
    item_dict = {
        "item_id": parsed_tuple[0],
        "title": parsed_tuple[1],
        "price": parsed_tuple[2],
        "currency": parsed_tuple[3],
        "condition": parsed_tuple[4],
        "seller_location": parsed_tuple[5],
        "description": parsed_tuple[6],
        # parsed_tuple[7] is a psycopg2.extras.Json wrapper (from
        # ingest.py's parse_item) -- unwrap back to a plain list so
        # db.insert_prediction_request can re-wrap it itself.
        "localized_aspects": parsed_tuple[7].adapted if parsed_tuple[7] else [],
    }

    # --- 3. Insert + run staging/intermediate SQL transform ---
    try:
        with get_connection() as conn:
            features = store_and_engineer_features(conn, item_dict)
    except Exception as e:
        logger.exception("Feature engineering (DB/SQL) step failed")
        raise HTTPException(status_code=500, detail=f"Feature engineering failed: {e}")

    # --- 4. Classifier ---
    title_vec, desc_vec = build_classifier_inputs(features, registry)
    logits = registry.classify(title_vec, desc_vec)
    probability_high_risk = torch.sigmoid(logits).item()
    risk_category_model = "High Risk" if probability_high_risk >= 0.5 else "Low Risk"

    risk = RiskAssessment(
        risk_category_model=risk_category_model,
        risk_category_rule_based=features["risk_category"],
        classifier_confidence=probability_high_risk,
        text_risk_score=features["text_risk_score"],
        price_risk_score=features["price_risk_score"],
        total_risk_score=features["total_risk_score"],
    )

    listing_features = ListingFeatures(
        title=features["title"],
        condition=features["condition"],
        price=features["price"],
        currency=features["currency"],
        seller_location=features["seller_location"],
        is_boxset=features["is_boxset"],
        volume_number=features["volume_number"],
        volume_number_end=features["volume_number_end"],
        volume_confidence=features["volume_confidence"],
        volume_count=features["volume_count"],
        price_per_volume=features["price_per_volume"],
        is_ambigous_bulk_pricing=features["is_ambigous_bulk_pricing"],
        is_special_edition=features["is_special_edition"],
        is_first_print=features["is_first_print"],
        has_signature=features["has_signature"],
        has_merch=features["has_merch"],
        has_paper_extra=features["has_paper_extra"],
        total_bonus_count=features["total_bonus_count"],
        is_out_of_scope=features.get("is_out_of_scope"),
        out_of_scope_reason=features.get("out_of_scope_reason"),
    )

    # --- 5. Gate ---
    if risk_category_model == "High Risk" and not request.override_risk_gate:
        _write_prediction_result(
            request_id=features["request_id"],
            risk=risk,
            prediction_blocked=True,
            override_risk_gate_used=request.override_risk_gate,
            predicted_price=None,
            lower=None,
            upper=None,
        )
        return PredictResponse(
            item_id=item_id,
            legacy_item_id=legacy_item_id,
            ebay_url=request.ebay_url,
            risk=risk,
            features=listing_features,
            predicted_price=None,
            prediction_blocked=True,
            prediction_disclaimer=None,
            message=(
                "This listing was classified as High Risk (likely unofficial/reprint) "
                "and no price prediction was made. If you believe this is a "
                "misclassification, retry with override_risk_gate=true to get a "
                "prediction anyway (results may be less reliable for High Risk items)."
            ),
        )

    # --- 6. Regressor (+ conformal interval, if calibration artifact loaded) ---
    try:
        regressor_df = build_regressor_dataframe(features, registry)
        predicted_price, lower, upper = registry.predict_price_with_interval(regressor_df)
    except ValueError as e:
        # Missing/unencodable feature -- explicit failure rather than
        # a fabricated prediction.
        raise HTTPException(status_code=422, detail=str(e))

    disclaimer = None
    if risk_category_model == "High Risk" and request.override_risk_gate:
        disclaimer = (
            "This listing was flagged High Risk by the classifier. The price "
            "prediction below was generated at your request despite that flag "
            "and may be unreliable -- the regressor was trained primarily on "
            "Low Risk listings."
        )

    message = "Prediction successful."
    if lower is None:
        # Conformal artifact wasn't found at startup -- degrade
        # gracefully to a point-prediction-only response rather than
        # failing the whole request.
        message += " (No calibrated interval available -- point estimate only.)"

    _write_prediction_result(
        request_id=features["request_id"],
        risk=risk,
        prediction_blocked=False,
        override_risk_gate_used=request.override_risk_gate,
        predicted_price=predicted_price,
        lower=lower,
        upper=upper,
    )

    return PredictResponse(
        item_id=item_id,
        legacy_item_id=legacy_item_id,
        ebay_url=request.ebay_url,
        risk=risk,
        features=listing_features,
        predicted_price=round(predicted_price, 2),
        price_interval_lower=round(lower, 2) if lower is not None else None,
        price_interval_upper=round(upper, 2) if upper is not None else None,
        price_interval_confidence=registry.conformal_confidence_level,
        prediction_blocked=False,
        prediction_disclaimer=disclaimer,
        message=message,
    )


@app.get("/health")
def health():
    return {"status": "ok"}