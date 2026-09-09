"""
db.py

psycopg2 (sync) connection helper for the FastAPI /predict endpoint.

Responsibilities:
  - open/reuse a Postgres connection
  - insert one row into raw.prediction_requests
  - run the staging+marts transformation SQL (staging_marts_sql.py)
    against that row and return the engineered feature dict

This module deliberately knows nothing about eBay API parsing, the
ML models, or the FastAPI request/response schema -- it is a thin,
single-purpose data-access layer.
"""

from __future__ import annotations

import os
from pathlib import Path
from contextlib import contextmanager
from typing import Any, Iterator

import psycopg2
import psycopg2.extras

ROOT_DIR = Path(__file__).resolve().parents[2]
from .staging_marts_sql import get_engineered_features

# ============================================================
# Connection
# ============================================================

def _get_dsn() -> str:
    """Reads connection params from environment variables.

    Expects the project's existing .env keys:
      DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
    """
    from dotenv import load_dotenv
    load_dotenv(ROOT_DIR / "config" / ".env")

    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT")
    dbname = os.getenv("DB_NAME")
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")

    return (
        f"host={host} port={port} dbname={dbname} "
        f"user={user} password={password}"
    )


@contextmanager
def get_connection() -> Iterator["psycopg2.extensions.connection"]:
    """Yields a psycopg2 connection, closing it on exit.

    NOTE: opens a fresh connection per call. For a low-traffic
    portfolio-project endpoint this is fine; if this becomes a
    bottleneck, swap in a psycopg2.pool.SimpleConnectionPool without
    changing any calling code (this function's signature stays the
    same).
    """
    conn = psycopg2.connect(_get_dsn())
    try:
        yield conn
    finally:
        conn.close()


# ============================================================
# Insert
# ============================================================

_INSERT_SQL = """
    insert into raw.prediction_requests
        (item_id, title, price, currency, condition, seller_location,
        description, localized_aspects)
    values
        (%(item_id)s, %(title)s, %(price)s, %(currency)s, %(condition)s,
        %(seller_location)s, %(description)s, %(localized_aspects)s)
    returning request_id;
"""


def insert_prediction_request(
    conn: "psycopg2.extensions.connection",
    item: dict[str, Any],
) -> str:
    """Inserts one raw eBay item (shape matching ingestion's
    parse_item() output) into raw.prediction_requests and returns
    the generated request_id.

    Expects `item` to have keys: item_id, title, price, currency,
    condition, seller_location, description, localized_aspects
    (the last one a Python list[dict], NOT pre-serialized -- psycopg2
    adapts it via Json()).
    """
    payload = {
        "item_id": item.get("item_id"),
        "title": item.get("title"),
        "price": item.get("price"),
        "currency": item.get("currency"),
        "condition": item.get("condition"),
        "seller_location": item.get("seller_location"),
        "description": item.get("description"),
        "localized_aspects": psycopg2.extras.Json(item.get("localized_aspects") or []),
    }

    with conn.cursor() as cur:
        cur.execute(_INSERT_SQL, payload)
        request_id = cur.fetchone()[0]
    conn.commit()

    return str(request_id)


# ============================================================
# Insert + transform in one call (what the FastAPI endpoint uses)
# ============================================================

def store_and_engineer_features(
    conn: "psycopg2.extensions.connection",
    item: dict[str, Any],
) -> dict[str, Any]:
    """Inserts the raw item, then runs the staging+marts SQL against
    it, returning the fully engineered feature row.

    Raises RuntimeError if the row was inserted but the subsequent
    fetch found nothing (should not happen under normal operation --
    would indicate a request_id mismatch or a transaction visibility
    issue).
    """
    request_id = insert_prediction_request(conn, item)

    features = get_engineered_features(
        conn, request_id=request_id, source_table="raw.prediction_requests",
    )

    if features is None:
        raise RuntimeError(
            f"Inserted prediction_requests row (request_id={request_id}) "
            "but the feature-engineering query returned no matching row. "
            "This should not happen -- check for a transaction visibility "
            "issue or a request_id type mismatch."
        )

    features["request_id"] = request_id
    return features


_INSERT_RESULT_SQL = """
    insert into raw.prediction_results
        (request_id, model_version, risk_category_model, risk_category_rule_based,
        classifier_confidence, text_risk_score, price_risk_score, total_risk_score,
        prediction_blocked, override_risk_gate_used,
        predicted_price, price_interval_lower, price_interval_upper, price_interval_confidence)
    values
        (%(request_id)s, %(model_version)s, %(risk_category_model)s, %(risk_category_rule_based)s,
        %(classifier_confidence)s, %(text_risk_score)s, %(price_risk_score)s, %(total_risk_score)s,
        %(prediction_blocked)s, %(override_risk_gate_used)s,
        %(predicted_price)s, %(price_interval_lower)s, %(price_interval_upper)s, %(price_interval_confidence)s)
    returning result_id;
"""
 
 
def insert_prediction_result(
    conn: "psycopg2.extensions.connection",
    result: dict[str, Any],
) -> str:
    """Writes back the model's output for a given request_id into
    raw.prediction_results, and returns the generated result_id.
 
    Expects `result` to have keys matching the columns above
    (predicted_price / price_interval_* may be None when
    prediction_blocked=True). model_version is a free-text tag (e.g.
    a git commit hash or "classifier_v1+regressor_v1") identifying
    which model artifacts produced this result -- see the migration
    file for rationale.
 
    Call this AFTER store_and_engineer_features() and after running
    the models -- this is a separate write, not part of that
    transaction, since the whole point of the two-table split is that
    a request can exist with zero, one, or many results over time.
    """
    with conn.cursor() as cur:
        cur.execute(_INSERT_RESULT_SQL, result)
        result_id = cur.fetchone()[0]
    conn.commit()
 
    return str(result_id)