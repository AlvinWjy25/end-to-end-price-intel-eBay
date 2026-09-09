from __future__ import annotations

from pathlib import Path

import psycopg2
import psycopg2.extras


_DBT_PROJECT_DIR = (
    Path(__file__).resolve().parents[2] / "price_intel_dbt" / "models"
)

_STAGING_SQL_PATH = _DBT_PROJECT_DIR / "staging" / "stg_ebay_listings.sql"
_INTERMEDIATE_SQL_PATH = (
    _DBT_PROJECT_DIR / "intermediate" / "int_ebay_listing_risk_analysis.sql"
)

def _load_staging_sql() -> str:
    """Reads stg_ebay_listings.sql, swapping the dbt source() macro
    for a plain {source_table} placeholder."""
    raw = _STAGING_SQL_PATH.read_text(encoding="utf-8")
    return raw.replace("{{ source('raw', 'ebay_listings') }}", "{source_table}")

def _load_intermediate_sql(staging_cte_name: str = "staged_listing") -> str:
    """Reads int_ebay_listing_risk_analysis.sql, swapping the dbt
    ref() macro for a reference to the staging CTE it will be
    chained after."""
    raw = _INTERMEDIATE_SQL_PATH.read_text(encoding="utf-8")
    return raw.replace(
        "{{ ref('stg_ebay_listings') }}", staging_cte_name,
    )

def _build_transform_sql() -> str:
    """Stitches staging + intermediate into a single query:
 
        with staged_listing as ( <stg_ebay_listings.sql body> )
        <int_ebay_listing_risk_analysis.sql, with its `source_data`
         CTE now selecting from staged_listing instead of a dbt ref>
 
    int_ebay_listing_risk_analysis.sql's own first CTE is:
        with source_data as (
            select * from {{ ref('stg_ebay_listings') }}
        ),
        ...
    After the ref() swap this becomes `select * from staged_listing`,
    so we wrap the staging query as a CTE named `staged_listing` and
    prepend it, then append the (already-`with`-prefixed)
    intermediate query as a continuation of the same CTE chain.
    """
    staging_sql = _load_staging_sql().strip().rstrip(";")
    intermediate_sql = _load_intermediate_sql(staging_cte_name="staged_listing").strip().rstrip(";")
 
    if not intermediate_sql.lower().startswith("with"):
        raise ValueError(
            "Expected int_ebay_listing_risk_analysis.sql to start with "
            "a WITH clause so it can be chained after the staging CTE. "
            "Update _build_transform_sql() if the file's structure changed."
        )
 
    # Turn "with source_data as (...)" into a continuation of our own
    # outer WITH chain: "<outer_with>, source_data as (...)"
    intermediate_body = intermediate_sql[len("with"):].lstrip()
 
    return (
        f"with staged_listing as (\n{staging_sql}\n),\n"
        f"{intermediate_body}"
    )

_TRANSFORM_SQL_TEMPLATE = _build_transform_sql()

def get_engineered_features(
        conn: "psycopg2.extensions.connection",
        request_id: str,
        source_table: str = "raw.prediction_requests",
    ) -> dict | None:

    """Runs the staging+intermediate transformation SQL against a
    single row (identified by request_id) in `source_table`, and
    returns the fully engineered feature row as a dict.
 
    Returns None if no matching row was found (e.g. insert failed
    silently, or wrong request_id).
 
    NOTE: does not compute condition_encoded, volume_tier_encoded, or
    seller_location_grouped -- those encoders live in the training
    notebook only. Caller must apply the saved joblib encoders to the
    relevant raw columns (condition, volume_confidence,
    seller_location) returned here.
    """

    escaped_template = _TRANSFORM_SQL_TEMPLATE.replace("%", "%%")
    
    query = escaped_template.replace(
        "{source_table}",
        # Filter to the single target row BEFORE the staging SQL runs,
        # not after -- stg_ebay_listings.sql does an explicit column
        # `select` (not `select *`) in its `cleaned` CTE, which drops
        # `request_id` (a prediction_requests-only column) before it
        # would reach any filter placed after the chained query.
        # Pre-filtering here means the staging/intermediate SQL only
        # ever sees the one row we care about, and we don't depend on
        # request_id surviving the dbt column list.
        f"(select * from {source_table} where request_id = %(request_id)s) as request_scoped_source",
    )
 
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(query, {"request_id": request_id})
        row = cur.fetchone()
 
    return dict(row) if row is not None else None
