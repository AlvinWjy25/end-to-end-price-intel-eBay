"""
test_staging_marts_sql.py

Manual smoke test: run this against your real Postgres instance to
validate staging_marts_sql.py before wiring it into FastAPI.

Usage:
    python test_staging_marts_sql.py

Requires .env with DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
already set (same as the rest of the project).
"""

import os
import sys
from pathlib import Path

# Adjust this import path to match your actual project layout
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT_DIR = Path(__file__).resolve().parents[2]
from dotenv import load_dotenv
load_dotenv(ROOT_DIR / "config" / ".env")

import psycopg2
from db import get_connection, store_and_engineer_features
from archive.logging import setup_logger

logger = setup_logger('test_staging_marts_run')

TEST_ITEM = {
    "item_id": "v1|198566223721|0",
    "title": "Re:ZERO Starting Life in Another World Light Novel Lot, English PB, Illustrated",
    "price": "220.00",
    "currency": "USD",
    "condition": "Good",
    "seller_location": "US",
    "description": "Whole set of 1-20 plus Ex 1-2 All very good condition except volume 8 with some discoloring on the spine.",
    "localized_aspects": [
        {"type": "STRING", "name": "Book Title", "value": "Re:ZERO -Starting Life in Another World-"},
        {"type": "STRING", "name": "Book Series", "value": "Re:ZERO -Starting Life in Another World-"},
        {"type": "STRING", "name": "Narrative Type", "value": "Fiction"},
        {"type": "STRING", "name": "Publisher", "value": "Yen Press"},
    ],
}

EXPECTED = {
    "is_boxset": True,
    "volume_confidence": "low",  # after the 2026-09 patch
    "risk_category": "Low Risk",  # condition="Good" present, no risk keywords
}


def main():
    logger.info(f"Connecting to Postgres...")
    with get_connection() as conn:
        logger.info(f"Connected. Running insert + staging/marts transform...")
        try:
            features = store_and_engineer_features(conn, TEST_ITEM)
        except psycopg2.Error as e:
            logger.info(f"\nFAILED with a Postgres error:\n{e}")
            logger.info(f"\nThis likely means a SQL syntax issue in staging_marts_sql.py.")
            logger.info(f"Check the error message above for the exact line/construct.")
            sys.exit(1)

    logger.info(f"\n--- Engineered features ---")
    for k, v in sorted(features.items()):
        logger.info(f"  {k}: {v}")

    logger.info(f"\n--- Sanity checks ---")
    failures = 0
    for key, expected_value in EXPECTED.items():
        got = features.get(key)
        status = "OK" if got == expected_value else "MISMATCH"
        if status == "MISMATCH":
            failures += 1
        logger.info(f"  [{status}] {key}: expected={expected_value}, got={got}")

    logger.info(f"\n{'PASSED' if failures == 0 else f'{failures} CHECK(S) FAILED'}")
    sys.exit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()