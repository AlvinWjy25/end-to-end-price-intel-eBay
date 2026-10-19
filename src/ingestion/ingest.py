"""
ingest.py
Fetch listing data from eBay Browse API, then insert to Postgres (schema: raw).

Process:
1. Authenticate to eBay using OAuth 2.0 Client Credentials flow -> get access_token
2. Call item_summary/search endpoint with keyword -> get list of item_id
3. For each item_id, call getItem endpoint -> get full description (not just shortDescription)
   This is important because disclaimers such as "NOT Original / reprint edition" are usually found
   in the free description written by the seller, not in the auto-generated shortDescription.
4. Parse & strip HTML from description -> get relevant field
5. Insert to tabel raw.ebay_listings
"""

import argparse
import os
import json
import time
import requests
import psycopg2
from datetime import datetime
from psycopg2.extras import Json
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from datetime import datetime
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------
# 1. Load credentials from .env
# ---------------------------------------------------------
load_dotenv(ROOT_DIR / "config" / ".env")

EBAY_CLIENT_ID = os.getenv("EBAY_CLIENT_ID")
EBAY_CLIENT_SECRET = os.getenv("EBAY_CLIENT_SECRET")

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "price_intelligence")
DB_USER = os.getenv("DB_USER", "alvin")
DB_PASSWORD = os.getenv("DB_PASSWORD")

EBAY_OAUTH_URL = "https://api.ebay.com/identity/v1/oauth2/token"
EBAY_SEARCH_URL = "https://api.ebay.com/buy/browse/v1/item_summary/search"
EBAY_ITEM_URL = "https://api.ebay.com/buy/browse/v1/item"


# ---------------------------------------------------------
# 2. Get OAuth access token
# ---------------------------------------------------------
def get_access_token() -> str:
    """
    eBay Browse API uses OAuth 2.0 Client Credentials flow.
    This token is temporary (usually ~2 hours), so it is generated
    anew each time the script is run -- no need to store permanently.
    """
    if not EBAY_CLIENT_ID or not EBAY_CLIENT_SECRET:
        raise ValueError(
            "EBAY_CLIENT_ID / EBAY_CLIENT_SECRET not filled in .env. "
            "Fill in first after eBay Developer Account is approved."
        )

    response = requests.post(
        EBAY_OAUTH_URL,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={"grant_type": "client_credentials", "scope": "https://api.ebay.com/oauth/api_scope"},
        auth=(EBAY_CLIENT_ID, EBAY_CLIENT_SECRET),
        timeout=10,
    )
    response.raise_for_status()
    return response.json()["access_token"]


# ---------------------------------------------------------
# 3. Search listing by keyword -> return list item_id
# ---------------------------------------------------------
def search_item_ids(
    session: requests.Session,
    access_token: str,
    keyword: str,
    limit: int = 200,
) -> list[str]:
    if not 1 <= limit <= 200:
        raise ValueError("limit harus antara 1 dan 200")

    headers = {
        "Authorization": f"Bearer {access_token}",
        "X-EBAY-C-MARKETPLACE-ID": "EBAY_US",
    }

    item_ids = []
    offset = 0
    max_offset = 400 #factor of 200 only

    while offset < max_offset:
        params = {
            "q": keyword,
            "limit": limit,
            "offset": offset,
        }

        response = session.get(
            EBAY_SEARCH_URL,
            headers=headers,
            params=params,
            timeout=(5, 10),
        )
        response.raise_for_status()

        data = response.json()
        items = data.get("itemSummaries", [])
        item_ids.extend(
            item["itemId"] for item in items if "itemId" in item
        )

        total = data.get("total", 0)
        next_offset = offset + len(items)

        if not items or next_offset >= total:
            break

        if next_offset <= offset or next_offset > max_offset:
            print(f"Warning: hasil untuk '{keyword}' melebihi batas offset eBay.")
            break

        offset = next_offset

    return item_ids

# ---------------------------------------------------------
# 4. Get item details
# ---------------------------------------------------------
def get_item_detail(session: requests.Session, access_token: str, item_id: str) -> dict | None:
    headers = {
        "Authorization": f"Bearer {access_token}",
        "X-EBAY-C-MARKETPLACE-ID": "EBAY_US",
    }
    url = f"{EBAY_ITEM_URL}/{item_id}"

    try:
        response = session.get(url, headers=headers, timeout=(5, 10))

        if response.status_code != 200:
            return None

        return response.json()
    except requests.exceptions.Timeout:
        print(f" Warning: Timeout getting item {item_id}, skipped.")
        return None
    except requests.exceptions.RequestException as e:
        print(f" Warning: Network error on item_id {item_id}: {e}")
        return None


def strip_html(raw_html: str | None) -> str | None:
    """eBay description is raw HTML (contains <b>, <br>, etc.).
    We strip it to get clean plain text for storage and parsing."""
    if not raw_html:
        return None
    soup = BeautifulSoup(raw_html, "html.parser")
    text = soup.get_text(separator=" ", strip=True)
    return text


def create_retry_session() -> requests.Session:
    session = requests.Session()
    retries = Retry( # Retry with short exponential backoff
        total=3,                
        connect=3,
        read=3,
        status=3,
        backoff_factor=0.2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
        respect_retry_after_header=False,
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

# ---------------------------------------------------------
# 5. Parse response getItem to tuple
# ---------------------------------------------------------
def parse_item(item: dict) -> tuple:
    item_id = item.get("itemId")
    title = item.get("title")

    price_info = item.get("price", {})
    price = price_info.get("value")
    currency = price_info.get("currency")

    condition = item.get("condition")

    location_info = item.get("itemLocation", {})
    seller_location = location_info.get("country")

    # eBay description is raw HTML -> must be stripped first.
    description_raw = item.get("description")
    description_clean = strip_html(description_raw)

    localized_aspects = item.get("localizedAspects", [])
    localized_aspects_json = Json(localized_aspects)

    return (item_id, title, price, currency, condition, seller_location, description_clean, localized_aspects_json)


# ---------------------------------------------------------
# 5. Insert batch to Postgres
# ---------------------------------------------------------
def insert_listings(rows: list[tuple]) -> int:
    if not rows:
        return 0

    initial_query = """
    CREATE SCHEMA IF NOT EXISTS raw;

    CREATE TABLE IF NOT EXISTS raw.ebay_listings (
        item_id VARCHAR(255) PRIMARY KEY,
        title TEXT,
        price NUMERIC,
        currency VARCHAR(10),
        condition TEXT,
        seller_location TEXT,
        description TEXT,
        localized_aspects JSONB,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    );

    ALTER TABLE raw.ebay_listings
        ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT now();
    """

    insert_query = """
        INSERT INTO raw.ebay_listings (item_id, title, price, currency, condition, seller_location, description, localized_aspects)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (item_id) DO UPDATE SET
            title = EXCLUDED.title,
            price = EXCLUDED.price,
            currency = EXCLUDED.currency,
            condition = EXCLUDED.condition,
            seller_location = EXCLUDED.seller_location,
            description = EXCLUDED.description,
            localized_aspects = EXCLUDED.localized_aspects,
            created_at = now();
    """

    try:
        # execute commit if success, rollback if fail
        with psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
        ) as conn:
            with conn.cursor() as cur:
                cur.execute(initial_query)
                cur.executemany(insert_query, rows)

        # returning length of row
        return len(rows)

    except Exception as e:
        print(f"Error when inserting data: {e}")
        return 0

def save_backup_to_json(rows: list[tuple], filename: str):
    # Save result to JSON file before insert to database.
    backup_data = []
    
    for row in rows:
        # Mapping tuple order from parse_item() back to dictionary
        item_dict = {
            "item_id": row[0],
            "title": row[1],
            "price": row[2],
            "currency": row[3],
            "condition": row[4],
            "seller_location": row[5],
            "description": row[6],
            # row[7] is psycopg2.extras.Json, use .adapted to get the original list
            "localized_aspects": row[7].adapted if row[7] else None 
        }
        backup_data.append(item_dict)

    # Create folder backup
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Write to JSON
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(backup_data, f, ensure_ascii=False, indent=4)


def load_backup_from_json(filename: Path) -> list[tuple]:
    with filename.open("r", encoding="utf-8") as f:
        backup_data = json.load(f)

    if not isinstance(backup_data, list):
        raise ValueError("Backup JSON must contain a list of listing objects.")

    rows = []
    for index, item in enumerate(backup_data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Backup row {index} must be a JSON object.")

        item_id = item.get("item_id")
        if not isinstance(item_id, str) or not item_id:
            raise ValueError(f"Backup row {index} is missing a valid item_id.")

        rows.append((
            item_id,
            item.get("title"),
            item.get("price"),
            item.get("currency"),
            item.get("condition"),
            item.get("seller_location"),
            item.get("description"),
            Json(item.get("localized_aspects", [])),
        ))

    return rows

# ---------------------------------------------------------
# 6. Main
# ---------------------------------------------------------
def main():
    keywords = [
        "re:zero light novel",
        "the angel next door light novel",
        "makine too many losing heroines light novel",
        "86 eighty six light novel",
        "gimai seikatsu light novel",
        "love unseen beneath the clear night sky light novel",
        "love unseen beneath the radiant night sky:volume",
        "even a replica can fall in love light novel",
        "classroom of the elite light novel",
        "mushoku tensei light novel",
        "overlord light novel",
        "konosuba light novel",
        "sword art online light novel",
        "spy classroom light novel",
        "rascal does not dream of bunny girl senpai light novel",
        "the eminence in shadow light novel"
        
    ]

    SEARCH_LIMIT = 200  # Item/keyword search limitation
    REQUEST_DELAY = 0.05  # preventing throttle and spam detection

    print(f"[{datetime.now()}] Starting ingestion process...")

    session = create_retry_session() #auto-retry session
    token = get_access_token()
    print("Access token successfully obtained.")

    # Stage 1: Gather all item_id from each keyword
    all_item_ids = []
    for base_kw in keywords:
        search_variations = [
            base_kw,
            f"{base_kw} special edition",           
            f"{base_kw} limited edition"     
        ]
        
        for kw in search_variations:
            ids = search_item_ids(session, token, kw, limit=SEARCH_LIMIT) 
            all_item_ids.extend(ids)
            print(f"  - '{kw}': {len(ids)} item_id found")

    # Deduplicate
    all_item_ids = list(dict.fromkeys(all_item_ids))
    print(f"Total unique item_id: {len(all_item_ids)}")

    # Stage 2: Get full detail of each item
    all_rows = []
    skipped = 0
    for i, item_id in enumerate(all_item_ids, start=1):
        detail = get_item_detail(session, token, item_id) # Oper session
        if detail is None:
            skipped += 1
            continue

        all_rows.append(parse_item(detail))
        time.sleep(REQUEST_DELAY)

        if i % 50 == 0:
            current_time = datetime.now().strftime("%H:%M:%S")
            print(f"[{current_time}] ...progress: {i}/{len(all_item_ids)} item processed")

    print(f"getItem finished. {len(all_rows)} item successfully processed, {skipped} item skipped (delisted/error).")

    if all_rows:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filepath = f"backups/ebay_raw_{timestamp_str}.json"
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Saving backup to {backup_filepath}...")
        try:
            save_backup_to_json(all_rows, backup_filepath)
            print(f"Backup successfully saved!")
        except Exception as e:
            print(f"Warning: Failed to save JSON backup file: {e}")

        # Stage 3: Insert data to database
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Saving {len(all_rows)} row to database...")
        inserted = insert_listings(all_rows)
        print(f"[{datetime.now()}] Finished. {inserted} row successfully inserted to raw.ebay_listings.")
    else:
        print("No data to insert.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest eBay listings into PostgreSQL.")
    parser.add_argument(
        "--from-json",
        type=Path,
        metavar="FILE",
        help="insert listings from a previously saved JSON backup instead of calling eBay",
    )
    
    args, unknown = parser.parse_args()
    
    if args.from_json:
        try:
            rows = load_backup_from_json(args.from_json)
        except (OSError, json.JSONDecodeError, ValueError) as e:
            parser.error(f"cannot load backup {args.from_json}: {e}")
    
        print(f"Loaded {len(rows)} listing rows from {args.from_json}.")
        inserted = insert_listings(rows)
        print(f"Finished. {inserted} row successfully inserted to raw.ebay_listings.")
        