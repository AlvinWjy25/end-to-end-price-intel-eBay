"""
legacy_id_resolver.py

eBay has two different item ID formats:
  - Legacy item ID: a plain 12-digit number, the one visible in a
    listing's browser URL (e.g. "137476531310"). This is what
    url_parser.py extracts.
  - RESTful item ID: the format required by the Browse API's getItem
    call, shaped "v1|{legacy_id}|0" (e.g. "v1|137476531310|0").

get_item_detail() in ingest.py calls getItem, which REQUIRES the
RESTful format -- passing a plain legacy ID (as returned by
url_parser.parse_item_id()) causes getItem to 404, since eBay is
looking for an item literally named "137476531310", not the item
whose legacy ID is 137476531310.

This resolves that gap via the Browse API's dedicated
get_item_by_legacy_id call, which is exactly what it exists for.

NOTE: search_item_ids() in ingest.py never hits this problem because
Browse API search results already return RESTful item IDs directly --
this conversion is only needed when the input comes from a URL (i.e.
the /predict endpoint's use case), not from ingest.py's normal batch
flow.
"""

from __future__ import annotations

import requests

EBAY_LEGACY_ID_URL = "https://api.ebay.com/buy/browse/v1/item/get_item_by_legacy_id"


def resolve_legacy_item_id(
    session: requests.Session,
    access_token: str,
    legacy_item_id: str,
    marketplace_id: str = "EBAY_US",
) -> str | None:
    """Converts a plain legacy eBay item ID (from a listing URL) into
    the RESTful item ID format required by getItem.

    Returns None if the legacy ID couldn't be resolved (invalid,
    delisted, or API error) -- mirroring get_item_detail()'s existing
    convention of returning None on failure rather than raising.
    """
    headers = {
        "Authorization": f"Bearer {access_token}",
        "X-EBAY-C-MARKETPLACE-ID": marketplace_id,
    }
    params = {"legacy_item_id": legacy_item_id}

    try:
        response = session.get(EBAY_LEGACY_ID_URL, headers=headers, params=params, timeout=20)
        if response.status_code != 200:
            return None
        data = response.json()
        return data.get("itemId")
    except requests.exceptions.RequestException:
        return None
