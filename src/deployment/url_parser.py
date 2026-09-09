"""
url_parser.py

Extracts an eBay item_id from a full-length listing URL, e.g.:

    https://www.ebay.com/itm/358790904803?_skw=...&itmmeta=...

MVP SCOPE (2026-09): only the long-form /itm/{item_id} URL is
supported. Short share links (e.g. https://ebay.io/m/jwQXeK) are NOT
resolved here -- those are redirect URLs with no item_id embedded in
the link itself, requiring an extra HTTP round-trip to follow the
redirect before an item_id is even available. Deferred as a follow-up
(see project backlog) rather than a hidden limitation: parse_item_id()
raises a clear, distinguishable error for share-link input so the
API layer can return an informative 400 rather than a confusing
"item not found".
"""

from __future__ import annotations

import re
from urllib.parse import urlparse


_ITEM_ID_RE = re.compile(r"/itm/(?:[^/]+/)?(\d+)")


class UnsupportedEbayUrlError(ValueError):
    """Raised when the URL is recognizably an eBay URL but not (yet)
    a supported format -- e.g. a share short-link."""


def parse_item_id(ebay_url: str) -> str:
    """Extracts the numeric item_id from a long-form eBay listing URL.

    Accepts both plain and title-slug forms:
        https://www.ebay.com/itm/358790904803
        https://www.ebay.com/itm/Some-Title-Slug/358790904803
    plus arbitrary query strings/fragments (ignored).

    Raises:
        UnsupportedEbayUrlError: URL looks like an eBay short/share
            link (e.g. ebay.io/...) -- not supported in this MVP.
        ValueError: URL doesn't look like a parseable eBay item URL
            at all.
    """
    if not ebay_url or not ebay_url.strip():
        raise ValueError("ebay_url is empty.")

    parsed = urlparse(ebay_url.strip())
    host = (parsed.netloc or "").lower()

    if "ebay.io" in host:
        raise UnsupportedEbayUrlError(
            "Short eBay share links (ebay.io/...) are not supported yet. "
            "Please paste the full listing URL instead (the one you get "
            "from your browser's address bar, containing '/itm/')."
        )

    if "ebay." not in host:
        raise ValueError(f"Doesn't look like an eBay URL: {ebay_url!r}")

    match = _ITEM_ID_RE.search(parsed.path)
    if not match:
        raise ValueError(
            f"Couldn't find an item ID in the URL path {parsed.path!r}. "
            "Expected a '/itm/{item_id}' segment."
        )

    return match.group(1)
