CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE SCHEMA IF NOT EXISTS raw;

CREATE TABLE raw.ebay_listings (
    item_id TEXT PRIMARY KEY,
    title TEXT,
    price NUMERIC,
    currency TEXT,
    condition TEXT,
    seller_location TEXT,
    description TEXT,
    localized_aspects JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);