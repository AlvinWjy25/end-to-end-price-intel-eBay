-- Migration: create raw.prediction_requests
--
-- Purpose: isolated table for real-time /predict endpoint requests.
-- Schema is intentionally identical to raw.ebay_listings so the same
-- staging + intermediate SQL transformations (stg_ebay_listings.sql,
-- int_ebay_listing_risk_analysis.sql) can run against either table
-- unmodified -- just pointed at a different source. This guarantees
-- feature parity with training data (no separate Python regex port
-- to keep in sync).
--
-- Kept separate from raw.ebay_listings (training data) so that:
--   - live user requests never contaminate the curated training set
--   - retention/cleanup policy can differ (e.g. purge after 30 days)
--   - batch ingestion + dbt runs are not lock-contended by live traffic
 
create table if not exists raw.prediction_requests (
    request_id      uuid primary key default gen_random_uuid(),
    requested_at    timestamptz not null default now(),
 
    -- identical to raw.ebay_listings columns
    item_id             text not null,
    title               text,
    price               numeric,
    currency            text,
    condition           text,
    seller_location     text,
    description         text,
    localized_aspects   jsonb,
    created_at          timestamptz not null default now()  -- mirrors ebay_listings.created_at semantics
);
 
-- item_id is NOT unique here (unlike training data intent) --
-- the same real-world listing may be requested multiple times by
-- different users, or re-checked later at a different price.
create index if not exists idx_prediction_requests_item_id
    on raw.prediction_requests (item_id);
 
create index if not exists idx_prediction_requests_requested_at
    on raw.prediction_requests (requested_at);