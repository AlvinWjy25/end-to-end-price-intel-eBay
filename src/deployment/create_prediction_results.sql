-- Migration: create raw.prediction_results
--
-- Stores model OUTPUT (risk classification, price prediction,
-- conformal interval) separately from raw.prediction_requests, which
-- stores model INPUT (the raw eBay listing data). This separation
-- lets one request have multiple results over time -- e.g. re-running
-- the same listing through a retrained model later, for drift
-- tracking or A/B comparison -- without re-inserting the (unchanged)
-- eBay listing data.
--
-- model_version is a free-text tag (not a foreign key to a formal
-- model registry table) -- keep it simple for now; e.g.
-- "classifier_v1+regressor_v1", a git commit hash, or an MLflow run
-- ID. Whatever's most useful to trace back to the exact artifacts
-- used, decided at insert time by the serving code, not enforced
-- here.

create table if not exists raw.prediction_results (
    result_id       uuid primary key default gen_random_uuid(),
    request_id      uuid not null references raw.prediction_requests(request_id),
    predicted_at    timestamptz not null default now(),

    -- which model artifacts produced this result (see note above)
    model_version   text,

    -- classifier output
    risk_category_model        text not null,   -- "Low Risk" / "High Risk"
    risk_category_rule_based   text,             -- from staging_marts_sql.py, for comparison
    classifier_confidence      numeric,          -- sigmoid probability of High Risk, 0-1
    text_risk_score            integer,
    price_risk_score           integer,
    total_risk_score           integer,

    -- gating outcome
    prediction_blocked         boolean not null default false,
    override_risk_gate_used    boolean not null default false,

    -- regressor output (null if prediction_blocked)
    predicted_price             numeric,
    price_interval_lower        numeric,
    price_interval_upper        numeric,
    price_interval_confidence   numeric  -- e.g. 0.90
);

create index if not exists idx_prediction_results_request_id
    on raw.prediction_results (request_id);

create index if not exists idx_prediction_results_predicted_at
    on raw.prediction_results (predicted_at);

create index if not exists idx_prediction_results_model_version
    on raw.prediction_results (model_version);