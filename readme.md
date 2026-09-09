# eBay Light Novel Price Intelligence

End-to-end ML/data engineering pipeline that ingests eBay light novel listings, classifies listing legitimacy (official vs. unofficial/reprint risk), and predicts fair market price with confidence intervals — built as a portfolio project to demonstrate the full path from raw API ingestion to a user-facing prediction tool.

**Given an eBay listing URL → the tool extracts structured metadata, flags legitimacy risk, and estimates a fair price range.**

## Why this project

Light novel listings on eBay are notoriously inconsistent: sellers mix official releases, unofficial reprints/bootlegs, single volumes, and box sets under near-identical titles, with price ranging from $3 to $2,000 for what looks like "the same item" at a glance. This project builds a pipeline to disambiguate that automatically — first by extracting reliable structured signal from noisy title/description text, then by scoring risk and predicting price on top of it. 

This project is built to classify if a light novel listing is an official light novel listing or not, and predict the price of an official light novel listing, given the region and condition of the item.

**Why exclude unofficial listings?** This project trains only on plausibly-official light novel listings — bootleg/reprint listings are filtered out via a text-based risk score before training. This isn't just a data-quality decision: unofficial listings don't reflect genuine market value (no licensing costs), and including them would bias price predictions downward while indirectly normalizing counterfeit goods. Official second-hand circulation, by contrast, is a legal, transparent market this project aims to support — see Section Q.2 [3.1.2 About Project FAQ: eBay Light Novel Price Intelligence Documentation](https://app.notion.com/p/eBay-Light-Novel-Price-Intelligence-Documentation-2-0-3bf16382ebe58040abeaeed49986dd13?source=copy_link#3c516382ebe580c2a46ef9c4ffff7394) for the full ethical/legal reasoning.

## Current status

| Component | Status |
|---|---|
| Ingestion (eBay Browse API → Postgres) | ✅ Done |
| dbt transformation layer (staging → intermediate → marts) | ✅ Done |
| Price regression model [LightGBM] | ✅ Finalized! — Multiple Random split metric: R² 0.677 ± 0.065, MAE $15.1457, SMAPE: 27.677% |
| Confidence intervals (quantile regression) | 📋 Planned  |
| Risk classification model (official vs. unofficial) [MLP] | ✅ Finalized! — Test metric: Accuracy: 98.9%, F1: 97.7%, Val -> Test F1 gap : -0.012, Val -> Test AUC gap: -0.011 |
| API (FastAPI) | ✅ Done |
| Frontend | 📋 Postponed |
| Containerization (Docker/K8s) | 📋 Functionally Done |

## Architecture

```
eBay Browse API → ingest.py → raw.ebay_listings (Postgres)
                                     │
                                dbt: staging → intermediate → marts
                                     │
                              fct_ebay_listings
                                     │
                    ┌────────────────┴────────────────┐
                    │                                 │
          Regression pipeline                Classification pipeline
          (price estimation)                 (legitimacy risk)
                    │
              FastAPI (Done) → Frontend (Postponed) → Docker (Functionally Done)
```

## Tech stack

- **Ingestion:** Python, eBay Browse API (OAuth 2.0 Client Credentials)
- **Storage:** PostgreSQL 16 (Docker, postgres=1.11.0)
- **Transformation:** dbt Core (dbt=1.12.0)
- **Modeling:** scikit-learn, LightGBM (Regression), ??? (Classification)
- **Serving (planned):** FastAPI, Docker, Kubernetes

## Setup

**Requirements:** Docker Desktop installed and running.

1. Clone this repo
2. Navigate to `config/` and create a `.env` file with the following:
   ```
   EBAY_CLIENT_ID=your_ebay_client_id
   EBAY_CLIENT_SECRET=your_ebay_client_secret
   DB_HOST=postgres
   DB_PORT=5432
   DB_NAME=price_intelligence
   DB_USER=your_db_user
   DB_PASSWORD=your_db_password
   ```
3. From the project root, run:
   ```
   docker compose -p end-to-end_price-intel-ebay up -d
   ```
   This will: spin up Postgres → run the dbt pipeline (staging → intermediate → marts) → run `pipeline.py` (feature prep + inference) → launch the API → launch the frontend.
4. Once containers are up, the app will be available at `http://localhost:<port>` (planned).

## Key design decisions (TL;DR)

- **Condition is structural, not inferred from text.** The `condition` field (New/Used/etc.) comes from eBay's own structured field, never parsed from the title — avoids conflicting signals from unstructured text.
- **Volume number extraction uses a 5-tier fallback chain** (structured API aspects → title regex → description regex → null), each tier ranked by confidence, because eBay's own fields are inconsistently populated across listings.
- **Risk scoring is binary (High/Low), not three-tier** — the data showed a natural score gap between 30 and 70 with nothing in between, so a Medium tier added no signal.
- **`total_risk_score`, `price_risk_score`, and `text_risk_score` are excluded from the price model** — both are mathematically derived from price itself, so including them would leak the target into the features. Excluding them dropped R² from 0.649 to 0.619 — a smaller, honest number instead of an inflated one.
- **Scaling (RobustScaler) is kept in the pipeline despite being inert for LightGBM** — tree splits depend on value ranking, not magnitude, so this is a consistency decision, not a modeling one (verified via ablation — identical metrics with/without scaling).

Full rationale for every decision above, plus EDA, failed hypotheses (e.g. why predicting `price_per_volume` and multiplying by volume count fails catastrophically for box sets), and model diagnostics live in the full documentation.

## Full documentation

📄 **[Problem Documentation — Notion](https://app.notion.com/p/eBay-Light-Novel-Price-Intelligence-Documentation-2-0-3bf16382ebe58040abeaeed49986dd13?source=copy_link)** 

Covers: Project overview, Problem Framing, and FAQ.

Pipeline logic is located at `notebook/01_EDA_x.ipynb`, and `notebook/02_EDA.ipynb`
Staging -> marts logic is located at `price_intel_dbt/models/../.yml`

## Known limitations

- Training set for regression is currently ~1,647 rows (54.88%) are qualified enough after filtering — small enough that hyperparameter tuning via 5-fold CV showed high variance across folds and was not adopted (default LightGBM params used instead; see full docs).
- Most of the data pulled from ebay listings are due to unofficial listings (automatically marked as high risk), inconsistent/incomplete metadata with title and description listings, or listings without condition information.
- `seller_location` is currently the dominant price signal (proxying for import/rarity/edition), but only 7 countries are represented and two (Germany, Canada) have fewer than 5 listings each — generalization to unseen seller countries is untested.
