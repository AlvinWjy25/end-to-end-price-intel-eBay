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
| dbt tests (`schema.yml`) | ✅ Passing |
| Price regression model [LightGBM] | ✅ Finalized! — Multiple Random split metric: R² 0.677 ± 0.065, MAE $15.1457, SMAPE: 27.677% |
| Confidence intervals (split conformal prediction) | ✅ Done — empirical coverage 0.894 at 90% target |
| Risk classification model (official vs. unofficial) [MLP] | ✅ Finalized! — Test metric: Accuracy: 98.9%, F1: 97.7%, Val → Test F1 gap: -0.012, Val → Test AUC gap: -0.011 |
| Serving layer (`/predict`: URL → risk + price) | ✅ Done — live end-to-end test validated |
| API (FastAPI) | ✅ Done |
| Frontend | 📋 Postponed |
| Containerization (Docker/K8s) | 📋 Functionally Done |
| CI/CD (GitHub Actions, `dbt-ci.yml`) | 📋 Planned — design finalized, not yet built |

## Architecture

```
eBay Browse API → ingest.py → raw.ebay_listings (Postgres)
                                     │
                                dbt: staging → intermediate → marts
                                     │
                              fct_ebay_listings
                                     │
                    ┌────────────────┴────────────────┐
                    │                                  │
          Regression pipeline                Classification pipeline
          (price estimation, LightGBM)       (legitimacy risk, MLP)
                    │                                  │
                    └────────────────┬─────────────────┘
                                     │
                         Serving layer (FastAPI /predict)
                    URL → resolve item → dbt SQL (verbatim) →
                    risk gate → price model → conformal interval
                                     │
                          Docker (Functionally Done) → Frontend (Postponed)
```

## Tech stack

- **Ingestion:** Python, eBay Browse API (OAuth 2.0 Client Credentials)
- **Storage:** PostgreSQL 16 (Docker, postgres=1.11.0)
- **Transformation:** dbt Core (dbt=1.12.0)
- **Modeling:** scikit-learn, LightGBM (Regression), PyTorch (Classification — two-branch MLP)
- **Experiment tracking:** MLflow
- **Serving:** FastAPI, uvicorn, Docker, Kubernetes (planned)

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
   DB_USER=test_user
   DB_PASSWORD=test_password
   ```
   NOTE: to get your `EBAY_CLIENT_ID` and `EBAY_CLIENT_SECRET`, navigate to: [1.3.A. Setup your .env](https://app.notion.com/p/eBay-Light-Novel-Price-Intelligence-Documentation-2-0-3bf16382ebe58040abeaeed49986dd13?source=copy_link#3bf16382ebe580de87c3d2ea89397301)
3. From the project root, run:
   ```
   docker compose -p end-to-end_price-intel-ebay up -d
   ```
   This will: spin up Postgres → run the dbt pipeline (staging → intermediate → marts) → run `pipeline.py` (feature prep + inference) → launch the API → launch the frontend.
4. Once container(s) is up, the app will be available at `http://localhost:8000/docs` (planned).
5. To predict your first listings: go to ebay, search for any light novels, copy it's link from the browser search bar.
Warning: pasting manga or any other product than light novel may result with error.

6. Navigate to `http://localhost:8000/docs#/default/predict_predict_post`, Click Try it out
7. Paste your link to: ebay_url

Example:
```
{
  "ebay_url": "https://www.ebay.com/itm/358790904803?_skw=Love+unseen+beneath+the+clear+night+sky&itmmeta=01M22X4QQWM5ZR93MYR10A3X68&hash=item53899abbe3:g:WJkAAeSwxghqfx3L&itmprp=enc%3AAQALAAABAGfYFPkwiKCW4ZNSs2u11xAFElMJdkLVZv0dGNqpH8wDpKkSEuTWms4VQQogd6FZk8Xx6ww5EuIy%2FvWUrwbq04dpXfJQQAWKXBKGenHr53V08CFmr7ub7NA5i%2FkR7hSQtFdO6AGCr5B8iysZsK9OHTkLaG%2FQmtfD6fmnP0REdJO58zVf4zB3WM4VgR%2B--LY0k4SqQYFWRpzHmbQAFKc7e4Pm%2FOFQQtr4donXskOMa%2BtUvS098nAute21yjYh%2Fv12j0WtJCd8eQTTncSaQAVjnzTRi24--lJ7RMQbA5JtubesZf4mvokxwz0O%2FVSpgxOZse1Nsw5HnXCJB%2FYP5kBlbR4%3D%7Ctkp%3ABk9SR4j8kt2QaA",
  "override_risk_gate": true
}
```

Live example response (validated end-to-end): item priced at $19.26 on eBay → predicted at $20.10, classified Low Risk (classifier confidence 0.043).

## How prediction serving works

Given a raw eBay URL, the `/predict` endpoint:

1. **Resolves the item** — extracts the legacy item ID from the URL and calls `getItem` (eBay Browse API, `fieldgroups=EXTENDED`) to fetch full listing metadata.
2. **Inserts into an isolated request table** (`raw.prediction_requests`) — same schema as training data (`raw.ebay_listings`), but physically separate so prediction traffic never contaminates the training set.
3. **Runs the exact same dbt SQL used in training, verbatim** — the staging → intermediate transformation logic (nested CTEs) is reused as-is rather than re-implemented in Python. This guarantees zero training-serving skew: whatever feature engineering happened during training happens identically at inference time.
4. **Risk gate** — the MLP classifier scores the listing; if it's flagged High Risk, prediction is blocked by default (bootleg/reprint pricing isn't meaningful market signal). An `override_risk_gate` flag is available for testing, with an explicit disclaimer in the response.
5. **Price prediction + interval** — the LightGBM regressor predicts price, and a split-conformal interval (calibrated in log-space, 90% target coverage, empirical coverage 0.894) is attached to the point estimate.

## Key design decisions (TL;DR)

**Regression:**
- **Condition is structural, not inferred from text.** The `condition` field (New/Used/etc.) comes from eBay's own structured field, never parsed from the title — avoids conflicting signals from unstructured text.
- **Volume number extraction uses a 5-tier fallback chain** (structured API aspects → title regex → description regex → null), each tier ranked by confidence, because eBay's own fields are inconsistently populated across listings.
- **Risk scoring is binary (High/Low), not three-tier** — the data showed a natural score gap between 30 and 70 with nothing in between, so a Medium tier added no signal.
- **`total_risk_score`, `price_risk_score`, and `text_risk_score` are excluded from the price model** — both are mathematically derived from price itself, so including them would leak the target into the features. Excluding them dropped R² from 0.649 to 0.619 — a smaller, honest number instead of an inflated one.
- **Scaling (RobustScaler) is kept in the pipeline despite being inert for LightGBM** — tree splits depend on value ranking, not magnitude, so this is a consistency decision, not a modeling one (verified via ablation — identical metrics with/without scaling).
- **Ambiguous bulk-pricing rows are flagged, not corrected.** No scalable automatic signal (title/description regex, structured unit-of-sale field) reliably tells a per-unit listing mislabeled as a lot apart from a genuine bulk lot. Rather than guess, an `is_ambiguous_bulk_pricing` flag (derived from a `price / volume_count` threshold sweep) excludes these rows from training eligibility, preserving auditability.

**Classification:**
- **Two-branch TF-IDF architecture.** Title and description text are vectorized separately (independent TF-IDF vectorizers), fed through their own small branch, then concatenated into a shared head — keeping the noisier, longer description text from drowning out the more reliable title signal.
- **Raw logits + `BCEWithLogitsLoss`** (with `pos_weight` to handle class imbalance) rather than a sigmoid output layer, for numerical stability during training.
- **Small by design.** ~28.6K parameters total — kept deliberately small relative to the training set size to control overfitting, with weight decay tuned via sweep (values above 5e-2 caused model collapse).
- **Generalization validated via Leave-One-Series-Out (LOSO), not just random split.** Random-split F1 sits around 0.958–0.964; LOSO F1 is 0.921 — a small, honest gap confirming the model isn't just memorizing series-specific phrasing rather than learning general risk signal.

Full rationale for every decision above, plus EDA, failed hypotheses (e.g. why predicting `price_per_volume` and multiplying by volume count fails catastrophically for box sets), and model diagnostics live in the notebook documentation.

## Full documentation

📄 **[Problem Documentation — Notion](https://app.notion.com/p/eBay-Light-Novel-Price-Intelligence-Documentation-2-0-3bf16382ebe58040abeaeed49986dd13?source=copy_link)**

Covers: Project overview, Problem Framing, and FAQ.

Pipeline logic is located at `notebook/01_EDA_x.ipynb`, and `notebook/02_EDA.ipynb`
Staging -> marts logic is located at `price_intel_dbt/models/../.yml`

## Known limitations

- Training set for regression is currently ~1,647 rows (54.88%) are qualified enough after filtering — small enough that hyperparameter tuning via 5-fold CV showed high variance across folds and was not adopted (default LightGBM params used instead; see full docs).
- Most of the data pulled from ebay listings are due to unofficial listings (automatically marked as high risk), inconsistent/incomplete metadata with title and description listings, or listings without condition information.
- `seller_location` is currently the dominant price signal (proxying for import/rarity/edition), but only 7 countries are represented and two (Germany, Canada) have fewer than 5 listings each — generalization to unseen seller countries is untested.
- A subset of High Risk training rows (roughly two-thirds) share near-identical seller phrasing patterns (e.g. "viz media" / "not suitable for collector" template language). This is a real, recurring pattern in the data but is narrow and template-specific — the classifier's risk signal may be partly reliant on this phrasing rather than more general markers, which is a documented generalization risk rather than a resolved one.
- CI/CD is designed (Postgres service container, seeded representative sample data, `profiles.yml.example` → runtime `ci` target, credentials via GitHub Secrets) but not yet implemented as a running GitHub Actions workflow.