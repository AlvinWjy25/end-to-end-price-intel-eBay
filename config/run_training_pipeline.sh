# READ Documentation: https://app.notion.com/p/eBay-Light-Novel-Price-Intelligence-Documentation-2-0-3bf16382ebe58040abeaeed49986dd13?source=copy_link#3bf16382ebe580cc878bc301b9a7228b
# HOW TO TRAIN YOUR PIPELINE WITH YOUR OWN DATA


# Stop execution if there is an error
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$PROJECT_ROOT/logs/training_pipeline"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/$(date +%Y%m%d_%H%M%S).log"

# Show every message in the terminal and save a copy for later review.
exec > >(tee -a "$LOG_FILE") 2>&1

echo "============================================================"
echo "Training pipeline started: $(date)"
echo "Log file: $LOG_FILE"
echo "============================================================"

trap 'exit_code=$?; echo "Training pipeline finished with exit code $exit_code: $(date)"' EXIT

conda activate ebay_price_intel_env

cd "$PROJECT_ROOT/price_intel_dbt"

dbt run

dbt test

cd "$PROJECT_ROOT"

# WARNING: This may consume your EBAY API Quota, SETUP YOUR EBAY API at https://developer.ebay.com/signin & Place as such:
# DIR: (parent dir)/config/.env
# NOTE: .env content:  #DO NOT FORGET TO PUT config/.env on .GITIGNORE!

cd "$PROJECT_ROOT/src/ingestion"
python ingest.py

cd "$PROJECT_ROOT/src/regression"
python pipeline.py

cd "$PROJECT_ROOT/src/classification"
python pipeline.py