# ============================================================
# Conformal Prediction: Calibration Split + Nonconformity Scores
# ============================================================
#
# Goal: split the EXISTING test set into calib + test (the trained
# regression model itself is NOT retrained -- it stays exactly as it
# is in production). This gives us a calibration set to compute
# nonconformity scores (absolute residuals in log scale, matching the
# TransformedTargetRegressor's log1p/expm1 target transform) and a
# smaller held-out test set to verify empirical coverage.
#
# Stratification: by price, bucketed into quantile bins (price itself
# is continuous, so train_test_split's stratify= needs a categorical
# proxy -- pd.qcut bins it into quantile-based groups so calib/test
# get comparable price distributions).

import sys
import os
import numpy as np
import pandas as pd
import joblib
import mlflow
from pathlib import Path
from sklearn.model_selection import train_test_split

root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir))

from config.config_script import (
    setup_logger,
    MODEL_REGRESSION_PATH,
    ARTIFACT_DIR,
    feature_cols,
    X_TEST_REGRESSION_PATH,
    Y_TEST_REGRESSION_PATH,
    MLFLOW_DB_PATH,
    MLFLOW_MLRUNS_PATH,
    MLFLOW_TRACKING_URI,
)


class Conformal_train:
    def __init__(self):
        self.logger = setup_logger("regression", "conformal_train")

        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        self.mlflow_experiment = "light_novel_price_regression"
        self.mlflow_run_name = "conformal_prediction_run"

        self.mlflow_tracking_path = MLFLOW_MLRUNS_PATH.resolve()
        self.mlflow_tracking_uri = MLFLOW_TRACKING_URI
        self.mlflow_db_path = MLFLOW_DB_PATH
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        self.logger.info(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
        self.logger.info(f"MLflow artifact root: {self.mlflow_tracking_path}")

        mlflow.set_experiment(self.mlflow_experiment)

    def fit_conformal(
        self,
        calib_size: float = 0.5,
        random_state: int = 42,
        n_price_bins: int = 5,
        alpha: float = 0.1
    ):
        """
        Executes conformal calibration split, nonconformity scoring,
        quantile computation, coverage verification, and MLflow logging.
        """
        self.logger.info("Starting conformal prediction calibration pipeline.")

        try:
            self.X_test_full = pd.read_parquet(X_TEST_REGRESSION_PATH)
            self.y_test_full = pd.read_parquet(Y_TEST_REGRESSION_PATH).squeeze()
            self.regression_pipeline = joblib.load(MODEL_REGRESSION_PATH)
            self.logger.info("Preprocessed test data and trained regression pipeline loaded successfully.")
            self.logger.info(f"Existing test set size: {len(self.X_test_full)}")
        except Exception as e:
            self.logger.error(f"Error loading required preprocessed data or model: {e}")
            raise

        with mlflow.start_run(run_name=self.mlflow_run_name) as run:
            self.logger.info(f"MLflow run started: {run.info.run_id}")
            self.logger.info(f"MLflow backend store: {mlflow.get_tracking_uri()}")

            mlflow.set_tag("model_family", "lightgbm_regression")
            mlflow.set_tag("pipeline_stage", "conformal_calibration")

            mlflow.log_param("calib_size", calib_size)
            mlflow.log_param("random_state", random_state)
            mlflow.log_param("n_price_bins", n_price_bins)
            mlflow.log_param("alpha", alpha)
            mlflow.log_param("confidence_level", 1 - alpha)

            # Stratified split by price quantile bins
            price_bins = pd.qcut(self.y_test_full, q=n_price_bins, duplicates="drop")
            self.logger.info(f"Price bin distribution in test set:\n{price_bins.value_counts().sort_index()}")

            X_calib, X_test_new, y_calib, y_test_new = train_test_split(
                self.X_test_full,
                self.y_test_full,
                test_size=(1 - calib_size),
                stratify=price_bins,
                random_state=random_state,
            )

            self.logger.info(f"Calibration set size: {len(X_calib)}")
            self.logger.info(f"New (smaller) test set size: {len(X_test_new)}")

            mlflow.log_param("n_calib_samples", len(X_calib))
            mlflow.log_param("n_heldout_test_samples", len(X_test_new))

            # Nonconformity scores calculation: |log1p(y_true) - log1p(y_pred)|
            y_calib_pred = self.regression_pipeline.predict(X_calib[feature_cols])
            y_calib_np = y_calib.values.ravel() if isinstance(y_calib, pd.Series) else y_calib.ravel()
            y_calib_pred_np = y_calib_pred.ravel()

            nonconformity_scores = np.abs(np.log1p(y_calib_np) - np.log1p(y_calib_pred_np))

            self.logger.info("Nonconformity scores (calibration set):")
            self.logger.info(f"  count: {len(nonconformity_scores)}")
            self.logger.info(f"  mean : {nonconformity_scores.mean():.4f}")
            self.logger.info(f"  std  : {nonconformity_scores.std():.4f}")
            self.logger.info(f"  min  : {nonconformity_scores.min():.4f}")
            self.logger.info(f"  max  : {nonconformity_scores.max():.4f}")

            # Compute small-sample corrected conformal quantile: ceil((n+1)*(1-alpha))/n
            n_calib = len(nonconformity_scores)
            quantile_level = np.ceil((n_calib + 1) * (1 - alpha)) / n_calib
            quantile_level = min(quantile_level, 1.0)

            conformal_quantile = np.quantile(nonconformity_scores, quantile_level)

            self.logger.info(f"90% conformal quantile (log scale): {conformal_quantile:.4f}")
            self.logger.info(f"Quantile level used: {quantile_level:.4f} (n_calib={n_calib})")

            # Verify empirical coverage on held-out test set
            y_test_pred = self.regression_pipeline.predict(X_test_new[feature_cols]).ravel()
            y_test_new_np = y_test_new.to_numpy().ravel() if hasattr(y_test_new, "to_numpy") else np.asarray(y_test_new).ravel()

            log_pred = np.log1p(y_test_pred)
            lower_log = log_pred - conformal_quantile
            upper_log = log_pred + conformal_quantile

            lower = np.expm1(lower_log)
            upper = np.expm1(upper_log)

            covered = (y_test_new_np >= lower) & (y_test_new_np <= upper)
            empirical_coverage = float(covered.mean())
            avg_interval_width = float((upper - lower).mean())
            median_interval_width = float(np.median(upper - lower))

            self.logger.info(f"Empirical coverage on held-out test set (target {1 - alpha:.0%}): {empirical_coverage:.4f}")
            self.logger.info(f"Average interval width (dollars): {avg_interval_width:.2f}")
            self.logger.info(f"Median interval width (dollars): {median_interval_width:.2f}")

            # Log metrics to MLflow
            mlflow.log_metrics({
                "nonconformity_mean": float(nonconformity_scores.mean()),
                "nonconformity_std": float(nonconformity_scores.std()),
                "nonconformity_min": float(nonconformity_scores.min()),
                "nonconformity_max": float(nonconformity_scores.max()),
                "quantile_level_used": float(quantile_level),
                "conformal_quantile_log": float(conformal_quantile),
                "empirical_coverage": empirical_coverage,
                "avg_interval_width_dollars": avg_interval_width,
                "median_interval_width_dollars": median_interval_width,
            })

            # Save conformal serving artifact
            conformal_artifact_path = ARTIFACT_DIR / "models" / "conformal_quantile_90.joblib"
            conformal_artifact = {
                "quantile": float(conformal_quantile),
                "alpha": alpha,
                "confidence_level": 1 - alpha,
                "n_calib": n_calib,
                "empirical_coverage_on_test": empirical_coverage,
                "scale": "log1p",
            }

            joblib.dump(conformal_artifact, conformal_artifact_path)
            self.logger.info(f"Saved conformal artifact to: {conformal_artifact_path}")
            self.logger.info(f"Conformal artifact content: {conformal_artifact}")

            if conformal_artifact_path.exists():
                mlflow.log_artifact(str(conformal_artifact_path), artifact_path="saved_model")
                mlflow.log_param("saved_conformal_artifact_path", str(conformal_artifact_path))

            self.logger.info(f"MLflow run completed: {run.info.run_id}")
            return run.info.run_id


if __name__ == "__main__":
    trainer = Conformal_train()
    trainer.fit_conformal()