import pandas as pd
import sys
import os
import shutil
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.compose import ColumnTransformer

root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir))
from config.config_script import load_dataframe, setup_logger, X_TRAIN_REGRESSION_PATH, X_TEST_REGRESSION_PATH, Y_TRAIN_REGRESSION_PATH, Y_TEST_REGRESSION_PATH, Y_PRED_REGRESSION_PATH, DF_META_REGRESSION_PATH
from preprocessor import preprocess_regression
from train import Pipeline_train
from evaluate import Pipeline_evaluate

if __name__ == "__main__":
    logger = setup_logger('regression' ,'pipeline_run')
    logger.info(f"[preprocessor.py]")
    logger.info("=" * 100)
    try:
        df_regression = preprocess_regression().fit('regression')
        
        X_train, X_test, y_train, y_test, indices_train, indices_test, df_meta, feature_cols, categorical_features, boolean_features, numeric_features = preprocess_regression().fit_transform(df_regression)

        pd.DataFrame(X_train).to_parquet(X_TRAIN_REGRESSION_PATH, index=False)
        pd.DataFrame(X_test).to_parquet(X_TEST_REGRESSION_PATH, index=False)
        pd.DataFrame(y_train).to_parquet(Y_TRAIN_REGRESSION_PATH, index=False)
        pd.DataFrame(y_test).to_parquet(Y_TEST_REGRESSION_PATH, index=False)
        pd.DataFrame(df_meta).to_parquet(DF_META_REGRESSION_PATH, index=False)

        logger.info("Preprocessed data saved to artifacts/preprocessed directory.")
    except Exception as e:
        logger.exception(f"Unhandled exception during direct run: {e}")
        raise e

    logger.info("[train.py & evaluate.py]")
    logger.info("=" * 100)
    pipeline_train = Pipeline_train()
    pipeline_train.fit_model_regression(verbose=0)

    logger.info("[conformal_train.py]")
    logger.info("=" * 100)

    from conformal_train import Conformal_train

    pipeline_train = Conformal_train()
    pipeline_train.fit_conformal()

    file_to_delete_1 = Path(root_dir / 'src' / 'regression' / 'mlruns')
    file_to_delete_2 = Path(root_dir / 'src' / 'regression' / '__pycache__')

    time.sleep(3) #letting mlflow output cache due to directory diff for a moment before deleting

    try:
        logger.info("[!FINALIZING]: Deleting local cache...")
        shutil.rmtree(file_to_delete_1, ignore_errors=True)
        shutil.rmtree(file_to_delete_2, ignore_errors=True)
        logger.info("[!FINALIZING]: Cache deleted succesfully!")
        logger.info("[FINALIZED]: Pipeline executed succesfully!")
    except Exception as e:
        logger.info("[!ERROR]: Error deleting cache")
        raise e

    logger.info('To view all pipeline run(s) on MLFLOW:')
    logger.info("Copy this to your IDE terminal: 'mlflow ui --backend-store-uri sqlite:///C:/Users/Alvin/Music/project_1/logs/mlflow.db --default-artifact-root file:///C:/Users/Alvin/Music/project_1/logs/mlartifacts --host 127.0.0.1 --port 5000'")
    
