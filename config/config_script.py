from pathlib import Path
import pandas as pd
import pyarrow
import fastparquet
import os
import mlflow
import torch

import logging
from datetime import datetime
from pathlib import Path

from sqlalchemy import create_engine
from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / 'config' / '.env')

ROOT_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT_DIR / "logs" / "pipeline_run" 

DBT_LOG_DIR = ROOT_DIR / "logs" / "dbt_logs"
DBT_LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
(LOG_DIR / 'regression').mkdir(parents=True, exist_ok=True)
(LOG_DIR / 'classification').mkdir(parents=True, exist_ok=True)

# print(f'ROOT DIR: {ROOT_DIR}\n')
ARTIFACT_DIR = Path(ROOT_DIR / 'src' / 'artifacts')
ARTIFACT_DIR.mkdir(parents = True, exist_ok = True)

PREPROCESSED_REGRESSION_DIR = Path(ARTIFACT_DIR / 'preprocessed' / 'regression')
PREPROCESSED_REGRESSION_DIR.mkdir(parents = True, exist_ok = True)

PREPROCESSED_CLASSIFICATION_DIR = Path(ARTIFACT_DIR / 'preprocessed' / 'classification')
PREPROCESSED_CLASSIFICATION_DIR.mkdir(parents = True, exist_ok = True)

#Regression
X_TRAIN_REGRESSION_PATH = PREPROCESSED_REGRESSION_DIR / 'X_train.parquet'
X_TEST_REGRESSION_PATH = PREPROCESSED_REGRESSION_DIR / 'X_test.parquet'
Y_TRAIN_REGRESSION_PATH = PREPROCESSED_REGRESSION_DIR / 'y_train.parquet'
Y_TEST_REGRESSION_PATH = PREPROCESSED_REGRESSION_DIR / 'y_test.parquet'
Y_PRED_REGRESSION_PATH = PREPROCESSED_REGRESSION_DIR / 'y_pred.parquet'
DF_META_REGRESSION_PATH = PREPROCESSED_REGRESSION_DIR / 'df_meta.parquet'

#Classification
DF_TRAIN_CLASSIFICATION_PATH = PREPROCESSED_CLASSIFICATION_DIR / 'df_train.parquet'
DF_VAL_CLASSIFICATION_PATH = PREPROCESSED_CLASSIFICATION_DIR / 'df_val.parquet'
DF_TEST_CLASSIFICATION_PATH = PREPROCESSED_CLASSIFICATION_DIR / 'df_test.parquet'

TITLE_TRAIN_PATH = PREPROCESSED_CLASSIFICATION_DIR / 'title_train.parquet'
TITLE_VAL_PATH = PREPROCESSED_CLASSIFICATION_DIR / 'title_val.parquet'
TITLE_TEST_PATH = PREPROCESSED_CLASSIFICATION_DIR / 'title_test.parquet'

DESCRIPTION_TRAIN_PATH = PREPROCESSED_CLASSIFICATION_DIR / 'description_train.parquet'
DESCRIPTION_VAL_PATH = PREPROCESSED_CLASSIFICATION_DIR / 'description_val.parquet'
DESCRIPTION_TEST_PATH = PREPROCESSED_CLASSIFICATION_DIR / 'description_test.parquet'

VECTORIZER_PATH = ARTIFACT_DIR / 'vectorizer'
VECTORIZER_PATH.mkdir(exist_ok=True, parents=True)

TITLE_VECTORIZER_PATH = VECTORIZER_PATH / 'TITLE_VECTORIZER.joblib'
DESCRIPTION_VECTORIZER_PATH = VECTORIZER_PATH /  'DESCRIPTION_VECTORIZER.joblib'

# Backward-compatible aliases for older imports during migration.
# PREPROCESSED_DIR = PREPROCESSED_REGRESSION_DIR
# X_TRAIN_PATH = X_TRAIN_REGRESSION_PATH
# X_TEST_PATH = X_TEST_REGRESSION_PATH
# Y_TRAIN_PATH = Y_TRAIN_REGRESSION_PATH
# Y_TEST_PATH = Y_TEST_REGRESSION_PATH
# Y_PRED_PATH = Y_PRED_REGRESSION_PATH
# DF_META_PATH = DF_META_REGRESSION_PATH

# REGRESSION EVALUATION
EVALUATION_DIR = Path(ARTIFACT_DIR / 'evaluation')
EVALUATION_DIR.mkdir(parents = True, exist_ok = True)
(EVALUATION_DIR / 'regression').mkdir(parents = True, exist_ok = True)
(EVALUATION_DIR / 'classification').mkdir(parents = True, exist_ok = True)

EVAL_SUMMARY_REGRESSION_PATH = EVALUATION_DIR / 'regression' / 'evaluation_report.json'
EVAL_DATAFRAME_REGRESSION_PATH = EVALUATION_DIR / 'regression' / 'evaluation_dataframe.json'
EVAL_RANDOM_SEED_REGRESSION_PATH = EVALUATION_DIR / 'regression' / 'evaluation_random_seed.json'

# CLASSIFICATION EVALUATION
EVAL_TRAINING_LOOP_CLASSIFICATION_PATH = EVALUATION_DIR / 'classification' / 'training_loop.png'
EVAL_HISTORY_CLASSIFICATION_PATH = EVALUATION_DIR / 'classification' / 'training_history.json'
EVAL_SUMMARY_CLASSIFICATION_PATH = EVALUATION_DIR / 'classification' / 'evaluation_report.json'

ARTIFACT_MODEL_PATH = Path(ARTIFACT_DIR / 'models')
ARTIFACT_MODEL_PATH.mkdir(parents=True, exist_ok=True)

MODEL_REGRESSION_PATH = Path(ARTIFACT_DIR / 'models' / 'final_regression.joblib')
MODEL_CLASSIFICATION_PATH = Path(ARTIFACT_DIR / 'models' / 'final_classification.pth')
CONFORMAL_ARTIFACT_PATH = Path(ARTIFACT_DIR / 'models' / 'conformal_quantile_90.joblib')

MLFLOW_MLRUNS_PATH = Path(ROOT_DIR / "logs" / "mlruns")
MLFLOW_MLRUNS_PATH.mkdir(parents=True, exist_ok=True)

# MLflow 3.x requires a database-backed store; local file-store mode is deprecated.
# Use SQLite under the project logs directory so the UI can start cleanly.
MLFLOW_DB_PATH = Path(ROOT_DIR / "logs" / "mlflow.db")
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
os.environ.setdefault("MLFLOW_TRACKING_URI", MLFLOW_TRACKING_URI)
mlflow.set_tracking_uri("sqlite:///C:/Users/Alvin/Music/project_1/logs/mlflow.db")

# MLP CONFIGURATION
# NOTE: SET_SEED is located at src/classification/preprocessor.py
# Why not on config_script?
# config_script is also used on regression, which could introduce overhead operation for regression pipeline
CLASSIFICATION_TARGET = "text_risk_score_v2"
BATCH_SIZE = 32
BRANCH_HIDDEN_DIM = 16
HEAD_HIDDEN_DIM = 8
DROPOUT_RATE = 0.3
LEARNING_RATE = 1e-3
N_EPOCHS = 100
WEIGHT_DECAY = 1e-2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def setup_logger(where:str, run_type: str) -> logging.Logger:
    """
    Configures a logger to output to both console and a timestamped file.
    Resolves directories dynamically to prevent path issues.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = LOG_DIR / where / f"{timestamp}_{run_type}.log"

    logger = logging.getLogger(run_type)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # File Handler
        file_handler = logging.FileHandler(log_filename)
        file_formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # Console Handler (Cleaner terminal output)
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s]: %(message)s",
            datefmt="%H:%M:%S"
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
    return logger

numeric_features = [
    'title_length', 
    'title_word_count', 
    'volume_count',
    'text_risk_score',
    'volume_tier_encoded',
    'condition_encoded',
    'total_bonus_count'
]
        
categorical_features = [
    'currency', 
    'seller_location_grouped'
]
        
boolean_features = [
    'is_boxset', 
    'is_special_edition', 
    'boxset_side_story_edition_included', 
    'standalone_side_story_edition',
    'is_first_print',
    'has_signature',
    'has_merch',
    'has_paper_extra',
]
        
feature_cols = numeric_features + categorical_features + boolean_features
target_col = 'price'

class load_dataframe:
    def __init__(self):
        ROOT_DIR = Path(__file__).resolve().parents[1]
        load_dotenv(ROOT_DIR / 'config' / '.env')

        self.DB_USER = os.getenv("DB_USER")
        self.DB_PASSWORD = os.getenv("DB_PASSWORD")
        self.DB_HOST = os.getenv("DB_HOST")
        self.DB_PORT = os.getenv("DB_PORT")
        self.DB_NAME = os.getenv("DB_NAME")

    def create_connection(self):
        self.QUERY_CREATE_ENGINE = (f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}")
        self.engine = create_engine(self.QUERY_CREATE_ENGINE)
        return self.engine
    
    def load_regression_data(self):
        QUERY_REGRESSION = "SELECT * FROM public.fct_ebay_listings"
        self.query_regression = QUERY_REGRESSION
        self.df_regression = pd.read_sql(self.query_regression, self.engine)
        return self.df_regression
    
    def load_classification_data(self):
        QUERY_CLASSIFICATION = "SELECT item_id, title, description, text_risk_score_v2 FROM public.int_ebay_listing_risk_analysis"

        self.query_classification = QUERY_CLASSIFICATION
        self.df_classification = pd.read_sql(self.query_classification, self.engine)
        return self.df_classification
    
    def fit(self, which):
        try:
            self.create_connection()
            if which == 'regression':
                df = self.load_regression_data()
            elif which == 'classification':
                df = self.load_classification_data()
        except Exception as e:
            print("Unable to connect to database, make sure it's Online on Docker!")
            raise e
            
        return df


