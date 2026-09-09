import sys
import os
import numpy as np
import pandas as pd
import mlflow
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

import warnings
warnings.filterwarnings('ignore')

#DB Postgres Connection
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir))
from config.config_script import load_dataframe, setup_logger, X_TRAIN_REGRESSION_PATH, X_TEST_REGRESSION_PATH, Y_TRAIN_REGRESSION_PATH, Y_TEST_REGRESSION_PATH, Y_PRED_REGRESSION_PATH, DF_META_REGRESSION_PATH
from config.config_script import numeric_features, categorical_features, boolean_features, feature_cols, target_col

class preprocess_regression(load_dataframe):
    def __init__(self, random_state=42):
        super().__init__()
        self.logger = setup_logger('regression', 'preprocessor_run')
        self.random_state = random_state
        self.logger.info(f"ROOT DIR: {root_dir}")
        self.logger.info("Initialized preprocessing class.")
    
    def ordinal_encoder(self, df):
        self.logger.info("Executing ordinal encoding process...")
        # A.2.1 Volume Tier

        def get_volume_tier_encoded(count):
            if count == 1:
                return 1
            elif 2 <= count <= 5:
                return 2
            elif 6 <= count <= 15:
                return 3
            else:
                return 4

        condition_map = {
            'Acceptable': 1,
            'Used': 3,
            'Good': 3,
            'Very Good': 4,
            'Like New': 5,
            'New': 6,
            'Brand New': 7,
            'unknown': 3  # Neutral fallback (middle ground)
        }
        
        # Encode Condition
        df['condition_encoded'] = df['condition'].map(condition_map).fillna(3)
        
        # Encode Volume Tier
        df['volume_tier_encoded'] = df['volume_count'].apply(get_volume_tier_encoded)

        location_counts = df['seller_location'].value_counts()
        rare_locations = location_counts[location_counts < 30].index

        df['seller_location_grouped'] = df['seller_location'].replace(
            rare_locations, 'Other'
        )

        df = df.drop(columns=['condition'])
        self.logger.info("Ordinal encoding completed.")
        return df
    
    def split_data(self, df_price_model):
        self.logger.info("Preparing model features and splitting data...")

        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.boolean_features = boolean_features
        self.feature_cols = feature_cols
        self.target_col = target_col

        self.X = df_price_model[self.feature_cols]
        self.y = df_price_model[self.target_col]

        self.df_meta = df_price_model[['item_id', 'title', 'is_special_edition', 'is_boxset']]

        self.X_train, self.X_test, self.y_train, self.y_test, self.indices_train, self.indices_test = train_test_split(
            self.X, self.y, df_price_model.index, test_size = 0.2, random_state = self.random_state
        )

        self.X_train[self.categorical_features] = self.X_train[self.categorical_features].astype(str)
        self.X_test[self.categorical_features] = self.X_test[self.categorical_features].astype(str)
        self.logger.info(f"Train: {len(self.X_train)}, Test: {len(self.X_test)}")
        self.logger.info("Train-test split completed.")

        return self.X_train, self.X_test, self.y_train, self.y_test, self.indices_train, self.indices_test, self.df_meta, self.feature_cols, self.categorical_features, self.boolean_features, self.numeric_features

    def final_filtering(self, df_price_model):

        try:
            self.logger.info("Stage 1 Filtering: Delete High Variance Item with Price >= 999...")
            bulk_lot_mask = (df_price_model['price'] >= 999) 
            self.logger.info(f"Total rows before exclusion: {len(df_price_model)}")
            self.logger.info(f"Rows with volume > 1000: {bulk_lot_mask.sum()}")
            df_price_model = df_price_model[~bulk_lot_mask].copy()

            self.logger.info("Stage 2 Filtering: Delete ambigous price_per_volume price...")
            self.logger.info(f"Rows after exclusion: {len(df_price_model)}")
            bulk_lot_mask = (df_price_model['volume_count'] >= 5) & (df_price_model['price_per_volume'] < 4.4) #cut off
            print(f"Total rows before exclusion: {len(df_price_model)}")
            print(f"Rows matching bulk-lot ambiguous pattern (volume_count >= 20, price_per_volume < 5): {bulk_lot_mask.sum()}")
            df_price_model = df_price_model[~bulk_lot_mask].copy()
            print(f"Rows after exclusion: {len(df_price_model)}")
        except Exception as e:
            self.logger.error("Final Dataframe Filtering is unsuccesfull..")

        return df_price_model
    @staticmethod
    def overview_dataframe(df):
        logger = setup_logger('preprocessor_run')
        logger.info("Generating dataframe overview.")
        print("=" * 50, 'DATAFRAME HEAD', "=" * 60)
        print(df.head(5))
        print("=" * 60, 'DATAFRAME INFO', "=" * 60)
        print(df.info())
        print("=" * 60, 'DATAFRAME STATISTICS', "=" * 60)
        print(df.describe())
        print("=" * 120)
    
    def final_check(self):
        self.logger.info("Executing pipeline diagnostic checks...")
        # Test [1]
        if(np.sum(np.sum(self.df_price_model[self.df_price_model['price'] < 0])) != 0):
            self.logger.error("[FAIL TEST - 1]: Negative values discovered in price target!")
            test_1 = 'Fail'
        else:
            self.logger.info("[PASS TEST - 1]: No negative values in price")
            test_1 = 'Pass'
        
        # Test [2]
        if self.X.isna().sum().sum() != 0:
            self.logger.error(f"[FAIL TEST - 2]: NaN values present in feature matrix X! (Count: {self.X.isna().sum().sum()})")
            test_2 = 'Fail'
        else:
            self.logger.info("[PASS TEST - 2]: No NaN values in X")
            test_2 = 'Pass'
        
        # Test [3]
        if (np.sum(self.X_train['volume_count'] < 0) + np.sum(self.X_test['volume_count'] < 0)) != 0:   
            self.logger.error("[FAIL TEST - 3]: Negative values discovered in volume_count!")
            test_3 = 'Fail'
        else:
            self.logger.info("[PASS TEST - 3]: No negative values in volume_count")
            test_3 = 'Pass'
        
        if test_1 == 'Pass' and test_2 == 'Pass' and test_3 == 'Pass':
            self.logger.info("[PASS TEST - ALL]: Preprocessing diagnostic verification complete.")
        else:
            self.logger.error("[FAIL TEST] Verification tests failed. Halting execution.")
            raise ValueError(f"Preprocessing checks failed: Test1={test_1}, Test2={test_2}, Test3={test_3}")

    def fit_transform(self, df_raw):
        self.logger.info("Starting comprehensive preprocessing pipeline.")
        self.df_price_model = df_raw[(df_raw['risk_category'] != 'High Risk') & (df_raw['volume_confidence'] != 'low')]
        self.df_price_model = self.ordinal_encoder(self.df_price_model)
        self.df_price_model = self.final_filtering(self.df_price_model)
        self.X_train, self.X_test, self.y_train, self.y_test, self.indices_train, self.indices_test, self.df_meta, self.feature_cols, self.categorical_features, self.boolean_features, self.numeric_features = self.split_data(self.df_price_model)
        
        self.final_check()

        if __name__ == "__main__":
            self.overview_dataframe(self.df_price_model)
        self.logger.info("Preprocessing pipeline completed successfully.")

        return self.X_train, self.X_test, self.y_train, self.y_test, self.indices_train, self.indices_test, self.df_meta, self.feature_cols, self.categorical_features, self.boolean_features, self.numeric_features
    
if __name__ == "__main__":
    logger = setup_logger('regression', 'preprocessor_run')
    logger.info("Running preprocessor.py directly...")
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
