import sys
import os
import numpy as np
import pandas as pd
import mlflow
from pathlib import Path
import time, shutil

import warnings
warnings.filterwarnings('ignore')

#DB Postgres Connection
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir))

from config.config_script import (
    load_dataframe, setup_logger, 
    DF_TRAIN_CLASSIFICATION_PATH, DF_VAL_CLASSIFICATION_PATH, DF_TEST_CLASSIFICATION_PATH, 
    TITLE_TRAIN_PATH, TITLE_VAL_PATH, TITLE_TEST_PATH,
    DESCRIPTION_TRAIN_PATH, DESCRIPTION_VAL_PATH, DESCRIPTION_TEST_PATH,
    TITLE_VECTORIZER_PATH, DESCRIPTION_VECTORIZER_PATH, 
    ARTIFACT_DIR, PREPROCESSED_CLASSIFICATION_DIR,
    BATCH_SIZE, BRANCH_HIDDEN_DIM, HEAD_HIDDEN_DIM, DROPOUT_RATE, LEARNING_RATE, N_EPOCHS, WEIGHT_DECAY, DEVICE,
    MODEL_CLASSIFICATION_PATH, CLASSIFICATION_TARGET
)

try:
    from .preprocessor import preprocess_classification, TextVectorizer, save_sparse_matrix_as_parquet, set_seed
    from .train import (RiskDataset, load_data, load_vectorizer, train_and_evaluate_model,
                        train_loop, evaluate_loop, save_model_artifact, model_setup,
                        configure_mlflow, log_classification_artifacts)
    from .evaluate import Evaluate_MLP
except ImportError:
    from preprocessor import preprocess_classification, TextVectorizer, save_sparse_matrix_as_parquet, set_seed
    from train import (RiskDataset, load_data, load_vectorizer, train_and_evaluate_model,
                       train_loop, evaluate_loop, save_model_artifact, model_setup,
                       configure_mlflow, log_classification_artifacts)
    from evaluate import Evaluate_MLP

if __name__ == "__main__":
    main_logger = setup_logger('classification', 'preprocessor_run')
    set_seed(42, main_logger)
    main_logger.info(f"=" * 75)
    main_logger.info("Running pipeline orchestration...")

    def preprocessor():
        logger = setup_logger('classification', 'preprocessor_run')
        logger.info(f"[preprocessor]: Preprocessing data...")

        try:
            logger.info("[ATTEMPTING]: Cleaning & Splitting the Dataframe...")
            df_train, df_val, df_test = preprocess_classification().fit_transform()
        except Exception as e:
            logger.error("[FAILED]: Fail to clean or splitting the Dataframe...")
            logger.error(f"{e}")
            raise e

        try:
            logger.info(f"[ATTEMPTING]: Saving splitted dataframe into {PREPROCESSED_CLASSIFICATION_DIR}")
            pd.DataFrame(df_train).to_parquet(DF_TRAIN_CLASSIFICATION_PATH, index=False)
            pd.DataFrame(df_val).to_parquet(DF_VAL_CLASSIFICATION_PATH, index=False)
            pd.DataFrame(df_test).to_parquet(DF_TEST_CLASSIFICATION_PATH, index=False)
            logger.info(f"[SUCCESS]: Splitted Dataframe succesfully saved!")
        except Exception as e:
            logger.error(f"[FAILED]: Error saving splitted dataframe!")
            logger.error(f"{e}")
            raise e

        try:
            logger.info("[ATTEMPTING]: Vectorizing title and description...")
            vectorizer = TextVectorizer(
                title_max_features=3000,
                desc_max_features=4000,
                ngram_range=(1, 2),
                min_df=2,
            )

            vectorizer.fit(df_train)
            
            title_train, desc_train = vectorizer.transform(df_train)
            title_val, desc_val = vectorizer.transform(df_val)
            title_test, desc_test = vectorizer.transform(df_test)

            logger.info("[SUCCESS]: Title & Description product succesfully vectorized!")
        except Exception as e:
            logger.error(f"[FAILED]: Error vectorizing title and description model!")
            logger.error(f"{e}")
            raise e

        try:
            logger.info("Saving title and description train/val/test..")
            save_sparse_matrix_as_parquet(title_train, TITLE_TRAIN_PATH)
            save_sparse_matrix_as_parquet(title_val, TITLE_VAL_PATH)
            save_sparse_matrix_as_parquet(title_test, TITLE_TEST_PATH)
            save_sparse_matrix_as_parquet(desc_train, DESCRIPTION_TRAIN_PATH)
            save_sparse_matrix_as_parquet(desc_val, DESCRIPTION_VAL_PATH)
            save_sparse_matrix_as_parquet(desc_test, DESCRIPTION_TEST_PATH)
            logger.info("Title and description matrices saved as Parquet.")
            
        except Exception as e:
            raise e

        try:
            logger.info(f"[ATTEMPTING]: Dumping title & description vectorizer into {ARTIFACT_DIR}...")
            vectorizer.save()
            logger.info(f"[SUCCESS]: Succesfully saved title & description vectorizer...")
        except Exception as e:
            logger.error(f"[FAILED]: Error saving title & description vectorizer caused by")
            logger.error(f'{e}')
            raise e

        logger.info(f"  Shapes:")
        logger.info(f"  title_train: {title_train.shape}, desc_train: {desc_train.shape}")
        logger.info(f"  title_val  : {title_val.shape}, desc_val  : {desc_val.shape}")
        logger.info(f"  title_test : {title_test.shape}, desc_test : {desc_test.shape}")

    def train_and_evaluate():
        logger = setup_logger('classification', 'train_run')
        logger.info(f"=" * 75)
        logger.info(f"[train & evaluate]: Training & Evaluting the model...")
        set_seed(42, logger)

        try:
            title_train = pd.read_parquet(TITLE_TRAIN_PATH).to_numpy()
            desc_train = pd.read_parquet(DESCRIPTION_TRAIN_PATH).to_numpy()
            title_val = pd.read_parquet(TITLE_VAL_PATH).to_numpy()
            desc_val = pd.read_parquet(DESCRIPTION_VAL_PATH).to_numpy()
            title_test = pd.read_parquet(TITLE_TEST_PATH).to_numpy()
            desc_test = pd.read_parquet(DESCRIPTION_TEST_PATH).to_numpy()

            df_train = pd.read_parquet(DF_TRAIN_CLASSIFICATION_PATH)
            df_val = pd.read_parquet(DF_VAL_CLASSIFICATION_PATH)
            df_test = pd.read_parquet(DF_TEST_CLASSIFICATION_PATH)
            
            train_dataset = RiskDataset(title_train, desc_train, df_train[CLASSIFICATION_TARGET])
            val_dataset = RiskDataset(title_val, desc_val, df_val[CLASSIFICATION_TARGET])
            test_dataset = RiskDataset(title_test, desc_test, df_test[CLASSIFICATION_TARGET])

            print(f"Train dataset size: {len(train_dataset)}")
            print(f"Val dataset size  : {len(val_dataset)}")
            print(f"Test dataset size : {len(test_dataset)}")

            # Sanity check: pull one sample and inspect shapes
            sample = train_dataset[0]
            print(f"\nSample shapes:")
            print(f"  title       : {sample['title'].shape}")
            print(f"  description : {sample['description'].shape}")
            print(f"  label       : {sample['label'].item()}")
        except Exception as e:
            logger.error(f"[FAILED] Error initiating data load and/or object import!")
            logger.error(f"{e}")
            raise e

        try:
            logger.info("[ATTEMPT - evaluate.py]: Estimating fraction of unique tokens in `texts` that do NOT exist in the vectorizer's fitted vocabulary")
            evaluate = Evaluate_MLP()
            evaluate.check_oov_rate()

        except Exception as e:
            logger.error(f"[FAILED]: Fail to estimate unique token in test set!")
            logger.error(f"{e}")
            raise e

        try:
            logger.info(f"[ATTEMPT]: Loading data_loader and vectorizer...")
            train_loader, val_loader, test_loader = load_data(
                train_dataset=train_dataset, 
                val_dataset=val_dataset, 
                test_dataset=test_dataset, 
                BATCH_SIZE=BATCH_SIZE
            )
            title_vectorizer, desc_vectorizer = load_vectorizer()
        except Exception as e:
            logger.error(f"[FAILED]: Fail to load vectorizer or data loader properly!")
            logger.error(f"{e}")
            raise e

        # TRAINING THE MODEL:

        try:
            logger.info("[ATTEMPT]: Setting up MLP model...")
            model, criterion, optimizer, scheduler, early_stopper, history = model_setup(
                logger, df_train, LEARNING_RATE, WEIGHT_DECAY
            )
            logger.info(f"Model Summary:\n{model}")
        except Exception as e:
            logger.error(f"[ERROR]: There is an error setting up the model!")
            logger.error(f"{e}")
            raise e

        # TRAINING LOOP:
        mlflow_experiment, mlflow_artifact_location = configure_mlflow()
        mlflow_run = mlflow.start_run(run_name="classification_pipeline_run")
        mlflow.set_tag("model_family", "two_branch_mlp")
        mlflow.set_tag("pipeline_stage", "classification_training")
        mlflow.log_params({
            "experiment_name": mlflow_experiment,
            "training_rows": len(df_train),
            "validation_rows": len(df_val),
            "test_rows": len(df_test),
            "title_input_dim": len(title_vectorizer.vocabulary_),
            "description_input_dim": len(desc_vectorizer.vocabulary_),
            "batch_size": BATCH_SIZE,
            "branch_hidden_dim": BRANCH_HIDDEN_DIM,
            "head_hidden_dim": HEAD_HIDDEN_DIM,
            "dropout_rate": DROPOUT_RATE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "max_epochs": N_EPOCHS,
            "random_state": 42,
            "device": DEVICE,
            "target_column": CLASSIFICATION_TARGET,
            "artifact_location": mlflow_artifact_location,
        })
        try:
            logger.info(f"[ATTEMPT]: Beggining Training Loop")
            model = train_and_evaluate_model(train_loader, criterion, optimizer, model, val_loader, 
                                             train_loop, evaluate_loop, scheduler, history, logger, early_stopper)
        except Exception as e:
            logger.error(f"[ERROR]: There is an error while running training loop!")
            logger.error(f"{e}")
            raise e
        
        #EVALUATE:
        try:
            logger.info(f"[ATTEMPT]: Evaluating trained model...")
            history_path = evaluate.save_history(history)
            logger.info(f"Training history saved to: {history_path}")
            evaluate.plot_training_model(history)
        except Exception as e:
            logger.error(f"[FAILED]: There is an error while running evaluation loop!")
            logger.error(f"{e}")
            raise e

        model_path = save_model_artifact(
            model,
            title_input_dim=len(title_vectorizer.vocabulary_),
            desc_input_dim=len(desc_vectorizer.vocabulary_),
            branch_hidden_dim=BRANCH_HIDDEN_DIM,
            head_hidden_dim=HEAD_HIDDEN_DIM,
            dropout_rate=DROPOUT_RATE,
        )

        test_metrics = evaluate.evaluate_test(
            history=history,
            model=model,
            test_loader=test_loader,
            criterion=criterion,
            device=DEVICE,
        )
        model_config = {
            "title_input_dim": len(title_vectorizer.vocabulary_),
            "desc_input_dim": len(desc_vectorizer.vocabulary_),
            "branch_hidden_dim": BRANCH_HIDDEN_DIM,
            "head_hidden_dim": HEAD_HIDDEN_DIM,
            "dropout_rate": DROPOUT_RATE,
            "classification_target": CLASSIFICATION_TARGET,
            "device": DEVICE,
        }
        log_classification_artifacts(model_config, history, test_metrics)
        logger.info(f"[SUCCESS] MLflow run saved: {mlflow_run.info.run_id}")
        mlflow.end_run()
        logger.info(f"[SUCCESS] Classification model saved to: {model_path}")

    def finalizing():
        logger = setup_logger('classification', 'evaluation_run')
        file_to_delete_1 = Path(root_dir / 'src' / 'classification' / 'mlruns')
        file_to_delete_2 = Path(root_dir / 'src' / 'classification' / '__pycache__')
        
        time.sleep(3) #letting mlflow output cache due to directory diff for a moment before deleting
        
        try:
            main_logger.info("[FINALIZING]: Deleting local cache...")
            shutil.rmtree(file_to_delete_1, ignore_errors=True)
            shutil.rmtree(file_to_delete_2, ignore_errors=True)
            main_logger.info("[FINALIZING]: Cache deleted succesfully!")
            main_logger.info("[FINALIZED]: Pipeline executed succesfully!")
        except Exception as e:
            main_logger.info("[ERROR]: Error deleting cache")
            raise e
        
        logger.info('To view all pipeline run(s) on MLFLOW:')
        logger.info("Copy this to your IDE terminal: 'mlflow ui --backend-store-uri sqlite:///C:/Users/Alvin/Music/project_1/logs/mlflow.db --default-artifact-root file:///C:/Users/Alvin/Music/project_1/logs/mlartifacts --host 127.0.0.1 --port 5000'")

    preprocessor()
    train_and_evaluate()
    finalizing()