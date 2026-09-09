import os
import sys
import json
import numpy as np
import pandas as pd
import mlflow
import joblib
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from torch.utils.data import DataLoader

import warnings
warnings.filterwarnings('ignore')

#DB Postgres Connection
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir))

from config.config_script import (
    load_dataframe, setup_logger, 
    DF_TRAIN_CLASSIFICATION_PATH, DF_VAL_CLASSIFICATION_PATH, DF_TEST_CLASSIFICATION_PATH, 
    TITLE_TEST_PATH, DESCRIPTION_TEST_PATH,
    EVAL_TRAINING_LOOP_CLASSIFICATION_PATH, EVAL_HISTORY_CLASSIFICATION_PATH,
    EVAL_SUMMARY_CLASSIFICATION_PATH, MODEL_CLASSIFICATION_PATH,
    DEVICE, BATCH_SIZE, CLASSIFICATION_TARGET
)

try:
    from .preprocessor import TextVectorizer, set_seed
except ImportError:
    from preprocessor import TextVectorizer, set_seed
from sklearn.feature_extraction.text import TfidfVectorizer

class Evaluate_MLP(TextVectorizer):

    def __init__(self):
        super().__init__()
        logger = setup_logger('classification', 'evaluation_run')
        self.logger = setup_logger('classification', 'evaluation_run')
        set_seed(42, self.logger)
        self.instance = TextVectorizer.load()

        self.text_vectorizer = self.instance.title_vectorizer
        self.desc_vectorizer = self.instance.desc_vectorizer

        self.df_train = pd.read_parquet(DF_TRAIN_CLASSIFICATION_PATH)
        self.df_val = pd.read_parquet(DF_VAL_CLASSIFICATION_PATH)
        self.df_test = pd.read_parquet(DF_TEST_CLASSIFICATION_PATH)

    @staticmethod
    def save_history(history, save_path=EVAL_HISTORY_CLASSIFICATION_PATH):
        """Persist epoch metrics so evaluation can run in a separate process."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("w", encoding="utf-8") as history_file:
            json.dump(history, history_file, indent=2)
        return save_path

    @staticmethod
    def load_history(load_path=EVAL_HISTORY_CLASSIFICATION_PATH):
        """Load the training history saved by the training process."""
        load_path = Path(load_path)
        if not load_path.exists():
            raise FileNotFoundError(
                f"Training history not found at {load_path}. Run train.py first."
            )
        with load_path.open("r", encoding="utf-8") as history_file:
            return json.load(history_file)

    @staticmethod
    def print_history(history):
        """Print the saved epoch metrics in a readable table."""
        metric_names = [
            "train_loss", "val_loss", "val_acc", "val_precision",
            "val_recall", "val_f1", "val_auc", "lr",
        ]
        epoch_count = len(history.get("train_loss", []))
        for epoch_index in range(epoch_count):
            values = [
                f"{history[name][epoch_index]:.4f}"
                for name in metric_names
            ]
            logger.info(f"Epoch {epoch_index + 1:>3}: " + " | ".join(values))

    @staticmethod
    def save_evaluation_report(
        history,
        test_metrics=None,
        save_path=EVAL_SUMMARY_CLASSIFICATION_PATH,
    ):
        """Save best validation metrics and optional final test metrics."""
        if not history.get("val_loss"):
            raise ValueError("Training history does not contain validation metrics.")

        best_epoch_idx = history["val_loss"].index(min(history["val_loss"]))
        best_epoch = best_epoch_idx + 1
        report = {
            "best_epoch": best_epoch,
            "best_validation_metrics": {
                "loss": history["val_loss"][best_epoch_idx],
                "accuracy": history["val_acc"][best_epoch_idx],
                "precision": history["val_precision"][best_epoch_idx],
                "recall": history["val_recall"][best_epoch_idx],
                "f1": history["val_f1"][best_epoch_idx],
                "auc": history["val_auc"][best_epoch_idx],
            },
            "epochs_recorded": len(history["train_loss"]),
        }

        if test_metrics is not None:
            report["test_metrics"] = test_metrics
            report["validation_to_test_gap"] = {
                "f1": report["best_validation_metrics"]["f1"] - test_metrics["f1"],
                "auc": report["best_validation_metrics"]["auc"] - test_metrics["auc"],
            }

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("w", encoding="utf-8") as report_file:
            json.dump(report, report_file, indent=2)
        return save_path

    def oov_rate(self, vectorizer: TfidfVectorizer, texts) -> float:
        """
        Estimate the out-of-vocabulary (OOV) rate: the fraction of
        unique tokens in `texts` that do NOT exist in the vectorizer's
        fitted vocabulary. Uses the vectorizer's own tokenizer/preprocessor
        so tokenization rules (lowercasing, ngram splitting) stay consistent
        with what transform() actually does.
        """
        # Build the analyzer callable using the SAME rules the vectorizer
        # itself uses internally (tokenization, lowercasing, ngram splitting)
        analyzer = vectorizer.build_analyzer()

        known_vocab = set(vectorizer.vocabulary_.keys())
        all_tokens = set()
        for text in texts:
            all_tokens.update(analyzer(text))

        oov_tokens = all_tokens - known_vocab
        oov_rate = len(oov_tokens) / len(all_tokens) if all_tokens else 0.0
        return oov_rate, oov_tokens

    def check_oov_rate(self):
        # Example usage on the description vectorizer
        oov_rate, oov_tokens = self.oov_rate(self.desc_vectorizer, self.df_test["description"])
        self.logger.info(f"OOV rate (unique tokens) in test description: {oov_rate:.2%}")
        self.logger.info(f"Sample OOV tokens: {list(oov_tokens)[:20]}")

    def plot_training_model(self, history, save_path=EVAL_TRAINING_LOOP_CLASSIFICATION_PATH):
        import os
        import matplotlib.pyplot as plt

        epochs = range(1, len(history["train_loss"]) + 1)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(epochs, history["train_loss"], label="Train Loss", marker="o")
        axes[0].plot(epochs, history["val_loss"], label="Val Loss", marker="o")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Train vs Val Loss")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        axes[1].plot(epochs, history["val_f1"], label="Val F1", marker="o")
        axes[1].plot(epochs, history["val_recall"], label="Val Recall", marker="o")
        axes[1].plot(epochs, history["val_precision"], label="Val Precision", marker="o")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Score")
        axes[1].set_title("Val Metrics Over Time")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        plt.tight_layout()

        if save_path:
            # Create output directory automatically if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            
            # Save figure with high DPI and bounded layout
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            self.logger.info(f"Plot saved to: {save_path}")
            plt.close(fig)  # Free up RAM memory after saving
        else:
            plt.show()

        best_epoch = history["val_loss"].index(min(history["val_loss"])) + 1
        self.logger.info(f"Lowest val_loss at epoch {best_epoch}: {min(history['val_loss']):.4f}")

    def evaluate_test(self, history = None, model = None, test_loader = None, criterion = None, device = DEVICE):
        try:
            from .train import evaluate_loop
        except ImportError:
            from train import evaluate_loop

        try:
            if model is None or history is None or criterion is None or test_loader is None:
                try:
                    from .train import RiskDataset, model_setup
                except ImportError:
                    from train import RiskDataset, model_setup

                title_test = pd.read_parquet(TITLE_TEST_PATH).to_numpy()
                desc_test = pd.read_parquet(DESCRIPTION_TEST_PATH).to_numpy()
                test_dataset = RiskDataset(
                    title_test,
                    desc_test,
                    self.df_test[CLASSIFICATION_TARGET],
                )
                test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

                model, criterion, optimizer, scheduler, early_stopper, temp_history = model_setup(
                    logger=self.logger,
                    df_train=self.df_train,
                )
                checkpoint = torch.load(MODEL_CLASSIFICATION_PATH, map_location=device)
                model.load_state_dict(checkpoint["model_state_dict"])
                model.to(device)

                history = self.load_history(EVAL_HISTORY_CLASSIFICATION_PATH)
        except Exception as e:
            self.logger.error("[FAILED]")
            self.logger.error(f"{e}")
            raise e
    
        test_metrics = evaluate_loop(model, test_loader, criterion, device)

        self.logger.info("=== MLP -- Final Test Set Metrics (best checkpoint) ===")
        self.logger.info(f"Test Loss      : {test_metrics['loss']:.4f}")
        self.logger.info(f"Test Accuracy  : {test_metrics['accuracy']:.3f}")
        self.logger.info(f"Test Precision : {test_metrics['precision']:.3f}")
        self.logger.info(f"Test Recall    : {test_metrics['recall']:.3f}")
        self.logger.info(f"Test F1        : {test_metrics['f1']:.3f}")
        self.logger.info(f"Test AUC       : {test_metrics['auc']:.3f}")

        # Direct comparison with the best validation metrics at the restored checkpoint
        # makes the generalization gap explicit, following the same pattern as the
        # LOSO versus random-split gap documented for XGBoost.
        best_epoch_idx = history["val_loss"].index(min(history["val_loss"]))
        self.logger.info("\n=== Validation metrics at the same checkpoint (for comparison) ===")
        self.logger.info(f"Val Loss      : {history['val_loss'][best_epoch_idx]:.4f}")
        self.logger.info(f"Val Accuracy  : {history['val_acc'][best_epoch_idx]:.3f}")
        self.logger.info(f"Val Precision : {history['val_precision'][best_epoch_idx]:.3f}")
        self.logger.info(f"Val Recall    : {history['val_recall'][best_epoch_idx]:.3f}")
        self.logger.info(f"Val F1        : {history['val_f1'][best_epoch_idx]:.3f}")
        self.logger.info(f"Val AUC       : {history['val_auc'][best_epoch_idx]:.3f}")

        self.logger.info(f"\nVal->Test F1 gap : {history['val_f1'][best_epoch_idx] - test_metrics['f1']:+.3f}")
        self.logger.info(f"Val->Test AUC gap: {history['val_auc'][best_epoch_idx] - test_metrics['auc']:+.3f}")

        report_path = self.save_evaluation_report(history, test_metrics)
        self.logger.info(f"Evaluation report saved to: {report_path}")
        return test_metrics

if __name__ == "__main__":
    logger = setup_logger('classification', 'evaluate_run')
    evaluate = Evaluate_MLP()
    
    saved_history = evaluate.load_history()
    evaluate.evaluate_test()
    

