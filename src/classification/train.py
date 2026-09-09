import sys
import os
import numpy as np
import pandas as pd
import mlflow
import joblib
import time
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix

import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import copy

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
    MODEL_CLASSIFICATION_PATH, CLASSIFICATION_TARGET, MLFLOW_MLRUNS_PATH,
    MLFLOW_TRACKING_URI, EVAL_TRAINING_LOOP_CLASSIFICATION_PATH,
    EVAL_HISTORY_CLASSIFICATION_PATH, EVAL_SUMMARY_CLASSIFICATION_PATH
)

try:
    from .preprocessor import preprocess_classification, TextVectorizer, set_seed
    from .evaluate import Evaluate_MLP
except ImportError:
    from preprocessor import preprocess_classification, TextVectorizer, set_seed
    from evaluate import Evaluate_MLP

import random
import torch
from torch.utils.data import Dataset
from scipy.sparse import csr_matrix

class RiskDataset(Dataset, preprocess_classification):

    def __init__(self, title_matrix: csr_matrix, desc_matrix: csr_matrix, labels):

        # Basic shape consistency check -- catches a mismatched split
        # (e.g. accidentally passing train titles with val labels) at
        # construction time instead of failing cryptically mid-training.
        assert title_matrix.shape[0] == desc_matrix.shape[0] == len(labels), (
            f"Mismatched row counts: title={title_matrix.shape[0]}, "
            f"desc={desc_matrix.shape[0]}, labels={len(labels)}"
        )

        self.title_matrix = title_matrix
        self.desc_matrix = desc_matrix
        self.logger = setup_logger('classification', 'train_run')

        # Convert raw text_risk_score (0 or 70) into binary label (0.0 or 1.0).
        # text_risk_score > 0 means a risk keyword was matched -> label 1
        # ("unofficial/risky"); text_risk_score == 0 -> label 0 ("official/safe").
        labels_array = np.asarray(labels)
        self.labels = (labels_array > 0).astype(np.float32)

    def __len__(self) -> int:
        """Total number of samples. Called by DataLoader to know how many
        batches to produce per epoch."""
        return self.title_matrix.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Return a single sample at position idx.

        .toarray() converts ONE ROW of the sparse matrix to a dense numpy
        array. This is cheap (one row, not the whole matrix) and is what
        keeps memory usage manageable -- the full dense conversion never
        happens all at once.

        [0] after .toarray() flattens the (1, vocab_size) row-slice result
        into a plain (vocab_size,) 1D array, since indexing a sparse matrix
        with a single integer still returns a 2D (1, N) shape.
        """
        title_row = self.title_matrix[idx]
        desc_row = self.desc_matrix[idx]

        title_dense = title_row.toarray()[0] if hasattr(title_row, "toarray") else np.asarray(title_row)
        desc_dense = desc_row.toarray()[0] if hasattr(desc_row, "toarray") else np.asarray(desc_row)

        return {
            "title": torch.tensor(title_dense, dtype=torch.float32),
            "description": torch.tensor(desc_dense, dtype=torch.float32),
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

class RiskClassifier(nn.Module):
    """
    Two-branch MLP for the official/unofficial risk classifier.

    Each branch (title, description) independently compresses its
    high-dimensional TF-IDF input into a smaller dense representation
    before the two are concatenated and passed through a shared head.

    Parameters
    ----------
    title_input_dim : int
        Size of the title TF-IDF vector (= title vocabulary size).
    desc_input_dim : int
        Size of the description TF-IDF vector (= description vocabulary size).
    branch_hidden_dim : int
        Output size of each branch's hidden layer (the "compressed"
        representation size per branch, before concatenation).
    head_hidden_dim : int
        Size of the hidden layer in the shared head, after concatenation.
    dropout_rate : float
        Fraction of neurons randomly zeroed during training, applied
        after every ReLU. Only active during model.train(); automatically
        disabled during model.eval().
    """

    def __init__(
        self,
        title_input_dim: int,
        desc_input_dim: int,
        branch_hidden_dim: int = 64,
        head_hidden_dim: int = 32,
        dropout_rate: float = 0.3,
    ):
        
        super().__init__()

        # ---- Title branch ----
        # Linear(in, out): a fully-connected layer. Every one of the
        # `title_input_dim` input features connects to every one of the
        # `branch_hidden_dim` output neurons, each connection having its
        # own learnable weight (plus one learnable bias per output neuron).
        self.title_branch = nn.Sequential(
            nn.Linear(title_input_dim, branch_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        # ---- Description branch ----
        # Structurally identical to the title branch, but with its OWN
        # independent weights -- nn.Sequential here is a separate object,
        # so title_branch and desc_branch never share parameters.
        self.desc_branch = nn.Sequential(
            nn.Linear(desc_input_dim, branch_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        # ---- Shared head ----
        # Input size = branch_hidden_dim * 2, because we concatenate the
        # two branch outputs (each of size branch_hidden_dim) side by side.
        self.head = nn.Sequential(
            nn.Linear(branch_hidden_dim * 2, head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(head_hidden_dim, 1),  # final layer: 1 output = raw logit
        )

    def forward(self, title_vec: torch.Tensor, desc_vec: torch.Tensor) -> torch.Tensor:
        """
        Define the actual data flow. Called automatically when you do
        model(title_batch, desc_batch) -- you never call forward() directly.

        Parameters
        ----------
        title_vec : torch.Tensor, shape (batch_size, title_input_dim)
        desc_vec  : torch.Tensor, shape (batch_size, desc_input_dim)

        Returns
        -------
        torch.Tensor, shape (batch_size, 1)
            Raw logits -- NOT probabilities. Sigmoid is applied later by
            BCEWithLogitsLoss during training, and manually during
            inference/evaluation (explained in the loss function section).
        """
        # Each branch processes its own input independently.
        title_repr = self.title_branch(title_vec)   # (batch_size, branch_hidden_dim)
        desc_repr = self.desc_branch(desc_vec)       # (batch_size, branch_hidden_dim)

        # Concatenate along dim=1 (the feature dimension, not the batch
        # dimension). dim=0 would be batch, dim=1 is features per sample --
        # getting this wrong is a very common bug, so it's worth double-
        # checking with .shape after this line while debugging.
        combined = torch.cat([title_repr, desc_repr], dim=1)  # (batch_size, branch_hidden_dim * 2)

        logits = self.head(combined)  # (batch_size, 1)
        return logits

class EarlyStopping:
    """
    Tracks validation loss across epochs and signals when training should
    stop. Encapsulated as a class (rather than loose variables in the loop)
    because it needs to persist state (best_loss, counter) across many
    calls -- one per epoch -- which is exactly the kind of state a class
    is meant to hold.

    Parameters
    ----------
    patience : int
        Number of consecutive epochs with no improvement to tolerate
        before signaling stop.
    min_delta : float
        Minimum decrease in val_loss to count as "improvement". Guards
        against stopping being triggered by negligible float noise
        (e.g. val_loss going from 0.2001 to 0.2000 shouldn't reset patience).
    """

    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.best_model_state = None  # holds the best weights seen so far

    def check(self, val_loss: float, model: nn.Module) -> bool:
        """
        Call once per epoch, after computing val_loss. Returns True if
        training should stop now.
        """
        if val_loss < self.best_loss - self.min_delta:
            # New best -- reset patience counter, checkpoint these weights.
            self.best_loss = val_loss
            self.counter = 0
            # deepcopy is essential here: model.state_dict() by default
            # returns references tied to the live model, which keeps
            # changing every epoch. Without deepcopy, "best_model_state"
            # would silently become whatever the CURRENT weights are by
            # the time you load it back, not the actual best epoch's weights.
            self.best_model_state = copy.deepcopy(model.state_dict())
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

def load_data(train_dataset, val_dataset, test_dataset, BATCH_SIZE):
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,   # reshuffle order every epoch -- prevents the model
                        # from learning any accidental pattern tied to row order
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,  # no need to shuffle -- we're not updating weights here
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # Sanity check: pull one batch and inspect shapes.
    # Expect: title -> (BATCH_SIZE, title_vocab_size)
    #         description -> (BATCH_SIZE, desc_vocab_size)
    #         label -> (BATCH_SIZE,)
    first_batch = next(iter(train_loader))
    print("Batch shapes:")
    print(f"  title       : {first_batch['title'].shape}")
    print(f"  description : {first_batch['description'].shape}")
    print(f"  label       : {first_batch['label'].shape}")

    sample = train_dataset[0]
    print(sample["title"].dtype)         # must -> torch.float32
    print(sample["label"].dtype)         # must -> torch.float32

    n_batches_per_epoch = len(train_loader)
    print(f"\nBatches per epoch (train): {n_batches_per_epoch}")
    print(f"(= ceil({len(train_dataset)} / {BATCH_SIZE}))")

    return train_loader, val_loader, test_loader
def load_vectorizer():
    instance = TextVectorizer().load()
    title_vectorizer = instance.title_vectorizer
    desc_vectorizer = instance.desc_vectorizer

    return title_vectorizer, desc_vectorizer
# MODEL CONFIGURATION BASED ON root_dir / config / config_script.py
def model_setup(logger = None, 
                df_train = None,
                LEARNING_RATE = LEARNING_RATE,
                WEIGHT_DECAY = WEIGHT_DECAY):
    
    title_vectorizer, desc_vectorizer = load_vectorizer()
    
    if logger is None:
        logger = setup_logger('classification', 'train_run')

    if df_train is None:
        df_train = pd.read_parquet(DF_TRAIN_CLASSIFICATION_PATH)

    TITLE_INPUT_DIM = len(title_vectorizer.vocabulary_)
    DESC_INPUT_DIM = len(desc_vectorizer.vocabulary_)
    
    model = RiskClassifier(
        title_input_dim=TITLE_INPUT_DIM,
        desc_input_dim=DESC_INPUT_DIM,
        branch_hidden_dim=BRANCH_HIDDEN_DIM,
        head_hidden_dim=HEAD_HIDDEN_DIM,
        dropout_rate=DROPOUT_RATE
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"\nTotal trainable parameters: {n_params:,}")

    n_negative = (df_train[CLASSIFICATION_TARGET] == 0).sum()
    n_positive = (df_train[CLASSIFICATION_TARGET] == 70).sum()
    pos_weight_value = n_negative / n_positive

    print(f"n_negative: {n_negative}, n_positive: {n_positive}")
    print(f"pos_weight: {pos_weight_value:.3f}")

    device = torch.device(DEVICE)
    print(f"Using device: {device}")

    # SETTING UP MODEL & OPTIMIZER
    pos_weight_tensor = torch.tensor([pos_weight_value], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2,
    )

    early_stopper = EarlyStopping(patience=8, min_delta=1e-4)

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": [], "val_recall": [], "val_precision": [], "val_auc": [], "lr": []}

    return model, criterion, optimizer, scheduler, early_stopper, history

# TRAINING & EVALUATION LOOP
def train_loop(model, loader, criterion, optimizer, device) -> float:
    model.train()

    running_loss = 0.0
    n_batches = len(loader)

    progress_bar = tqdm(loader, desc = 'training', leave = True)

    for batch in progress_bar:
        # Move this batch's tensors to the same device as the model (into GPU memory for cuda).
        title_batch = batch["title"].to(device)
        desc_batch = batch["description"].to(device)

        # .unsqueeze(1) reshapes labels from (batch_size,) to (batch_size, 1),
        # matching BCEWithLogitsLoss format prediction and target.
        labels_batch = batch["label"].to(device).unsqueeze(1)

        # 5 main training step
        optimizer.zero_grad()                   #1. reset gradients
        logits = model(title_batch, desc_batch)  #2. forward pass
        loss = criterion(logits, labels_batch)    #3. compute loss
        loss.backward()                            #4. backward pass
        optimizer.step()                            #5. update weights

        running_loss += loss.item()         #.item() extract python float from 1-element tensor

        progress_bar.set_postfix(loss = f"{running_loss / (progress_bar.n + 1):.4f}")

    avg_loss = running_loss / n_batches
    return avg_loss
def evaluate_loop(model, loader, criterion, device) -> dict:
    """
    Run one full pass over a dataset WITHOUT updating weights -- used for
    both validation (during training, to monitor progress) and final test
    evaluation. Returns average loss plus classification metrics.

    Note: labels and predictions are collected across the WHOLE loader
    before computing precision/recall/F1, rather than averaging per-batch
    metrics -- per-batch F1 averaging would be mathematically incorrect
    (F1 is not a simple average-friendly metric across unevenly-sized
    or class-imbalanced batches).
    """

    model.eval() #disable dropout in eval mode

    running_loss = 0.0
    all_labels = []
    all_preds = [] # binary predictions (0/1), after thresholding at 0.5

    with torch.no_grad():
        progress_bar = tqdm(loader, desc = 'Evaluating', leave = False)
        for batch in progress_bar:
            title_batch = batch["title"].to(device)
            desc_batch = batch["description"].to(device)
            labels_batch = batch["label"].to(device).unsqueeze(1)

            logits = model(title_batch, desc_batch)
            loss = criterion(logits, labels_batch)
            running_loss += loss.item()

            # Manually apply sigmoid here -- the model only outputs raw
            # logits (explained earlier), so converting to probability is
            # our responsibility at inference/evaluation time.
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float() # threshold at 0.5 -> binary prediction

            all_labels.extend(labels_batch.cpu().numpy().flatten())
            all_preds.extend(preds.cpu().numpy().flatten())

        avg_loss = running_loss / len(loader)

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

        return {
        "loss": avg_loss,
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
        "auc": roc_auc_score(all_labels, all_preds)
    }

def save_model_artifact(model, path=MODEL_CLASSIFICATION_PATH, **model_config):
    """Save the trained classifier weights and reconstruction configuration."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model_state_dict": model.state_dict(),
        "model_config": model_config,
    }

    torch.save(artifact, path)
    return path

def configure_mlflow():
    experiment_name = "light_novel_risk_classification"
    artifact_location = MLFLOW_MLRUNS_PATH.resolve().as_uri()
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(
            experiment_name,
            artifact_location=artifact_location,
        )
    mlflow.set_experiment(experiment_name)
    return experiment_name, artifact_location

def log_classification_artifacts(model_config, history, test_metrics):
    mlflow.log_dict(model_config, "artifacts/model_config.json")
    mlflow.log_dict(history, "artifacts/training_history.json")

    for artifact_path in [
        EVAL_TRAINING_LOOP_CLASSIFICATION_PATH,
        EVAL_HISTORY_CLASSIFICATION_PATH,
        EVAL_SUMMARY_CLASSIFICATION_PATH,
        MODEL_CLASSIFICATION_PATH,
        TITLE_VECTORIZER_PATH,
        DESCRIPTION_VECTORIZER_PATH,
    ]:
        if Path(artifact_path).exists():
            mlflow.log_artifact(str(artifact_path), artifact_path="artifacts")

    mlflow.log_metrics({f"test_{key}": float(value) for key, value in test_metrics.items()})
def load_model_artifact():
    artifact = torch.load(path=MODEL_CLASSIFICATION_PATH, map_location="cpu")
    return artifact

# USING TRAINING LOOP V2 on 02_EDA.ipynb
def train_and_evaluate_model(train_loader, criterion, optimizer, model, val_loader, train_model, 
                             evaluate_model, scheduler, history, logger, early_stopper):
    for epoch in range(1, N_EPOCHS + 1):
        train_loss = train_model(model, train_loader, criterion, optimizer, DEVICE)
        val_metrics = evaluate_model(model, val_loader, criterion, DEVICE)

        # scheduler.step() must be called with the metric it's monitoring --
        # this is what lets it decide whether to shrink the learning rate
        scheduler.step(val_metrics["loss"])
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_f1"].append(val_metrics["f1"])
        history["val_recall"].append(val_metrics["recall"])
        history["val_precision"].append(val_metrics["precision"])
        history["val_auc"].append(val_metrics["auc"])
        history["lr"].append(current_lr)

        if mlflow.active_run() is not None:
            mlflow.log_metrics(
                {
                    "train_loss": float(train_loss),
                    "val_loss": float(val_metrics["loss"]),
                    "val_accuracy": float(val_metrics["accuracy"]),
                    "val_precision": float(val_metrics["precision"]),
                    "val_recall": float(val_metrics["recall"]),
                    "val_f1": float(val_metrics["f1"]),
                    "val_auc": float(val_metrics["auc"]),
                    "learning_rate": float(current_lr),
                },
                step=epoch,
            )

        if epoch % 5 == 0 or epoch == 1:
            logger.info(
                f"Epoch {epoch:2d} | Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Accuracy: {val_metrics['accuracy']:.3f} |"
                f"Val Precision: {val_metrics['precision']:.3f} | "
                f"Val Recall: {val_metrics['recall']:.3f} | "
                f"Val F1: {val_metrics['f1']:.3f} | "
                f"Val AUC: {val_metrics['auc']:.3f}"
            )

        # check() both updates the best checkpoint internally AND tells us
        # whether patience has run out
        should_stop = early_stopper.check(val_metrics["loss"], model)
        if should_stop:
            logger.info(f"\nEarly stopping triggered at epoch {epoch} (best val_loss: {early_stopper.best_loss:.4f})")
            break

    # ============================================================
    # Restore the BEST weights (not necessarily the last epoch's weights)
    # ============================================================
    model.load_state_dict(early_stopper.best_model_state)
    logger.info(f"\nRestored model weights from best epoch (val_loss = {early_stopper.best_loss:.4f})")
    return model

if __name__ == "__main__":

    logger = setup_logger('classification', 'train_run')
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
    N_EPOCHS = N_EPOCHS
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
        logger.error(f"[ERROR]: There is an error while running evaluation loop!")
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
    

    


