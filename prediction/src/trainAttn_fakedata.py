import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import datetime
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import torchmetrics
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryStatScores,
    BinaryAccuracy,
    BinaryPrecision,
    BinaryRecall,
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryF1Score,
)

from torch.utils.data import TensorDataset, DataLoader
from data_generation import prepare_data
from models.self_attention_lightning import SelfAttentionLightning

from pytorch_lightning.loggers import TensorBoardLogger

os.environ["CUDA_VISIBLE_DEVICES"]=""

def main(n_features,name):
    logger_path = f"/media/dan/Data/outputs/ubiquitous-spork/attention_prediction/varying_feature_counts/logs"
    os.makedirs(logger_path, exist_ok=True)

    logger = TensorBoardLogger(save_dir=logger_path, name=name)

    # Prepare data
    train_dataset, val_dataset, test_dataset = prepare_data(n_features)

    n_features = train_dataset[0][0].shape[0]
    
    # Create data loaders with workers
    num_workers = min(63, os.cpu_count() or 1)  # Follow Lightning's suggestion but cap at CPU count
    train_loader = DataLoader(
        train_dataset, 
        batch_size=2048, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=2048, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Initialize model
    model = SelfAttentionLightning(N_value=n_features, learning_rate=1e-3)
    
    # Create unique timestamp for model saving
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # model_save_path = f"/media/dan/Data/outputs/ubiquitous-spork/attention_prediction/varying_feature_counts/attention_model_{timestamp}"

    model_save_path = f"/media/dan/Data/outputs/ubiquitous-spork/attention_prediction/varying_feature_counts/models/{n_features}/{name}/"
    os.makedirs(model_save_path, exist_ok=True)
    
    # Early stopping callback
    early_stopping = pl.callbacks.EarlyStopping(
        monitor='val/auprc',
        mode='max',
        patience=10,
        min_delta=0.001,
        verbose=True
    )
    
    # # Model checkpoint callback
    # checkpoint_callback = ModelCheckpoint(
    #     dirpath=model_save_path,
    #     filename='model-{epoch:02d}-{val/auprc:.4f}',
    #     monitor='val/auprc',
    #     mode='max',
    #     save_top_k=1,
    #     verbose=True
    # )
    
    # Progress bar callback for more verbose output
    progress_bar = pl.callbacks.TQDMProgressBar(
        refresh_rate=1
    )
    
    # Initialize trainer with verbose progress bar
    trainer = pl.Trainer(
        max_epochs=200,
        accelerator='auto',
        devices=1,
        # callbacks=[early_stopping, checkpoint_callback, progress_bar],
        callbacks=[early_stopping, progress_bar],
        enable_progress_bar=True,
        enable_model_summary=True,
        log_every_n_steps=1,
        logger=logger
    )
    
    # Train model
    trainer.fit(model, train_loader, val_loader)
    
    # Save final model
    final_model_path = os.path.join(model_save_path, f"{n_features}_features~{name}~final_model.ckpt")
    trainer.save_checkpoint(final_model_path)
    print(f"\nModel saved to:\n{final_model_path}")
    
    return model, trainer

if __name__ == "__main__":
    for n_features in range(2,6):
        for i in range(10):
            model, trainer = main(n_features=n_features,name=f"{n_features}_features~{i}")
