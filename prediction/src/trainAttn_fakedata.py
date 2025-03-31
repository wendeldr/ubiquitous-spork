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

def attention_fakedata():
    rng = np.random.default_rng(42)
    ratio = 11.0

    # Generate synthetic data with 
    n_samples = 100000
    n_pos = int(n_samples / (ratio + 1))  # Calculate number of positive samples
    n_neg = n_samples - n_pos  # Remaining are negative samples

    # Create labels array with the correct ratio
    labels = np.concatenate([np.zeros(n_neg), np.ones(n_pos)])

    # Shuffle the labels
    rng.shuffle(labels)

    # Create a DataFrame with the synthetic data
    raw_df = pd.DataFrame({'label': labels})

    # Using boolean indexing:
    mask0 = raw_df['label'] == 0
    mask1 = raw_df['label'] == 1

    raw_df.loc[mask0, 'f1'] = rng.normal(6, 0.3, size=mask0.sum())
    raw_df.loc[mask1, 'f1'] = rng.gamma(1, .11, size=mask1.sum())


    raw_df['f2'] = rng.normal(10, 5, size=n_samples)
    # raw_df['f3'] = rng.poisson(lam=3, size=n_samples)

    # # plot features on one 1,3 plot grouped by label
    # fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # # Plot feature 1 distributions
    # sns.histplot(data=raw_df, hue='label', x='f1', ax=axes[0])
    # axes[0].set_title('Feature 1 Distribution by SOZ')
    # axes[0].set_xlabel('SOZ')
    # axes[0].set_ylabel('Value')

    # # Plot feature 2 distributions  
    # sns.histplot(data=raw_df, hue='label', x='f2', ax=axes[1])
    # axes[1].set_title('Feature 2 Distribution by SOZ')
    # axes[1].set_xlabel('SOZ')
    # axes[1].set_ylabel('Value')

    # # Plot feature 3 distributions
    # sns.histplot(data=raw_df, hue='label', x='f3', ax=axes[2])
    # axes[2].set_title('Feature 3 Distribution by SOZ')
    # axes[2].set_xlabel('SOZ')
    # axes[2].set_ylabel('Value')

    # plt.tight_layout()
    # plt.show()

    neg, pos = np.bincount(raw_df['label'])
    total = neg + pos
    print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
        total, pos, 100 * pos / total))

    initial_bias = np.log([pos/neg])

    print('data shape (+1 for label)')
    print(raw_df.shape)
    return raw_df,initial_bias

class SelfAttentionLightning(pl.LightningModule):
    def __init__(self, N_value, projection_dim=3, learning_rate=1e-3):
        super(SelfAttentionLightning, self).__init__()
        self.save_hyperparameters()
        self.N_value = N_value
        self.learning_rate = learning_rate
        
        # Model layers
        self.proj_keys = nn.Linear(1, projection_dim)
        self.proj_queries = nn.Linear(1, projection_dim)
        self.proj_values = nn.Linear(1, projection_dim)
        self.attn = nn.MultiheadAttention(embed_dim=projection_dim, num_heads=1, batch_first=True)
        self.ln = nn.LayerNorm(N_value * projection_dim)
        self.fc1 = nn.Linear(N_value * projection_dim, 4 * N_value)
        self.fc2 = nn.Linear(4 * N_value, 1)
        
        # Loss function
        self.criterion = nn.BCELoss()
        
        # Binary stat scores (handled separately as they return multiple values)
        self.train_binary_stats = BinaryStatScores()
        self.val_binary_stats = BinaryStatScores()
        
        # Create metric collections for training and validation
        metrics = MetricCollection({
            'accuracy': BinaryAccuracy(threshold=0.5),
            'precision': BinaryPrecision(threshold=0.5),
            'recall': BinaryRecall(threshold=0.5),
            'f1': BinaryF1Score(threshold=0.5),
            'auroc': BinaryAUROC(),
            'auprc': BinaryAveragePrecision()
        })
        
        # Clone the collection for training and validation with prefixes
        self.train_metrics = metrics.clone(prefix='train/')
        self.val_metrics = metrics.clone(prefix='val/')

    def forward(self, x):
        x = x.unsqueeze(-1)
        keys = self.proj_keys(x)
        queries = self.proj_queries(x)
        values = self.proj_values(x)
        attn_output, attn_weights = self.attn(queries, keys, values)
        x = attn_output.reshape(attn_output.size(0), -1)
        x = self.ln(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x, attn_weights

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        features, labels = batch
        outputs, _ = self(features)
        outputs = outputs.squeeze()
        labels = labels.squeeze()
        
        # Calculate loss
        loss = self.criterion(outputs, labels)
        
        # Update and log metrics collection
        metrics_dict = self.train_metrics(outputs, labels.int())
        self.log_dict(metrics_dict, prog_bar=True, on_step=True, on_epoch=True)
        
        # Update and log binary stats separately - convert to float32
        stats = self.train_binary_stats(outputs, labels.int())
        self.log('train/tp', stats[0].float(), prog_bar=False, on_step=True, on_epoch=True)
        self.log('train/fp', stats[1].float(), prog_bar=False, on_step=True, on_epoch=True)
        self.log('train/tn', stats[2].float(), prog_bar=False, on_step=True, on_epoch=True)
        self.log('train/fn', stats[3].float(), prog_bar=False, on_step=True, on_epoch=True)
        
        # Log loss
        self.log('train/loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        features, labels = batch
        outputs, _ = self(features)
        outputs = outputs.squeeze()
        labels = labels.squeeze()
        
        # Calculate loss
        loss = self.criterion(outputs, labels)
        
        # Update metrics collection
        metrics_dict = self.val_metrics(outputs, labels.int())
        self.log_dict(metrics_dict, prog_bar=True, on_step=True, on_epoch=True)
        
        # Update and log binary stats separately - convert to float32
        stats = self.val_binary_stats(outputs, labels.int())
        self.log('val/tp', stats[0].float(), prog_bar=False, on_step=True, on_epoch=True)
        self.log('val/fp', stats[1].float(), prog_bar=False, on_step=True, on_epoch=True)
        self.log('val/tn', stats[2].float(), prog_bar=False, on_step=True, on_epoch=True)
        self.log('val/fn', stats[3].float(), prog_bar=False, on_step=True, on_epoch=True)
        
        # Log loss
        self.log('val/loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        
        return loss

    def on_train_epoch_end(self):
        # Compute and log epoch-level metrics
        self.log_dict(self.train_metrics.compute(), prog_bar=True)
        
        # Reset all metrics
        self.train_metrics.reset()
        self.train_binary_stats.reset()

    def on_validation_epoch_end(self):
        # Compute and log epoch-level metrics
        self.log_dict(self.val_metrics.compute(), prog_bar=True)
        
        # Reset all metrics
        self.val_metrics.reset()
        self.val_binary_stats.reset()

# Data preparation and training
def prepare_data():
    raw_df, initial_bias = attention_fakedata()
    
    # Split dataset
    train_df, test_df = train_test_split(raw_df, test_size=0.2)
    train_df, val_df = train_test_split(train_df, test_size=0.2)
    
    # Prepare labels and features
    train_labels = np.array(train_df.pop('label')).reshape(-1, 1)
    val_labels = np.array(val_df.pop('label')).reshape(-1, 1)
    test_labels = np.array(test_df.pop('label')).reshape(-1, 1)
    
    train_features = np.array(train_df)
    val_features = np.array(val_df)
    test_features = np.array(test_df)
    
    # Scale features
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    val_features = scaler.transform(val_features)
    test_features = scaler.transform(test_features)
    
    # Convert to tensors
    train_features_tensor = torch.tensor(train_features, dtype=torch.float32)
    train_labels_tensor = torch.tensor(train_labels, dtype=torch.float32)
    
    val_features_tensor = torch.tensor(val_features, dtype=torch.float32)
    val_labels_tensor = torch.tensor(val_labels, dtype=torch.float32)
    
    # Create datasets
    train_dataset = TensorDataset(train_features_tensor, train_labels_tensor)
    val_dataset = TensorDataset(val_features_tensor, val_labels_tensor)
    
    return train_dataset, val_dataset, train_features.shape[1]

def main():
    # Prepare data
    train_dataset, val_dataset, n_features = prepare_data()
    
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
    model_save_path = f"/media/dan/Data/git/ubiquitous-spork/prediction/models/attention_model_{timestamp}"
    
    # Early stopping callback
    early_stopping = pl.callbacks.EarlyStopping(
        monitor='val/auprc',
        mode='max',
        patience=10,
        min_delta=0.001,
        verbose=True
    )
    
    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=model_save_path,
        filename='model-{epoch:02d}-{val/auprc:.4f}',
        monitor='val/auprc',
        mode='max',
        save_top_k=3,
        verbose=True
    )
    
    # Progress bar callback for more verbose output
    progress_bar = pl.callbacks.TQDMProgressBar(
        refresh_rate=1
    )
    
    # Initialize trainer with verbose progress bar
    trainer = pl.Trainer(
        max_epochs=200,
        accelerator='auto',
        devices=1,
        callbacks=[early_stopping, checkpoint_callback, progress_bar],
        enable_progress_bar=True,
        enable_model_summary=True,
        log_every_n_steps=1
    )
    
    # Train model
    trainer.fit(model, train_loader, val_loader)
    
    # Save final model
    final_model_path = f"{model_save_path}/final_model.ckpt"
    trainer.save_checkpoint(final_model_path)
    print(f"\nModel saved to: {final_model_path}")
    
    return model, trainer

if __name__ == "__main__":
    model, trainer = main()
