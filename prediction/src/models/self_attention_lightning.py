import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
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

class SelfAttentionLightning(pl.LightningModule):
    def __init__(self, N_value, projection_dim=1, learning_rate=1e-3):
        super(SelfAttentionLightning, self).__init__()
        self.save_hyperparameters()
        self.N_value = N_value
        self.learning_rate = learning_rate
        
        # Model layers
        self.attn = nn.MultiheadAttention(embed_dim=projection_dim, num_heads=1, batch_first=True)
        self.ln = nn.LayerNorm(N_value * projection_dim)
        self.fc1 = nn.Linear(N_value * projection_dim, 1)
        
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
        attn_output, attn_weights = self.attn(x, x, x)
        x = attn_output.reshape(attn_output.size(0), -1)
        x = self.ln(x)
        x = torch.sigmoid(self.fc1(x))
        return x, attn_weights

    def forward_outputs(self, x):
        with torch.no_grad():
            x = x.unsqueeze(-1)
            attn_output, attn_weights = self.attn(x, x, x)
            ln_input = attn_output.reshape(attn_output.size(0), -1)
            ln_output = self.ln(ln_input)
            fc1_output = self.fc1(ln_output)
            out = torch.sigmoid(fc1_output)
        return x, attn_output, attn_weights, ln_input, ln_output, fc1_output, out

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