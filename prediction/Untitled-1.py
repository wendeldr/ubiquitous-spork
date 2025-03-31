import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics import (
    MeanMetric,
)
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


# Self-attention model mimicking your Keras architecture.
class SelfAttentionModel(nn.Module):
    def __init__(self, N_value, projection_dim=3):
        super(SelfAttentionModel, self).__init__()
        self.N_value = N_value
        # Project each feature (token) from 1 -> 3 to mimic key_dim=3.
        self.proj_keys = nn.Linear(1, projection_dim)
        self.proj_queries = nn.Linear(1, projection_dim)
        self.proj_values = nn.Linear(1, projection_dim)
        # MultiheadAttention: embed_dim=3, one head, batch_first=True.
        self.attn = nn.MultiheadAttention(embed_dim=projection_dim, num_heads=1, batch_first=True)
        # Layer normalization after flattening the attention output.
        self.ln = nn.LayerNorm(N_value * projection_dim)
        # MLP block: first dense layer with 4 * N_value units.
        self.fc1 = nn.Linear(N_value * projection_dim, 4 * N_value)
        # Output layer with one unit.
        self.fc2 = nn.Linear(4 * N_value, 1)
    
    def forward(self, x):
        # x: (batch, N_value)
        # Reshape to (batch, N_value, 1) so each feature is a token.
        x = x.unsqueeze(-1)
        # Project tokens to dimension projection_dim.
        keys = self.proj_keys(x)
        queries = self.proj_queries(x)
        values = self.proj_values(x)
        # Self-attention (queries, keys, and values are the same).
        attn_output, attn_weights = self.attn(queries, keys, values)
        # Flatten the attention output to (batch, N_value * projection_dim).
        x = attn_output.reshape(attn_output.size(0), -1)
        # Apply layer normalization.
        x = self.ln(x)
        # MLP block with ReLU activation.
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # Sigmoid activation to get probabilities.
        x = torch.sigmoid(x)
        return x, attn_weights

# Early stopping callback implementation.
class EarlyStopping:
    def __init__(self, patience=10, verbose=False, mode='max', delta=0.001):
        self.patience = patience
        self.verbose = verbose
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.best_state = None
        self.delta = delta
    
    def __call__(self, metric, model):
        score = metric
        if self.best_score is None:
            self.best_score = score
            self.best_state = model.state_dict()
        elif (self.mode == 'max' and score < self.best_score - self.delta) or (self.mode == 'min' and score > self.best_score + self.delta):
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_state = model.state_dict()
            self.counter = 0



# Training loop using torchmetrics.
def train_model(model, train_loader, val_loader, epochs=200, lr=1e-3, device='cpu'):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()  # Binary crossentropy loss

    # Initialize torchmetrics for training
    train_loss_metric = MeanMetric().to(device)
    train_tp_fp_tn_fn_sup = BinaryStatScores().to(device)
    train_accuracy = BinaryAccuracy().to(device)
    train_precision = BinaryPrecision().to(device)
    train_recall = BinaryRecall().to(device)
    train_f1 = BinaryF1Score().to(device)
    train_auroc = BinaryAUROC().to(device)
    train_auprc = BinaryAveragePrecision().to(device)

    # Initialize torchmetrics for validation
    val_loss_metric = MeanMetric().to(device)
    val_tp_fp_tn_fn_sup = BinaryStatScores().to(device)
    val_accuracy = BinaryAccuracy().to(device)
    val_precision = BinaryPrecision().to(device)
    val_recall = BinaryRecall().to(device)
    val_f1 = BinaryF1Score().to(device)
    val_auroc = BinaryAUROC().to(device)
    val_auprc = BinaryAveragePrecision().to(device)

    # Initialize early stopping
    early_stopping = EarlyStopping(patience=10, mode='max', delta=0.001)
    
    for epoch in tqdm(range(epochs)):
        # Reset metrics at the start of each epoch
        model.train()
        train_loss_metric.reset()
        train_tp_fp_tn_fn_sup.reset()
        train_accuracy.reset()
        train_precision.reset()
        train_recall.reset()
        train_f1.reset()
        train_auroc.reset()
        train_auprc.reset()

        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device).float().squeeze()
            optimizer.zero_grad()
            outputs, _ = model(features)
            outputs = outputs.squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Update training metrics
            train_loss_metric.update(loss.item())
            train_tp_fp_tn_fn_sup.update(outputs, labels.int())
            train_accuracy.update(outputs, labels.int())
            train_precision.update(outputs, labels.int())
            train_recall.update(outputs, labels.int())
            train_f1.update(outputs, labels.int())
            train_auroc.update(outputs, labels.int())
            train_auprc.update(outputs, labels.int())
        
        # Compute average metrics for the epoch
        avg_train_loss = train_loss_metric.compute()
        avg_train_accuracy = train_accuracy.compute()
        avg_train_precision = train_precision.compute()
        avg_train_recall = train_recall.compute()
        avg_train_f1 = train_f1.compute()
        avg_train_auroc = train_auroc.compute()
        avg_train_auprc = train_auprc.compute()

        # Evaluate on validation set
        model.eval()
        val_loss_metric.reset()
        val_tp_fp_tn_fn_sup.reset()
        val_accuracy.reset()
        val_precision.reset()
        val_recall.reset()
        val_f1.reset()
        val_auroc.reset()
        val_auprc.reset()

        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(device)
                labels = labels.to(device).float().squeeze()
                outputs, _ = model(features)
                outputs = outputs.squeeze()
                loss = criterion(outputs, labels)
                
                val_loss_metric.update(loss.item())
                val_tp_fp_tn_fn_sup.update(outputs, labels.int())
                val_accuracy.update(outputs, labels.int())
                val_precision.update(outputs, labels.int())
                val_recall.update(outputs, labels.int())
                val_f1.update(outputs, labels.int())
                val_auroc.update(outputs, labels.int())
                val_auprc.update(outputs, labels.int())
        
        # Validation metrics
        avg_val_loss = val_loss_metric.compute()
        avg_val_accuracy = val_accuracy.compute()
        avg_val_precision = val_precision.compute()
        avg_val_recall = val_recall.compute()
        avg_val_f1 = val_f1.compute()
        avg_val_auroc = val_auroc.compute()
        avg_val_auprc = val_auprc.compute()

        # print val metrics
        # Early stopping check (using validation F1 score)
        early_stopping(avg_val_auprc, model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            model.load_state_dict(early_stopping.best_state)
            break
    
    return model




raw_df, initial_bias = attention_fakedata()


# Use a utility from sklearn to split and shuffle your dataset.
train_df, test_df = train_test_split(raw_df, test_size=0.2)
train_df, val_df = train_test_split(train_df, test_size=0.2)

# Form np arrays of labels and features.
train_labels = np.array(train_df.pop('label')).reshape(-1, 1)
bool_train_labels = train_labels[:, 0] != 0
val_labels = np.array(val_df.pop('label')).reshape(-1, 1)
test_labels = np.array(test_df.pop('label')).reshape(-1, 1)

train_features = np.array(train_df)
val_features = np.array(val_df)
test_features = np.array(test_df)


scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)

val_features = scaler.transform(val_features)
test_features = scaler.transform(test_features)


# train_features = np.clip(train_features, -5, 5)
# val_features = np.clip(val_features, -5, 5)
# test_features = np.clip(test_features, -5, 5)




# Convert your NumPy arrays to PyTorch tensors.
train_features_tensor = torch.tensor(train_features, dtype=torch.float32)
train_labels_tensor = torch.tensor(train_labels, dtype=torch.float32)  # shape (n, 1)

val_features_tensor = torch.tensor(val_features, dtype=torch.float32)
val_labels_tensor = torch.tensor(val_labels, dtype=torch.float32)

# Create TensorDatasets for training and validation.
train_dataset = TensorDataset(train_features_tensor, train_labels_tensor)
val_dataset = TensorDataset(val_features_tensor, val_labels_tensor)

# Define your batch size.
BATCH_SIZE = 2048

# Create DataLoaders. Note: shuffle is True for training.
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_value = train_features.shape[1]
model = SelfAttentionModel(N_value)
trained_model = train_model(model, train_loader, val_loader, epochs=200, lr=1e-3, device=device)
